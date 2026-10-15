"""DFW-E102: language-model family/domain adaptive HPO audit.

E099 gave one positive tiny causal-LM domain-shift result. E102 broadens the
same fixed-compute question across multiple character-level LM model families
and multiple text-domain shifts:

* internal HPO trains LR-arm candidate next states from one shared model each
  round and uses target-domain probes for same-round one-hot selection;
* shuffled probes preserve the candidate updates but break the score-to-arm
  assignment;
* external HPO spends the same candidate-update budget on separate fixed-LR
  trajectories and selects one by target-domain validation loss.

This is still a lightweight LM audit, but it directly attacks the "one tiny
Transformer on one code shift" weakness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import math
import os
import platform
import statistics
import sys
from typing import Iterable

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None


DATE = "20261015"
OUT_DIR = Path(os.environ.get("DFW_E102_OUT_DIR", Path(__file__).resolve().parent)).resolve()
RESULT_PATH = OUT_DIR / f"dfw_e102_lm_family_domain_hpo_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e102_lm_family_domain_hpo_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e102_lm_family_domain_hpo_LEGITIMACY_CHECK_{DATE}.md"

ROUNDS = int(os.environ.get("DFW_E102_ROUNDS", "12"))
SWITCH_ROUND = int(os.environ.get("DFW_E102_SWITCH_ROUND", str(ROUNDS // 2)))
BLOCK_SIZE = int(os.environ.get("DFW_E102_BLOCK_SIZE", "40"))
BATCH_SIZE = int(os.environ.get("DFW_E102_BATCH_SIZE", "24"))
PROBE_BATCH_SIZE = int(os.environ.get("DFW_E102_PROBE_BATCH_SIZE", "12"))
EVAL_BATCHES = int(os.environ.get("DFW_E102_EVAL_BATCHES", "2"))
BASE_LR = float(os.environ.get("DFW_E102_BASE_LR", "0.004"))
WEIGHT_DECAY = float(os.environ.get("DFW_E102_WEIGHT_DECAY", "0.01"))
MODEL_FAMILIES = tuple(
    part.strip()
    for part in os.environ.get(
        "DFW_E102_MODEL_FAMILIES",
        "transformer_small,transformer_wide,gru,causal_conv",
    ).split(",")
    if part.strip()
)
DOMAIN_NAMES = tuple(
    part.strip()
    for part in os.environ.get(
        "DFW_E102_DOMAINS",
        "prose_to_code,dialogue_to_json,manual_to_sql,recipe_to_math",
    ).split(",")
    if part.strip()
)
HELDOUT_SEEDS = tuple(
    int(part)
    for part in os.environ.get(
        "DFW_E102_HELDOUT_SEEDS",
        ",".join(str(seed) for seed in range(999_300, 999_308)),
    ).split(",")
    if part
)
LR_MULTS = tuple(
    float(part)
    for part in os.environ.get("DFW_E102_LR_MULTS", "0.0,0.25,0.5,1.0,2.0,4.0").split(",")
    if part
)


PROSE = (
    """
The keeper walked along the ridge before sunrise. Below him the harbor lights
were still bright, and every mast drew a narrow black line against the water.
He wrote the weather in a ledger, folded a letter into his coat, and waited for
the first ferry to cross the bay.

In the village library, a child found an atlas with pages softened by rain. The
mountains were printed in brown, the rivers in blue, and the old railway lines
looked like threads pulled across a quilt.
"""
    * 90
)

CODE = (
    """
def update_state(step, lr, weight_decay):
    grad = read_gradient(step)
    state["weight"] = state["weight"] - lr * (grad + weight_decay * state["weight"])
    return state["weight"]

for item in batch:
    token_count += len(item["tokens"])
    if item["loss"] < best_loss:
        best_loss = item["loss"]
        best_config = {"lr": lr, "seed": seed, "round": round_id}

class Router:
    def route(self, request):
        key = request.get("domain", "default")
        return self.tables.get(key, self.tables["default"])
"""
    * 90
)

DIALOGUE = (
    """
agent: thanks for calling the service desk, what can I help with today?
user: the report opens, but the filter panel is blank after refresh.
agent: I will check the account flags and the latest deployment window.
user: please keep the export settings, because the finance team needs them.
agent: understood, I am adding that to the ticket notes now.
"""
    * 100
)

JSON_LOGS = (
    """
{"event":"login","region":"iad","latency_ms":42,"status":"ok"}
{"event":"search","region":"sfo","latency_ms":87,"status":"ok","items":14}
{"event":"checkout","region":"ams","latency_ms":133,"status":"retry","items":3}
{"event":"alert","severity":"warn","service":"router","count":2}
{"event":"batch_done","tokens":4096,"loss":2.37,"lr":0.004}
"""
    * 100
)

MANUAL = (
    """
Before replacing the filter, disconnect the power cord and wait for the lamp to
cool. Remove the rear panel with a small screwdriver. Keep the screws in order,
because the upper pair is shorter than the lower pair. After cleaning the seal,
press the cartridge into place until the latch clicks.

The device should be tested on a level table. If the indicator flashes twice,
repeat calibration and check that the inlet valve is open.
"""
    * 90
)

SQL_TEXT = (
    """
select account_id, sum(spend) as total_spend
from ad_events
where event_date between '2026-01-01' and '2026-01-31'
group by account_id
having sum(spend) > 1000
order by total_spend desc;

with ranked as (
  select user_id, item_id, score,
         row_number() over (partition by user_id order by score desc) as rank_id
  from recommendations
)
select * from ranked where rank_id <= 5;
"""
    * 90
)

RECIPE = (
    """
Warm the pan slowly and add the onions with a pinch of salt. Stir until the
edges soften, then fold in the lentils, tomatoes, and broth. Simmer until the
grain is tender. Finish with lemon, parsley, and black pepper. Serve the stew
with toasted bread and a spoon of yogurt.

For the dough, mix flour and water until smooth, rest it under a towel, then
roll thin rounds before cooking on a dry skillet.
"""
    * 90
)

MATH_TEXT = (
    """
Lemma. If a sequence a_n is bounded and monotone increasing, then it converges.
Proof. Let L be the least upper bound. For every epsilon > 0 there is an index
k with L - epsilon < a_k <= L. Since the sequence is increasing, all n >= k
satisfy L - epsilon < a_n <= L, and therefore |a_n - L| < epsilon.

For x in R and integer m >= 1, define p_m(x) = sum_{i=0}^{m} x^i / i!.
The recurrence p_{m+1}(x) = p_m(x) + x^{m+1}/(m+1)! follows immediately.
"""
    * 90
)


@dataclass(frozen=True)
class DomainSpec:
    name: str
    source_text: str
    target_text: str
    description: str


def cipher_text(text: str, shift: int) -> str:
    chars = sorted(set(text))
    mapping = {char: chars[(idx + shift) % len(chars)] for idx, char in enumerate(chars)}
    return "".join(mapping[char] for char in text)


@dataclass(frozen=True)
class DataSplits:
    source_train: object
    target_train: object
    target_select: object
    target_eval: object
    vocab_size: int


@dataclass(frozen=True)
class CellScores:
    family: str
    domain: str
    initial_loss: list[float]
    default: list[float]
    internal_probe: list[float]
    internal_shuffled: list[float]
    internal_oracle: list[float]
    external_valid: list[float]
    external_oracle: list[float]


DOMAINS = {
    "prose_to_code": DomainSpec(
        name="prose_to_code",
        source_text=PROSE,
        target_text=CODE,
        description="narrative prose early stream, Python-like code target",
    ),
    "dialogue_to_json": DomainSpec(
        name="dialogue_to_json",
        source_text=DIALOGUE,
        target_text=JSON_LOGS,
        description="support dialogue early stream, JSON/log target",
    ),
    "manual_to_sql": DomainSpec(
        name="manual_to_sql",
        source_text=MANUAL,
        target_text=SQL_TEXT,
        description="procedural manual early stream, SQL target",
    ),
    "recipe_to_math": DomainSpec(
        name="recipe_to_math",
        source_text=RECIPE,
        target_text=MATH_TEXT,
        description="recipe prose early stream, theorem/math target",
    ),
    "cipher_to_code": DomainSpec(
        name="cipher_to_code",
        source_text=cipher_text(CODE, 17),
        target_text=CODE,
        description="substitution-cipher code early stream, real code target",
    ),
    "cipher_to_json": DomainSpec(
        name="cipher_to_json",
        source_text=cipher_text(JSON_LOGS, 13),
        target_text=JSON_LOGS,
        description="substitution-cipher JSON early stream, real JSON/log target",
    ),
    "cipher_to_sql": DomainSpec(
        name="cipher_to_sql",
        source_text=cipher_text(SQL_TEXT, 19),
        target_text=SQL_TEXT,
        description="substitution-cipher SQL early stream, real SQL target",
    ),
    "cipher_to_math": DomainSpec(
        name="cipher_to_math",
        source_text=cipher_text(MATH_TEXT, 23),
        target_text=MATH_TEXT,
        description="substitution-cipher math early stream, real theorem/math target",
    ),
}


def mean(vals: Iterable[float]) -> float:
    rows = tuple(vals)
    return sum(rows) / len(rows) if rows else 0.0


def ci95(vals: Iterable[float]) -> float:
    rows = tuple(vals)
    if len(rows) < 2:
        return 0.0
    return 1.96 * statistics.stdev(rows) / math.sqrt(len(rows))


def fmt_ci(vals: Iterable[float]) -> str:
    rows = tuple(vals)
    return f"{mean(rows):7.3f}+/-{ci95(rows):.3f}"


def paired(lhs: list[float], rhs: list[float]) -> list[float]:
    return [a - b for a, b in zip(lhs, rhs)]


def positive_ci(vals: Iterable[float]) -> bool:
    rows = tuple(vals)
    return bool(rows) and mean(rows) - ci95(rows) > 0.0


def compute_status() -> tuple[str, str, str]:
    if torch is None:
        return "NO_TORCH", "PyTorch import failed", "cpu"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        detail = f"torch={torch.__version__} cuda_available=True device={device_name}"
        return "REAL_GPU", detail, "cuda"
    return "TORCH_CPU", f"torch={torch.__version__} cuda_available=False", "cpu"


def encode(text: str, stoi: dict[str, int]):
    return torch.tensor([stoi[char] for char in text], dtype=torch.long)


def make_splits(spec: DomainSpec, device: str) -> DataSplits:
    chars = sorted(set(spec.source_text + spec.target_text))
    stoi = {char: idx for idx, char in enumerate(chars)}
    source = encode(spec.source_text, stoi).to(device)
    target = encode(spec.target_text, stoi).to(device)
    train_end = int(0.68 * len(target))
    select_end = int(0.84 * len(target))
    return DataSplits(
        source_train=source,
        target_train=target[:train_end],
        target_select=target[train_end:select_end],
        target_eval=target[select_end:],
        vocab_size=len(stoi),
    )


def make_batch(data, batch_size: int, block_size: int, seed: int, device: str) -> tuple[object, object]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,), generator=gen)
    x_rows = [data[start : start + block_size] for start in starts]
    y_rows = [data[start + 1 : start + block_size + 1] for start in starts]
    return torch.stack(x_rows).to(device), torch.stack(y_rows).to(device)


class TransformerLM(nn.Module if nn is not None else object):
    def __init__(self, vocab_size: int, width: int, layers: int) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, width)
        self.pos_embed = nn.Embedding(BLOCK_SIZE, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 3,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, idx):
        batch, seq = idx.shape
        positions = torch.arange(seq, device=idx.device).unsqueeze(0).expand(batch, seq)
        hidden = self.token_embed(idx) + self.pos_embed(positions)
        mask = torch.triu(torch.ones(seq, seq, device=idx.device, dtype=torch.bool), diagonal=1)
        hidden = self.encoder(hidden, mask=mask)
        return self.head(self.norm(hidden))


class GRULM(nn.Module if nn is not None else object):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        width = 56
        self.token_embed = nn.Embedding(vocab_size, width)
        self.rnn = nn.GRU(width, 72, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(72)
        self.head = nn.Linear(72, vocab_size)

    def forward(self, idx):
        hidden = self.token_embed(idx)
        hidden, _state = self.rnn(hidden)
        return self.head(self.norm(hidden))


class CausalConvLM(nn.Module if nn is not None else object):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        width = 56
        self.token_embed = nn.Embedding(vocab_size, width)
        self.pos_embed = nn.Embedding(BLOCK_SIZE, width)
        self.conv1 = nn.Conv1d(width, width, kernel_size=3, padding=2)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, padding=2)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def causal_conv(self, conv: nn.Conv1d, hidden):
        seq = hidden.shape[-1]
        return conv(hidden)[..., :seq]

    def forward(self, idx):
        batch, seq = idx.shape
        positions = torch.arange(seq, device=idx.device).unsqueeze(0).expand(batch, seq)
        hidden = self.token_embed(idx) + self.pos_embed(positions)
        hidden = hidden.transpose(1, 2)
        hidden = F.gelu(self.causal_conv(self.conv1, hidden))
        hidden = F.gelu(self.causal_conv(self.conv2, hidden))
        hidden = hidden.transpose(1, 2)
        return self.head(self.norm(hidden))


def make_model(family: str, vocab_size: int) -> nn.Module:
    if family == "transformer_small":
        return TransformerLM(vocab_size, width=48, layers=1)
    if family == "transformer_wide":
        return TransformerLM(vocab_size, width=72, layers=2)
    if family == "gru":
        return GRULM(vocab_size)
    if family == "causal_conv":
        return CausalConvLM(vocab_size)
    raise ValueError(f"unknown model family: {family}")


def loss_on_batch(model: nn.Module, x, y) -> object:
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def clone_state(model: nn.Module) -> dict[str, object]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def load_state(model: nn.Module, state: dict[str, object]) -> None:
    model.load_state_dict(state)


def eval_loss(model: nn.Module, data, seed: int, batch_size: int, device: str) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for idx in range(EVAL_BATCHES):
            x, y = make_batch(data, batch_size, BLOCK_SIZE, seed + 10_000 * idx, device)
            losses.append(float(loss_on_batch(model, x, y).detach().cpu()))
    return mean(losses)


def state_score(
    model: nn.Module,
    state: dict[str, object],
    data,
    seed: int,
    batch_size: int,
    device: str,
) -> float:
    load_state(model, state)
    return eval_loss(model, data, seed, batch_size, device)


def train_worker_candidate(
    worker: nn.Module,
    opt: object,
    state: dict[str, object],
    batch,
) -> dict[str, object]:
    load_state(worker, state)
    worker.train()
    x, y = batch
    opt.zero_grad(set_to_none=True)
    loss = loss_on_batch(worker, x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(worker.parameters(), 1.0)
    opt.step()
    return clone_state(worker)


def argmin_weights(losses: list[float]) -> list[float]:
    best_idx = min(range(len(losses)), key=lambda idx: losses[idx])
    return [1.0 if idx == best_idx else 0.0 for idx in range(len(losses))]


def aggregate_states(
    base_state: dict[str, object],
    candidate_states: list[dict[str, object]],
    weights: list[float],
) -> dict[str, object]:
    out = {}
    for key, base_value in base_state.items():
        if not torch.is_floating_point(base_value):
            out[key] = base_value.clone()
            continue
        agg = torch.zeros_like(base_value)
        for weight, state in zip(weights, candidate_states):
            agg = agg + weight * state[key]
        out[key] = agg
    return out


def train_data_for_round(splits: DataSplits, round_idx: int):
    return splits.source_train if round_idx < SWITCH_ROUND else splits.target_train


def run_internal(
    seed: int,
    family: str,
    splits: DataSplits,
    arms: tuple[float, ...],
    device: str,
    mode: str,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    model = make_model(family, splits.vocab_size).to(device)
    workers = [copy.deepcopy(model).to(device) for _arm in arms]
    opts = [
        torch.optim.AdamW(worker.parameters(), lr=BASE_LR * arm, weight_decay=WEIGHT_DECAY)
        for worker, arm in zip(workers, arms)
    ]
    initial = eval_loss(model, splits.target_eval, seed + 900_000, PROBE_BATCH_SIZE, device)
    for round_idx in range(ROUNDS):
        base_state = clone_state(model)
        batch = make_batch(
            train_data_for_round(splits, round_idx),
            BATCH_SIZE,
            BLOCK_SIZE,
            seed + 1000 * round_idx,
            device,
        )
        candidate_states = [
            train_worker_candidate(worker, opt, base_state, batch)
            for worker, opt in zip(workers, opts)
        ]
        score_data = splits.target_eval if mode == "oracle" else splits.target_select
        losses = [
            state_score(
                model,
                state,
                score_data,
                seed + 200_000 + 31 * round_idx,
                PROBE_BATCH_SIZE,
                device,
            )
            for state in candidate_states
        ]
        if mode == "shuffled":
            shift = (seed + round_idx) % len(losses)
            losses = losses[shift:] + losses[:shift]
        load_state(model, aggregate_states(base_state, candidate_states, argmin_weights(losses)))
    final = eval_loss(model, splits.target_eval, seed + 901_000, PROBE_BATCH_SIZE, device)
    return initial, initial - final


def run_fixed_arm(
    seed: int,
    family: str,
    splits: DataSplits,
    arm: float,
    device: str,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    model = make_model(family, splits.vocab_size).to(device)
    initial = eval_loss(model, splits.target_eval, seed + 900_000, PROBE_BATCH_SIZE, device)
    opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR * arm, weight_decay=WEIGHT_DECAY)
    for round_idx in range(ROUNDS):
        x, y = make_batch(
            train_data_for_round(splits, round_idx),
            BATCH_SIZE,
            BLOCK_SIZE,
            seed + 1000 * round_idx,
            device,
        )
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_on_batch(model, x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    select = eval_loss(model, splits.target_select, seed + 902_000, PROBE_BATCH_SIZE, device)
    target = eval_loss(model, splits.target_eval, seed + 901_000, PROBE_BATCH_SIZE, device)
    return initial - select, initial - target


def run_external(
    seed: int,
    family: str,
    splits: DataSplits,
    arms: tuple[float, ...],
    device: str,
) -> tuple[float, float, float]:
    scores = [run_fixed_arm(seed, family, splits, arm, device) for arm in arms]
    valid_idx = max(range(len(scores)), key=lambda idx: scores[idx][0])
    oracle_idx = max(range(len(scores)), key=lambda idx: scores[idx][1])
    default_idx = min(range(len(arms)), key=lambda idx: abs(arms[idx] - 1.0))
    return scores[default_idx][1], scores[valid_idx][1], scores[oracle_idx][1]


def diagnose_cell(family: str, domain: DomainSpec, device: str) -> CellScores:
    splits = make_splits(domain, device)
    initial_loss: list[float] = []
    default: list[float] = []
    internal_probe: list[float] = []
    internal_shuffled: list[float] = []
    internal_oracle: list[float] = []
    external_valid: list[float] = []
    external_oracle: list[float] = []
    for idx, seed in enumerate(HELDOUT_SEEDS):
        initial, probe_score = run_internal(seed, family, splits, LR_MULTS, device, "probe")
        _initial_s, shuffled_score = run_internal(seed, family, splits, LR_MULTS, device, "shuffled")
        _initial_o, oracle_score = run_internal(seed, family, splits, LR_MULTS, device, "oracle")
        default_score, valid_score, oracle_ext_score = run_external(seed, family, splits, LR_MULTS, device)
        initial_loss.append(initial)
        default.append(default_score)
        internal_probe.append(probe_score)
        internal_shuffled.append(shuffled_score)
        internal_oracle.append(oracle_score)
        external_valid.append(valid_score)
        external_oracle.append(oracle_ext_score)
        print(
            f"CELL family={family} domain={domain.name} rep={idx:02d} seed={seed} "
            f"default={default_score:.4f} internal={probe_score:.4f} "
            f"shuffled={shuffled_score:.4f} external_valid={valid_score:.4f} "
            f"external_oracle={oracle_ext_score:.4f}",
            flush=True,
        )
    return CellScores(
        family=family,
        domain=domain.name,
        initial_loss=initial_loss,
        default=default,
        internal_probe=internal_probe,
        internal_shuffled=internal_shuffled,
        internal_oracle=internal_oracle,
        external_valid=external_valid,
        external_oracle=external_oracle,
    )


def flatten(cells: list[CellScores], attr: str) -> list[float]:
    out: list[float] = []
    for cell in cells:
        out.extend(getattr(cell, attr))
    return out


def subset(cells: list[CellScores], *, family: str | None = None, domain: str | None = None) -> list[CellScores]:
    return [
        cell
        for cell in cells
        if (family is None or cell.family == family) and (domain is None or cell.domain == domain)
    ]


def margins(cells: list[CellScores]) -> tuple[list[float], list[float], list[float], list[float]]:
    internal = flatten(cells, "internal_probe")
    default = flatten(cells, "default")
    shuffled = flatten(cells, "internal_shuffled")
    external = flatten(cells, "external_valid")
    oracle = flatten(cells, "external_oracle")
    return paired(internal, default), paired(internal, shuffled), paired(internal, external), paired(internal, oracle)


def cell_clears(cell: CellScores) -> bool:
    _d, shuffled, external, _o = margins([cell])
    return positive_ci(shuffled) and positive_ci(external)


def decide(cells: list[CellScores], status: str) -> str:
    if status == "NO_TORCH":
        return "NO_TORCH"
    default, shuffled, external, _oracle = margins(cells)
    clear_count = sum(1 for cell in cells if cell_clears(cell))
    if positive_ci(default) and positive_ci(shuffled) and positive_ci(external) and clear_count >= math.ceil(0.6 * len(cells)):
        return "LM_FAMILY_DOMAIN_HPO_WIN"
    if positive_ci(default) and positive_ci(shuffled) and mean(external) > 0.0:
        return "LM_FAMILY_DOMAIN_HPO_LIVE"
    if positive_ci(default) and positive_ci(shuffled):
        return "LM_FAMILY_DOMAIN_LABEL_CONTENT"
    return "LM_FAMILY_DOMAIN_NOT_CLEARED"


def no_torch_artifacts(detail: str) -> tuple[str, str, str]:
    run_log = "\n".join(
        (
            "DFW-E102 LM family/domain adaptive HPO",
            f"host={platform.node()} python={sys.executable}",
            f"DECISION NO_TORCH detail={detail}",
        )
    )
    result = "# DFW-E102 Result - LM Family/Domain Adaptive HPO\n\nDecision label: `NO_TORCH`\n"
    legit = "# DFW-E102 Legitimacy Check\n\n- PyTorch workload: not run.\n"
    return run_log + "\n", result + "\n", legit + "\n"


def format_summary_rows(cells: list[CellScores]) -> list[str]:
    rows = [
        "| slice | internal-default | internal-shuffled | internal-external valid | internal-external oracle | cleared cells |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    default, shuffled, external, oracle = margins(cells)
    rows.append(
        f"| pooled | `{fmt_ci(default).strip()}` | `{fmt_ci(shuffled).strip()}` | "
        f"`{fmt_ci(external).strip()}` | `{fmt_ci(oracle).strip()}` | "
        f"`{sum(1 for cell in cells if cell_clears(cell))}/{len(cells)}` |"
    )
    for family in MODEL_FAMILIES:
        group = subset(cells, family=family)
        d, s, e, o = margins(group)
        rows.append(
            f"| family={family} | `{fmt_ci(d).strip()}` | `{fmt_ci(s).strip()}` | "
            f"`{fmt_ci(e).strip()}` | `{fmt_ci(o).strip()}` | "
            f"`{sum(1 for cell in group if cell_clears(cell))}/{len(group)}` |"
        )
    for domain in DOMAIN_NAMES:
        group = subset(cells, domain=domain)
        d, s, e, o = margins(group)
        rows.append(
            f"| domain={domain} | `{fmt_ci(d).strip()}` | `{fmt_ci(s).strip()}` | "
            f"`{fmt_ci(e).strip()}` | `{fmt_ci(o).strip()}` | "
            f"`{sum(1 for cell in group if cell_clears(cell))}/{len(group)}` |"
        )
    return rows


def format_cell_rows(cells: list[CellScores]) -> list[str]:
    rows = [
        "| family | domain | internal-default | internal-shuffled | internal-external valid | clear? |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cell in cells:
        d, s, e, _o = margins([cell])
        rows.append(
            f"| {cell.family} | {cell.domain} | `{fmt_ci(d).strip()}` | "
            f"`{fmt_ci(s).strip()}` | `{fmt_ci(e).strip()}` | "
            f"`{'yes' if cell_clears(cell) else 'no'}` |"
        )
    return rows


def run_experiment() -> tuple[str, str, str]:
    status, detail, device = compute_status()
    if status == "NO_TORCH":
        return no_torch_artifacts(detail)
    torch.set_num_threads(int(os.environ.get("DFW_E102_TORCH_THREADS", "1")))
    missing_domains = [name for name in DOMAIN_NAMES if name not in DOMAINS]
    if missing_domains:
        raise ValueError(f"unknown domains: {missing_domains}")
    cells: list[CellScores] = []
    for family in MODEL_FAMILIES:
        for domain_name in DOMAIN_NAMES:
            cells.append(diagnose_cell(family, DOMAINS[domain_name], device))
    decision = decide(cells, status)
    default, shuffled, external, oracle = margins(cells)
    arms = ", ".join(f"{mult:g}" for mult in LR_MULTS)
    probe_cost = 100.0 * PROBE_BATCH_SIZE / (3.0 * BATCH_SIZE)
    run_lines = [
        "DFW-E102 LM family/domain adaptive HPO",
        f"host={platform.node()} python={sys.executable} compute_status={status} {detail}",
        (
            f"rounds={ROUNDS} switch_round={SWITCH_ROUND} block={BLOCK_SIZE} "
            f"batch={BATCH_SIZE} probe_batch={PROBE_BATCH_SIZE} eval_batches={EVAL_BATCHES}"
        ),
        (
            f"base_lr={BASE_LR} weight_decay={WEIGHT_DECAY} lr_mults=({arms}) "
            f"families=({', '.join(MODEL_FAMILIES)}) domains=({', '.join(DOMAIN_NAMES)}) "
            f"seeds={len(HELDOUT_SEEDS)}"
        ),
        f"POOLED internal-default={fmt_ci(default)} "
        f"internal-shuffled={fmt_ci(shuffled)} "
        f"internal-external_valid={fmt_ci(external)} "
        f"internal-external_oracle={fmt_ci(oracle)}",
        f"clear_cells={sum(1 for cell in cells if cell_clears(cell))}/{len(cells)}",
        f"DECISION {decision}",
        f"compute_status={status}",
    ]
    result_lines = [
        "# DFW-E102 Result - LM Family/Domain Adaptive HPO",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{status}`: `{detail}`.",
        "",
        "## Design",
        "",
        (
            "E102 broadens E099 from one tiny Transformer/code-shift run to a "
            "family/domain sweep. Each cell trains a character-level causal LM "
            "whose early stream is off-target and whose later stream, probe/select "
            "split, and target split are target-domain text. Internal HPO uses "
            "one-hot same-round probe selection over LR-arm candidate next states. "
            "External HPO trains separate fixed-LR trajectories and selects by "
            "target-domain validation loss under the same candidate-update budget."
        ),
        (
            "The `cipher_*` domains are adversarial textual shifts: the early stream "
            "has the target alphabet but a substitution-cipher transition structure. "
            "They are stress tests for whether same-round probes can suppress harmful "
            "early LM updates and exploit later target-domain rounds."
        ),
        "",
        f"- model families: `{', '.join(MODEL_FAMILIES)}`",
        f"- domain shifts: `{', '.join(DOMAIN_NAMES)}`",
        f"- held-out seeds per cell: `{len(HELDOUT_SEEDS)}`",
        f"- LR arms: `({arms})`",
        f"- probe overhead estimate: `{probe_cost:.2f}%`",
        "",
        "## Pooled and Slice Margins",
        "",
        *format_summary_rows(cells),
        "",
        "## Cell Margins",
        "",
        *format_cell_rows(cells),
        "",
        "## Interpretation",
        "",
    ]
    if decision == "LM_FAMILY_DOMAIN_HPO_WIN":
        result_lines.append(
            "The LM-family/domain audit clears the main fixed-compute adaptive-HPO "
            "comparators: same-round internal probe selection beats default, "
            "shuffled probes, and validation-selected external fixed-LR HPO in "
            "the pooled result, with most model/domain cells clearing individually."
        )
    elif decision == "LM_FAMILY_DOMAIN_HPO_LIVE":
        result_lines.append(
            "The pooled LM-family/domain sweep is positive against external HPO but "
            "not yet strong enough cell-by-cell to be a headline result."
        )
    else:
        result_lines.append(
            "The LM-family/domain sweep should be treated as a boundary result under "
            "this budget; use the slice margins to choose the next rerun."
        )
    result_lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])
    legit_lines = [
        "# DFW-E102 Legitimacy Check",
        "",
        f"- Compute: `{status}` / `{detail}`.",
        "- Models: character-level causal LMs from transformer, recurrent, and causal-convolution families.",
        "- Training stream: off-target text before the switch, target-domain text after the switch.",
        "- Probe/select and target streams: held-out target-domain text split disjointly.",
        "- Same candidate-update budget: internal and external HPO both train one candidate update per LR arm per round.",
        "- Internal HPO selects one candidate next state from a shared base model each round.",
        "- External HPO keeps separate fixed-LR trajectories and selects by held-out validation loss.",
        "- Shuffled control preserves candidate updates but permutes probe-to-arm labels.",
        "- Target-eval split is used for scoring and oracle diagnostics, not for the probe controller.",
        "- This is a lightweight LM-family audit, not a full-scale LLM pretraining run.",
        "",
    ]
    return "\n".join(run_lines) + "\n", "\n".join(result_lines) + "\n", "\n".join(legit_lines) + "\n"


def main() -> None:
    run_log, result, legit = run_experiment()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_PATH.write_text(run_log)
    RESULT_PATH.write_text(result)
    LEGITIMACY_PATH.write_text(legit)
    print(result)
    print(f"wrote {RESULT_PATH}")
    print(f"wrote {RUN_LOG_PATH}")
    print(f"wrote {LEGITIMACY_PATH}")


if __name__ == "__main__":
    main()
