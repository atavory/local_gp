"""DFW-E103: token-level LM scale-up adaptive HPO audit.

E102 strengthened the LM section, but it is still character-level and the best
branch uses adversarial substitution-cipher shifts. E103 moves one notch closer
to the LM experiment we actually want:

* word/punctuation tokenization instead of raw character prediction;
* larger causal Transformer, recurrent, and convolutional model families;
* natural text-domain shifts before any adversarial stress setting;
* the same fixed-compute internal-vs-external HPO comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import math
import os
import platform
import re
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

import dfw_e102_lm_family_domain_hpo as e102


DATE = "20261015"
OUT_DIR = Path(os.environ.get("DFW_E103_OUT_DIR", Path(__file__).resolve().parent)).resolve()
RESULT_PATH = OUT_DIR / f"dfw_e103_token_lm_scale_hpo_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e103_token_lm_scale_hpo_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e103_token_lm_scale_hpo_LEGITIMACY_CHECK_{DATE}.md"

ROUNDS = int(os.environ.get("DFW_E103_ROUNDS", "16"))
SWITCH_ROUND = int(os.environ.get("DFW_E103_SWITCH_ROUND", str(ROUNDS // 2)))
BLOCK_SIZE = int(os.environ.get("DFW_E103_BLOCK_SIZE", "32"))
BATCH_SIZE = int(os.environ.get("DFW_E103_BATCH_SIZE", "32"))
PROBE_BATCH_SIZE = int(os.environ.get("DFW_E103_PROBE_BATCH_SIZE", "8"))
EVAL_BATCHES = int(os.environ.get("DFW_E103_EVAL_BATCHES", "2"))
BASE_LR = float(os.environ.get("DFW_E103_BASE_LR", "0.003"))
WEIGHT_DECAY = float(os.environ.get("DFW_E103_WEIGHT_DECAY", "0.01"))
MODEL_FAMILIES = tuple(
    part.strip()
    for part in os.environ.get(
        "DFW_E103_MODEL_FAMILIES",
        "token_transformer,token_gru,token_lstm,token_causal_conv",
    ).split(",")
    if part.strip()
)
DOMAIN_NAMES = tuple(
    part.strip()
    for part in os.environ.get(
        "DFW_E103_DOMAINS",
        "prose_to_code,dialogue_to_json,manual_to_sql,recipe_to_math",
    ).split(",")
    if part.strip()
)
HELDOUT_SEEDS = tuple(
    int(part)
    for part in os.environ.get(
        "DFW_E103_HELDOUT_SEEDS",
        ",".join(str(seed) for seed in range(999_400, 999_408)),
    ).split(",")
    if part
)
LR_MULTS = tuple(
    float(part)
    for part in os.environ.get("DFW_E103_LR_MULTS", "0.0,0.25,0.5,1.0,2.0,4.0").split(",")
    if part
)

TOKEN_RE = re.compile(
    r"[a-zA-Z_][a-zA-Z_0-9]*|\d+(?:\.\d+)?|==|!=|<=|>=|->|[-+*/%=<>]|[{}()[\],.:;\"']|\S"
)


@dataclass(frozen=True)
class DomainSpec:
    name: str
    source_text: str
    target_text: str
    description: str


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
        source_text=e102.PROSE,
        target_text=e102.CODE,
        description="narrative prose early stream, Python-like code target",
    ),
    "dialogue_to_json": DomainSpec(
        name="dialogue_to_json",
        source_text=e102.DIALOGUE,
        target_text=e102.JSON_LOGS,
        description="support dialogue early stream, JSON/log target",
    ),
    "manual_to_sql": DomainSpec(
        name="manual_to_sql",
        source_text=e102.MANUAL,
        target_text=e102.SQL_TEXT,
        description="procedural manual early stream, SQL target",
    ),
    "recipe_to_math": DomainSpec(
        name="recipe_to_math",
        source_text=e102.RECIPE,
        target_text=e102.MATH_TEXT,
        description="recipe prose early stream, theorem/math target",
    ),
    "cipher_to_code": DomainSpec(
        name="cipher_to_code",
        source_text=e102.cipher_text(e102.CODE, 17),
        target_text=e102.CODE,
        description="substitution-cipher code early stream, real code target",
    ),
    "cipher_to_math": DomainSpec(
        name="cipher_to_math",
        source_text=e102.cipher_text(e102.MATH_TEXT, 23),
        target_text=e102.MATH_TEXT,
        description="substitution-cipher math early stream, real theorem/math target",
    ),
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


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


def encode(tokens: list[str], stoi: dict[str, int]):
    return torch.tensor([stoi[token] for token in tokens], dtype=torch.long)


def make_splits(spec: DomainSpec, device: str) -> DataSplits:
    source_tokens = tokenize(spec.source_text)
    target_tokens = tokenize(spec.target_text)
    vocab = sorted(set(source_tokens + target_tokens))
    stoi = {token: idx for idx, token in enumerate(vocab)}
    source = encode(source_tokens, stoi).to(device)
    target = encode(target_tokens, stoi).to(device)
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


class TokenTransformerLM(nn.Module if nn is not None else object):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        width = 96
        self.token_embed = nn.Embedding(vocab_size, width)
        self.pos_embed = nn.Embedding(BLOCK_SIZE, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=288,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, idx):
        batch, seq = idx.shape
        positions = torch.arange(seq, device=idx.device).unsqueeze(0).expand(batch, seq)
        hidden = self.token_embed(idx) + self.pos_embed(positions)
        mask = torch.triu(torch.ones(seq, seq, device=idx.device, dtype=torch.bool), diagonal=1)
        hidden = self.encoder(hidden, mask=mask)
        return self.head(self.norm(hidden))


class TokenGRULM(nn.Module if nn is not None else object):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        width = 96
        hidden_width = 128
        self.token_embed = nn.Embedding(vocab_size, width)
        self.rnn = nn.GRU(width, hidden_width, num_layers=2, batch_first=True)
        self.norm = nn.LayerNorm(hidden_width)
        self.head = nn.Linear(hidden_width, vocab_size)

    def forward(self, idx):
        hidden = self.token_embed(idx)
        hidden, _state = self.rnn(hidden)
        return self.head(self.norm(hidden))


class TokenLSTMLM(nn.Module if nn is not None else object):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        width = 96
        hidden_width = 128
        self.token_embed = nn.Embedding(vocab_size, width)
        self.rnn = nn.LSTM(width, hidden_width, num_layers=2, batch_first=True)
        self.norm = nn.LayerNorm(hidden_width)
        self.head = nn.Linear(hidden_width, vocab_size)

    def forward(self, idx):
        hidden = self.token_embed(idx)
        hidden, _state = self.rnn(hidden)
        return self.head(self.norm(hidden))


class TokenCausalConvLM(nn.Module if nn is not None else object):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        width = 96
        self.token_embed = nn.Embedding(vocab_size, width)
        self.pos_embed = nn.Embedding(BLOCK_SIZE, width)
        self.conv1 = nn.Conv1d(width, width, kernel_size=3, padding=2)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, padding=2)
        self.conv3 = nn.Conv1d(width, width, kernel_size=3, padding=2)
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
        hidden = F.gelu(self.causal_conv(self.conv3, hidden))
        hidden = hidden.transpose(1, 2)
        return self.head(self.norm(hidden))


def make_model(family: str, vocab_size: int) -> nn.Module:
    if family == "token_transformer":
        return TokenTransformerLM(vocab_size)
    if family == "token_gru":
        return TokenGRULM(vocab_size)
    if family == "token_lstm":
        return TokenLSTMLM(vocab_size)
    if family == "token_causal_conv":
        return TokenCausalConvLM(vocab_size)
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
        return "TOKEN_LM_SCALE_HPO_WIN"
    if positive_ci(default) and positive_ci(shuffled) and mean(external) > 0.0:
        return "TOKEN_LM_SCALE_HPO_LIVE"
    if positive_ci(default) and positive_ci(shuffled):
        return "TOKEN_LM_SCALE_LABEL_CONTENT"
    return "TOKEN_LM_SCALE_NOT_CLEARED"


def no_torch_artifacts(detail: str) -> tuple[str, str, str]:
    run_log = "\n".join(
        (
            "DFW-E103 token-level LM scale adaptive HPO",
            f"host={platform.node()} python={sys.executable}",
            f"DECISION NO_TORCH detail={detail}",
        )
    )
    result = "# DFW-E103 Result - Token-Level LM Scale Adaptive HPO\n\nDecision label: `NO_TORCH`\n"
    legit = "# DFW-E103 Legitimacy Check\n\n- PyTorch workload: not run.\n"
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
    torch.set_num_threads(int(os.environ.get("DFW_E103_TORCH_THREADS", "1")))
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
        "DFW-E103 token-level LM scale adaptive HPO",
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
        "# DFW-E103 Result - Token-Level LM Scale Adaptive HPO",
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
            "E103 moves from character-level prediction to word/punctuation token "
            "prediction, with larger Transformer, recurrent, and convolutional "
            "causal LMs. Each cell uses the same fixed-compute protocol as E102: "
            "early rounds train on a source domain, later rounds train on "
            "target-domain tokens, and probe/select plus target-eval splits are "
            "held-out target-domain tokens."
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
    if decision == "TOKEN_LM_SCALE_HPO_WIN":
        result_lines.append(
            "The token-level LM scale audit clears the fixed-compute adaptive-HPO "
            "comparators. This is stronger than the character-level stress test, "
            "though still far below foundation-scale pretraining."
        )
    elif decision == "TOKEN_LM_SCALE_HPO_LIVE":
        result_lines.append(
            "The pooled token-level LM run is positive against external HPO but not "
            "yet broad enough cell-by-cell for the headline table."
        )
    else:
        result_lines.append(
            "The token-level LM run should be treated as a boundary result under this "
            "budget; use the slice margins to choose the next rerun."
        )
    result_lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])
    legit_lines = [
        "# DFW-E103 Legitimacy Check",
        "",
        f"- Compute: `{status}` / `{detail}`.",
        "- Models: token-level Transformer, recurrent, and causal-convolution language models.",
        "- Tokenization: deterministic word/punctuation tokenizer implemented in the script.",
        "- Training stream: source-domain tokens before the switch, target-domain tokens after the switch.",
        "- Probe/select and target streams: held-out target-domain token splits.",
        "- Same candidate-update budget: internal and external HPO both train one candidate update per LR arm per round.",
        "- Internal HPO selects one candidate next state from a shared base model each round.",
        "- External HPO keeps separate fixed-LR trajectories and selects by held-out validation loss.",
        "- Shuffled control preserves candidate updates but permutes probe-to-arm labels.",
        "- Target-eval split is used for scoring and oracle diagnostics, not for the probe controller.",
        "- This is a token-level scale-up audit, not a foundation-scale LLM pretraining run.",
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
