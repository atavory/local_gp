"""DFW-E096: tiny causal-LM fixed-compute HPO probe.

This is a lightweight language-model tuning audit for the TMLR packet.  It uses
an embedded character-level causal language-model corpus and a tiny Transformer
so the run is reproducible inside the Buck PyTorch environment.

The comparison mirrors the main fixed-compute HPO claim:

* internal HPO trains several LR-arm candidate next states from the same base
  model, scores them on held-out probe text, and aggregates the candidate
  parameter deltas back into one shared LM trajectory;
* external HPO spends the same candidate-update budget on separate fixed-LR
  trajectories and selects one by held-out validation loss;
* shuffled probes preserve the candidate updates but break the probe-to-arm
  assignment.
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


DATE = "20261014"
OUT_DIR = Path(os.environ.get("DFW_E096_OUT_DIR", Path(__file__).resolve().parent)).resolve()
RESULT_PATH = OUT_DIR / f"dfw_e096_tiny_lm_probe_hpo_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e096_tiny_lm_probe_hpo_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e096_tiny_lm_probe_hpo_LEGITIMACY_CHECK_{DATE}.md"

ROUNDS = int(os.environ.get("DFW_E096_ROUNDS", "20"))
BLOCK_SIZE = int(os.environ.get("DFW_E096_BLOCK_SIZE", "32"))
BATCH_SIZE = int(os.environ.get("DFW_E096_BATCH_SIZE", "16"))
PROBE_BATCH_SIZE = int(os.environ.get("DFW_E096_PROBE_BATCH_SIZE", "32"))
EVAL_BATCHES = int(os.environ.get("DFW_E096_EVAL_BATCHES", "2"))
BASE_LR = float(os.environ.get("DFW_E096_BASE_LR", "0.003"))
WEIGHT_DECAY = float(os.environ.get("DFW_E096_WEIGHT_DECAY", "0.01"))
TAU_SCALE = float(os.environ.get("DFW_E096_TAU_SCALE", "0.5"))
POLICY = os.environ.get("DFW_E096_POLICY", "soft")
HELDOUT_SEEDS = tuple(
    int(part)
    for part in os.environ.get(
        "DFW_E096_HELDOUT_SEEDS",
        ",".join(str(seed) for seed in range(999_000, 999_016)),
    ).split(",")
    if part
)
LR_MULTS = tuple(
    float(part)
    for part in os.environ.get("DFW_E096_LR_MULTS", "0.5,1.0,2.0,4.0").split(",")
    if part
)
EPS = 1e-12

CORPUS = (
    """
Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do. Once or twice she had peeped into the book her
sister was reading, but it had no pictures or conversations in it, and what is
the use of a book, thought Alice, without pictures or conversation?

The rabbit started when it heard her voice, and then hurried on. Alice followed
it across the field, down a long passage, and into a room where every door was
locked. She tried the tiny key in the golden lock, and the door opened upon a
garden bright with flowers.

In another part of the story, a traveler carried a notebook through a cold
morning. The road bent by the river, the mill wheel turned slowly, and every
house sent a narrow thread of smoke into the clean air.

The old captain told the children that maps are promises, not answers. A map
can mark a harbor, a reef, or a shoal, but the sailor still has to read the
wind and the tide before choosing a course.

Language models learn by predicting the next sign from a stream of signs. A
small model sees only a little context, yet even a small context contains
syntax, rhythm, spelling, and the repeated habits of a text.
"""
    * 80
)


@dataclass(frozen=True)
class Arm:
    name: str
    lr_mult: float


@dataclass(frozen=True)
class Scores:
    default: list[float]
    internal_probe: list[float]
    internal_shuffled: list[float]
    internal_oracle: list[float]
    external_valid: list[float]
    external_oracle: list[float]
    probe_cost_pct: float
    initial_loss: list[float]


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


def build_vocab() -> tuple[dict[str, int], list[str]]:
    chars = sorted(set(CORPUS))
    return {char: idx for idx, char in enumerate(chars)}, chars


def encode_data() -> tuple[object, object, object, dict[str, int]]:
    stoi, _itos = build_vocab()
    ids = torch.tensor([stoi[char] for char in CORPUS], dtype=torch.long)
    train_end = int(0.72 * len(ids))
    select_end = int(0.86 * len(ids))
    return ids[:train_end], ids[train_end:select_end], ids[select_end:], stoi


def make_batch(data, batch_size: int, block_size: int, seed: int, device: str) -> tuple[object, object]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,), generator=gen)
    x_rows = [data[start : start + block_size] for start in starts]
    y_rows = [data[start + 1 : start + block_size + 1] for start in starts]
    return torch.stack(x_rows).to(device), torch.stack(y_rows).to(device)


class TinyCausalLM(nn.Module if nn is not None else object):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        width = 48
        self.token_embed = nn.Embedding(vocab_size, width)
        self.pos_embed = nn.Embedding(BLOCK_SIZE, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, idx):
        batch, seq = idx.shape
        positions = torch.arange(seq, device=idx.device).unsqueeze(0).expand(batch, seq)
        hidden = self.token_embed(idx) + self.pos_embed(positions)
        mask = torch.triu(torch.ones(seq, seq, device=idx.device, dtype=torch.bool), diagonal=1)
        hidden = self.encoder(hidden, mask=mask)
        return self.head(self.norm(hidden))


def loss_on_batch(model: nn.Module, x, y) -> object:
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def clone_state(model: nn.Module) -> dict[str, object]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def load_state(model: nn.Module, state: dict[str, object]) -> None:
    model.load_state_dict(state)


def train_candidate(model: nn.Module, state: dict[str, object], arm: Arm, batch, device: str) -> dict[str, object]:
    candidate = copy.deepcopy(model)
    candidate.to(device)
    load_state(candidate, state)
    candidate.train()
    opt = torch.optim.AdamW(candidate.parameters(), lr=BASE_LR * arm.lr_mult, weight_decay=WEIGHT_DECAY)
    x, y = batch
    opt.zero_grad(set_to_none=True)
    loss = loss_on_batch(candidate, x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
    opt.step()
    return clone_state(candidate)


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


def weights_from_losses(losses: list[float]) -> list[float]:
    if POLICY == "argmin":
        best_idx = min(range(len(losses)), key=lambda idx: losses[idx])
        return [1.0 if idx == best_idx else 0.0 for idx in range(len(losses))]
    center = mean(losses)
    spread = statistics.pstdev(losses) if len(losses) > 1 else 0.0
    tau = max(TAU_SCALE * spread, 0.005)
    raw = [math.exp(-(loss - center) / tau) for loss in losses]
    total = sum(raw)
    return [val / total for val in raw]


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


def run_internal(
    seed: int,
    arms: tuple[Arm, ...],
    train_data,
    select_data,
    target_data,
    vocab_size: int,
    device: str,
    mode: str,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    model = TinyCausalLM(vocab_size).to(device)
    workers = [copy.deepcopy(model).to(device) for _arm in arms]
    opts = [
        torch.optim.AdamW(worker.parameters(), lr=BASE_LR * arm.lr_mult, weight_decay=WEIGHT_DECAY)
        for worker, arm in zip(workers, arms)
    ]
    initial = eval_loss(model, target_data, seed + 900_000, PROBE_BATCH_SIZE, device)
    for round_idx in range(ROUNDS):
        base_state = clone_state(model)
        batch = make_batch(train_data, BATCH_SIZE, BLOCK_SIZE, seed + 1000 * round_idx, device)
        candidate_states = [
            train_worker_candidate(worker, opt, base_state, batch) for worker, opt in zip(workers, opts)
        ]
        select_seed = seed + 200_000 + round_idx
        target_seed = seed + 300_000 + round_idx
        if mode == "probe":
            losses = [
                state_score(model, state, select_data, select_seed, PROBE_BATCH_SIZE, device)
                for state in candidate_states
            ]
        elif mode == "shuffled":
            losses = [
                state_score(model, state, select_data, select_seed, PROBE_BATCH_SIZE, device)
                for state in candidate_states
            ]
            shift = (seed + round_idx) % len(losses)
            losses = losses[shift:] + losses[:shift]
        elif mode == "oracle":
            losses = [
                state_score(model, state, target_data, target_seed, PROBE_BATCH_SIZE, device)
                for state in candidate_states
            ]
        else:
            raise ValueError(mode)
        load_state(model, aggregate_states(base_state, candidate_states, weights_from_losses(losses)))
    final = eval_loss(model, target_data, seed + 901_000, PROBE_BATCH_SIZE, device)
    return initial, initial - final


def run_fixed_arm(
    seed: int,
    arm: Arm,
    train_data,
    select_data,
    target_data,
    vocab_size: int,
    device: str,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    model = TinyCausalLM(vocab_size).to(device)
    initial = eval_loss(model, target_data, seed + 900_000, PROBE_BATCH_SIZE, device)
    opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR * arm.lr_mult, weight_decay=WEIGHT_DECAY)
    for round_idx in range(ROUNDS):
        x, y = make_batch(train_data, BATCH_SIZE, BLOCK_SIZE, seed + 1000 * round_idx, device)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_on_batch(model, x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    select = eval_loss(model, select_data, seed + 902_000, PROBE_BATCH_SIZE, device)
    target = eval_loss(model, target_data, seed + 901_000, PROBE_BATCH_SIZE, device)
    return initial - select, initial - target


def run_external(
    seed: int,
    arms: tuple[Arm, ...],
    train_data,
    select_data,
    target_data,
    vocab_size: int,
    device: str,
) -> tuple[float, float, float]:
    scores = [run_fixed_arm(seed, arm, train_data, select_data, target_data, vocab_size, device) for arm in arms]
    valid_idx = max(range(len(scores)), key=lambda idx: scores[idx][0])
    oracle_idx = max(range(len(scores)), key=lambda idx: scores[idx][1])
    default_idx = min(range(len(arms)), key=lambda idx: abs(arms[idx].lr_mult - 1.0))
    return scores[default_idx][1], scores[valid_idx][1], scores[oracle_idx][1]


def diagnose(device: str) -> Scores:
    train_data, select_data, target_data, stoi = encode_data()
    train_data = train_data.to(device)
    select_data = select_data.to(device)
    target_data = target_data.to(device)
    arms = tuple(Arm(f"lr_x{mult:g}", mult) for mult in LR_MULTS)
    default: list[float] = []
    internal_probe: list[float] = []
    internal_shuffled: list[float] = []
    internal_oracle: list[float] = []
    external_valid: list[float] = []
    external_oracle: list[float] = []
    initial_loss: list[float] = []
    for idx, seed in enumerate(HELDOUT_SEEDS):
        initial, probe_score = run_internal(seed, arms, train_data, select_data, target_data, len(stoi), device, "probe")
        _initial_s, shuffled_score = run_internal(
            seed,
            arms,
            train_data,
            select_data,
            target_data,
            len(stoi),
            device,
            "shuffled",
        )
        _initial_o, oracle_score = run_internal(
            seed,
            arms,
            train_data,
            select_data,
            target_data,
            len(stoi),
            device,
            "oracle",
        )
        default_score, valid_score, external_oracle_score = run_external(
            seed,
            arms,
            train_data,
            select_data,
            target_data,
            len(stoi),
            device,
        )
        initial_loss.append(initial)
        internal_probe.append(probe_score)
        internal_shuffled.append(shuffled_score)
        internal_oracle.append(oracle_score)
        default.append(default_score)
        external_valid.append(valid_score)
        external_oracle.append(external_oracle_score)
        print(
            f"REP {idx:02d} seed={seed} "
            f"default={default_score:.4f} internal={probe_score:.4f} "
            f"shuffled={shuffled_score:.4f} external_valid={valid_score:.4f} "
            f"external_oracle={external_oracle_score:.4f}",
            flush=True,
        )
    probe_cost = 100.0 * PROBE_BATCH_SIZE / (3.0 * BATCH_SIZE)
    return Scores(
        default=default,
        internal_probe=internal_probe,
        internal_shuffled=internal_shuffled,
        internal_oracle=internal_oracle,
        external_valid=external_valid,
        external_oracle=external_oracle,
        probe_cost_pct=probe_cost,
        initial_loss=initial_loss,
    )


def decide(scores: Scores, status: str) -> str:
    if status == "NO_TORCH":
        return "NO_TORCH"
    internal_default = paired(scores.internal_probe, scores.default)
    internal_shuffled = paired(scores.internal_probe, scores.internal_shuffled)
    internal_external = paired(scores.internal_probe, scores.external_valid)
    if positive_ci(internal_default) and positive_ci(internal_shuffled) and positive_ci(internal_external):
        return "TINY_LM_FIXED_COMPUTE_HPO_WIN"
    if positive_ci(internal_default) and positive_ci(internal_shuffled) and mean(internal_external) > 0.0:
        return "TINY_LM_HPO_LIVE_EXTERNAL_BOUNDARY"
    if positive_ci(internal_default) and positive_ci(internal_shuffled):
        return "TINY_LM_PROBE_LABEL_CONTENT"
    return "TINY_LM_HPO_NOT_CLEARED"


def no_torch_artifacts(detail: str) -> tuple[str, str, str]:
    run_log = "\n".join(
        (
            "DFW-E096 tiny causal-LM probe HPO",
            f"host={platform.node()} python={sys.executable}",
            f"DECISION NO_TORCH detail={detail}",
        )
    )
    result = "# DFW-E096 Result - Tiny Causal-LM Probe HPO\n\nDecision label: `NO_TORCH`\n"
    legit = "# DFW-E096 Legitimacy Check\n\n- PyTorch workload: not run.\n"
    return run_log + "\n", result + "\n", legit + "\n"


def run_experiment() -> tuple[str, str, str]:
    status, detail, device = compute_status()
    if status == "NO_TORCH":
        return no_torch_artifacts(detail)
    torch.set_num_threads(int(os.environ.get("DFW_E096_TORCH_THREADS", "1")))
    scores = diagnose(device)
    decision = decide(scores, status)
    internal_default = paired(scores.internal_probe, scores.default)
    internal_shuffled = paired(scores.internal_probe, scores.internal_shuffled)
    internal_external = paired(scores.internal_probe, scores.external_valid)
    internal_external_oracle = paired(scores.internal_probe, scores.external_oracle)
    external_default = paired(scores.external_valid, scores.default)
    internal_oracle_gap = paired(scores.internal_probe, scores.internal_oracle)
    arms = ", ".join(f"{mult:g}" for mult in LR_MULTS)
    run_lines = [
        "DFW-E096 tiny causal-LM probe HPO",
        f"host={platform.node()} python={sys.executable} compute_status={status} {detail}",
        (
            f"rounds={ROUNDS} block={BLOCK_SIZE} batch={BATCH_SIZE} "
            f"probe_batch={PROBE_BATCH_SIZE} eval_batches={EVAL_BATCHES}"
        ),
        (
            f"base_lr={BASE_LR} weight_decay={WEIGHT_DECAY} policy={POLICY} tau_scale={TAU_SCALE} "
            f"lr_mults=({arms}) seeds={len(HELDOUT_SEEDS)}"
        ),
        f"POOLED internal-default={fmt_ci(internal_default)} "
        f"internal-shuffled={fmt_ci(internal_shuffled)} "
        f"internal-external_valid={fmt_ci(internal_external)} "
        f"internal-external_oracle={fmt_ci(internal_external_oracle)} "
        f"external_valid-default={fmt_ci(external_default)} "
        f"internal-internal_oracle={fmt_ci(internal_oracle_gap)}",
        f"DECISION {decision}",
        f"compute_status={status}",
    ]
    lines = [
        "# DFW-E096 Result - Tiny Causal-LM Probe HPO",
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
            "A tiny character-level causal Transformer language model is trained on an "
            "embedded held-out text corpus. Internal HPO evaluates LR-arm candidate "
            "next states from the same base model each round, scores them on held-out "
            "probe text, and probe-weights the parameter deltas into one shared "
            "trajectory. External HPO spends the same candidate-update budget on "
            "separate fixed-LR trajectories and selects by held-out validation loss."
        ),
        "",
        "## Pooled Target-Loss Improvement",
        "",
        f"- initial target loss: `{fmt_ci(scores.initial_loss).strip()}`",
        f"- default fixed LR: `{fmt_ci(scores.default).strip()}`",
        f"- internal probe-weighted HPO: `{fmt_ci(scores.internal_probe).strip()}`",
        f"- shuffled-probe internal: `{fmt_ci(scores.internal_shuffled).strip()}`",
        f"- external validation-selected HPO: `{fmt_ci(scores.external_valid).strip()}`",
        f"- external target oracle HPO: `{fmt_ci(scores.external_oracle).strip()}`",
        f"- internal minus default: `{fmt_ci(internal_default).strip()}`",
        f"- internal minus shuffled: `{fmt_ci(internal_shuffled).strip()}`",
        f"- internal minus external validation-selected HPO: `{fmt_ci(internal_external).strip()}`",
        f"- internal minus external oracle HPO: `{fmt_ci(internal_external_oracle).strip()}`",
        f"- external validation-selected HPO minus default: `{fmt_ci(external_default).strip()}`",
        f"- probe overhead estimate: `{scores.probe_cost_pct:.2f}%`",
        "",
        "## Interpretation",
        "",
    ]
    if decision == "TINY_LM_FIXED_COMPUTE_HPO_WIN":
        lines.append(
            "The lightweight LM audit clears the fixed-compute comparator: internal "
            "probe-weighted aggregation beats the default fixed LR, shuffled probes, "
            "and validation-selected external HPO under the same candidate-update budget."
        )
    else:
        lines.append(
            "The lightweight LM audit is a boundary result for language modeling; use "
            "the pooled margins rather than promoting it as a main result."
        )
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])
    legit = "\n".join(
        [
            "# DFW-E096 Legitimacy Check",
            "",
            f"- Compute: `{status}` / `{detail}`.",
            "- Model: tiny character-level causal Transformer language model.",
            "- Dataset: embedded text corpus split into train, probe/select, and target text.",
            "- Same candidate-update budget: internal and external HPO both train one candidate trajectory per LR arm per round.",
            "- Internal HPO aggregates candidate parameter deltas from one shared base state each round.",
            "- External HPO keeps separate fixed-LR trajectories and selects by held-out validation loss.",
            "- Shuffled control preserves candidate updates but permutes probe-to-arm labels.",
            "- Target text is used for scoring and oracle diagnostics, not for the probe controller.",
            "- This is a lightweight LM tuning result, not a full-scale LLM pretraining run.",
            "",
        ]
    )
    return "\n".join(run_lines) + "\n", "\n".join(lines) + "\n", legit + "\n"


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
