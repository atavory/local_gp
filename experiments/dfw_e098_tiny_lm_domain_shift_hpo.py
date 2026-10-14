"""DFW-E098: tiny causal-LM domain-shift adaptive HPO.

E096/E097 showed that a tiny LM with a stationary fixed-LR grid is too easy for
external HPO: one-hot internal probing recovers the same arm but does not beat
validation-selected fixed-LR HPO.  E098 adds the regime where internal HPO
should matter: the training stream shifts from off-domain prose to code-like
text while the target distribution is code-like text.

The internal controller probes candidate LR-arm next states on code-like heldout
text every round, so it can suppress harmful/high-LR off-domain updates early
and use larger updates after the stream shifts.  External HPO spends the same
candidate-update budget on separate fixed-LR trajectories and selects one by
heldout validation loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import platform
import statistics
import sys
from typing import Iterable

import dfw_e096_tiny_lm_probe_hpo as e96


DATE = "20261014"
OUT_DIR = Path(os.environ.get("DFW_E098_OUT_DIR", Path(__file__).resolve().parent)).resolve()
RESULT_PATH = OUT_DIR / f"dfw_e098_tiny_lm_domain_shift_hpo_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e098_tiny_lm_domain_shift_hpo_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e098_tiny_lm_domain_shift_hpo_LEGITIMACY_CHECK_{DATE}.md"

ROUNDS = int(os.environ.get("DFW_E098_ROUNDS", "16"))
SWITCH_ROUND = int(os.environ.get("DFW_E098_SWITCH_ROUND", str(ROUNDS // 2)))
BLOCK_SIZE = int(os.environ.get("DFW_E098_BLOCK_SIZE", "32"))
BATCH_SIZE = int(os.environ.get("DFW_E098_BATCH_SIZE", "32"))
PROBE_BATCH_SIZE = int(os.environ.get("DFW_E098_PROBE_BATCH_SIZE", "16"))
EVAL_BATCHES = int(os.environ.get("DFW_E098_EVAL_BATCHES", "2"))
BASE_LR = float(os.environ.get("DFW_E098_BASE_LR", "0.004"))
WEIGHT_DECAY = float(os.environ.get("DFW_E098_WEIGHT_DECAY", "0.01"))
HELDOUT_SEEDS = tuple(
    int(part)
    for part in os.environ.get(
        "DFW_E098_HELDOUT_SEEDS",
        ",".join(str(seed) for seed in range(999_200, 999_216)),
    ).split(",")
    if part
)
LR_MULTS = tuple(
    float(part)
    for part in os.environ.get("DFW_E098_LR_MULTS", "0.0,0.25,0.5,1.0,2.0,4.0").split(",")
    if part
)

PROSE = (
    """
The winter road crossed a quiet field and climbed toward the old observatory.
Every evening the keeper wrote a page in his ledger: cloud cover, wind, the
small changes in the river ice, and the hour when the lamps below the hill were
lit. Children from the village brought questions about comets and clocks, and
he answered by drawing careful circles in chalk.

Letters arrived slowly in those days. A traveler might wait a week for news,
then read the same paragraph three times beside the stove. The page smelled of
ink and rain. Outside, horses shook snow from their manes and the road vanished
under a blue dusk.
"""
    * 120
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

metrics = {"train_loss": 2.31, "valid_loss": 2.44, "tokens": 4096}
"""
    * 120
)


@dataclass(frozen=True)
class DataSplits:
    prose_train: object
    code_train: object
    code_select: object
    code_target: object
    vocab_size: int


@dataclass(frozen=True)
class Scores:
    initial_loss: list[float]
    default: list[float]
    internal_probe: list[float]
    internal_shuffled: list[float]
    internal_oracle: list[float]
    external_valid: list[float]
    external_oracle: list[float]
    probe_cost_pct: float


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


def configure_e96() -> None:
    e96.BLOCK_SIZE = BLOCK_SIZE
    e96.BATCH_SIZE = BATCH_SIZE
    e96.PROBE_BATCH_SIZE = PROBE_BATCH_SIZE
    e96.EVAL_BATCHES = EVAL_BATCHES
    e96.BASE_LR = BASE_LR
    e96.WEIGHT_DECAY = WEIGHT_DECAY
    e96.POLICY = "argmin"


def encode(text: str, stoi: dict[str, int]):
    return e96.torch.tensor([stoi[char] for char in text], dtype=e96.torch.long)


def make_splits(device: str) -> DataSplits:
    chars = sorted(set(PROSE + CODE))
    stoi = {char: idx for idx, char in enumerate(chars)}
    prose_ids = encode(PROSE, stoi)
    code_ids = encode(CODE, stoi)
    code_train_end = int(0.70 * len(code_ids))
    code_select_end = int(0.85 * len(code_ids))
    return DataSplits(
        prose_train=prose_ids.to(device),
        code_train=code_ids[:code_train_end].to(device),
        code_select=code_ids[code_train_end:code_select_end].to(device),
        code_target=code_ids[code_select_end:].to(device),
        vocab_size=len(stoi),
    )


def train_data_for_round(splits: DataSplits, round_idx: int):
    return splits.prose_train if round_idx < SWITCH_ROUND else splits.code_train


def candidate_losses(model, candidate_states: list[dict[str, object]], data, seed: int, device: str) -> list[float]:
    return [
        e96.state_score(model, state, data, seed, PROBE_BATCH_SIZE, device)
        for state in candidate_states
    ]


def run_internal(seed: int, arms: tuple[e96.Arm, ...], splits: DataSplits, device: str, mode: str) -> tuple[float, float]:
    e96.torch.manual_seed(seed)
    model = e96.TinyCausalLM(splits.vocab_size).to(device)
    workers = [e96.copy.deepcopy(model).to(device) for _arm in arms]
    opts = [
        e96.torch.optim.AdamW(worker.parameters(), lr=BASE_LR * arm.lr_mult, weight_decay=WEIGHT_DECAY)
        for worker, arm in zip(workers, arms)
    ]
    initial = e96.eval_loss(model, splits.code_target, seed + 900_000, PROBE_BATCH_SIZE, device)
    for round_idx in range(ROUNDS):
        base_state = e96.clone_state(model)
        train_data = train_data_for_round(splits, round_idx)
        batch = e96.make_batch(train_data, BATCH_SIZE, BLOCK_SIZE, seed + 1000 * round_idx, device)
        candidate_states = [
            e96.train_worker_candidate(worker, opt, base_state, batch)
            for worker, opt in zip(workers, opts)
        ]
        score_data = splits.code_target if mode == "oracle" else splits.code_select
        losses = candidate_losses(model, candidate_states, score_data, seed + 200_000 + round_idx, device)
        if mode == "shuffled":
            shift = (seed + round_idx) % len(losses)
            losses = losses[shift:] + losses[:shift]
        weights = e96.weights_from_losses(losses)
        e96.load_state(model, e96.aggregate_states(base_state, candidate_states, weights))
    final = e96.eval_loss(model, splits.code_target, seed + 901_000, PROBE_BATCH_SIZE, device)
    return initial, initial - final


def run_fixed_arm(seed: int, arm: e96.Arm, splits: DataSplits, device: str) -> tuple[float, float]:
    e96.torch.manual_seed(seed)
    model = e96.TinyCausalLM(splits.vocab_size).to(device)
    opt = e96.torch.optim.AdamW(model.parameters(), lr=BASE_LR * arm.lr_mult, weight_decay=WEIGHT_DECAY)
    initial = e96.eval_loss(model, splits.code_target, seed + 900_000, PROBE_BATCH_SIZE, device)
    for round_idx in range(ROUNDS):
        train_data = train_data_for_round(splits, round_idx)
        x, y = e96.make_batch(train_data, BATCH_SIZE, BLOCK_SIZE, seed + 1000 * round_idx, device)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = e96.loss_on_batch(model, x, y)
        loss.backward()
        e96.torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    select = e96.eval_loss(model, splits.code_select, seed + 902_000, PROBE_BATCH_SIZE, device)
    target = e96.eval_loss(model, splits.code_target, seed + 901_000, PROBE_BATCH_SIZE, device)
    return initial - select, initial - target


def run_external(seed: int, arms: tuple[e96.Arm, ...], splits: DataSplits, device: str) -> tuple[float, float, float]:
    scores = [run_fixed_arm(seed, arm, splits, device) for arm in arms]
    valid_idx = max(range(len(scores)), key=lambda idx: scores[idx][0])
    oracle_idx = max(range(len(scores)), key=lambda idx: scores[idx][1])
    default_idx = min(range(len(arms)), key=lambda idx: abs(arms[idx].lr_mult - 1.0))
    return scores[default_idx][1], scores[valid_idx][1], scores[oracle_idx][1]


def diagnose(device: str) -> Scores:
    configure_e96()
    splits = make_splits(device)
    arms = tuple(e96.Arm(f"lr_x{mult:g}", mult) for mult in LR_MULTS)
    initial_loss: list[float] = []
    default: list[float] = []
    internal_probe: list[float] = []
    internal_shuffled: list[float] = []
    internal_oracle: list[float] = []
    external_valid: list[float] = []
    external_oracle: list[float] = []
    for idx, seed in enumerate(HELDOUT_SEEDS):
        initial, probe_score = run_internal(seed, arms, splits, device, "probe")
        _initial_s, shuffled_score = run_internal(seed, arms, splits, device, "shuffled")
        _initial_o, oracle_score = run_internal(seed, arms, splits, device, "oracle")
        default_score, valid_score, oracle_ext_score = run_external(seed, arms, splits, device)
        initial_loss.append(initial)
        default.append(default_score)
        internal_probe.append(probe_score)
        internal_shuffled.append(shuffled_score)
        internal_oracle.append(oracle_score)
        external_valid.append(valid_score)
        external_oracle.append(oracle_ext_score)
        print(
            f"REP {idx:02d} seed={seed} default={default_score:.4f} "
            f"internal={probe_score:.4f} shuffled={shuffled_score:.4f} "
            f"external_valid={valid_score:.4f} external_oracle={oracle_ext_score:.4f}",
            flush=True,
        )
    probe_cost = 100.0 * PROBE_BATCH_SIZE / (3.0 * BATCH_SIZE)
    return Scores(
        initial_loss=initial_loss,
        default=default,
        internal_probe=internal_probe,
        internal_shuffled=internal_shuffled,
        internal_oracle=internal_oracle,
        external_valid=external_valid,
        external_oracle=external_oracle,
        probe_cost_pct=probe_cost,
    )


def decide(scores: Scores, status: str) -> str:
    if status == "NO_TORCH":
        return "NO_TORCH"
    internal_default = paired(scores.internal_probe, scores.default)
    internal_shuffled = paired(scores.internal_probe, scores.internal_shuffled)
    internal_external = paired(scores.internal_probe, scores.external_valid)
    if positive_ci(internal_default) and positive_ci(internal_shuffled) and positive_ci(internal_external):
        return "TINY_LM_DOMAIN_SHIFT_HPO_WIN"
    if positive_ci(internal_default) and positive_ci(internal_shuffled) and mean(internal_external) > 0.0:
        return "TINY_LM_DOMAIN_SHIFT_HPO_LIVE"
    if positive_ci(internal_default) and positive_ci(internal_shuffled):
        return "TINY_LM_DOMAIN_SHIFT_LABEL_CONTENT"
    return "TINY_LM_DOMAIN_SHIFT_NOT_CLEARED"


def no_torch_artifacts(detail: str) -> tuple[str, str, str]:
    run_log = "\n".join(
        (
            "DFW-E098 tiny causal-LM domain-shift HPO",
            f"host={platform.node()} python={sys.executable}",
            f"DECISION NO_TORCH detail={detail}",
        )
    )
    result = "# DFW-E098 Result - Tiny Causal-LM Domain-Shift HPO\n\nDecision label: `NO_TORCH`\n"
    legit = "# DFW-E098 Legitimacy Check\n\n- PyTorch workload: not run.\n"
    return run_log + "\n", result + "\n", legit + "\n"


def run_experiment() -> tuple[str, str, str]:
    status, detail, device = e96.compute_status()
    if status == "NO_TORCH":
        return no_torch_artifacts(detail)
    e96.torch.set_num_threads(int(os.environ.get("DFW_E098_TORCH_THREADS", "1")))
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
        "DFW-E098 tiny causal-LM domain-shift HPO",
        f"host={platform.node()} python={sys.executable} compute_status={status} {detail}",
        (
            f"rounds={ROUNDS} switch_round={SWITCH_ROUND} block={BLOCK_SIZE} "
            f"batch={BATCH_SIZE} probe_batch={PROBE_BATCH_SIZE} eval_batches={EVAL_BATCHES}"
        ),
        f"base_lr={BASE_LR} weight_decay={WEIGHT_DECAY} lr_mults=({arms}) seeds={len(HELDOUT_SEEDS)}",
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
        "# DFW-E098 Result - Tiny Causal-LM Domain-Shift HPO",
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
            "The tiny causal-LM training stream shifts from off-domain prose to "
            "code-like text, while probe/select and target text are code-like. "
            "Internal HPO uses one-hot probe selection over LR-arm candidate "
            "next states from the shared model each round. External HPO spends "
            "the same candidate-update budget on separate fixed-LR trajectories "
            "and selects one by held-out validation loss."
        ),
        "",
        "## Pooled Target-Loss Improvement",
        "",
        f"- initial target loss: `{fmt_ci(scores.initial_loss).strip()}`",
        f"- default fixed LR: `{fmt_ci(scores.default).strip()}`",
        f"- internal probe HPO: `{fmt_ci(scores.internal_probe).strip()}`",
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
    if decision == "TINY_LM_DOMAIN_SHIFT_HPO_WIN":
        lines.append(
            "The domain-shift LM audit clears the fixed-compute comparator: "
            "same-round probe selection beats the default fixed LR, shuffled "
            "probes, and validation-selected fixed-LR external HPO."
        )
    else:
        lines.append(
            "The domain-shift LM audit should be treated as a boundary or "
            "supporting result according to the pooled margins above."
        )
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])
    legit = "\n".join(
        [
            "# DFW-E098 Legitimacy Check",
            "",
            f"- Compute: `{status}` / `{detail}`.",
            "- Model: tiny character-level causal Transformer language model.",
            "- Training stream: off-domain prose before the switch, code-like text after the switch.",
            "- Probe/select and target streams: held-out code-like text.",
            "- Same candidate-update budget: internal and external HPO both train one candidate update per LR arm per round.",
            "- Internal HPO aggregates by one-hot probe selection from one shared base state each round.",
            "- External HPO keeps separate fixed-LR trajectories and selects by held-out validation loss.",
            "- Shuffled control preserves candidate updates but permutes probe-to-arm labels.",
            "- Target text is used for scoring and oracle diagnostics, not for the probe controller.",
            "- This is a tiny-LM domain-shift tuning result, not a full-scale LLM pretraining run.",
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
