"""DFW-E067: default-anchored admission for grouped LR probes.

E066 showed that grouped LR probes rescue a deliberately bad average over
exploratory LR arms, but lose to a known default-only LR arm.  This follow-up
tests the direct repair: keep the default arm as the fallback action, and only
move toward exploratory arms when same-round probes support them.

The question is not whether exploration beats a tuned default for free.  The
question is whether probe-gated exploration can tie the default when probes are
uninformative and improve when an exploratory arm is genuinely better.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import platform
import statistics
import sys

import dfw_e030_gpu_real_path_local_map as e30
import dfw_e032_gpu_probe_outer_control_real_path as e32
import dfw_e059_gpu_stationary_inner_lr_probe_admission as e59
import dfw_e060_gpu_stationary_inner_lr_group_probe_admission as e60


DATE = "20261005"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e067_gpu_default_anchored_probe_admission_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e067_gpu_default_anchored_probe_admission_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e067_gpu_default_anchored_probe_admission_LEGITIMACY_CHECK_{DATE}.md"

K_VALUES = e32.env_int_tuple("DFW_E067_K_VALUES", (8, 16, 32))
N_WORKERS = int(os.environ.get("DFW_E067_N_WORKERS", "8"))
ROUNDS = int(os.environ.get("DFW_E067_ROUNDS", "36"))
SPREAD = float(os.environ.get("DFW_E067_SPREAD", "16.0"))
MAX_SELECT_BATCH = int(os.environ.get("DFW_E067_MAX_SELECT_BATCH", "384"))
PROBE_BATCH = int(os.environ.get("DFW_E067_PROBE_BATCH", "96"))
PROBE_SLICES = int(os.environ.get("DFW_E067_PROBE_SLICES", "4"))
PILOT_SEEDS = e32.env_seed_tuple("DFW_E067_PILOT_SEEDS", tuple(range(937_000, 937_003)))
HELDOUT_SEEDS = e32.env_seed_tuple("DFW_E067_HELDOUT_SEEDS", tuple(range(938_000, 938_012)))
LAM = float(os.environ.get("DFW_E067_LAM", "0.4"))
TAU = float(os.environ.get("DFW_E067_TAU", "2.0"))
EPS = 1e-12


mean = e59.mean
fmt_ci = e59.fmt_ci
paired = e59.paired
positive_ci = e59.positive_ci
spread_text = e59.spread_text


@dataclass(frozen=True)
class Cell:
    k_value: int
    group_values: tuple[float, ...]
    fixed_pair: tuple[float, float]
    default_pair: tuple[float, float]
    fixed_all: list[float]
    default_only: list[float]
    oracle_subset: list[float]
    oracle_single: list[float]
    allprior_raw: list[float]
    anchored: list[float]
    anchored_shuffled: list[float]
    cost_pct: float


def default_update(row: e60.GroupRoundRow):
    for value, update, _prior in e60.group_updates(row):
        if abs(value - 1.0) < 1e-12:
            return update
    raise ValueError("default group lr_multiplier=1.0 missing")


def score_default(rows: list[e60.GroupRoundRow], pair: tuple[float, float]) -> float:
    alpha, beta = pair
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=rows[0].theta.device)
    raw = 0.0
    total = 0.0
    for row in rows:
        update = default_update(row)
        direction = (update + beta * velocity).detach()
        raw += e60.improvement(row, row.fedavg)
        total += e60.improvement(row, alpha * direction)
        velocity = direction
    return total / raw if raw > EPS else total


def tune_default_pair(groups: list[list[e60.GroupRoundRow]]) -> tuple[float, float]:
    best_pair = (1.0, 0.0)
    best_score = -float("inf")
    for alpha in e60.ALPHAS:
        for beta in e60.BETAS:
            score = mean(score_default(rows, (alpha, beta)) for rows in groups)
            if score > best_score:
                best_score = score
                best_pair = (alpha, beta)
    return best_pair


def budget_loss(row: e60.GroupRoundRow, action) -> float:
    b = PROBE_BATCH
    losses = [e30.loss_float(row.theta + action, row.select_x[:b], row.select_y[:b])]
    if PROBE_SLICES >= 2:
        losses.append(e30.loss_float(row.theta + action, row.select2_x[:b], row.select2_y[:b]))
    if PROBE_SLICES >= 4:
        losses.append(e30.loss_float(row.theta + action, row.select_x[b : 2 * b], row.select_y[b : 2 * b]))
        losses.append(e30.loss_float(row.theta + action, row.select2_x[b : 2 * b], row.select2_y[b : 2 * b]))
    return mean(losses)


def group_losses(row: e60.GroupRoundRow, velocity, pair: tuple[float, float]) -> list[float]:
    alpha, beta = pair
    return [budget_loss(row, alpha * (update + beta * velocity)) for _value, update, _prior in e60.group_updates(row)]


def score_probe(
    rows: list[e60.GroupRoundRow],
    pair: tuple[float, float],
    rep_idx: int,
    shuffled: bool,
) -> float:
    alpha, beta = pair
    spec = e59.DampingSpec("e066_l40_t2", LAM, TAU, PROBE_SLICES)
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=rows[0].theta.device)
    raw = 0.0
    total = 0.0
    for row_idx, row in enumerate(rows):
        losses = group_losses(row, velocity, pair)
        if shuffled:
            losses = e60.shuffled_losses(losses, row_idx, rep_idx)
        weights = e60.soft_damped_group_weights(row, losses, spec)
        agg = e60.aggregate_groups(e60.group_updates(row), weights)
        direction = (agg + beta * velocity).detach()
        raw += e60.improvement(row, row.fedavg)
        total += e60.improvement(row, alpha * direction)
        velocity = direction
    return total / raw if raw > EPS else total


def default_index(groups: tuple[tuple[float, object, float], ...]) -> int:
    for idx, (value, _update, _prior) in enumerate(groups):
        if abs(value - 1.0) < 1e-12:
            return idx
    raise ValueError("default group lr_multiplier=1.0 missing")


def default_anchored_weights(
    row: e60.GroupRoundRow,
    losses: list[float],
) -> list[float]:
    groups = e60.group_updates(row)
    idx0 = default_index(groups)
    prior = [0.0] * len(groups)
    prior[idx0] = 1.0
    if len(losses) <= 1:
        return prior
    spread = statistics.stdev(losses)
    temp = max(TAU * spread, 1e-5)
    loc = sum(losses) / max(len(losses), 1)
    logits = [max(-12.0, min(12.0, (loc - loss) / temp)) for loss in losses]
    offset = max(logits)
    exps = [math.exp(logit - offset) for logit in logits]
    denom = max(sum(exps), EPS)
    soft = [value / denom for value in exps]
    return [(1.0 - LAM) * base + LAM * value for base, value in zip(prior, soft)]


def score_default_anchored(
    rows: list[e60.GroupRoundRow],
    pair: tuple[float, float],
    rep_idx: int,
    shuffled: bool,
) -> float:
    alpha, beta = pair
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=rows[0].theta.device)
    raw = 0.0
    total = 0.0
    for row_idx, row in enumerate(rows):
        losses = group_losses(row, velocity, pair)
        if shuffled:
            losses = e60.shuffled_losses(losses, row_idx, rep_idx)
        weights = default_anchored_weights(row, losses)
        agg = e60.aggregate_groups(e60.group_updates(row), weights)
        direction = (agg + beta * velocity).detach()
        raw += e60.improvement(row, row.fedavg)
        total += e60.improvement(row, alpha * direction)
        velocity = direction
    return total / raw if raw > EPS else total


def cost_pct(group_count: int, k_value: int) -> float:
    return 100.0 * group_count * PROBE_SLICES * PROBE_BATCH / (
        e30.TRAIN_SIZE * 3.0 * N_WORKERS * k_value
    )


def capture(vals: list[float], fixed: list[float], oracle: list[float]) -> list[float]:
    out = []
    for val, fix, ora in zip(vals, fixed, oracle):
        gap = ora - fix
        out.append((val - fix) / gap if abs(gap) > EPS else 0.0)
    return out


def diagnose_cell(k_value: int, device: str) -> Cell:
    e30.ROUNDS = ROUNDS
    e60.SELECT_BATCH = MAX_SELECT_BATCH
    if PROBE_SLICES >= 4 and 2 * PROBE_BATCH > MAX_SELECT_BATCH:
        raise ValueError("MAX_SELECT_BATCH must cover two chunks from each select draw")
    pilot_groups = e60.make_groups(k_value, N_WORKERS, SPREAD, PILOT_SEEDS, device)
    fixed_pair = e60.tune_fixed_pair(pilot_groups)
    default_pair = tune_default_pair(pilot_groups)
    heldout_groups = e60.make_groups(k_value, N_WORKERS, SPREAD, HELDOUT_SEEDS, device)
    fixed_vals = [e60.score_fixed_group(rows, fixed_pair) for rows in heldout_groups]
    default_vals = [score_default(rows, default_pair) for rows in heldout_groups]
    oracle_vals = []
    oracle_single_vals = []
    allprior_raw_vals = []
    anchored_vals = []
    anchored_shuffled_vals = []
    for rep_idx, rows in enumerate(heldout_groups):
        subset, single = e60.score_group_oracle(rows, default_pair)
        oracle_vals.append(subset)
        oracle_single_vals.append(single)
        allprior_raw_vals.append(score_probe(rows, fixed_pair, rep_idx, False))
        anchored_vals.append(score_default_anchored(rows, default_pair, rep_idx, False))
        anchored_shuffled_vals.append(score_default_anchored(rows, default_pair, rep_idx, True))
    groups = e60.group_updates(heldout_groups[0][0])
    return Cell(
        k_value=k_value,
        group_values=tuple(value for value, _update, _prior in groups),
        fixed_pair=fixed_pair,
        default_pair=default_pair,
        fixed_all=fixed_vals,
        default_only=default_vals,
        oracle_subset=oracle_vals,
        oracle_single=oracle_single_vals,
        allprior_raw=allprior_raw_vals,
        anchored=anchored_vals,
        anchored_shuffled=anchored_shuffled_vals,
        cost_pct=cost_pct(len(groups), k_value),
    )


def decide(cells: list[Cell], compute_status: str) -> str:
    if compute_status != "REAL_GPU":
        return "NO_GPU_HARNESS"
    anchored_fixed = []
    anchored_default = []
    anchored_shuffled = []
    allprior_default = []
    default_fixed = []
    for cell in cells:
        anchored_fixed.extend(paired(cell.anchored, cell.fixed_all))
        anchored_default.extend(paired(cell.anchored, cell.default_only))
        anchored_shuffled.extend(paired(cell.anchored, cell.anchored_shuffled))
        allprior_default.extend(paired(cell.allprior_raw, cell.default_only))
        default_fixed.extend(paired(cell.default_only, cell.fixed_all))
    if positive_ci(anchored_default) and positive_ci(anchored_shuffled):
        return "DEFAULT_ANCHORED_PROBE_BEATS_DEFAULT"
    if positive_ci(anchored_fixed) and positive_ci(anchored_shuffled):
        return "DEFAULT_ANCHORED_PROBE_BEATS_PARALLEL_AVERAGE_ONLY"
    if positive_ci(default_fixed) and not positive_ci(allprior_default):
        return "DEFAULT_ONLY_EXPLAINS_ALLPRIOR_PROBES"
    if positive_ci(default_fixed):
        return "DEFAULT_ONLY_EXPLAINS_LR_GROUP"
    return "DEFAULT_ANCHORED_PROBE_NEGATIVE"


def no_gpu_artifacts(diag: e30.ComputeDiagnostic) -> tuple[str, str, str]:
    run_log = "\n".join(
        (
            "DFW-E067 default-anchored probe admission",
            f"host={platform.node()} python={sys.executable}",
            f"DECISION NO_GPU_HARNESS compute_status={diag.status} detail={diag.detail}",
        )
    )
    result = "\n".join(
        (
            "# DFW-E067 Result - Default-Anchored Probe Admission",
            "",
            "Decision label: `NO_GPU_HARNESS`",
            "",
            f"Diagnostic: `{diag.detail}`.",
            "",
        )
    )
    legit = "# DFW-E067 Legitimacy Check\n\n- GPU workload: not run.\n"
    return run_log + "\n", result + "\n", legit


def run_experiment() -> tuple[str, str, str]:
    diag = e30.compute_diagnostic()
    if diag.status != "REAL_GPU":
        return no_gpu_artifacts(diag)
    e30.torch.set_default_dtype(e30.torch.float32)
    try:
        e30.torch.set_num_threads(1)
    except Exception:
        pass
    cells = []
    run_lines = [
        "DFW-E067 default-anchored probe admission",
        f"host={platform.node()} python={sys.executable} compute_status={diag.status} {diag.detail}",
        f"model=torch_two_layer_binary_classifier dim={e30.DIM} N={N_WORKERS} K={K_VALUES} rounds={ROUNDS} eta={e30.ETA}",
        f"spread={spread_text(SPREAD)} probe_batch={PROBE_BATCH} slices={PROBE_SLICES} lam={LAM} tau={TAU}",
        f"pilot_seeds={len(PILOT_SEEDS)} heldout_seeds={len(HELDOUT_SEEDS)}",
        "",
    ]
    for k_value in K_VALUES:
        cell = diagnose_cell(k_value, diag.device)
        cells.append(cell)
        line = (
            f"CELL K={k_value} cost={cell.cost_pct:.2f}% "
            f"fixed_pair=({cell.fixed_pair[0]:.1f},{cell.fixed_pair[1]:.2f}) "
            f"default_pair=({cell.default_pair[0]:.1f},{cell.default_pair[1]:.2f}) "
            f"groups={','.join(f'{value:.3g}' for value in cell.group_values)} "
            f"fixed_all={fmt_ci(cell.fixed_all)} default_only={fmt_ci(cell.default_only)} "
            f"oracle_subset={fmt_ci(cell.oracle_subset)} oracle_single={fmt_ci(cell.oracle_single)} "
            f"allprior_raw={fmt_ci(cell.allprior_raw)} "
            f"allprior-default={fmt_ci(paired(cell.allprior_raw, cell.default_only))} "
            f"anchored={fmt_ci(cell.anchored)} "
            f"anchored-fixed={fmt_ci(paired(cell.anchored, cell.fixed_all))} "
            f"anchored-default={fmt_ci(paired(cell.anchored, cell.default_only))} "
            f"default-fixed={fmt_ci(paired(cell.default_only, cell.fixed_all))} "
            f"anchored-shuffled={fmt_ci(paired(cell.anchored, cell.anchored_shuffled))} "
            f"capture_vs_default={fmt_ci(capture(cell.anchored, cell.default_only, cell.oracle_subset))}"
        )
        run_lines.append(line)
        print(line, flush=True)
    decision = decide(cells, diag.status)
    run_lines.append("")
    run_lines.append(f"DECISION {decision}")
    run_lines.append(f"compute_status={diag.status}")

    lines = [
        "# DFW-E067 Result - Default-Anchored Probe Admission",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{diag.status}` via PyTorch CUDA: `{diag.detail}`.",
        "",
        "## Held-Out Scores",
        "",
        "| K | cost | fixed `(alpha,beta)` | default `(alpha,beta)` | fixed all | default only | oracle subset | oracle single | all-prior raw | all-prior minus default | anchored | anchored minus fixed | anchored minus default | default minus fixed | anchored minus shuffled | capture vs default |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{cell.k_value}",
                    f"{cell.cost_pct:.2f}%",
                    f"({cell.fixed_pair[0]:.1f},{cell.fixed_pair[1]:.2f})",
                    f"({cell.default_pair[0]:.1f},{cell.default_pair[1]:.2f})",
                    fmt_ci(cell.fixed_all).strip(),
                    fmt_ci(cell.default_only).strip(),
                    fmt_ci(cell.oracle_subset).strip(),
                    fmt_ci(cell.oracle_single).strip(),
                    fmt_ci(cell.allprior_raw).strip(),
                    fmt_ci(paired(cell.allprior_raw, cell.default_only)).strip(),
                    fmt_ci(cell.anchored).strip(),
                    fmt_ci(paired(cell.anchored, cell.fixed_all)).strip(),
                    fmt_ci(paired(cell.anchored, cell.default_only)).strip(),
                    fmt_ci(paired(cell.default_only, cell.fixed_all)).strip(),
                    fmt_ci(paired(cell.anchored, cell.anchored_shuffled)).strip(),
                    fmt_ci(capture(cell.anchored, cell.default_only, cell.oracle_subset)).strip(),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])
    legit = "\n".join(
        [
            "# DFW-E067 Legitimacy Check",
            "",
            f"- Compute: `{diag.status}` / `{diag.detail}`.",
            "- Target batches are used only for scoring and oracle diagnostics.",
            "- Default-only control uses only the lr_multiplier=1 group with its own tuned outer pair.",
            "- All-prior raw rule repeats the E064/E066 grouped soft average.",
            "- Default-anchored rule uses a one-hot default prior plus frozen E064 damping.",
            "- Shuffled control rotates probe losses across LR groups for the anchored rule.",
            "",
        ]
    )
    return "\n".join(run_lines) + "\n", "\n".join(lines) + "\n", legit + "\n"


def main() -> None:
    run_log, result, legit = run_experiment()
    RUN_LOG_PATH.write_text(run_log)
    RESULT_PATH.write_text(result)
    LEGITIMACY_PATH.write_text(legit)
    print(result)
    print(f"wrote {RESULT_PATH}")
    print(f"wrote {RUN_LOG_PATH}")
    print(f"wrote {LEGITIMACY_PATH}")


if __name__ == "__main__":
    main()
