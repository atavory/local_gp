"""DFW-E070: validation-gradient marginal-value grouped probes.

E066-E069 all used forward finite differences: evaluate theta + candidate on a
held-out/probe shard and soft-weight arms from those noisy scalar losses.  That
extractor did not beat a good default, did not improve with doubled forward
budget, and did not reliably leave a bad/stale anchor.

This test changes the measurement, not the anchoring rule.  One validation
backward gives g_V at theta, then every group candidate gets the first-order
marginal score <g_V, alpha * (u_i + beta v)>.  This costs roughly one backward
per probe slice, independent of the number of LR arms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import platform
import sys

import dfw_e030_gpu_real_path_local_map as e30
import dfw_e032_gpu_probe_outer_control_real_path as e32
import dfw_e059_gpu_stationary_inner_lr_probe_admission as e59
import dfw_e060_gpu_stationary_inner_lr_group_probe_admission as e60
import dfw_e067_gpu_default_anchored_probe_admission as e67


DATE = "20261005"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e070_gpu_validation_gradient_group_probe_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e070_gpu_validation_gradient_group_probe_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e070_gpu_validation_gradient_group_probe_LEGITIMACY_CHECK_{DATE}.md"

K_VALUES = e32.env_int_tuple("DFW_E070_K_VALUES", (8, 16, 32))
N_WORKERS = int(os.environ.get("DFW_E070_N_WORKERS", "8"))
ROUNDS = int(os.environ.get("DFW_E070_ROUNDS", "36"))
SPREAD = float(os.environ.get("DFW_E070_SPREAD", "16.0"))
MAX_SELECT_BATCH = int(os.environ.get("DFW_E070_MAX_SELECT_BATCH", "384"))
PROBE_BATCH = int(os.environ.get("DFW_E070_PROBE_BATCH", "96"))
PROBE_SLICES = int(os.environ.get("DFW_E070_PROBE_SLICES", "4"))
PILOT_SEEDS = e32.env_seed_tuple("DFW_E070_PILOT_SEEDS", tuple(range(939_000, 939_003)))
HELDOUT_SEEDS = e32.env_seed_tuple("DFW_E070_HELDOUT_SEEDS", tuple(range(940_000, 940_012)))
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
    forward_anchor: list[float]
    gradient_anchor: list[float]
    gradient_shuffled: list[float]
    forward_cost_pct: float
    gradient_cost_pct: float


def select_slices(row: e60.GroupRoundRow) -> list[tuple[object, object]]:
    b = PROBE_BATCH
    slices = [(row.select_x[:b], row.select_y[:b])]
    if PROBE_SLICES >= 2:
        slices.append((row.select2_x[:b], row.select2_y[:b]))
    if PROBE_SLICES >= 4:
        slices.append((row.select_x[b : 2 * b], row.select_y[b : 2 * b]))
        slices.append((row.select2_x[b : 2 * b], row.select2_y[b : 2 * b]))
    return slices


def validation_grad(row: e60.GroupRoundRow):
    grads = []
    for xs, ys in select_slices(row):
        _loss, grad = e30.loss_and_grad(row.theta, xs, ys)
        grads.append(grad)
    return e30.mean_tensors(grads).detach()


def gradient_losses(
    row: e60.GroupRoundRow,
    velocity,
    pair: tuple[float, float],
) -> list[float]:
    alpha, beta = pair
    grad = validation_grad(row)
    losses = []
    for _value, update, _prior in e60.group_updates(row):
        action = alpha * (update + beta * velocity)
        # First-order predicted loss change; lower is better.
        losses.append(e30.dot_update(grad, action))
    return losses


def score_gradient_anchored(
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
        losses = gradient_losses(row, velocity, pair)
        if shuffled:
            losses = e60.shuffled_losses(losses, row_idx, rep_idx)
        weights = e67.default_anchored_weights(row, losses)
        agg = e60.aggregate_groups(e60.group_updates(row), weights)
        direction = (agg + beta * velocity).detach()
        raw += e60.improvement(row, row.fedavg)
        total += e60.improvement(row, alpha * direction)
        velocity = direction
    return total / raw if raw > EPS else total


def forward_cost_pct(group_count: int, k_value: int) -> float:
    return 100.0 * group_count * PROBE_SLICES * PROBE_BATCH / (
        e30.TRAIN_SIZE * 3.0 * N_WORKERS * k_value
    )


def gradient_cost_pct(k_value: int) -> float:
    return 100.0 * PROBE_SLICES * PROBE_BATCH / (e30.TRAIN_SIZE * N_WORKERS * k_value)


def capture(vals: list[float], fixed: list[float], oracle: list[float]) -> list[float]:
    out = []
    for val, fix, ora in zip(vals, fixed, oracle):
        gap = ora - fix
        out.append((val - fix) / gap if abs(gap) > EPS else 0.0)
    return out


def diagnose_cell(k_value: int, device: str) -> Cell:
    e30.ROUNDS = ROUNDS
    e60.SELECT_BATCH = MAX_SELECT_BATCH
    e67.PROBE_BATCH = PROBE_BATCH
    e67.PROBE_SLICES = PROBE_SLICES
    e67.MAX_SELECT_BATCH = MAX_SELECT_BATCH
    if PROBE_SLICES >= 4 and 2 * PROBE_BATCH > MAX_SELECT_BATCH:
        raise ValueError("MAX_SELECT_BATCH must cover two chunks from each select draw")
    pilot_groups = e60.make_groups(k_value, N_WORKERS, SPREAD, PILOT_SEEDS, device)
    fixed_pair = e60.tune_fixed_pair(pilot_groups)
    default_pair = e67.tune_default_pair(pilot_groups)
    heldout_groups = e60.make_groups(k_value, N_WORKERS, SPREAD, HELDOUT_SEEDS, device)
    fixed_vals = [e60.score_fixed_group(rows, fixed_pair) for rows in heldout_groups]
    default_vals = [e67.score_default(rows, default_pair) for rows in heldout_groups]
    oracle_vals = []
    forward_vals = []
    grad_vals = []
    grad_shuf_vals = []
    for rep_idx, rows in enumerate(heldout_groups):
        subset, _single = e60.score_group_oracle(rows, default_pair)
        oracle_vals.append(subset)
        forward_vals.append(e67.score_default_anchored(rows, default_pair, rep_idx, False))
        grad_vals.append(score_gradient_anchored(rows, default_pair, rep_idx, False))
        grad_shuf_vals.append(score_gradient_anchored(rows, default_pair, rep_idx, True))
    groups = e60.group_updates(heldout_groups[0][0])
    return Cell(
        k_value=k_value,
        group_values=tuple(value for value, _update, _prior in groups),
        fixed_pair=fixed_pair,
        default_pair=default_pair,
        fixed_all=fixed_vals,
        default_only=default_vals,
        oracle_subset=oracle_vals,
        forward_anchor=forward_vals,
        gradient_anchor=grad_vals,
        gradient_shuffled=grad_shuf_vals,
        forward_cost_pct=forward_cost_pct(len(groups), k_value),
        gradient_cost_pct=gradient_cost_pct(k_value),
    )


def decide(cells: list[Cell], compute_status: str) -> str:
    if compute_status != "REAL_GPU":
        return "NO_GPU_HARNESS"
    grad_default = []
    grad_forward = []
    grad_shuffled = []
    for cell in cells:
        grad_default.extend(paired(cell.gradient_anchor, cell.default_only))
        grad_forward.extend(paired(cell.gradient_anchor, cell.forward_anchor))
        grad_shuffled.extend(paired(cell.gradient_anchor, cell.gradient_shuffled))
    if positive_ci(grad_default) and positive_ci(grad_shuffled):
        return "VALIDATION_GRADIENT_PROBE_BEATS_DEFAULT"
    if positive_ci(grad_forward):
        return "VALIDATION_GRADIENT_IMPROVES_FORWARD_EXTRACTOR"
    return "VALIDATION_GRADIENT_PROBE_NEGATIVE"


def no_gpu_artifacts(diag: e30.ComputeDiagnostic) -> tuple[str, str, str]:
    run_log = "\n".join(
        (
            "DFW-E070 validation-gradient group probe",
            f"host={platform.node()} python={sys.executable}",
            f"DECISION NO_GPU_HARNESS compute_status={diag.status} detail={diag.detail}",
        )
    )
    result = "\n".join(
        (
            "# DFW-E070 Result - Validation-Gradient Group Probe",
            "",
            "Decision label: `NO_GPU_HARNESS`",
            "",
            f"Diagnostic: `{diag.detail}`.",
            "",
        )
    )
    legit = "# DFW-E070 Legitimacy Check\n\n- GPU workload: not run.\n"
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
        "DFW-E070 validation-gradient group probe",
        f"host={platform.node()} python={sys.executable} compute_status={diag.status} {diag.detail}",
        f"model=torch_two_layer_binary_classifier dim={e30.DIM} N={N_WORKERS} K={K_VALUES} rounds={ROUNDS} eta={e30.ETA}",
        f"spread={spread_text(SPREAD)} probe_batch={PROBE_BATCH} slices={PROBE_SLICES}",
        f"pilot_seeds={len(PILOT_SEEDS)} heldout_seeds={len(HELDOUT_SEEDS)}",
        "",
    ]
    for k_value in K_VALUES:
        cell = diagnose_cell(k_value, diag.device)
        cells.append(cell)
        line = (
            f"CELL K={k_value} fwd_cost={cell.forward_cost_pct:.2f}% grad_cost={cell.gradient_cost_pct:.2f}% "
            f"fixed_pair=({cell.fixed_pair[0]:.1f},{cell.fixed_pair[1]:.2f}) "
            f"default_pair=({cell.default_pair[0]:.1f},{cell.default_pair[1]:.2f}) "
            f"groups={','.join(f'{value:.3g}' for value in cell.group_values)} "
            f"default_only={fmt_ci(cell.default_only)} oracle_subset={fmt_ci(cell.oracle_subset)} "
            f"forward_anchor={fmt_ci(cell.forward_anchor)} "
            f"forward-default={fmt_ci(paired(cell.forward_anchor, cell.default_only))} "
            f"grad_anchor={fmt_ci(cell.gradient_anchor)} "
            f"grad-default={fmt_ci(paired(cell.gradient_anchor, cell.default_only))} "
            f"grad-forward={fmt_ci(paired(cell.gradient_anchor, cell.forward_anchor))} "
            f"grad-shuffled={fmt_ci(paired(cell.gradient_anchor, cell.gradient_shuffled))} "
            f"grad_capture_vs_default={fmt_ci(capture(cell.gradient_anchor, cell.default_only, cell.oracle_subset))}"
        )
        run_lines.append(line)
        print(line, flush=True)
    decision = decide(cells, diag.status)
    run_lines.append("")
    run_lines.append(f"DECISION {decision}")
    run_lines.append(f"compute_status={diag.status}")

    lines = [
        "# DFW-E070 Result - Validation-Gradient Group Probe",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{diag.status}` via PyTorch CUDA: `{diag.detail}`.",
        "",
        "## Held-Out Scores",
        "",
        "| K | forward cost | gradient cost | default `(alpha,beta)` | default only | oracle subset | forward anchor | forward minus default | gradient anchor | gradient minus default | gradient minus forward | gradient minus shuffled | gradient capture vs default |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{cell.k_value}",
                    f"{cell.forward_cost_pct:.2f}%",
                    f"{cell.gradient_cost_pct:.2f}%",
                    f"({cell.default_pair[0]:.1f},{cell.default_pair[1]:.2f})",
                    fmt_ci(cell.default_only).strip(),
                    fmt_ci(cell.oracle_subset).strip(),
                    fmt_ci(cell.forward_anchor).strip(),
                    fmt_ci(paired(cell.forward_anchor, cell.default_only)).strip(),
                    fmt_ci(cell.gradient_anchor).strip(),
                    fmt_ci(paired(cell.gradient_anchor, cell.default_only)).strip(),
                    fmt_ci(paired(cell.gradient_anchor, cell.forward_anchor)).strip(),
                    fmt_ci(paired(cell.gradient_anchor, cell.gradient_shuffled)).strip(),
                    fmt_ci(capture(cell.gradient_anchor, cell.default_only, cell.oracle_subset)).strip(),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])
    legit = "\n".join(
        [
            "# DFW-E070 Legitimacy Check",
            "",
            f"- Compute: `{diag.status}` / `{diag.detail}`.",
            "- Target batches are used only for scoring and oracle diagnostics.",
            "- Validation-gradient scores use select shards only and compute gradients at theta, not at target labels.",
            "- Default-only control uses only the lr_multiplier=1 group with its own tuned outer pair.",
            "- Forward anchor repeats E067's forward finite-difference default-anchored rule.",
            "- Gradient anchor uses one validation backward per slice and scores all LR groups by `<g_V, action>`.",
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
