"""DFW-E064: cost/noise frontier for frozen grouped LR probes.

E063 replicated the favorable stationary result: if aligned workers deliberately
run different inner learning rates, a frozen group-level probe rule extracts a
small but repeatable held-out gain and collapses exactly when the groups are
exchangeable.

This experiment asks the next economic question.  How much probe evidence is
needed?  We keep the method fixed (`lambda=0.4`, `tau=2.0`) and sweep only the
probe budget: examples per forward slice and number of independent slices.
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


DATE = "20261005"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e064_gpu_group_probe_cost_frontier_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e064_gpu_group_probe_cost_frontier_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e064_gpu_group_probe_cost_frontier_LEGITIMACY_CHECK_{DATE}.md"

K_VALUES = e32.env_int_tuple("DFW_E064_K_VALUES", (8, 16, 32))
N_WORKERS = e32.env_int_tuple("DFW_E064_N_WORKERS", (8,))
ROUNDS = int(os.environ.get("DFW_E064_ROUNDS", "36"))
MAX_SELECT_BATCH = int(os.environ.get("DFW_E064_MAX_SELECT_BATCH", "768"))
BATCH_SIZES = e32.env_int_tuple("DFW_E064_BATCH_SIZES", (48, 96, 192, 384))
SLICE_COUNTS = e32.env_int_tuple("DFW_E064_SLICE_COUNTS", (1, 2, 4))
PILOT_SEEDS = e32.env_seed_tuple("DFW_E064_PILOT_SEEDS", tuple(range(931_000, 931_003)))
HELDOUT_SEEDS = e32.env_seed_tuple("DFW_E064_HELDOUT_SEEDS", tuple(range(932_000, 932_012)))
SPREAD = float(os.environ.get("DFW_E064_SPREAD", "16.0"))
LAM = float(os.environ.get("DFW_E064_LAM", "0.4"))
TAU = float(os.environ.get("DFW_E064_TAU", "2.0"))
EPS = 1e-12


mean = e59.mean
fmt_ci = e59.fmt_ci
paired = e59.paired
positive_ci = e59.positive_ci
spread_text = e59.spread_text


@dataclass(frozen=True)
class Budget:
    batch_size: int
    slices: int


@dataclass(frozen=True)
class BudgetScore:
    n_workers: int
    k_value: int
    budget: Budget
    group_values: tuple[float, ...]
    fixed_pair: tuple[float, float]
    fixed: list[float]
    oracle_subset: list[float]
    raw: list[float]
    shuffled: list[float]
    cost_pct: float


def available_budgets() -> tuple[Budget, ...]:
    budgets: list[Budget] = []
    for batch_size in BATCH_SIZES:
        for slices in SLICE_COUNTS:
            if slices == 4 and 2 * batch_size > MAX_SELECT_BATCH:
                continue
            if batch_size > MAX_SELECT_BATCH:
                continue
            budgets.append(Budget(batch_size=batch_size, slices=slices))
    return tuple(budgets)


def budget_loss(row: e60.GroupRoundRow, action, batch_size: int, slices: int) -> float:
    losses = [
        e30.loss_float(row.theta + action, row.select_x[:batch_size], row.select_y[:batch_size])
    ]
    if slices >= 2:
        losses.append(
            e30.loss_float(row.theta + action, row.select2_x[:batch_size], row.select2_y[:batch_size])
        )
    if slices >= 4:
        lo = batch_size
        hi = 2 * batch_size
        losses.append(e30.loss_float(row.theta + action, row.select_x[lo:hi], row.select_y[lo:hi]))
        losses.append(e30.loss_float(row.theta + action, row.select2_x[lo:hi], row.select2_y[lo:hi]))
    return mean(losses)


def group_losses(
    row: e60.GroupRoundRow,
    velocity,
    pair: tuple[float, float],
    budget: Budget,
) -> list[float]:
    alpha, beta = pair
    losses: list[float] = []
    for _value, update, _prior in e60.group_updates(row):
        action = alpha * (update + beta * velocity)
        losses.append(budget_loss(row, action, budget.batch_size, budget.slices))
    return losses


def score_policy(
    rows: list[e60.GroupRoundRow],
    pair: tuple[float, float],
    budget: Budget,
    rep_idx: int,
    shuffled: bool,
) -> float:
    alpha, beta = pair
    spec = e59.DampingSpec("e064_frozen_l40_t2", LAM, TAU, budget.slices)
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=rows[0].theta.device)
    raw = 0.0
    total = 0.0
    for row_idx, row in enumerate(rows):
        losses = group_losses(row, velocity, pair, budget)
        if shuffled:
            losses = e60.shuffled_losses(losses, row_idx, rep_idx)
        weights = e60.soft_damped_group_weights(row, losses, spec)
        agg = e60.aggregate_groups(e60.group_updates(row), weights)
        direction = (agg + beta * velocity).detach()
        raw += e60.improvement(row, row.fedavg)
        total += e60.improvement(row, alpha * direction)
        velocity = direction
    return total / raw if raw > EPS else total


def cost_pct(group_count: int, n_workers: int, k_value: int, budget: Budget) -> float:
    return 100.0 * group_count * budget.slices * budget.batch_size / (
        e30.TRAIN_SIZE * 3.0 * n_workers * k_value
    )


def capture(vals: list[float], fixed: list[float], oracle: list[float]) -> list[float]:
    out: list[float] = []
    for val, fix, ora in zip(vals, fixed, oracle):
        gap = ora - fix
        out.append((val - fix) / gap if abs(gap) > EPS else 0.0)
    return out


def diagnose_k(k_value: int, n_workers: int, device: str) -> list[BudgetScore]:
    e30.ROUNDS = ROUNDS
    e60.SELECT_BATCH = MAX_SELECT_BATCH
    pilot_groups = e60.make_groups(k_value, n_workers, SPREAD, PILOT_SEEDS, device)
    fixed_pair = e60.tune_fixed_pair(pilot_groups)
    heldout_groups = e60.make_groups(k_value, n_workers, SPREAD, HELDOUT_SEEDS, device)
    fixed_vals = [e60.score_fixed_group(rows, fixed_pair) for rows in heldout_groups]
    oracle_vals = [e60.score_group_oracle(rows, fixed_pair)[0] for rows in heldout_groups]
    groups = e60.group_updates(heldout_groups[0][0])
    out: list[BudgetScore] = []
    for budget in available_budgets():
        raw_vals = []
        shuffled_vals = []
        for rep_idx, rows in enumerate(heldout_groups):
            raw_vals.append(score_policy(rows, fixed_pair, budget, rep_idx, False))
            shuffled_vals.append(score_policy(rows, fixed_pair, budget, rep_idx, True))
        out.append(
            BudgetScore(
                n_workers=n_workers,
                k_value=k_value,
                budget=budget,
                group_values=tuple(value for value, _update, _prior in groups),
                fixed_pair=fixed_pair,
                fixed=fixed_vals,
                oracle_subset=oracle_vals,
                raw=raw_vals,
                shuffled=shuffled_vals,
                cost_pct=cost_pct(len(groups), n_workers, k_value, budget),
            )
        )
    return out


def decide(scores: list[BudgetScore], compute_status: str) -> str:
    if compute_status != "REAL_GPU":
        return "NO_GPU_HARNESS"
    sub1 = [score for score in scores if score.cost_pct <= 1.0 + 1e-12]
    sub1_raw = []
    sub1_shuf = []
    for score in sub1:
        sub1_raw.extend(paired(score.raw, score.fixed))
        sub1_shuf.extend(paired(score.raw, score.shuffled))
    if sub1 and positive_ci(sub1_raw) and positive_ci(sub1_shuf):
        return "SUB1_PERCENT_COST_FRONTIER_LIVE"
    all_raw = []
    all_shuf = []
    for score in scores:
        all_raw.extend(paired(score.raw, score.fixed))
        all_shuf.extend(paired(score.raw, score.shuffled))
    if positive_ci(all_raw) and positive_ci(all_shuf):
        return "HIGHER_COST_FRONTIER_ONLY"
    return "COST_FRONTIER_NEGATIVE"


def no_gpu_artifacts(diag: e30.ComputeDiagnostic) -> tuple[str, str, str]:
    run_log = "\n".join(
        (
            "DFW-E064 grouped LR probe cost frontier",
            f"host={platform.node()} python={sys.executable}",
            f"DECISION NO_GPU_HARNESS compute_status={diag.status} detail={diag.detail}",
        )
    )
    result = "\n".join(
        (
            "# DFW-E064 Result - Group Probe Cost Frontier",
            "",
            "Decision label: `NO_GPU_HARNESS`",
            "",
            f"Diagnostic: `{diag.detail}`.",
            "",
        )
    )
    legit = "# DFW-E064 Legitimacy Check\n\n- GPU workload: not run.\n"
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
    scores: list[BudgetScore] = []
    run_lines = [
        "DFW-E064 grouped LR probe cost frontier",
        f"host={platform.node()} python={sys.executable} compute_status={diag.status} {diag.detail}",
        f"model=torch_two_layer_binary_classifier dim={e30.DIM} N={N_WORKERS} K={K_VALUES} rounds={ROUNDS} eta={e30.ETA}",
        f"pilot_seeds={len(PILOT_SEEDS)} heldout_seeds={len(HELDOUT_SEEDS)} spread={spread_text(SPREAD)} max_select_batch={MAX_SELECT_BATCH}",
        f"batch_sizes={BATCH_SIZES} slices={SLICE_COUNTS} lam={LAM} tau={TAU}",
        "",
    ]
    for n_workers in N_WORKERS:
        for k_value in K_VALUES:
            for score in diagnose_k(k_value, n_workers, diag.device):
                scores.append(score)
                line = (
                    f"CELL K={k_value} N={n_workers} batch={score.budget.batch_size} "
                    f"slices={score.budget.slices} cost={score.cost_pct:.2f}% "
                    f"fixed_pair=({score.fixed_pair[0]:.1f},{score.fixed_pair[1]:.2f}) "
                    f"groups={','.join(f'{value:.3g}' for value in score.group_values)} "
                    f"fixed={fmt_ci(score.fixed)} oracle={fmt_ci(score.oracle_subset)} "
                    f"raw={fmt_ci(score.raw)} raw-fixed={fmt_ci(paired(score.raw, score.fixed))} "
                    f"raw-shuffled={fmt_ci(paired(score.raw, score.shuffled))} "
                    f"capture={fmt_ci(capture(score.raw, score.fixed, score.oracle_subset))}"
                )
                run_lines.append(line)
                print(line, flush=True)
    decision = decide(scores, diag.status)
    run_lines.append("")
    run_lines.append(f"DECISION {decision}")
    run_lines.append(f"compute_status={diag.status}")

    lines = [
        "# DFW-E064 Result - Group Probe Cost Frontier",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{diag.status}` via PyTorch CUDA: `{diag.detail}`.",
        "",
        "## Held-Out Scores",
        "",
        "| K | batch | slices | cost | fixed `(alpha,beta)` | fixed | oracle | raw | raw minus fixed | raw minus shuffled | capture |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for score in scores:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{score.k_value}",
                    f"{score.budget.batch_size}",
                    f"{score.budget.slices}",
                    f"{score.cost_pct:.2f}%",
                    f"({score.fixed_pair[0]:.1f},{score.fixed_pair[1]:.2f})",
                    fmt_ci(score.fixed).strip(),
                    fmt_ci(score.oracle_subset).strip(),
                    fmt_ci(score.raw).strip(),
                    fmt_ci(paired(score.raw, score.fixed)).strip(),
                    fmt_ci(paired(score.raw, score.shuffled)).strip(),
                    fmt_ci(capture(score.raw, score.fixed, score.oracle_subset)).strip(),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])
    legit = "\n".join(
        [
            "# DFW-E064 Legitimacy Check",
            "",
            f"- Compute: `{diag.status}` / `{diag.detail}`.",
            "- Target batches are used only for scoring and oracle diagnostics.",
            "- Method frozen to `lambda=0.4`, `tau=2.0`; only probe cost varies.",
            "- Four-slice budgets use disjoint prefixes from the two independent select draws.",
            "- Shuffled control rotates probe losses across LR groups.",
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
