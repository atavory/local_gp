"""DFW-E065: two-dimensional grouped hyperparameter probes.

E060-E064 made a narrow but repeatable result: aligned workers running
different inner learning rates create first-order non-exchangeability, and
group-level probes can conservatively extract a small value signal at sub-1%
probe cost.

This experiment tests whether the mechanism survives a second optimizer
hyperparameter.  Workers are grouped by `(inner_lr_multiplier, inner_momentum)`.
The probe rule is still the frozen soft damped group rule; no response surface
or GP is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools
import math
import os
import platform
import statistics
import sys

import dfw_e030_gpu_real_path_local_map as e30
import dfw_e032_gpu_probe_outer_control_real_path as e32
import dfw_e059_gpu_stationary_inner_lr_probe_admission as e59


DATE = "20261005"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e065_gpu_lr_momentum_group_probe_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e065_gpu_lr_momentum_group_probe_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e065_gpu_lr_momentum_group_probe_LEGITIMACY_CHECK_{DATE}.md"

K_VALUES = e32.env_int_tuple("DFW_E065_K_VALUES", (8, 16))
N_WORKERS = int(os.environ.get("DFW_E065_N_WORKERS", "18"))
ROUNDS = int(os.environ.get("DFW_E065_ROUNDS", "36"))
MAX_SELECT_BATCH = int(os.environ.get("DFW_E065_MAX_SELECT_BATCH", "384"))
PROBE_BATCH = int(os.environ.get("DFW_E065_PROBE_BATCH", "96"))
PROBE_SLICES = int(os.environ.get("DFW_E065_PROBE_SLICES", "4"))
PILOT_SEEDS = e32.env_seed_tuple("DFW_E065_PILOT_SEEDS", tuple(range(933_000, 933_003)))
HELDOUT_SEEDS = e32.env_seed_tuple("DFW_E065_HELDOUT_SEEDS", tuple(range(934_000, 934_008)))
LR_ARMS = e59.env_float_tuple("DFW_E065_LR_ARMS", (0.5, 1.0, 2.0))
MOMENTUM_ARMS = e59.env_float_tuple("DFW_E065_MOMENTUM_ARMS", (0.0, 0.5, 0.9))
LAM = float(os.environ.get("DFW_E065_LAM", "0.4"))
TAU = float(os.environ.get("DFW_E065_TAU", "2.0"))
ALPHAS = e59.ALPHAS
BETAS = e59.BETAS
EPS = 1e-12


mean = e59.mean
fmt_ci = e59.fmt_ci
paired = e59.paired
positive_ci = e59.positive_ci


Arm = tuple[float, float]


@dataclass(frozen=True)
class ArmRoundRow:
    theta: object
    fedavg: object
    worker_updates: tuple[object, ...]
    arms: tuple[Arm, ...]
    select_x: object
    select_y: object
    select2_x: object
    select2_y: object
    target_x: object
    target_y: object
    target_start_loss: float


@dataclass(frozen=True)
class Cell:
    n_workers: int
    k_value: int
    arms: tuple[Arm, ...]
    fixed_pair: tuple[float, float]
    default_pair: tuple[float, float]
    fixed_all: list[float]
    default_only: list[float]
    oracle_subset: list[float]
    oracle_single: list[float]
    raw: list[float]
    shuffled: list[float]
    cost_pct: float


def arm_grid() -> tuple[Arm, ...]:
    return tuple((lr, mom) for lr in LR_ARMS for mom in MOMENTUM_ARMS)


def worker_arms(n_workers: int) -> tuple[Arm, ...]:
    arms = arm_grid()
    if n_workers < len(arms):
        raise ValueError(f"N_WORKERS={n_workers} must be >= arm count {len(arms)}")
    reps = math.ceil(n_workers / len(arms))
    return tuple((list(arms) * reps)[:n_workers])


def make_rows(seed: int, n_workers: int, k_value: int, device: str) -> list[ArmRoundRow]:
    data = e30.make_dataset(seed, device)
    shards = e30.worker_shards(e30.TRAIN_SIZE, n_workers)
    arms = worker_arms(n_workers)
    theta = e30.init_theta(seed, 0.11, device)
    rows: list[ArmRoundRow] = []
    for round_idx in range(e30.ROUNDS):
        start = theta.detach().clone()
        local_thetas = [start.detach().clone() for _ in range(n_workers)]
        local_vels = [e30.torch.zeros_like(start) for _ in range(n_workers)]
        for step_idx in range(k_value):
            for worker_id, (lr_mult, momentum) in enumerate(arms):
                bx, by = e30.sample_batch(
                    data.train_x,
                    data.train_y,
                    e30.rng_for(seed, 53, n_workers, round_idx, step_idx, worker_id),
                    e30.LOCAL_BATCH,
                    shards[worker_id],
                )
                _loss, grad = e30.loss_and_grad(local_thetas[worker_id], bx, by)
                local_vels[worker_id] = (momentum * local_vels[worker_id] - e30.ETA * lr_mult * grad).detach()
                local_thetas[worker_id] = (local_thetas[worker_id] + local_vels[worker_id]).detach()
        worker_updates = tuple((worker_theta - start).detach() for worker_theta in local_thetas)
        fedavg = e30.mean_tensors(list(worker_updates)).detach()
        select_x, select_y = e30.sample_batch(
            data.select_x,
            data.select_y,
            e30.rng_for(seed, 57, n_workers, round_idx),
            MAX_SELECT_BATCH,
        )
        select2_x, select2_y = e30.sample_batch(
            data.select_x,
            data.select_y,
            e30.rng_for(seed, 59, n_workers, round_idx),
            MAX_SELECT_BATCH,
        )
        target_x, target_y = e30.sample_batch(
            data.target_x,
            data.target_y,
            e30.rng_for(seed, 61, n_workers, round_idx),
            e30.TARGET_BATCH,
        )
        rows.append(
            ArmRoundRow(
                theta=start,
                fedavg=fedavg,
                worker_updates=worker_updates,
                arms=arms,
                select_x=select_x,
                select_y=select_y,
                select2_x=select2_x,
                select2_y=select2_y,
                target_x=target_x,
                target_y=target_y,
                target_start_loss=e30.loss_float(start, target_x, target_y),
            )
        )
        theta = (start + fedavg).detach()
    return rows


def make_groups(k_value: int, seeds: tuple[int, ...], device: str) -> list[list[ArmRoundRow]]:
    return [make_rows(seed, N_WORKERS, k_value, device) for seed in seeds]


def improvement(row: ArmRoundRow, update) -> float:
    return row.target_start_loss - e30.loss_float(row.theta + update, row.target_x, row.target_y)


def group_updates(row: ArmRoundRow) -> tuple[tuple[Arm, object, float], ...]:
    out = []
    unique = tuple(sorted(set(row.arms)))
    for arm in unique:
        idxs = [idx for idx, worker_arm in enumerate(row.arms) if worker_arm == arm]
        update = e30.mean_tensors([row.worker_updates[idx] for idx in idxs]).detach()
        out.append((arm, update, len(idxs) / len(row.worker_updates)))
    return tuple(out)


def aggregate_groups(groups: tuple[tuple[Arm, object, float], ...], weights: list[float]):
    total = e30.torch.zeros_like(groups[0][1])
    for weight, (_arm, update, _prior) in zip(weights, groups):
        total = total + float(weight) * update
    return total.detach()


def score_fixed(rows: list[ArmRoundRow], pair: tuple[float, float]) -> float:
    alpha, beta = pair
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=rows[0].theta.device)
    raw = 0.0
    total = 0.0
    for row in rows:
        direction = (row.fedavg + beta * velocity).detach()
        raw += improvement(row, row.fedavg)
        total += improvement(row, alpha * direction)
        velocity = direction
    return total / raw if raw > EPS else total


def default_update(row: ArmRoundRow):
    default = (1.0, 0.0)
    matches = [update for arm, update, _prior in group_updates(row) if arm == default]
    if not matches:
        raise ValueError("default arm (1.0, 0.0) missing")
    return matches[0]


def score_default(rows: list[ArmRoundRow], pair: tuple[float, float]) -> float:
    alpha, beta = pair
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=rows[0].theta.device)
    raw = 0.0
    total = 0.0
    for row in rows:
        update = default_update(row)
        direction = (update + beta * velocity).detach()
        raw += improvement(row, row.fedavg)
        total += improvement(row, alpha * direction)
        velocity = direction
    return total / raw if raw > EPS else total


def tune_pair(groups: list[list[ArmRoundRow]], scorer) -> tuple[float, float]:
    best_pair = (1.0, 0.0)
    best_score = -float("inf")
    for alpha in ALPHAS:
        for beta in BETAS:
            score = mean(scorer(rows, (alpha, beta)) for rows in groups)
            if score > best_score:
                best_score = score
                best_pair = (alpha, beta)
    return best_pair


def budget_loss(row: ArmRoundRow, action) -> float:
    b = PROBE_BATCH
    losses = [e30.loss_float(row.theta + action, row.select_x[:b], row.select_y[:b])]
    if PROBE_SLICES >= 2:
        losses.append(e30.loss_float(row.theta + action, row.select2_x[:b], row.select2_y[:b]))
    if PROBE_SLICES >= 4:
        losses.append(e30.loss_float(row.theta + action, row.select_x[b : 2 * b], row.select_y[b : 2 * b]))
        losses.append(e30.loss_float(row.theta + action, row.select2_x[b : 2 * b], row.select2_y[b : 2 * b]))
    return mean(losses)


def group_losses(row: ArmRoundRow, velocity, pair: tuple[float, float]) -> list[float]:
    alpha, beta = pair
    return [budget_loss(row, alpha * (update + beta * velocity)) for _arm, update, _prior in group_updates(row)]


def soft_weights(row: ArmRoundRow, losses: list[float]) -> list[float]:
    groups = group_updates(row)
    priors = [prior for _arm, _update, prior in groups]
    if len(losses) <= 1:
        return priors
    loc = mean(losses)
    scale = statistics.stdev(losses)
    temp = max(TAU * scale, 1e-5)
    logits = [max(-12.0, min(12.0, (loc - loss) / temp)) for loss in losses]
    offset = max(logits)
    exps = [math.exp(logit - offset) for logit in logits]
    denom = max(sum(exps), EPS)
    soft = [value / denom for value in exps]
    return [(1.0 - LAM) * prior + LAM * weight for prior, weight in zip(priors, soft)]


def shuffled_losses(losses: list[float], row_idx: int, rep_idx: int) -> list[float]:
    shift = (row_idx + 5 * rep_idx + 1) % len(losses)
    return losses[shift:] + losses[:shift]


def score_probe(rows: list[ArmRoundRow], pair: tuple[float, float], rep_idx: int, shuffled: bool) -> float:
    alpha, beta = pair
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=rows[0].theta.device)
    raw = 0.0
    total = 0.0
    for row_idx, row in enumerate(rows):
        losses = group_losses(row, velocity, pair)
        if shuffled:
            losses = shuffled_losses(losses, row_idx, rep_idx)
        weights = soft_weights(row, losses)
        agg = aggregate_groups(group_updates(row), weights)
        direction = (agg + beta * velocity).detach()
        raw += improvement(row, row.fedavg)
        total += improvement(row, alpha * direction)
        velocity = direction
    return total / raw if raw > EPS else total


def score_oracle(rows: list[ArmRoundRow], pair: tuple[float, float]) -> tuple[float, float]:
    alpha, beta = pair
    group_count = len(group_updates(rows[0]))
    masks = [tuple(idx for idx in range(group_count) if (mask >> idx) & 1) for mask in range(1, 1 << group_count)]
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=rows[0].theta.device)
    single_velocity = e30.torch.zeros_like(velocity)
    raw = 0.0
    total = 0.0
    single_total = 0.0
    for row in rows:
        groups = group_updates(row)
        priors = [prior for _arm, _update, prior in groups]
        raw += improvement(row, row.fedavg)
        best_loss = float("inf")
        best_agg = row.fedavg
        best_single_loss = float("inf")
        best_single = groups[0][1]
        for mask in masks:
            denom = sum(priors[idx] for idx in mask)
            weights = [0.0] * group_count
            for idx in mask:
                weights[idx] = priors[idx] / max(denom, EPS)
            agg = aggregate_groups(groups, weights)
            loss = e30.loss_float(row.theta + alpha * (agg + beta * velocity), row.target_x, row.target_y)
            if loss < best_loss:
                best_loss = loss
                best_agg = agg
            if len(mask) == 1:
                single_loss = e30.loss_float(row.theta + alpha * (agg + beta * single_velocity), row.target_x, row.target_y)
                if single_loss < best_single_loss:
                    best_single_loss = single_loss
                    best_single = agg
        total += row.target_start_loss - best_loss
        single_total += row.target_start_loss - best_single_loss
        velocity = (best_agg + beta * velocity).detach()
        single_velocity = (best_single + beta * single_velocity).detach()
    return (total / raw if raw > EPS else total), (single_total / raw if raw > EPS else single_total)


def cost_pct(group_count: int, k_value: int) -> float:
    return 100.0 * group_count * PROBE_SLICES * PROBE_BATCH / (
        e30.TRAIN_SIZE * 3.0 * N_WORKERS * k_value
    )


def diagnose_cell(k_value: int, device: str) -> Cell:
    e30.ROUNDS = ROUNDS
    if 2 * PROBE_BATCH > MAX_SELECT_BATCH and PROBE_SLICES >= 4:
        raise ValueError("MAX_SELECT_BATCH must cover two chunks from each select draw")
    pilot = make_groups(k_value, PILOT_SEEDS, device)
    fixed_pair = tune_pair(pilot, score_fixed)
    default_pair = tune_pair(pilot, score_default)
    heldout = make_groups(k_value, HELDOUT_SEEDS, device)
    fixed_vals = [score_fixed(rows, fixed_pair) for rows in heldout]
    default_vals = [score_default(rows, default_pair) for rows in heldout]
    oracle_vals = []
    oracle_single_vals = []
    raw_vals = []
    shuffled_vals = []
    for rep_idx, rows in enumerate(heldout):
        subset, single = score_oracle(rows, fixed_pair)
        oracle_vals.append(subset)
        oracle_single_vals.append(single)
        raw_vals.append(score_probe(rows, fixed_pair, rep_idx, False))
        shuffled_vals.append(score_probe(rows, fixed_pair, rep_idx, True))
    first_groups = group_updates(heldout[0][0])
    return Cell(
        n_workers=N_WORKERS,
        k_value=k_value,
        arms=tuple(arm for arm, _update, _prior in first_groups),
        fixed_pair=fixed_pair,
        default_pair=default_pair,
        fixed_all=fixed_vals,
        default_only=default_vals,
        oracle_subset=oracle_vals,
        oracle_single=oracle_single_vals,
        raw=raw_vals,
        shuffled=shuffled_vals,
        cost_pct=cost_pct(len(first_groups), k_value),
    )


def decide(cells: list[Cell], compute_status: str) -> str:
    if compute_status != "REAL_GPU":
        return "NO_GPU_HARNESS"
    raw_fixed = []
    raw_shuf = []
    raw_default = []
    oracle_fixed = []
    for cell in cells:
        raw_fixed.extend(paired(cell.raw, cell.fixed_all))
        raw_shuf.extend(paired(cell.raw, cell.shuffled))
        raw_default.extend(paired(cell.raw, cell.default_only))
        oracle_fixed.extend(paired(cell.oracle_subset, cell.fixed_all))
    if positive_ci(raw_fixed) and positive_ci(raw_shuf) and positive_ci(raw_default):
        return "SECOND_DIM_GROUP_PROBE_BEATS_DEFAULT"
    if positive_ci(raw_fixed) and positive_ci(raw_shuf):
        return "SECOND_DIM_GROUP_PROBE_BEATS_PARALLEL_AVERAGE"
    if positive_ci(oracle_fixed):
        return "SECOND_DIM_ORACLE_HEADROOM_EXTRACTION_WEAK"
    return "SECOND_DIM_GROUP_PROBE_NEGATIVE"


def capture(vals: list[float], fixed: list[float], oracle: list[float]) -> list[float]:
    out = []
    for val, fix, ora in zip(vals, fixed, oracle):
        gap = ora - fix
        out.append((val - fix) / gap if abs(gap) > EPS else 0.0)
    return out


def no_gpu_artifacts(diag: e30.ComputeDiagnostic) -> tuple[str, str, str]:
    run_log = "\n".join(
        (
            "DFW-E065 LR x momentum group probe",
            f"host={platform.node()} python={sys.executable}",
            f"DECISION NO_GPU_HARNESS compute_status={diag.status} detail={diag.detail}",
        )
    )
    result = "\n".join(
        (
            "# DFW-E065 Result - LR x Momentum Group Probe",
            "",
            "Decision label: `NO_GPU_HARNESS`",
            "",
            f"Diagnostic: `{diag.detail}`.",
            "",
        )
    )
    legit = "# DFW-E065 Legitimacy Check\n\n- GPU workload: not run.\n"
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
    cells: list[Cell] = []
    run_lines = [
        "DFW-E065 LR x momentum group probe",
        f"host={platform.node()} python={sys.executable} compute_status={diag.status} {diag.detail}",
        f"model=torch_two_layer_binary_classifier dim={e30.DIM} N={N_WORKERS} K={K_VALUES} rounds={ROUNDS} eta={e30.ETA}",
        f"lr_arms={LR_ARMS} momentum_arms={MOMENTUM_ARMS} probe_batch={PROBE_BATCH} slices={PROBE_SLICES} lam={LAM} tau={TAU}",
        f"pilot_seeds={len(PILOT_SEEDS)} heldout_seeds={len(HELDOUT_SEEDS)}",
        "",
    ]
    for k_value in K_VALUES:
        cell = diagnose_cell(k_value, diag.device)
        cells.append(cell)
        line = (
            f"CELL K={k_value} N={cell.n_workers} arms={len(cell.arms)} cost={cell.cost_pct:.2f}% "
            f"fixed_pair=({cell.fixed_pair[0]:.1f},{cell.fixed_pair[1]:.2f}) "
            f"default_pair=({cell.default_pair[0]:.1f},{cell.default_pair[1]:.2f}) "
            f"fixed_all={fmt_ci(cell.fixed_all)} default_only={fmt_ci(cell.default_only)} "
            f"oracle_subset={fmt_ci(cell.oracle_subset)} oracle_single={fmt_ci(cell.oracle_single)} "
            f"raw={fmt_ci(cell.raw)} raw-fixed={fmt_ci(paired(cell.raw, cell.fixed_all))} "
            f"raw-default={fmt_ci(paired(cell.raw, cell.default_only))} "
            f"raw-shuffled={fmt_ci(paired(cell.raw, cell.shuffled))} "
            f"capture={fmt_ci(capture(cell.raw, cell.fixed_all, cell.oracle_subset))}"
        )
        run_lines.append(line)
        print(line, flush=True)
    decision = decide(cells, diag.status)
    run_lines.append("")
    run_lines.append(f"DECISION {decision}")
    run_lines.append(f"compute_status={diag.status}")

    lines = [
        "# DFW-E065 Result - LR x Momentum Group Probe",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{diag.status}` via PyTorch CUDA: `{diag.detail}`.",
        "",
        "## Held-Out Scores",
        "",
        "| K | arms | cost | fixed `(alpha,beta)` | default `(alpha,beta)` | fixed all | default only | oracle subset | oracle single | raw | raw minus fixed | raw minus default | raw minus shuffled | capture |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{cell.k_value}",
                    f"{len(cell.arms)}",
                    f"{cell.cost_pct:.2f}%",
                    f"({cell.fixed_pair[0]:.1f},{cell.fixed_pair[1]:.2f})",
                    f"({cell.default_pair[0]:.1f},{cell.default_pair[1]:.2f})",
                    fmt_ci(cell.fixed_all).strip(),
                    fmt_ci(cell.default_only).strip(),
                    fmt_ci(cell.oracle_subset).strip(),
                    fmt_ci(cell.oracle_single).strip(),
                    fmt_ci(cell.raw).strip(),
                    fmt_ci(paired(cell.raw, cell.fixed_all)).strip(),
                    fmt_ci(paired(cell.raw, cell.default_only)).strip(),
                    fmt_ci(paired(cell.raw, cell.shuffled)).strip(),
                    fmt_ci(capture(cell.raw, cell.fixed_all, cell.oracle_subset)).strip(),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])
    legit = "\n".join(
        [
            "# DFW-E065 Legitimacy Check",
            "",
            f"- Compute: `{diag.status}` / `{diag.detail}`.",
            "- Target batches are used only for scoring and oracle diagnostics.",
            "- Workers are aligned at the same base point each round.",
            "- All arms have the same local step count; only LR and inner momentum vary.",
            "- Default-only control uses the `(lr=1.0, momentum=0.0)` arm.",
            "- Shuffled control rotates probe losses across LR/momentum groups.",
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
