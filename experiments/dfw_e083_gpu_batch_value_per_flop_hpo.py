"""DFW-E083: batch-size/value-per-FLOP internal-HPO arm.

Workers start from one shared state, but worker groups run different local
batch sizes.  The server probes the candidate group updates and chooses a
default-anchored mixture by estimated value per unit training FLOP.  Target
batches are used only for scoring and oracle diagnostics.
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


DATE = "20261006"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e083_gpu_batch_value_per_flop_hpo_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e083_gpu_batch_value_per_flop_hpo_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e083_gpu_batch_value_per_flop_hpo_LEGITIMACY_CHECK_{DATE}.md"

K_VALUES = e32.env_int_tuple("DFW_E083_K_VALUES", (8, 16))
N_WORKERS = int(os.environ.get("DFW_E083_N_WORKERS", "12"))
ROUNDS = int(os.environ.get("DFW_E083_ROUNDS", "32"))
BATCH_ARMS = e32.env_int_tuple("DFW_E083_BATCH_ARMS", (8, 16, 32))
DEFAULT_BATCH = int(os.environ.get("DFW_E083_DEFAULT_BATCH", "16"))
SELECT_BATCH = int(os.environ.get("DFW_E083_SELECT_BATCH", "384"))
PROBE_BATCH = int(os.environ.get("DFW_E083_PROBE_BATCH", "96"))
PROBE_SLICES = int(os.environ.get("DFW_E083_PROBE_SLICES", "2"))
PILOT_SEEDS = e32.env_seed_tuple("DFW_E083_PILOT_SEEDS", tuple(range(976_000, 976_002)))
HELDOUT_SEEDS = e32.env_seed_tuple("DFW_E083_HELDOUT_SEEDS", tuple(range(977_000, 977_008)))
LAM = float(os.environ.get("DFW_E083_LAM", "0.5"))
TAU = float(os.environ.get("DFW_E083_TAU", "2.0"))
EPS = 1e-12

mean = e59.mean
fmt_ci = e59.fmt_ci
paired = e59.paired
positive_ci = e59.positive_ci


@dataclass(frozen=True)
class BatchRoundRow:
    theta: object
    arm_updates: dict[int, object]
    select_x: object
    select_y: object
    select2_x: object
    select2_y: object
    target_x: object
    target_y: object
    target_start_loss: float


@dataclass(frozen=True)
class Cell:
    k_value: int
    fixed_pair: tuple[float, float]
    default: list[float]
    equal_mix: list[float]
    raw_probe: list[float]
    vpf_probe: list[float]
    vpf_shuffled: list[float]
    oracle_arm: list[float]
    oracle_subset: list[float]
    selected_cost_ratio: list[float]
    default_cost_ratio: float
    mixed_cost_ratio: float
    probe_overhead_pct: float


def arm_counts() -> dict[int, int]:
    if DEFAULT_BATCH not in BATCH_ARMS:
        raise ValueError("DEFAULT_BATCH must be in BATCH_ARMS")
    base = N_WORKERS // len(BATCH_ARMS)
    rem = N_WORKERS % len(BATCH_ARMS)
    counts = {arm: base for arm in BATCH_ARMS}
    for arm in BATCH_ARMS[:rem]:
        counts[arm] += 1
    return counts


def arm_for_workers() -> tuple[int, ...]:
    arms: list[int] = []
    for arm, count in arm_counts().items():
        arms.extend([arm] * count)
    if len(arms) != N_WORKERS:
        raise ValueError("batch-arm worker assignment has wrong length")
    return tuple(arms)


def make_rows(seed: int, k_value: int, device: str) -> list[BatchRoundRow]:
    data = e30.make_dataset(seed, device)
    shards = e30.worker_shards(e30.TRAIN_SIZE, N_WORKERS)
    worker_arms = arm_for_workers()
    theta = e30.init_theta(seed, 0.11, device)
    rows: list[BatchRoundRow] = []
    for round_idx in range(e30.ROUNDS):
        start = theta.detach().clone()
        local_thetas = [start.detach().clone() for _ in range(N_WORKERS)]
        for step_idx in range(k_value):
            for worker_id, batch_size in enumerate(worker_arms):
                bx, by = e30.sample_batch(
                    data.train_x,
                    data.train_y,
                    e30.rng_for(seed, 31, N_WORKERS, k_value, round_idx, step_idx, worker_id),
                    batch_size,
                    shards[worker_id],
                )
                _loss, grad = e30.loss_and_grad(local_thetas[worker_id], bx, by)
                local_thetas[worker_id] = (local_thetas[worker_id] - e30.ETA * grad).detach()

        updates = tuple((worker_theta - start).detach() for worker_theta in local_thetas)
        arm_updates = {}
        for arm in BATCH_ARMS:
            idxs = [idx for idx, value in enumerate(worker_arms) if value == arm]
            arm_updates[arm] = e30.mean_tensors([updates[idx] for idx in idxs]).detach()
        select_x, select_y = e30.sample_batch(
            data.select_x,
            data.select_y,
            e30.rng_for(seed, 43, N_WORKERS, k_value, round_idx),
            SELECT_BATCH,
        )
        select2_x, select2_y = e30.sample_batch(
            data.select_x,
            data.select_y,
            e30.rng_for(seed, 44, N_WORKERS, k_value, round_idx),
            SELECT_BATCH,
        )
        target_x, target_y = e30.sample_batch(
            data.target_x,
            data.target_y,
            e30.rng_for(seed, 47, N_WORKERS, k_value, round_idx),
            e30.TARGET_BATCH,
        )
        rows.append(
            BatchRoundRow(
                theta=start,
                arm_updates=arm_updates,
                select_x=select_x,
                select_y=select_y,
                select2_x=select2_x,
                select2_y=select2_y,
                target_x=target_x,
                target_y=target_y,
                target_start_loss=e30.loss_float(start, target_x, target_y),
            )
        )
        theta = (start + arm_updates[DEFAULT_BATCH]).detach()
    return rows


def make_groups(k_value: int, seeds: tuple[int, ...], device: str) -> list[list[BatchRoundRow]]:
    return [make_rows(seed, k_value, device) for seed in seeds]


def improvement(row: BatchRoundRow, update) -> float:
    return row.target_start_loss - e30.loss_float(row.theta + update, row.target_x, row.target_y)


def update_from_weights(row: BatchRoundRow, weights: dict[int, float]):
    total = e30.torch.zeros_like(next(iter(row.arm_updates.values())))
    for arm, weight in weights.items():
        total = total + float(weight) * row.arm_updates[arm]
    return total.detach()


def score_policy(rows: list[BatchRoundRow], choose_weights) -> float:
    raw = 0.0
    total = 0.0
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=rows[0].theta.device)
    for idx, row in enumerate(rows):
        weights, pair = choose_weights(row, velocity, idx)
        alpha, beta = pair
        update = update_from_weights(row, weights)
        direction = (update + beta * velocity).detach()
        raw += improvement(row, row.arm_updates[DEFAULT_BATCH])
        total += improvement(row, alpha * direction)
        velocity = direction
    return total / raw if raw > EPS else total


def tune_default_pair(groups: list[list[BatchRoundRow]]) -> tuple[float, float]:
    best_pair = (1.0, 0.0)
    best_score = -float("inf")
    default_weights = {arm: float(arm == DEFAULT_BATCH) for arm in BATCH_ARMS}
    for alpha in e32.ALPHAS:
        for beta in e32.BETAS:
            pair = (alpha, beta)
            score = mean(
                score_policy(rows, lambda _row, _vel, _idx, w=default_weights, p=pair: (w, p))
                for rows in groups
            )
            if score > best_score:
                best_score = score
                best_pair = pair
    return best_pair


def budget_losses(row: BatchRoundRow, velocity, pair: tuple[float, float]) -> list[float]:
    alpha, beta = pair
    out = []
    for arm in BATCH_ARMS:
        action = alpha * (row.arm_updates[arm] + beta * velocity)
        losses = [e30.loss_float(row.theta + action, row.select_x[:PROBE_BATCH], row.select_y[:PROBE_BATCH])]
        if PROBE_SLICES >= 2:
            losses.append(
                e30.loss_float(row.theta + action, row.select2_x[:PROBE_BATCH], row.select2_y[:PROBE_BATCH])
            )
        out.append(mean(losses))
    return out


def soft_weights(scores: list[float]) -> dict[int, float]:
    if len(scores) <= 1:
        return {BATCH_ARMS[0]: 1.0}
    spread = statistics.pstdev(scores)
    temp = max(TAU * spread, 1e-5)
    offset = max(scores)
    exps = [math.exp(max(-12.0, min(12.0, (score - offset) / temp))) for score in scores]
    denom = max(sum(exps), EPS)
    soft = [value / denom for value in exps]
    prior = [float(arm == DEFAULT_BATCH) for arm in BATCH_ARMS]
    return {
        arm: (1.0 - LAM) * base + LAM * weight
        for arm, base, weight in zip(BATCH_ARMS, prior, soft)
    }


def shuffled(scores: list[float], row_idx: int, rep_idx: int) -> list[float]:
    shift = (row_idx + 5 * rep_idx + 1) % len(scores)
    return scores[shift:] + scores[:shift]


def selector_weights(
    row: BatchRoundRow,
    velocity,
    pair: tuple[float, float],
    row_idx: int,
    rep_idx: int,
    *,
    value_per_flop: bool,
    shuffle: bool,
) -> dict[int, float]:
    losses = budget_losses(row, velocity, pair)
    start_loss = e30.loss_float(row.theta, row.select_x[:PROBE_BATCH], row.select_y[:PROBE_BATCH])
    scores = [start_loss - loss for loss in losses]
    if value_per_flop:
        scores = [score / max(arm / DEFAULT_BATCH, EPS) for score, arm in zip(scores, BATCH_ARMS)]
    if shuffle:
        scores = shuffled(scores, row_idx, rep_idx)
    return soft_weights(scores)


def oracle_arm_weights(row: BatchRoundRow, velocity, pair: tuple[float, float]) -> dict[int, float]:
    alpha, beta = pair
    best_arm = DEFAULT_BATCH
    best_score = -float("inf")
    for arm in BATCH_ARMS:
        action = alpha * (row.arm_updates[arm] + beta * velocity)
        gain = improvement(row, action)
        score = gain / max(arm / DEFAULT_BATCH, EPS)
        if score > best_score:
            best_score = score
            best_arm = arm
    return {arm: float(arm == best_arm) for arm in BATCH_ARMS}


def oracle_subset_weights(row: BatchRoundRow, velocity, pair: tuple[float, float]) -> dict[int, float]:
    alpha, beta = pair
    best_weights = {arm: float(arm == DEFAULT_BATCH) for arm in BATCH_ARMS}
    best_gain = -float("inf")
    for mask in range(1, 1 << len(BATCH_ARMS)):
        chosen = [arm for idx, arm in enumerate(BATCH_ARMS) if (mask >> idx) & 1]
        inv = [DEFAULT_BATCH / arm for arm in chosen]
        denom = max(sum(inv), EPS)
        weights = {arm: 0.0 for arm in BATCH_ARMS}
        for arm, value in zip(chosen, inv):
            weights[arm] = value / denom
        action = alpha * (update_from_weights(row, weights) + beta * velocity)
        gain = improvement(row, action)
        if gain > best_gain:
            best_gain = gain
            best_weights = weights
    return best_weights


def score_selector(
    rows: list[BatchRoundRow],
    pair: tuple[float, float],
    rep_idx: int,
    *,
    value_per_flop: bool,
    shuffle: bool,
) -> tuple[float, float]:
    raw = 0.0
    total = 0.0
    costs = []
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=rows[0].theta.device)
    for idx, row in enumerate(rows):
        weights = selector_weights(row, velocity, pair, idx, rep_idx, value_per_flop=value_per_flop, shuffle=shuffle)
        update = update_from_weights(row, weights)
        alpha, beta = pair
        direction = (update + beta * velocity).detach()
        raw += improvement(row, row.arm_updates[DEFAULT_BATCH])
        total += improvement(row, alpha * direction)
        costs.append(sum(weights[arm] * arm for arm in BATCH_ARMS) / DEFAULT_BATCH)
        velocity = direction
    return (total / raw if raw > EPS else total), mean(costs)


def score_oracle(rows: list[BatchRoundRow], pair: tuple[float, float], subset: bool) -> float:
    raw = 0.0
    total = 0.0
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=rows[0].theta.device)
    for row in rows:
        weights = oracle_subset_weights(row, velocity, pair) if subset else oracle_arm_weights(row, velocity, pair)
        update = update_from_weights(row, weights)
        alpha, beta = pair
        direction = (update + beta * velocity).detach()
        raw += improvement(row, row.arm_updates[DEFAULT_BATCH])
        total += improvement(row, alpha * direction)
        velocity = direction
    return total / raw if raw > EPS else total


def cost_ledger(k_value: int) -> tuple[float, float, float]:
    counts = arm_counts()
    default_units = N_WORKERS * k_value * DEFAULT_BATCH
    mixed_units = sum(counts[arm] * k_value * arm for arm in BATCH_ARMS)
    probe_units = len(BATCH_ARMS) * PROBE_SLICES * PROBE_BATCH
    return 1.0, mixed_units / max(default_units, EPS), 100.0 * probe_units / max(3.0 * mixed_units, EPS)


def diagnose_cell(k_value: int, device: str) -> Cell:
    e30.ROUNDS = ROUNDS
    pilot = make_groups(k_value, PILOT_SEEDS, device)
    fixed_pair = tune_default_pair(pilot)
    heldout = make_groups(k_value, HELDOUT_SEEDS, device)
    default_weights = {arm: float(arm == DEFAULT_BATCH) for arm in BATCH_ARMS}
    equal_weights = {arm: 1.0 / len(BATCH_ARMS) for arm in BATCH_ARMS}
    default_vals = [score_policy(rows, lambda _row, _vel, _idx, w=default_weights, p=fixed_pair: (w, p)) for rows in heldout]
    equal_vals = [score_policy(rows, lambda _row, _vel, _idx, w=equal_weights, p=fixed_pair: (w, p)) for rows in heldout]
    raw_vals = []
    vpf_vals = []
    shuf_vals = []
    costs = []
    oracle_arm_vals = []
    oracle_subset_vals = []
    for rep_idx, rows in enumerate(heldout):
        raw_value, _raw_cost = score_selector(rows, fixed_pair, rep_idx, value_per_flop=False, shuffle=False)
        vpf_value, vpf_cost = score_selector(rows, fixed_pair, rep_idx, value_per_flop=True, shuffle=False)
        shuf_value, _shuf_cost = score_selector(rows, fixed_pair, rep_idx, value_per_flop=True, shuffle=True)
        raw_vals.append(raw_value)
        vpf_vals.append(vpf_value)
        shuf_vals.append(shuf_value)
        costs.append(vpf_cost)
        oracle_arm_vals.append(score_oracle(rows, fixed_pair, subset=False))
        oracle_subset_vals.append(score_oracle(rows, fixed_pair, subset=True))
    default_cost, mixed_cost, probe_overhead = cost_ledger(k_value)
    return Cell(
        k_value=k_value,
        fixed_pair=fixed_pair,
        default=default_vals,
        equal_mix=equal_vals,
        raw_probe=raw_vals,
        vpf_probe=vpf_vals,
        vpf_shuffled=shuf_vals,
        oracle_arm=oracle_arm_vals,
        oracle_subset=oracle_subset_vals,
        selected_cost_ratio=costs,
        default_cost_ratio=default_cost,
        mixed_cost_ratio=mixed_cost,
        probe_overhead_pct=probe_overhead,
    )


def decide(cells: list[Cell], compute_status: str) -> str:
    if compute_status != "REAL_GPU":
        return "NO_GPU_HARNESS"
    probe_default = []
    probe_shuffled = []
    oracle_default = []
    for cell in cells:
        probe_default.extend(paired(cell.vpf_probe, cell.default))
        probe_shuffled.extend(paired(cell.vpf_probe, cell.vpf_shuffled))
        oracle_default.extend(paired(cell.oracle_subset, cell.default))
    if positive_ci(probe_default) and positive_ci(probe_shuffled):
        return "BATCH_VALUE_PER_FLOP_PROBE_LIVE"
    if positive_ci(oracle_default):
        return "BATCH_VALUE_PER_FLOP_ORACLE_ONLY"
    return "BATCH_VALUE_PER_FLOP_NEGATIVE"


def no_gpu_artifacts(diag: e30.ComputeDiagnostic) -> tuple[str, str, str]:
    run_log = "\n".join(
        (
            "DFW-E083 batch-size value-per-FLOP HPO",
            f"host={platform.node()} python={sys.executable}",
            f"DECISION NO_GPU_HARNESS compute_status={diag.status} detail={diag.detail}",
        )
    )
    result = "# DFW-E083 Result - Batch Value-per-FLOP HPO\n\nDecision label: `NO_GPU_HARNESS`\n"
    legit = "# DFW-E083 Legitimacy Check\n\n- GPU workload: not run.\n"
    return run_log + "\n", result + "\n", legit + "\n"


def run_experiment() -> tuple[str, str, str]:
    diag = e30.compute_diagnostic()
    if diag.status != "REAL_GPU":
        return no_gpu_artifacts(diag)
    e30.torch.set_default_dtype(e30.torch.float32)
    try:
        e30.torch.set_num_threads(1)
    except Exception:
        pass
    cells = [diagnose_cell(k_value, diag.device) for k_value in K_VALUES]
    decision = decide(cells, diag.status)
    run_lines = [
        "DFW-E083 batch-size value-per-FLOP HPO",
        f"host={platform.node()} python={sys.executable} compute_status={diag.status} {diag.detail}",
        f"N={N_WORKERS} K={K_VALUES} rounds={ROUNDS} batch_arms={BATCH_ARMS} default_batch={DEFAULT_BATCH}",
        f"pilot_seeds={len(PILOT_SEEDS)} heldout_seeds={len(HELDOUT_SEEDS)} probe_batch={PROBE_BATCH} slices={PROBE_SLICES}",
        "",
    ]
    for cell in cells:
        run_lines.append(
            f"CELL K={cell.k_value} fixed_pair=({cell.fixed_pair[0]:.1f},{cell.fixed_pair[1]:.2f}) "
            f"default={fmt_ci(cell.default)} equal_mix={fmt_ci(cell.equal_mix)} "
            f"raw_probe={fmt_ci(cell.raw_probe)} vpf_probe={fmt_ci(cell.vpf_probe)} "
            f"vpf_shuffled={fmt_ci(cell.vpf_shuffled)} oracle_arm={fmt_ci(cell.oracle_arm)} "
            f"oracle_subset={fmt_ci(cell.oracle_subset)} "
            f"vpf-default={fmt_ci(paired(cell.vpf_probe, cell.default))} "
            f"vpf-shuffled={fmt_ci(paired(cell.vpf_probe, cell.vpf_shuffled))} "
            f"selected_cost={fmt_ci(cell.selected_cost_ratio)} mixed_cost={cell.mixed_cost_ratio:.3f} "
            f"probe_overhead={cell.probe_overhead_pct:.2f}%"
        )
    run_lines.append("")
    run_lines.append(f"DECISION {decision}")
    run_lines.append(f"compute_status={diag.status}")

    lines = [
        "# DFW-E083 Result - Batch Value-per-FLOP HPO",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{diag.status}` via PyTorch CUDA: `{diag.detail}`.",
        "",
        "## Held-Out Scores",
        "",
        "| K | fixed `(alpha,beta)` | default | equal mix | raw probe | value/FLOP probe | shuffled | oracle arm | oracle subset | vpf-default | vpf-shuffled | selected cost | mixed cost | probe overhead |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(cell.k_value),
                    f"({cell.fixed_pair[0]:.1f},{cell.fixed_pair[1]:.2f})",
                    fmt_ci(cell.default).strip(),
                    fmt_ci(cell.equal_mix).strip(),
                    fmt_ci(cell.raw_probe).strip(),
                    fmt_ci(cell.vpf_probe).strip(),
                    fmt_ci(cell.vpf_shuffled).strip(),
                    fmt_ci(cell.oracle_arm).strip(),
                    fmt_ci(cell.oracle_subset).strip(),
                    fmt_ci(paired(cell.vpf_probe, cell.default)).strip(),
                    fmt_ci(paired(cell.vpf_probe, cell.vpf_shuffled)).strip(),
                    fmt_ci(cell.selected_cost_ratio).strip(),
                    f"{cell.mixed_cost_ratio:.3f}x",
                    f"{cell.probe_overhead_pct:.2f}%",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```", ""])
    legit = "\n".join(
        [
            "# DFW-E083 Legitimacy Check",
            "",
            f"- Compute: `{diag.status}` / `{diag.detail}`.",
            "- Same base state per round; worker groups differ only by local batch-size arm.",
            "- Default batch arm is present and used as the conservative prior.",
            "- Probe selector uses select batches only; target batches score policies and oracle diagnostics.",
            "- Shuffled value/FLOP labels are included.",
            "- FLOP ledger reports mixed training cost and probe overhead relative to batch-size work.",
            "",
        ]
    )
    return "\n".join(run_lines) + "\n", "\n".join(lines), legit


def main() -> None:
    run_log, result, legit = run_experiment()
    RUN_LOG_PATH.write_text(run_log, encoding="utf-8")
    RESULT_PATH.write_text(result, encoding="utf-8")
    LEGITIMACY_PATH.write_text(legit, encoding="utf-8")
    print(result)
    print(f"wrote {RESULT_PATH}")
    print(f"wrote {RUN_LOG_PATH}")
    print(f"wrote {LEGITIMACY_PATH}")


if __name__ == "__main__":
    main()
