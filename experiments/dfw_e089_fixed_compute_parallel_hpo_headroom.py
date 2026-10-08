"""DFW-E089: fixed-compute parallel-HPO comparison in E063 headroom regime.

E088 tested the right comparator but only as an ultra-small smoke.  This run
keeps the fixed-compute accounting and moves the workload to the grouped-LR
spread-16 regime where E063 already showed real same-round HPO content.

The comparator is deliberately narrow:

  * internal HPO uses all workers in one shared trajectory as grouped LR arms;
  * external HPO splits the same total workers across separate LR trajectories
    and selects one by validation loss;
  * target-oracle selection is diagnostic only.
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

import dfw_e030_gpu_real_path_local_map as e30
import dfw_e032_gpu_probe_outer_control_real_path as e32
import dfw_e059_gpu_stationary_inner_lr_probe_admission as e59


DATE = "20261008"
OUT_DIR = Path(os.environ.get("DFW_E089_OUT_DIR", Path(__file__).resolve().parent)).resolve()
RESULT_PATH = OUT_DIR / f"dfw_e089_fixed_compute_parallel_hpo_headroom_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e089_fixed_compute_parallel_hpo_headroom_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e089_fixed_compute_parallel_hpo_headroom_LEGITIMACY_CHECK_{DATE}.md"

K_VALUES = e32.env_int_tuple("DFW_E089_K_VALUES", (8, 16, 32))
N_WORKERS = int(os.environ.get("DFW_E089_N_WORKERS", "8"))
ROUNDS = int(os.environ.get("DFW_E089_ROUNDS", "36"))
SELECT_BATCH = int(os.environ.get("DFW_E089_SELECT_BATCH", "384"))
PILOT_SEEDS = e32.env_seed_tuple("DFW_E089_PILOT_SEEDS", tuple(range(993_000, 993_003)))
HELDOUT_SEEDS = e32.env_seed_tuple("DFW_E089_HELDOUT_SEEDS", tuple(range(994_000, 994_008)))
SPREAD = float(os.environ.get("DFW_E089_SPREAD", "16"))
LR_ARMS = tuple(
    float(part)
    for part in os.environ.get(
        "DFW_E089_LR_ARMS",
        f"{1.0 / math.sqrt(SPREAD):.12g},1.0,{math.sqrt(SPREAD):.12g}",
    ).split(",")
)
ARM_COUNTS = tuple(int(part) for part in os.environ.get("DFW_E089_ARM_COUNTS", "2,4,2").split(","))
PAIR_MODE = os.environ.get("DFW_E089_PAIR_MODE", "e063_fixed").strip()
ALPHAS = tuple(float(part) for part in os.environ.get("DFW_E089_ALPHAS", "1,2,3,4,5,6").split(","))
BETAS = tuple(float(part) for part in os.environ.get("DFW_E089_BETAS", "0,0.1,0.2,0.3,0.4,0.5,0.6").split(","))
SPEC = e59.DampingSpec("group_mean2_l40_t2", 0.40, 2.0, 2)
E063_PAIR_BY_K = {
    8: (3.0, 0.40),
    16: (2.0, 0.20),
    32: (2.0, 0.00),
}
EPS = 1e-12

mean = e59.mean
fmt_ci = e59.fmt_ci
mean_ci = e59.mean_ci
paired = e59.paired
positive_ci = e59.positive_ci


@dataclass(frozen=True)
class DatasetState:
    data: e30.Dataset
    theta0: object
    target_start_loss: float


@dataclass(frozen=True)
class ArmUpdate:
    lr: float
    count: int
    update: object
    prior: float


@dataclass(frozen=True)
class TrajectoryScore:
    target_improvement: float
    select_loss: float
    target_loss: float


@dataclass(frozen=True)
class Cell:
    k_value: int
    pair: tuple[float, float]
    pair_source: str
    low_full_worker_arm: list[float]
    default_full_worker_arm: list[float]
    parallel_average: list[float]
    internal_probe: list[float]
    internal_shuffled: list[float]
    external_valid: list[float]
    external_oracle: list[float]
    best_full_worker_arm: list[float]
    cost_pct: float


def compute_status() -> tuple[str, str, str]:
    diag = e30.compute_diagnostic()
    if diag.status == "REAL_GPU":
        return diag.status, diag.detail, diag.device
    if e30.torch is None:
        return "NO_TORCH", diag.detail, "cpu"
    version = getattr(e30.torch, "__version__", "unknown")
    return "TORCH_CPU", f"torch={version} cuda_available=False", "cpu"


def check_arm_budget() -> None:
    if len(LR_ARMS) != len(ARM_COUNTS):
        raise ValueError("LR_ARMS and ARM_COUNTS must have the same length")
    if sum(ARM_COUNTS) != N_WORKERS:
        raise ValueError("sum(ARM_COUNTS) must equal N_WORKERS for fixed-compute comparison")
    if len(LR_ARMS) < 2:
        raise ValueError("E089 needs at least two LR arms")


def make_state(seed: int, device: str) -> DatasetState:
    data = e30.make_dataset(seed, device)
    theta0 = e30.init_theta(seed, 0.11, device)
    target_start_loss = e30.loss_float(theta0, data.target_x, data.target_y)
    return DatasetState(data=data, theta0=theta0, target_start_loss=target_start_loss)


def score_theta(state: DatasetState, theta) -> TrajectoryScore:
    select_loss = e30.loss_float(theta, state.data.select_x, state.data.select_y)
    target_loss = e30.loss_float(theta, state.data.target_x, state.data.target_y)
    return TrajectoryScore(
        target_improvement=state.target_start_loss - target_loss,
        select_loss=select_loss,
        target_loss=target_loss,
    )


def local_updates(
    theta,
    state: DatasetState,
    seed: int,
    round_idx: int,
    k_value: int,
    lr_multipliers: tuple[float, ...],
    namespace: int,
) -> tuple[object, ...]:
    n_workers = len(lr_multipliers)
    shards = e30.worker_shards(e30.TRAIN_SIZE, n_workers)
    local_thetas = [theta.detach().clone() for _ in range(n_workers)]
    for step_idx in range(k_value):
        for worker_id, lr_mult in enumerate(lr_multipliers):
            bx, by = e30.sample_batch(
                state.data.train_x,
                state.data.train_y,
                e30.rng_for(seed, namespace, round_idx, step_idx, worker_id),
                e30.LOCAL_BATCH,
                shards[worker_id],
            )
            _loss, grad = e30.loss_and_grad(local_thetas[worker_id], bx, by)
            local_thetas[worker_id] = (
                local_thetas[worker_id] - e30.ETA * lr_mult * grad
            ).detach()
    return tuple((worker_theta - theta).detach() for worker_theta in local_thetas)


def arm_assignment() -> tuple[float, ...]:
    vals: list[float] = []
    for lr, count in zip(LR_ARMS, ARM_COUNTS):
        vals.extend([lr] * count)
    return tuple(vals)


def group_arm_updates(worker_updates: tuple[object, ...]) -> tuple[ArmUpdate, ...]:
    out: list[ArmUpdate] = []
    pos = 0
    for lr, count in zip(LR_ARMS, ARM_COUNTS):
        updates = list(worker_updates[pos : pos + count])
        pos += count
        out.append(
            ArmUpdate(
                lr=lr,
                count=count,
                update=e30.mean_tensors(updates).detach(),
                prior=count / N_WORKERS,
            )
        )
    return tuple(out)


def aggregate(updates: tuple[ArmUpdate, ...], weights: list[float]):
    total = e30.torch.zeros_like(updates[0].update)
    for arm, weight in zip(updates, weights):
        total = total + float(weight) * arm.update
    return total.detach()


def sample_probe_batches(state: DatasetState, seed: int, round_idx: int) -> tuple[object, object, object, object]:
    x1, y1 = e30.sample_batch(
        state.data.select_x,
        state.data.select_y,
        e30.rng_for(seed, 701, round_idx),
        SELECT_BATCH,
    )
    x2, y2 = e30.sample_batch(
        state.data.select_x,
        state.data.select_y,
        e30.rng_for(seed, 702, round_idx),
        SELECT_BATCH,
    )
    return x1, y1, x2, y2


def arm_losses(theta, velocity, pair: tuple[float, float], arms: tuple[ArmUpdate, ...], batches) -> list[float]:
    alpha, beta = pair
    x1, y1, x2, y2 = batches
    losses: list[float] = []
    for arm in arms:
        action = alpha * (arm.update + beta * velocity)
        loss1 = e30.loss_float(theta + action, x1, y1)
        loss2 = e30.loss_float(theta + action, x2, y2)
        losses.append(0.5 * (loss1 + loss2))
    return losses


def soft_weights(arms: tuple[ArmUpdate, ...], losses: list[float]) -> list[float]:
    priors = [arm.prior for arm in arms]
    if len(losses) <= 1:
        return priors
    loc = mean(losses)
    spread = statistics.stdev(losses)
    temp = max(SPEC.tau * spread, 1e-5)
    logits = [max(-12.0, min(12.0, (loc - loss) / temp)) for loss in losses]
    offset = max(logits)
    exps = [math.exp(logit - offset) for logit in logits]
    denom = max(sum(exps), EPS)
    soft = [value / denom for value in exps]
    return [(1.0 - SPEC.lam) * prior + SPEC.lam * weight for prior, weight in zip(priors, soft)]


def shuffled(losses: list[float], round_idx: int, seed: int) -> list[float]:
    if not losses:
        return losses
    shift = (round_idx + seed + 1) % len(losses)
    return losses[shift:] + losses[:shift]


def run_single_lr(
    seed: int,
    state: DatasetState,
    k_value: int,
    pair: tuple[float, float],
    n_workers: int,
    lr: float,
    namespace: int,
) -> TrajectoryScore:
    alpha, beta = pair
    theta = state.theta0.detach().clone()
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=theta.device)
    lr_multipliers = tuple(lr for _ in range(n_workers))
    for round_idx in range(ROUNDS):
        updates = local_updates(theta, state, seed, round_idx, k_value, lr_multipliers, namespace)
        fedavg = e30.mean_tensors(list(updates)).detach()
        direction = (fedavg + beta * velocity).detach()
        theta = (theta + alpha * direction).detach()
        velocity = direction
    return score_theta(state, theta)


def run_internal(
    seed: int,
    state: DatasetState,
    k_value: int,
    pair: tuple[float, float],
    mode: str,
) -> TrajectoryScore:
    alpha, beta = pair
    theta = state.theta0.detach().clone()
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=theta.device)
    lr_multipliers = arm_assignment()
    for round_idx in range(ROUNDS):
        updates = local_updates(theta, state, seed, round_idx, k_value, lr_multipliers, 811)
        arms = group_arm_updates(updates)
        if mode == "parallel_average":
            weights = [arm.prior for arm in arms]
        else:
            losses = arm_losses(theta, velocity, pair, arms, sample_probe_batches(state, seed, round_idx))
            if mode == "shuffled":
                losses = shuffled(losses, round_idx, seed)
            weights = soft_weights(arms, losses)
        aggregate_update = aggregate(arms, weights)
        direction = (aggregate_update + beta * velocity).detach()
        theta = (theta + alpha * direction).detach()
        velocity = direction
    return score_theta(state, theta)


def run_external_hpo(
    seed: int,
    state: DatasetState,
    k_value: int,
    pair: tuple[float, float],
) -> tuple[TrajectoryScore, TrajectoryScore]:
    scores: list[TrajectoryScore] = []
    for arm_idx, (lr, count) in enumerate(zip(LR_ARMS, ARM_COUNTS)):
        scores.append(run_single_lr(seed, state, k_value, pair, count, lr, 901 + 17 * arm_idx))
    valid = min(scores, key=lambda score: score.select_loss)
    oracle = min(scores, key=lambda score: score.target_loss)
    return valid, oracle


def run_full_worker_arms(
    seed: int,
    state: DatasetState,
    k_value: int,
    pair: tuple[float, float],
) -> list[TrajectoryScore]:
    return [
        run_single_lr(seed, state, k_value, pair, N_WORKERS, lr, 1001 + 19 * arm_idx)
        for arm_idx, lr in enumerate(LR_ARMS)
    ]


def score_parallel_average_for_pair(
    seed: int,
    state: DatasetState,
    k_value: int,
    pair: tuple[float, float],
) -> float:
    return run_internal(seed, state, k_value, pair, "parallel_average").target_improvement


def tune_pair(k_value: int, device: str) -> tuple[tuple[float, float], str]:
    if PAIR_MODE == "e063_fixed" and k_value in E063_PAIR_BY_K:
        return E063_PAIR_BY_K[k_value], "E063 spread-16 fixed pair"
    best_pair = (1.0, 0.0)
    best_score = -float("inf")
    for alpha in ALPHAS:
        for beta in BETAS:
            pair = (alpha, beta)
            scores: list[float] = []
            for seed in PILOT_SEEDS:
                state = make_state(seed, device)
                scores.append(score_parallel_average_for_pair(seed, state, k_value, pair))
            score_mean = mean(scores)
            if score_mean > best_score:
                best_score = score_mean
                best_pair = pair
    return best_pair, "pilot parallel-arm-average tune"


def probe_cost_pct(k_value: int) -> float:
    return 100.0 * len(LR_ARMS) * SPEC.slices * SELECT_BATCH / (
        e30.TRAIN_SIZE * 3.0 * N_WORKERS * k_value
    )


def pick_default_full(scores: list[TrajectoryScore]) -> TrajectoryScore:
    default_idx = min(range(len(LR_ARMS)), key=lambda idx: abs(LR_ARMS[idx] - 1.0))
    return scores[default_idx]


def diagnose_cell(k_value: int, device: str) -> Cell:
    pair, pair_source = tune_pair(k_value, device)
    low_full: list[float] = []
    default: list[float] = []
    parallel_average: list[float] = []
    internal_probe: list[float] = []
    internal_shuffled: list[float] = []
    external_valid: list[float] = []
    external_oracle: list[float] = []
    best_full: list[float] = []
    for seed in HELDOUT_SEEDS:
        state = make_state(seed, device)
        full_scores = run_full_worker_arms(seed, state, k_value, pair)
        low_full.append(full_scores[0].target_improvement)
        default.append(pick_default_full(full_scores).target_improvement)
        best_full.append(min(full_scores, key=lambda score: score.target_loss).target_improvement)
        parallel_average.append(run_internal(seed, state, k_value, pair, "parallel_average").target_improvement)
        internal_probe.append(run_internal(seed, state, k_value, pair, "probe").target_improvement)
        internal_shuffled.append(run_internal(seed, state, k_value, pair, "shuffled").target_improvement)
        valid, oracle = run_external_hpo(seed, state, k_value, pair)
        external_valid.append(valid.target_improvement)
        external_oracle.append(oracle.target_improvement)
    return Cell(
        k_value=k_value,
        pair=pair,
        pair_source=pair_source,
        low_full_worker_arm=low_full,
        default_full_worker_arm=default,
        parallel_average=parallel_average,
        internal_probe=internal_probe,
        internal_shuffled=internal_shuffled,
        external_valid=external_valid,
        external_oracle=external_oracle,
        best_full_worker_arm=best_full,
        cost_pct=probe_cost_pct(k_value),
    )


def decide(cells: list[Cell], status: str) -> str:
    if status == "NO_TORCH":
        return "NO_TORCH"
    probe_external: list[float] = []
    probe_shuffled: list[float] = []
    probe_parallel: list[float] = []
    for cell in cells:
        probe_external.extend(paired(cell.internal_probe, cell.external_valid))
        probe_shuffled.extend(paired(cell.internal_probe, cell.internal_shuffled))
        probe_parallel.extend(paired(cell.internal_probe, cell.parallel_average))
    if positive_ci(probe_external) and positive_ci(probe_shuffled):
        return "FIXED_COMPUTE_INTERNAL_HPO_BEATS_PARALLEL_HPO"
    if positive_ci(probe_shuffled) and positive_ci(probe_parallel):
        return "INTERNAL_HPO_CONTENT_LIVE_EXTERNAL_HPO_TIE"
    if positive_ci(probe_parallel):
        return "RECOMBINATION_BEATS_PARALLEL_AVERAGE_ONLY"
    return "FIXED_COMPUTE_BASELINE_NOT_CLEARED"


def extend_pooled(cells: Iterable[Cell], attr: str, baseline: str) -> list[float]:
    out: list[float] = []
    for cell in cells:
        out.extend(paired(getattr(cell, attr), getattr(cell, baseline)))
    return out


def run_experiment() -> tuple[str, str, str]:
    check_arm_budget()
    status, detail, device = compute_status()
    if status == "NO_TORCH":
        run_log = "\n".join(
            (
                "DFW-E089 fixed-compute parallel HPO headroom",
                f"host={platform.node()} python={sys.executable}",
                f"DECISION NO_TORCH detail={detail}",
            )
        )
        result = "# DFW-E089 Result - Fixed-Compute Parallel HPO Headroom\n\nDecision label: `NO_TORCH`\n"
        legit = "# DFW-E089 Legitimacy Check\n\n- PyTorch workload: not run.\n"
        return run_log + "\n", result + "\n", legit + "\n"

    e30.torch.set_default_dtype(e30.torch.float32)
    try:
        e30.torch.set_num_threads(int(os.environ.get("DFW_E089_TORCH_THREADS", "1")))
    except Exception:
        pass

    cells: list[Cell] = []
    run_lines = [
        "DFW-E089 fixed-compute parallel HPO headroom",
        f"host={platform.node()} python={sys.executable} compute_status={status} {detail}",
        f"N={N_WORKERS} spread={SPREAD:g} arms={LR_ARMS} counts={ARM_COUNTS} K={K_VALUES} rounds={ROUNDS}",
        f"pair_mode={PAIR_MODE} pilot_seeds={len(PILOT_SEEDS)} heldout_seeds={len(HELDOUT_SEEDS)} select_batch={SELECT_BATCH}",
        f"alphas={ALPHAS} betas={BETAS} spec={SPEC.name}",
        "",
    ]
    for k_value in K_VALUES:
        cell = diagnose_cell(k_value, device)
        cells.append(cell)
        run_lines.append(
            f"CELL K={k_value} pair=({cell.pair[0]:.2f},{cell.pair[1]:.2f}) source={cell.pair_source} "
            f"low_full={fmt_ci(cell.low_full_worker_arm)} default_full={fmt_ci(cell.default_full_worker_arm)} "
            f"parallel_avg={fmt_ci(cell.parallel_average)} internal={fmt_ci(cell.internal_probe)} "
            f"internal-shuffled={fmt_ci(paired(cell.internal_probe, cell.internal_shuffled))} "
            f"internal-parallel_avg={fmt_ci(paired(cell.internal_probe, cell.parallel_average))} "
            f"external_valid={fmt_ci(cell.external_valid)} internal-external_valid={fmt_ci(paired(cell.internal_probe, cell.external_valid))} "
            f"external_oracle={fmt_ci(cell.external_oracle)} internal-external_oracle={fmt_ci(paired(cell.internal_probe, cell.external_oracle))} "
            f"best_full={fmt_ci(cell.best_full_worker_arm)} internal-best_full={fmt_ci(paired(cell.internal_probe, cell.best_full_worker_arm))} "
            f"cost={cell.cost_pct:.2f}%"
        )
        print(run_lines[-1], flush=True)

    decision = decide(cells, status)
    pooled_internal_external = extend_pooled(cells, "internal_probe", "external_valid")
    pooled_internal_external_oracle = extend_pooled(cells, "internal_probe", "external_oracle")
    pooled_internal_shuffled = extend_pooled(cells, "internal_probe", "internal_shuffled")
    pooled_internal_parallel = extend_pooled(cells, "internal_probe", "parallel_average")
    pooled_internal_default = extend_pooled(cells, "internal_probe", "default_full_worker_arm")
    pooled_internal_best_full = extend_pooled(cells, "internal_probe", "best_full_worker_arm")
    run_lines.extend(
        [
            "",
            f"POOLED internal-external_valid={fmt_ci(pooled_internal_external)} "
            f"internal-external_oracle={fmt_ci(pooled_internal_external_oracle)} "
            f"internal-shuffled={fmt_ci(pooled_internal_shuffled)} "
            f"internal-parallel_avg={fmt_ci(pooled_internal_parallel)} "
            f"internal-default_full={fmt_ci(pooled_internal_default)} "
            f"internal-best_full={fmt_ci(pooled_internal_best_full)}",
            f"DECISION {decision}",
            f"compute_status={status}",
        ]
    )

    lines = [
        "# DFW-E089 Result - Fixed-Compute Parallel HPO Headroom",
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
            f"E089 reuses the E063 grouped-LR spread-{SPREAD:g} headroom regime: "
            f"`N={N_WORKERS}`, arms `{LR_ARMS}`, counts `{ARM_COUNTS}`, "
            f"`K={K_VALUES}`, and `{ROUNDS}` rounds."
        ),
        "",
        "Internal HPO spends all workers in one shared trajectory and recombines grouped candidate updates. "
        "External HPO spends the same total workers on separate LR-arm trajectories and selects one by validation loss.",
        "",
        "## Pooled",
        "",
        f"- internal minus external validation-selected HPO: `{fmt_ci(pooled_internal_external).strip()}`",
        f"- internal minus external oracle HPO: `{fmt_ci(pooled_internal_external_oracle).strip()}`",
        f"- internal minus shuffled internal: `{fmt_ci(pooled_internal_shuffled).strip()}`",
        f"- internal minus parallel-arm average: `{fmt_ci(pooled_internal_parallel).strip()}`",
        f"- internal minus default full-worker arm: `{fmt_ci(pooled_internal_default).strip()}`",
        f"- internal minus best full-worker arm: `{fmt_ci(pooled_internal_best_full).strip()}`",
        "",
        "## Cells",
        "",
        "| K | pair | pair source | low full-worker | default full-worker | parallel arm average | internal probe | internal minus shuffled | internal minus ext-valid | internal minus ext-oracle | internal minus best full-worker arm | probe cost |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{cell.k_value}",
                    f"({cell.pair[0]:.2f},{cell.pair[1]:.2f})",
                    cell.pair_source,
                    fmt_ci(cell.low_full_worker_arm).strip(),
                    fmt_ci(cell.default_full_worker_arm).strip(),
                    fmt_ci(cell.parallel_average).strip(),
                    fmt_ci(cell.internal_probe).strip(),
                    fmt_ci(paired(cell.internal_probe, cell.internal_shuffled)).strip(),
                    fmt_ci(paired(cell.internal_probe, cell.external_valid)).strip(),
                    fmt_ci(paired(cell.internal_probe, cell.external_oracle)).strip(),
                    fmt_ci(paired(cell.internal_probe, cell.best_full_worker_arm)).strip(),
                    f"{cell.cost_pct:.2f}%",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    if decision == "FIXED_COMPUTE_INTERNAL_HPO_BEATS_PARALLEL_HPO":
        lines.append(
            "The internal grouped-LR rule clears the deployable fixed-compute baseline: "
            "with the same worker budget, recombining probed arms beats selecting one "
            "separate external trajectory by validation loss."
        )
    elif decision == "FIXED_COMPUTE_BASELINE_NOT_CLEARED":
        lines.append(
            "This is a larger negative fixed-compute result than E088 ultra: the "
            "E063 headroom regime still does not clear validation-selected external "
            "parallel HPO under this final-trajectory comparison."
        )
    else:
        lines.append(
            "The run shows internal HPO content against one or more internal controls, "
            "but the fixed-compute external-HPO comparison remains the binding check."
        )
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])

    legit = "\n".join(
        [
            "# DFW-E089 Legitimacy Check",
            "",
            f"- Compute: `{status}` / `{detail}`.",
            f"- E063 headroom regime: spread `{SPREAD:g}`, LR arms `{LR_ARMS}`, worker counts `{ARM_COUNTS}`.",
            f"- Seriousness beyond E088 ultra: `K={K_VALUES}`, `{ROUNDS}` rounds, `{len(HELDOUT_SEEDS)}` held-out seeds.",
            "- Same total worker count per round: `sum(ARM_COUNTS) == N_WORKERS`.",
            "- External HPO splits workers across independent arm trajectories and discards non-selected arms.",
            "- Internal HPO starts all arms from one shared theta each round and recombines by probe-derived weights.",
            "- Select loss chooses the deployable external-HPO winner; target loss is used only for oracle diagnostics.",
            "- Default full-worker and best full-worker arms are diagnostics, not part of the fixed-compute external-HPO budget.",
            "- TORCH_CPU is an algorithmic PyTorch run, not a GPU-kernel performance claim.",
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
