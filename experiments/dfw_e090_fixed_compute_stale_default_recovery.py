"""DFW-E090: fixed-compute stale-default recovery.

E085 made the practical stale-default story legible, but it was not yet priced
against the obvious "just run HPO in parallel" baseline.  This experiment uses
the E089 fixed-compute accounting around the E085 guarded abandon/fallback rule:

  * internal HPO uses the same workers as stale/fresh/high LR arms inside one
    shared trajectory;
  * external HPO splits the same total workers across independent stale/fresh/high
    trajectories and selects one by validation loss;
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
OUT_DIR = Path(os.environ.get("DFW_E090_OUT_DIR", Path(__file__).resolve().parent)).resolve()
RESULT_PATH = OUT_DIR / f"dfw_e090_fixed_compute_stale_default_recovery_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e090_fixed_compute_stale_default_recovery_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e090_fixed_compute_stale_default_recovery_LEGITIMACY_CHECK_{DATE}.md"

K_VALUES = e32.env_int_tuple("DFW_E090_K_VALUES", (8, 16, 32))
N_WORKERS = int(os.environ.get("DFW_E090_N_WORKERS", "8"))
ROUNDS = int(os.environ.get("DFW_E090_ROUNDS", "36"))
SELECT_BATCH = int(os.environ.get("DFW_E090_SELECT_BATCH", "384"))
PROBE_BATCH = int(os.environ.get("DFW_E090_PROBE_BATCH", "192"))
PROBE_SLICES = int(os.environ.get("DFW_E090_PROBE_SLICES", "4"))
GUARD_Z = float(os.environ.get("DFW_E090_GUARD_Z", "0.5"))
MIN_AGREE = float(os.environ.get("DFW_E090_MIN_AGREE", "0.75"))
PILOT_SEEDS = e32.env_seed_tuple("DFW_E090_PILOT_SEEDS", tuple(range(996_000, 996_003)))
HELDOUT_SEEDS = e32.env_seed_tuple("DFW_E090_HELDOUT_SEEDS", tuple(range(997_000, 997_012)))
LR_ARMS = tuple(float(part) for part in os.environ.get("DFW_E090_LR_ARMS", "0.25,1.0,4.0").split(","))
ARM_COUNTS = tuple(int(part) for part in os.environ.get("DFW_E090_ARM_COUNTS", "2,4,2").split(","))
STALE_DEFAULT = float(os.environ.get("DFW_E090_STALE_DEFAULT", "0.25"))
FRESH_DEFAULT = float(os.environ.get("DFW_E090_FRESH_DEFAULT", "1.0"))
PAIR_MODE = os.environ.get("DFW_E090_PAIR_MODE", "e085_stale").strip()
ALPHAS = tuple(float(part) for part in os.environ.get("DFW_E090_ALPHAS", "1,2,3,4,5,6,8,10,12,14,16,18").split(","))
BETAS = tuple(float(part) for part in os.environ.get("DFW_E090_BETAS", "0,0.1,0.2,0.3,0.4,0.5,0.6").split(","))
E085_STALE_PAIR_BY_K = {
    8: (18.0, 0.45),
    16: (11.0, 0.30),
    32: (6.0, 0.20),
}
E085_FRESH_PAIR_BY_K = {
    8: (6.0, 0.35),
    16: (3.0, 0.20),
    32: (2.0, 0.00),
}
EPS = 1e-12

mean = e59.mean
fmt_ci = e59.fmt_ci
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
class GuardedTrace:
    score: TrajectoryScore
    abandon_rate: float
    mass_by_arm: dict[float, float]


@dataclass(frozen=True)
class Cell:
    k_value: int
    pair: tuple[float, float]
    pair_source: str
    fresh_pair: tuple[float, float]
    stale_full: list[float]
    fresh_full: list[float]
    high_full: list[float]
    fresh_tuned_full: list[float]
    best_full: list[float]
    internal_stale_arm: list[float]
    internal_fresh_arm: list[float]
    internal_guarded: list[float]
    internal_shuffled: list[float]
    external_valid: list[float]
    external_oracle: list[float]
    external_stale: list[float]
    external_fresh: list[float]
    external_high: list[float]
    abandon_rate: list[float]
    shuffled_abandon_rate: list[float]
    stale_mass: list[float]
    fresh_mass: list[float]
    high_mass: list[float]
    cost_pct: float


def compute_status() -> tuple[str, str, str]:
    diag = e30.compute_diagnostic()
    if diag.status == "REAL_GPU":
        return diag.status, diag.detail, diag.device
    if e30.torch is None:
        return "NO_TORCH", diag.detail, "cpu"
    version = getattr(e30.torch, "__version__", "unknown")
    return "TORCH_CPU", f"torch={version} cuda_available=False", "cpu"


def check_config() -> None:
    if len(LR_ARMS) != len(ARM_COUNTS):
        raise ValueError("LR_ARMS and ARM_COUNTS must have the same length")
    if sum(ARM_COUNTS) != N_WORKERS:
        raise ValueError("sum(ARM_COUNTS) must equal N_WORKERS")
    if STALE_DEFAULT not in LR_ARMS:
        raise ValueError("stale default must be one of the LR arms")
    if FRESH_DEFAULT not in LR_ARMS:
        raise ValueError("fresh default must be one of the LR arms")
    if PROBE_SLICES not in (1, 2, 4):
        raise ValueError("PROBE_SLICES must be 1, 2, or 4")
    if PROBE_SLICES >= 4 and 2 * PROBE_BATCH > SELECT_BATCH:
        raise ValueError("SELECT_BATCH must cover two probe chunks from each select draw")


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


def arm_index(lr: float) -> int:
    for idx, arm in enumerate(LR_ARMS):
        if abs(arm - lr) < 1e-12:
            return idx
    raise ValueError(f"lr arm {lr} is missing")


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


def sample_probe_chunks(state: DatasetState, seed: int, round_idx: int) -> list[tuple[object, object]]:
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
    chunks = [(x1[:PROBE_BATCH], y1[:PROBE_BATCH]), (x2[:PROBE_BATCH], y2[:PROBE_BATCH])]
    if PROBE_SLICES >= 4:
        chunks.extend(
            [
                (x1[PROBE_BATCH : 2 * PROBE_BATCH], y1[PROBE_BATCH : 2 * PROBE_BATCH]),
                (x2[PROBE_BATCH : 2 * PROBE_BATCH], y2[PROBE_BATCH : 2 * PROBE_BATCH]),
            ]
        )
    return chunks[:PROBE_SLICES]


def arm_slice_losses(
    theta,
    velocity,
    pair: tuple[float, float],
    arms: tuple[ArmUpdate, ...],
    chunks: list[tuple[object, object]],
) -> list[list[float]]:
    alpha, beta = pair
    losses: list[list[float]] = []
    for arm in arms:
        action = alpha * (arm.update + beta * velocity)
        losses.append([e30.loss_float(theta + action, xs, ys) for xs, ys in chunks])
    return losses


def shuffled(losses: list[list[float]], round_idx: int, seed: int) -> list[list[float]]:
    if not losses:
        return losses
    shift = (round_idx + seed + 1) % len(losses)
    return losses[shift:] + losses[:shift]


def guarded_choice(
    losses: list[list[float]],
) -> tuple[int, float]:
    stale_idx = arm_index(STALE_DEFAULT)
    means = [mean(row) for row in losses]
    best_idx = min(range(len(means)), key=lambda idx: means[idx])
    if best_idx == stale_idx:
        return stale_idx, 0.0
    diffs = [stale_loss - best_loss for stale_loss, best_loss in zip(losses[stale_idx], losses[best_idx])]
    gain = mean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    agree = sum(1 for diff in diffs if diff > 0.0) / len(diffs)
    if gain > GUARD_Z * se and agree >= MIN_AGREE:
        return best_idx, 1.0
    return stale_idx, 0.0


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


def run_internal_fixed_arm(
    seed: int,
    state: DatasetState,
    k_value: int,
    pair: tuple[float, float],
    lr: float,
    namespace: int,
) -> TrajectoryScore:
    alpha, beta = pair
    theta = state.theta0.detach().clone()
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=theta.device)
    lr_multipliers = arm_assignment()
    idx = arm_index(lr)
    for round_idx in range(ROUNDS):
        updates = local_updates(theta, state, seed, round_idx, k_value, lr_multipliers, namespace)
        arms = group_arm_updates(updates)
        direction = (arms[idx].update + beta * velocity).detach()
        theta = (theta + alpha * direction).detach()
        velocity = direction
    return score_theta(state, theta)


def run_internal_guarded(
    seed: int,
    state: DatasetState,
    k_value: int,
    pair: tuple[float, float],
    mode: str,
) -> GuardedTrace:
    alpha, beta = pair
    theta = state.theta0.detach().clone()
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=theta.device)
    lr_multipliers = arm_assignment()
    mass: dict[float, float] = {}
    abandons = 0.0
    for round_idx in range(ROUNDS):
        updates = local_updates(theta, state, seed, round_idx, k_value, lr_multipliers, 811)
        arms = group_arm_updates(updates)
        chunks = sample_probe_chunks(state, seed, round_idx)
        losses = arm_slice_losses(theta, velocity, pair, arms, chunks)
        if mode == "shuffled":
            losses = shuffled(losses, round_idx, seed)
        chosen_idx, abandoned = guarded_choice(losses)
        chosen = arms[chosen_idx]
        direction = (chosen.update + beta * velocity).detach()
        theta = (theta + alpha * direction).detach()
        velocity = direction
        abandons += abandoned
        mass[chosen.lr] = mass.get(chosen.lr, 0.0) + 1.0
    rounds = max(ROUNDS, 1)
    return GuardedTrace(
        score=score_theta(state, theta),
        abandon_rate=abandons / rounds,
        mass_by_arm={lr: count / rounds for lr, count in mass.items()},
    )


def run_external_hpo(
    seed: int,
    state: DatasetState,
    k_value: int,
    pair: tuple[float, float],
) -> tuple[TrajectoryScore, TrajectoryScore, list[TrajectoryScore]]:
    scores: list[TrajectoryScore] = []
    for arm_idx, (lr, count) in enumerate(zip(LR_ARMS, ARM_COUNTS)):
        scores.append(run_single_lr(seed, state, k_value, pair, count, lr, 901 + 17 * arm_idx))
    valid = min(scores, key=lambda score: score.select_loss)
    oracle = min(scores, key=lambda score: score.target_loss)
    return valid, oracle, scores


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


def score_stale_for_pair(seed: int, device: str, k_value: int, pair: tuple[float, float]) -> float:
    state = make_state(seed, device)
    return run_single_lr(seed, state, k_value, pair, N_WORKERS, STALE_DEFAULT, 601).target_improvement


def tune_stale_pair(k_value: int, device: str) -> tuple[tuple[float, float], str]:
    if PAIR_MODE == "e085_stale" and k_value in E085_STALE_PAIR_BY_K:
        return E085_STALE_PAIR_BY_K[k_value], "E085 stale guarded pair"
    best_pair = (1.0, 0.0)
    best_score = -float("inf")
    for alpha in ALPHAS:
        for beta in BETAS:
            pair = (alpha, beta)
            score = mean(score_stale_for_pair(seed, device, k_value, pair) for seed in PILOT_SEEDS)
            if score > best_score:
                best_score = score
                best_pair = pair
    return best_pair, "pilot stale-full tune"


def fresh_pair_for_k(k_value: int) -> tuple[float, float]:
    return E085_FRESH_PAIR_BY_K.get(k_value, (1.0, 0.0))


def probe_cost_pct(k_value: int) -> float:
    return 100.0 * len(LR_ARMS) * PROBE_SLICES * PROBE_BATCH / (
        e30.TRAIN_SIZE * 3.0 * N_WORKERS * k_value
    )


def extend_pooled(cells: Iterable[Cell], attr: str, baseline: str) -> list[float]:
    out: list[float] = []
    for cell in cells:
        out.extend(paired(getattr(cell, attr), getattr(cell, baseline)))
    return out


def capture(vals: list[float], base: list[float], oracle: list[float]) -> list[float]:
    out = []
    for val, bval, oval in zip(vals, base, oracle):
        gap = oval - bval
        out.append((val - bval) / gap if abs(gap) > EPS else 0.0)
    return out


def diagnose_cell(k_value: int, device: str) -> Cell:
    pair, pair_source = tune_stale_pair(k_value, device)
    fresh_pair = fresh_pair_for_k(k_value)
    stale_full: list[float] = []
    fresh_full: list[float] = []
    high_full: list[float] = []
    fresh_tuned_full: list[float] = []
    best_full: list[float] = []
    internal_stale_arm: list[float] = []
    internal_fresh_arm: list[float] = []
    internal_guarded: list[float] = []
    internal_shuffled: list[float] = []
    external_valid: list[float] = []
    external_oracle: list[float] = []
    external_stale: list[float] = []
    external_fresh: list[float] = []
    external_high: list[float] = []
    abandon_rate: list[float] = []
    shuffled_abandon_rate: list[float] = []
    stale_mass: list[float] = []
    fresh_mass: list[float] = []
    high_mass: list[float] = []
    high_lr = max(LR_ARMS)

    for seed in HELDOUT_SEEDS:
        state = make_state(seed, device)
        full_scores = run_full_worker_arms(seed, state, k_value, pair)
        stale_full.append(full_scores[arm_index(STALE_DEFAULT)].target_improvement)
        fresh_full.append(full_scores[arm_index(FRESH_DEFAULT)].target_improvement)
        high_full.append(full_scores[arm_index(high_lr)].target_improvement)
        best_full.append(min(full_scores, key=lambda score: score.target_loss).target_improvement)
        fresh_tuned_full.append(
            run_single_lr(seed, state, k_value, fresh_pair, N_WORKERS, FRESH_DEFAULT, 613).target_improvement
        )
        internal_stale_arm.append(
            run_internal_fixed_arm(seed, state, k_value, pair, STALE_DEFAULT, 821).target_improvement
        )
        internal_fresh_arm.append(
            run_internal_fixed_arm(seed, state, k_value, pair, FRESH_DEFAULT, 823).target_improvement
        )
        guarded = run_internal_guarded(seed, state, k_value, pair, "guarded")
        shuf = run_internal_guarded(seed, state, k_value, pair, "shuffled")
        internal_guarded.append(guarded.score.target_improvement)
        internal_shuffled.append(shuf.score.target_improvement)
        abandon_rate.append(guarded.abandon_rate)
        shuffled_abandon_rate.append(shuf.abandon_rate)
        stale_mass.append(guarded.mass_by_arm.get(STALE_DEFAULT, 0.0))
        fresh_mass.append(guarded.mass_by_arm.get(FRESH_DEFAULT, 0.0))
        high_mass.append(guarded.mass_by_arm.get(high_lr, 0.0))
        valid, oracle, external_scores = run_external_hpo(seed, state, k_value, pair)
        external_valid.append(valid.target_improvement)
        external_oracle.append(oracle.target_improvement)
        external_stale.append(external_scores[arm_index(STALE_DEFAULT)].target_improvement)
        external_fresh.append(external_scores[arm_index(FRESH_DEFAULT)].target_improvement)
        external_high.append(external_scores[arm_index(high_lr)].target_improvement)

    return Cell(
        k_value=k_value,
        pair=pair,
        pair_source=pair_source,
        fresh_pair=fresh_pair,
        stale_full=stale_full,
        fresh_full=fresh_full,
        high_full=high_full,
        fresh_tuned_full=fresh_tuned_full,
        best_full=best_full,
        internal_stale_arm=internal_stale_arm,
        internal_fresh_arm=internal_fresh_arm,
        internal_guarded=internal_guarded,
        internal_shuffled=internal_shuffled,
        external_valid=external_valid,
        external_oracle=external_oracle,
        external_stale=external_stale,
        external_fresh=external_fresh,
        external_high=external_high,
        abandon_rate=abandon_rate,
        shuffled_abandon_rate=shuffled_abandon_rate,
        stale_mass=stale_mass,
        fresh_mass=fresh_mass,
        high_mass=high_mass,
        cost_pct=probe_cost_pct(k_value),
    )


def decide(cells: list[Cell], status: str) -> str:
    if status == "NO_TORCH":
        return "NO_TORCH"
    guarded_stale = extend_pooled(cells, "internal_guarded", "stale_full")
    guarded_shuffled = extend_pooled(cells, "internal_guarded", "internal_shuffled")
    guarded_external = extend_pooled(cells, "internal_guarded", "external_valid")
    guarded_fresh = extend_pooled(cells, "internal_guarded", "fresh_full")
    if positive_ci(guarded_stale) and positive_ci(guarded_shuffled) and positive_ci(guarded_external):
        return "FIXED_COMPUTE_STALE_DEFAULT_INTERNAL_HPO_WIN"
    if (
        positive_ci(guarded_stale)
        and positive_ci(guarded_shuffled)
        and mean(guarded_external) > 0.0
    ):
        return "FIXED_COMPUTE_STALE_DEFAULT_INTERNAL_HPO_LIVE"
    if positive_ci(guarded_stale) and positive_ci(guarded_shuffled):
        if positive_ci(guarded_fresh):
            return "STALE_DEFAULT_INTERNAL_HPO_LIVE_BEATS_FRESH_CONTROL"
        return "STALE_DEFAULT_INTERNAL_HPO_LIVE_EXTERNAL_BOUNDARY"
    if positive_ci(guarded_stale):
        return "STALE_DEFAULT_RECOVERY_ONLY"
    return "FIXED_COMPUTE_STALE_DEFAULT_NOT_CLEARED"


def no_torch_artifacts(detail: str) -> tuple[str, str, str]:
    run_log = "\n".join(
        (
            "DFW-E090 fixed-compute stale-default recovery",
            f"host={platform.node()} python={sys.executable}",
            f"DECISION NO_TORCH detail={detail}",
        )
    )
    result = "# DFW-E090 Result - Fixed-Compute Stale-Default Recovery\n\nDecision label: `NO_TORCH`\n"
    legit = "# DFW-E090 Legitimacy Check\n\n- PyTorch workload: not run.\n"
    return run_log + "\n", result + "\n", legit + "\n"


def run_experiment() -> tuple[str, str, str]:
    check_config()
    status, detail, device = compute_status()
    if status == "NO_TORCH":
        return no_torch_artifacts(detail)

    e30.torch.set_default_dtype(e30.torch.float32)
    try:
        e30.torch.set_num_threads(int(os.environ.get("DFW_E090_TORCH_THREADS", "1")))
    except Exception:
        pass

    cells: list[Cell] = []
    run_lines = [
        "DFW-E090 fixed-compute stale-default recovery",
        f"host={platform.node()} python={sys.executable} compute_status={status} {detail}",
        f"N={N_WORKERS} arms={LR_ARMS} counts={ARM_COUNTS} K={K_VALUES} rounds={ROUNDS}",
        (
            f"stale_default={STALE_DEFAULT:g} fresh_default={FRESH_DEFAULT:g} "
            f"pair_mode={PAIR_MODE} select_batch={SELECT_BATCH} probe_batch={PROBE_BATCH} "
            f"slices={PROBE_SLICES} guard_z={GUARD_Z} min_agree={MIN_AGREE}"
        ),
        f"pilot_seeds={len(PILOT_SEEDS)} heldout_seeds={len(HELDOUT_SEEDS)}",
        "",
    ]
    for k_value in K_VALUES:
        cell = diagnose_cell(k_value, device)
        cells.append(cell)
        run_lines.append(
            f"CELL K={k_value} pair=({cell.pair[0]:.2f},{cell.pair[1]:.2f}) source={cell.pair_source} "
            f"stale_full={fmt_ci(cell.stale_full)} fresh_full={fmt_ci(cell.fresh_full)} "
            f"high_full={fmt_ci(cell.high_full)} fresh_tuned_full={fmt_ci(cell.fresh_tuned_full)} "
            f"best_full={fmt_ci(cell.best_full)} internal_stale_arm={fmt_ci(cell.internal_stale_arm)} "
            f"internal_fresh_arm={fmt_ci(cell.internal_fresh_arm)} guarded={fmt_ci(cell.internal_guarded)} "
            f"guarded-stale_full={fmt_ci(paired(cell.internal_guarded, cell.stale_full))} "
            f"guarded-fresh_full={fmt_ci(paired(cell.internal_guarded, cell.fresh_full))} "
            f"guarded-shuffled={fmt_ci(paired(cell.internal_guarded, cell.internal_shuffled))} "
            f"external_valid={fmt_ci(cell.external_valid)} guarded-external_valid={fmt_ci(paired(cell.internal_guarded, cell.external_valid))} "
            f"external_oracle={fmt_ci(cell.external_oracle)} guarded-external_oracle={fmt_ci(paired(cell.internal_guarded, cell.external_oracle))} "
            f"external_stale={fmt_ci(cell.external_stale)} external_fresh={fmt_ci(cell.external_fresh)} "
            f"external_high={fmt_ci(cell.external_high)} abandon={fmt_ci(cell.abandon_rate)} "
            f"shuffled_abandon={fmt_ci(cell.shuffled_abandon_rate)} mass_stale={fmt_ci(cell.stale_mass)} "
            f"mass_fresh={fmt_ci(cell.fresh_mass)} mass_high={fmt_ci(cell.high_mass)} "
            f"capture_vs_stale_external_oracle={fmt_ci(capture(cell.internal_guarded, cell.stale_full, cell.external_oracle))} "
            f"cost={cell.cost_pct:.2f}%"
        )
        print(run_lines[-1], flush=True)

    pooled_guarded_stale = extend_pooled(cells, "internal_guarded", "stale_full")
    pooled_guarded_fresh = extend_pooled(cells, "internal_guarded", "fresh_full")
    pooled_guarded_shuffled = extend_pooled(cells, "internal_guarded", "internal_shuffled")
    pooled_guarded_external = extend_pooled(cells, "internal_guarded", "external_valid")
    pooled_guarded_oracle = extend_pooled(cells, "internal_guarded", "external_oracle")
    pooled_external_stale = extend_pooled(cells, "external_valid", "stale_full")
    pooled_guarded_best_full = extend_pooled(cells, "internal_guarded", "best_full")
    decision = decide(cells, status)
    run_lines.extend(
        [
            "",
            f"POOLED guarded-stale_full={fmt_ci(pooled_guarded_stale)} "
            f"guarded-fresh_full={fmt_ci(pooled_guarded_fresh)} "
            f"guarded-shuffled={fmt_ci(pooled_guarded_shuffled)} "
            f"guarded-external_valid={fmt_ci(pooled_guarded_external)} "
            f"guarded-external_oracle={fmt_ci(pooled_guarded_oracle)} "
            f"external_valid-stale_full={fmt_ci(pooled_external_stale)} "
            f"guarded-best_full={fmt_ci(pooled_guarded_best_full)}",
            f"DECISION {decision}",
            f"compute_status={status}",
        ]
    )

    lines = [
        "# DFW-E090 Result - Fixed-Compute Stale-Default Recovery",
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
            "E090 prices the E085 stale-default guard under fixed-compute accounting. "
            f"The stale user default is LR multiplier `{STALE_DEFAULT:g}`; the fresh/default "
            f"candidate is `{FRESH_DEFAULT:g}`; and the high diagnostic arm is `{max(LR_ARMS):g}`."
        ),
        "",
        (
            "Internal HPO spends all workers in one shared trajectory as grouped LR arms and "
            "abandons the stale fallback only when the same-round probe slices agree. External "
            "HPO spends the same total workers on independent stale/fresh/high trajectories "
            "with counts matching the internal arm allocation, then chooses one by validation loss."
        ),
        "",
        "## Pooled",
        "",
        f"- internal guarded minus stale full-worker default: `{fmt_ci(pooled_guarded_stale).strip()}`",
        f"- internal guarded minus fresh/default full-worker control: `{fmt_ci(pooled_guarded_fresh).strip()}`",
        f"- internal guarded minus shuffled guard: `{fmt_ci(pooled_guarded_shuffled).strip()}`",
        f"- internal guarded minus external validation-selected HPO: `{fmt_ci(pooled_guarded_external).strip()}`",
        f"- internal guarded minus external oracle HPO: `{fmt_ci(pooled_guarded_oracle).strip()}`",
        f"- external validation-selected HPO minus stale full-worker default: `{fmt_ci(pooled_external_stale).strip()}`",
        f"- internal guarded minus best full-worker arm: `{fmt_ci(pooled_guarded_best_full).strip()}`",
        "",
        "## Cells",
        "",
        "| K | pair | stale full | fresh full | fresh tuned full | internal guarded | guarded-stale | guarded-shuffled | guarded-ext-valid | ext-valid | ext-oracle | abandon | stale mass | fresh mass | high mass | probe cost |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{cell.k_value}",
                    f"({cell.pair[0]:.2f},{cell.pair[1]:.2f})",
                    fmt_ci(cell.stale_full).strip(),
                    fmt_ci(cell.fresh_full).strip(),
                    fmt_ci(cell.fresh_tuned_full).strip(),
                    fmt_ci(cell.internal_guarded).strip(),
                    fmt_ci(paired(cell.internal_guarded, cell.stale_full)).strip(),
                    fmt_ci(paired(cell.internal_guarded, cell.internal_shuffled)).strip(),
                    fmt_ci(paired(cell.internal_guarded, cell.external_valid)).strip(),
                    fmt_ci(cell.external_valid).strip(),
                    fmt_ci(cell.external_oracle).strip(),
                    fmt_ci(cell.abandon_rate).strip(),
                    fmt_ci(cell.stale_mass).strip(),
                    fmt_ci(cell.fresh_mass).strip(),
                    fmt_ci(cell.high_mass).strip(),
                    f"{cell.cost_pct:.2f}%",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    if decision == "FIXED_COMPUTE_STALE_DEFAULT_INTERNAL_HPO_WIN":
        lines.append(
            "The guarded internal HPO rule clears the fixed-compute stale-default comparator: "
            "it beats the stale default, shuffled guards, and validation-selected external HPO "
            "while paying only the reported probe overhead."
        )
    elif "LIVE" in decision:
        lines.append(
            "The guarded stale-default rule is live in the fixed-compute ledger: it beats the "
            "stale default and the shuffled probe-label control, while the external-HPO columns "
            "show how much of the same-budget comparator is cleared."
        )
    else:
        lines.append(
            "This run does not promote stale-default recovery beyond the fixed-compute baseline."
        )
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])

    legit = "\n".join(
        [
            "# DFW-E090 Legitimacy Check",
            "",
            f"- Compute: `{status}` / `{detail}`.",
            "- Fixed-compute accounting: `sum(ARM_COUNTS) == N_WORKERS`; internal and external comparisons both spend that worker budget per round.",
            "- Stale-default recovery setting: the fallback arm is the stale LR multiplier, and the guard can abandon it only on same-round probe evidence.",
            "- External HPO uses independent arm trajectories and validation loss for the deployable choice; target loss is oracle-only.",
            "- Shuffled guard rotates probe-loss labels across LR arms before applying the same guarded decision rule.",
            "- Reports stale, fresh/default, high, best full-worker, validation-selected external HPO, external oracle, abandonment, arm mass, and probe cost.",
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
