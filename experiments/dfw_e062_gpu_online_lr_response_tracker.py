"""DFW-E062: online response tracker over inner-LR groups.

E060/E061 made the first stationary positive by assigning aligned workers to
different inner learning-rate groups and probing group-mean updates.  The
remaining failure was extraction: a fixed damped rule captures only a small
piece of oracle headroom, and pilot selection was brittle at K=32.

This experiment keeps the same aligned grouped-LR setup but adds a simple
online response-surface tracker.  It fits a ridge quadratic in
log(inner_lr_multiplier) to normalized probe improvements accumulated with
exponential forgetting, then uses the fitted values for damped group weights.
This is the minimal GP-adjacent test: does pooling hyperparameter response over
rounds help before we reach for a full GP?
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
import dfw_e060_gpu_stationary_inner_lr_group_probe_admission as e60


DATE = "20261005"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e062_gpu_online_lr_response_tracker_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e062_gpu_online_lr_response_tracker_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e062_gpu_online_lr_response_tracker_LEGITIMACY_CHECK_{DATE}.md"

K_VALUES = e32.env_int_tuple("DFW_E062_K_VALUES", (8, 16, 32))
N_WORKERS = e32.env_int_tuple("DFW_E062_N_WORKERS", (8,))
ROUNDS = int(os.environ.get("DFW_E062_ROUNDS", "36"))
SELECT_BATCH = int(os.environ.get("DFW_E062_SELECT_BATCH", "384"))
PILOT_SEEDS = e32.env_seed_tuple("DFW_E062_PILOT_SEEDS", tuple(range(927_000, 927_002)))
HELDOUT_SEEDS = e32.env_seed_tuple("DFW_E062_HELDOUT_SEEDS", tuple(range(928_000, 928_006)))
SPREAD_VALUES = e59.env_float_tuple("DFW_E062_SPREAD_VALUES", (4.0, 16.0))
EPS = 1e-12


@dataclass(frozen=True)
class TrackerSpec:
    name: str
    lam: float
    tau: float
    slices: int
    decay: float
    ridge: float


@dataclass(frozen=True)
class Score:
    name: str
    values: list[float]
    cost_pct: float
    weight_l1: list[float]
    entropy_frac: list[float]


@dataclass(frozen=True)
class Cell:
    spread: float
    n_workers: int
    k_value: int
    group_values: tuple[float, ...]
    fixed_pair: tuple[float, float]
    fixed: list[float]
    oracle_subset: list[float]
    raw_frozen: Score
    raw_shuffled: Score
    trackers: tuple[Score, ...]
    shuffled_trackers: tuple[Score, ...]


TRACKER_SPECS = (
    TrackerSpec("quad_current_l40_t2", 0.40, 2.0, 2, 0.0, 1e-3),
    TrackerSpec("quad_decay80_l40_t2", 0.40, 2.0, 2, 0.80, 1e-3),
    TrackerSpec("quad_decay92_l40_t2", 0.40, 2.0, 2, 0.92, 1e-3),
    TrackerSpec("quad_decay97_l40_t2", 0.40, 2.0, 2, 0.97, 1e-3),
)
RAW_SPEC = next(spec for spec in e60.GROUP_SPECS if spec.name == "group_mean2_l40_t2")


mean = e59.mean
ci95 = e59.ci95
fmt_ci = e59.fmt_ci
paired = e59.paired
positive_ci = e59.positive_ci
ratio_of_means = e59.ratio_of_means
spread_text = e59.spread_text


def solve3(a: list[list[float]], b: list[float]) -> list[float]:
    mat = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(mat[row][col]))
        mat[col], mat[pivot] = mat[pivot], mat[col]
        denom = mat[col][col]
        if abs(denom) < EPS:
            return [0.0, 0.0, 0.0]
        for j in range(col, 4):
            mat[col][j] /= denom
        for row in range(3):
            if row == col:
                continue
            factor = mat[row][col]
            for j in range(col, 4):
                mat[row][j] -= factor * mat[col][j]
    return [mat[row][3] for row in range(3)]


def fit_quadratic(obs: list[tuple[float, float, int]], round_idx: int, spec: TrackerSpec) -> list[float]:
    a = [[0.0, 0.0, 0.0] for _ in range(3)]
    b = [0.0, 0.0, 0.0]
    for x, y, obs_round in obs:
        weight = spec.decay ** max(0, round_idx - obs_round)
        phi = [1.0, x, x * x]
        for i in range(3):
            b[i] += weight * phi[i] * y
            for j in range(3):
                a[i][j] += weight * phi[i] * phi[j]
    for i in range(3):
        a[i][i] += spec.ridge
    return solve3(a, b)


def eval_quad(coef: list[float], x: float) -> float:
    return coef[0] + coef[1] * x + coef[2] * x * x


def normalized(vals: list[float]) -> list[float]:
    if len(vals) <= 1:
        return [0.0 for _ in vals]
    loc = mean(vals)
    scale = statistics.stdev(vals)
    if scale <= EPS:
        return [0.0 for _ in vals]
    return [(value - loc) / scale for value in vals]


def soft_weights_from_scores(row: e60.GroupRoundRow, scores: list[float], lam: float, tau: float) -> list[float]:
    groups = e60.group_updates(row)
    priors = [prior for _value, _update, prior in groups]
    if len(scores) <= 1:
        return priors
    loc = mean(scores)
    scale = statistics.stdev(scores)
    if scale <= EPS:
        return priors
    temp = max(tau * scale, 1e-5)
    logits = [max(-12.0, min(12.0, (score - loc) / temp)) for score in scores]
    offset = max(logits)
    exps = [math.exp(logit - offset) for logit in logits]
    denom = max(sum(exps), EPS)
    soft = [value / denom for value in exps]
    return [(1.0 - lam) * prior + lam * weight for prior, weight in zip(priors, soft)]


def probe_values(row: e60.GroupRoundRow, velocity, pair: tuple[float, float], slices: int) -> list[float]:
    alpha, beta = pair
    base1 = e30.loss_float(row.theta, row.select_x, row.select_y)
    if slices == 1:
        base = base1
    else:
        base2 = e30.loss_float(row.theta, row.select2_x, row.select2_y)
        base = 0.5 * (base1 + base2)
    out = []
    for _value, update, _prior in e60.group_updates(row):
        action = alpha * (update + beta * velocity)
        loss1 = e30.loss_float(row.theta + action, row.select_x, row.select_y)
        if slices == 1:
            loss = loss1
        else:
            loss2 = e30.loss_float(row.theta + action, row.select2_x, row.select2_y)
            loss = 0.5 * (loss1 + loss2)
        out.append(base - loss)
    return out


def shuffled(vals: list[float], row_idx: int, rep_idx: int) -> list[float]:
    if not vals:
        return []
    shift = (row_idx + 3 * rep_idx + 1) % len(vals)
    return vals[shift:] + vals[:shift]


def entropy_fraction(weights: list[float]) -> float:
    if len(weights) <= 1:
        return 1.0
    ent = -sum(weight * math.log(max(weight, EPS)) for weight in weights)
    return ent / math.log(len(weights))


def score_tracker(
    rows: list[e60.GroupRoundRow],
    pair: tuple[float, float],
    spec: TrackerSpec,
    rep_idx: int,
    shuffle_probe: bool,
) -> tuple[float, list[float], list[float]]:
    alpha, beta = pair
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=rows[0].theta.device)
    obs: list[tuple[float, float, int]] = []
    raw = 0.0
    total = 0.0
    l1s: list[float] = []
    ents: list[float] = []
    for row_idx, row in enumerate(rows):
        groups = e60.group_updates(row)
        xs = [math.log(max(value, EPS)) for value, _update, _prior in groups]
        vals = probe_values(row, velocity, pair, spec.slices)
        if shuffle_probe:
            vals = shuffled(vals, row_idx, rep_idx)
        zvals = normalized(vals)
        for x, y in zip(xs, zvals):
            obs.append((x, y, row_idx))
        coef = fit_quadratic(obs, row_idx, spec)
        preds = [eval_quad(coef, x) for x in xs]
        weights = soft_weights_from_scores(row, preds, spec.lam, spec.tau)
        agg = e60.aggregate_groups(groups, weights)
        direction = (agg + beta * velocity).detach()
        raw += e60.improvement(row, row.fedavg)
        total += e60.improvement(row, alpha * direction)
        l1s.append(e60.l1_from_prior(weights, row))
        ents.append(entropy_fraction(weights))
        velocity = direction
    return total / raw if raw > EPS else total, l1s, ents


def policy_cost_pct(slices: int, group_count: int, n_workers: int, k_value: int) -> float:
    return 100.0 * group_count * slices * SELECT_BATCH / (e30.TRAIN_SIZE * 3.0 * n_workers * k_value)


def make_score(name: str, values: list[float], cost_pct: float, l1s: list[float], ents: list[float]) -> Score:
    return Score(name=name, values=values, cost_pct=cost_pct, weight_l1=l1s, entropy_frac=ents)


def diagnose_cell(spread: float, k_value: int, n_workers: int, device: str) -> Cell:
    e30.ROUNDS = ROUNDS
    e60.SELECT_BATCH = SELECT_BATCH
    pilot_groups = e60.make_groups(k_value, n_workers, spread, PILOT_SEEDS, device)
    fixed_pair = e60.tune_fixed_pair(pilot_groups)
    heldout_groups = e60.make_groups(k_value, n_workers, spread, HELDOUT_SEEDS, device)
    fixed_vals = [e60.score_fixed_group(rows, fixed_pair) for rows in heldout_groups]
    oracle_vals = [e60.score_group_oracle(rows, fixed_pair)[0] for rows in heldout_groups]

    raw_vals: list[float] = []
    raw_shuf_vals: list[float] = []
    raw_l1s: list[float] = []
    raw_ents: list[float] = []
    raw_shuf_l1s: list[float] = []
    raw_shuf_ents: list[float] = []
    group_count = len(e60.group_updates(heldout_groups[0][0]))
    for rep_idx, rows in enumerate(heldout_groups):
        value, l1, ent = e60.score_damped_group_policy(rows, fixed_pair, RAW_SPEC, rep_idx, False)
        shuf, shuf_l1, shuf_ent = e60.score_damped_group_policy(rows, fixed_pair, RAW_SPEC, rep_idx, True)
        raw_vals.append(value)
        raw_shuf_vals.append(shuf)
        raw_l1s.extend(l1)
        raw_ents.extend(ent)
        raw_shuf_l1s.extend(shuf_l1)
        raw_shuf_ents.extend(shuf_ent)
    raw_cost = policy_cost_pct(RAW_SPEC.slices, group_count, n_workers, k_value)
    raw_score = make_score("raw_group_mean2_l40_t2", raw_vals, raw_cost, raw_l1s, raw_ents)
    raw_shuffled = make_score("shuffled_raw_group_mean2_l40_t2", raw_shuf_vals, raw_cost, raw_shuf_l1s, raw_shuf_ents)

    trackers: list[Score] = []
    shuffled_trackers: list[Score] = []
    for spec in TRACKER_SPECS:
        vals: list[float] = []
        shuf_vals: list[float] = []
        l1s: list[float] = []
        ents: list[float] = []
        shuf_l1s: list[float] = []
        shuf_ents: list[float] = []
        for rep_idx, rows in enumerate(heldout_groups):
            value, l1, ent = score_tracker(rows, fixed_pair, spec, rep_idx, False)
            shuf, shuf_l1, shuf_ent = score_tracker(rows, fixed_pair, spec, rep_idx, True)
            vals.append(value)
            shuf_vals.append(shuf)
            l1s.extend(l1)
            ents.extend(ent)
            shuf_l1s.extend(shuf_l1)
            shuf_ents.extend(shuf_ent)
        cost = policy_cost_pct(spec.slices, group_count, n_workers, k_value)
        trackers.append(make_score(spec.name, vals, cost, l1s, ents))
        shuffled_trackers.append(make_score(f"shuffled_{spec.name}", shuf_vals, cost, shuf_l1s, shuf_ents))
    first = heldout_groups[0][0]
    return Cell(
        spread=spread,
        n_workers=n_workers,
        k_value=k_value,
        group_values=tuple(value for value, _update, _prior in e60.group_updates(first)),
        fixed_pair=fixed_pair,
        fixed=fixed_vals,
        oracle_subset=oracle_vals,
        raw_frozen=raw_score,
        raw_shuffled=raw_shuffled,
        trackers=tuple(trackers),
        shuffled_trackers=tuple(shuffled_trackers),
    )


def best_tracker(cell: Cell) -> Score:
    return max(cell.trackers, key=lambda score: mean(score.values))


def matching_shuffled(cell: Cell, score: Score) -> Score:
    return next(item for item in cell.shuffled_trackers if item.name == f"shuffled_{score.name}")


def capture(vals: list[float], fixed: list[float], oracle: list[float]) -> list[float]:
    out = []
    for val, fix, ora in zip(vals, fixed, oracle):
        gap = ora - fix
        out.append((val - fix) / gap if abs(gap) > EPS else 0.0)
    return out


def decide(cells: list[Cell], compute_status: str) -> str:
    if compute_status != "REAL_GPU":
        return "NO_GPU_HARNESS"
    tracker_gains: list[float] = []
    tracker_vs_shuffled: list[float] = []
    tracker_vs_raw: list[float] = []
    raw_gains: list[float] = []
    raw_vs_shuffled: list[float] = []
    oracle_gains: list[float] = []
    for cell in cells:
        tracker = best_tracker(cell)
        shuf = matching_shuffled(cell, tracker)
        tracker_gains.extend(paired(tracker.values, cell.fixed))
        tracker_vs_shuffled.extend(paired(tracker.values, shuf.values))
        tracker_vs_raw.extend(paired(tracker.values, cell.raw_frozen.values))
        raw_gains.extend(paired(cell.raw_frozen.values, cell.fixed))
        raw_vs_shuffled.extend(paired(cell.raw_frozen.values, cell.raw_shuffled.values))
        oracle_gains.extend(paired(cell.oracle_subset, cell.fixed))
    if positive_ci(tracker_gains) and positive_ci(tracker_vs_shuffled) and positive_ci(tracker_vs_raw):
        return "REAL_ONLINE_LR_RESPONSE_TRACKER_LIVE"
    if positive_ci(raw_gains) and positive_ci(raw_vs_shuffled):
        return "FROZEN_GROUP_PROBE_LIVE_TRACKER_TIES"
    if positive_ci(oracle_gains):
        return "ORACLE_GROUP_HEADROOM_NO_EXTRACTION"
    return "AVERAGING_EXPLAINS"


def no_gpu_artifacts(diag: e30.ComputeDiagnostic) -> tuple[str, str, str]:
    run_log = "\n".join(
        (
            "DFW-E062 online LR response tracker",
            f"host={platform.node()} python={sys.executable}",
            f"DECISION NO_GPU_HARNESS compute_status={diag.status} detail={diag.detail}",
        )
    )
    result = "\n".join(
        (
            "# DFW-E062 Result - Online LR Response Tracker",
            "",
            "Decision label: `NO_GPU_HARNESS`",
            "",
            f"Diagnostic: `{diag.detail}`.",
            "",
        )
    )
    legit = "# DFW-E062 Legitimacy Check\n\n- GPU workload: not run.\n"
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
    device = diag.device
    cells: list[Cell] = []
    run_lines = [
        "DFW-E062 online LR response tracker",
        f"host={platform.node()} python={sys.executable} compute_status={diag.status} {diag.detail}",
        f"model=torch_two_layer_binary_classifier dim={e30.DIM} N={N_WORKERS} K={K_VALUES} rounds={ROUNDS} eta={e30.ETA}",
        f"pilot_seeds={len(PILOT_SEEDS)} heldout_seeds={len(HELDOUT_SEEDS)} spread_values={SPREAD_VALUES} select_batch={SELECT_BATCH}",
        "",
    ]
    for n_workers in N_WORKERS:
        for k_value in K_VALUES:
            for spread in SPREAD_VALUES:
                cell = diagnose_cell(spread, k_value, n_workers, device)
                cells.append(cell)
                tracker = best_tracker(cell)
                shuf = matching_shuffled(cell, tracker)
                run_lines.append(
                    f"CELL spread={spread_text(spread)} N={n_workers} K={k_value} "
                    f"fixed_pair=({cell.fixed_pair[0]:.1f},{cell.fixed_pair[1]:.2f}) "
                    f"groups={','.join(f'{v:.3g}' for v in cell.group_values)} "
                    f"fixed={fmt_ci(cell.fixed)} oracle={fmt_ci(cell.oracle_subset)} "
                    f"raw={fmt_ci(cell.raw_frozen.values)} raw-fixed={fmt_ci(paired(cell.raw_frozen.values, cell.fixed))} "
                    f"raw-shuffled={fmt_ci(paired(cell.raw_frozen.values, cell.raw_shuffled.values))} "
                    f"best_tracker={tracker.name} tracker={fmt_ci(tracker.values)} "
                    f"tracker-fixed={fmt_ci(paired(tracker.values, cell.fixed))} "
                    f"tracker-shuffled={fmt_ci(paired(tracker.values, shuf.values))} "
                    f"tracker-raw={fmt_ci(paired(tracker.values, cell.raw_frozen.values))}"
                )
                print(run_lines[-1], flush=True)
    decision = decide(cells, diag.status)
    run_lines.append("")
    run_lines.append(f"DECISION {decision}")
    run_lines.append(f"compute_status={diag.status}")

    lines = [
        "# DFW-E062 Result - Online LR Response Tracker",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{diag.status}` via PyTorch CUDA: `{diag.detail}`.",
        "",
        "## Held-Out Scores",
        "",
        "| spread | N | K | groups | fixed `(alpha,beta)` | fixed | oracle | raw frozen | raw minus fixed | raw minus shuffled | best tracker | tracker | tracker minus fixed | tracker minus shuffled | tracker minus raw | capture |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        tracker = best_tracker(cell)
        shuf = matching_shuffled(cell, tracker)
        lines.append(
            "| "
            + " | ".join(
                [
                    spread_text(cell.spread),
                    f"{cell.n_workers}",
                    f"{cell.k_value}",
                    ",".join(f"{value:.3g}" for value in cell.group_values),
                    f"({cell.fixed_pair[0]:.1f},{cell.fixed_pair[1]:.2f})",
                    fmt_ci(cell.fixed).strip(),
                    fmt_ci(cell.oracle_subset).strip(),
                    fmt_ci(cell.raw_frozen.values).strip(),
                    fmt_ci(paired(cell.raw_frozen.values, cell.fixed)).strip(),
                    fmt_ci(paired(cell.raw_frozen.values, cell.raw_shuffled.values)).strip(),
                    tracker.name,
                    fmt_ci(tracker.values).strip(),
                    fmt_ci(paired(tracker.values, cell.fixed)).strip(),
                    fmt_ci(paired(tracker.values, shuf.values)).strip(),
                    fmt_ci(paired(tracker.values, cell.raw_frozen.values)).strip(),
                    fmt_ci(capture(tracker.values, cell.fixed, cell.oracle_subset)).strip(),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])
    legit = "\n".join(
        [
            "# DFW-E062 Legitimacy Check",
            "",
            f"- Compute: `{diag.status}` / `{diag.detail}`.",
            "- Target batches are used only for scoring and oracle diagnostics.",
            "- Aligned stationary workers differ only by inner LR group.",
            "- Trackers fit probe observations only; shuffled trackers rotate probe values across groups.",
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
