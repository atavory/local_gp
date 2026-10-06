"""DFW-E085: guarded stale-default HP-arm abandonment test.

E067/E069 showed that a soft default-anchored mixture is not enough.  This
test models the sharper customer case: a stale hyperparameter default remains
the fallback, and the server abandons it only when same-round probes clear a
guarded comparison against that stale arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import platform
import statistics
import sys
from typing import Iterable

import dfw_e030_gpu_real_path_local_map as e30
import dfw_e032_gpu_probe_outer_control_real_path as e32
import dfw_e059_gpu_stationary_inner_lr_probe_admission as e59
import dfw_e060_gpu_stationary_inner_lr_group_probe_admission as e60


DATE = "20261006"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e085_gpu_stale_default_guarded_hp_arm_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e085_gpu_stale_default_guarded_hp_arm_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e085_gpu_stale_default_guarded_hp_arm_LEGITIMACY_CHECK_{DATE}.md"

K_VALUES = (8, 16, 32)
N_WORKERS = 8
ROUNDS = 36
SPREAD = 16.0
STALE_DEFAULT = 0.25
FRESH_DEFAULT = 1.0
SELECT_BATCH = 384
PROBE_BATCH = 192
PROBE_SLICES = 4
GUARD_Z = 0.50
MIN_AGREE = 0.75
PILOT_SEEDS = tuple(range(980_000, 980_003))
HELDOUT_SEEDS = tuple(range(981_000, 981_012))
EPS = 1e-12

mean = e59.mean
fmt_ci = e59.fmt_ci
paired = e59.paired
positive_ci = e59.positive_ci
spread_text = e59.spread_text


@dataclass(frozen=True)
class GuardedScore:
    values: list[float]
    shuffled: list[float]
    abandon_rate: list[float]
    shuffled_abandon_rate: list[float]
    stale_mass: list[float]
    fresh_mass: list[float]
    high_mass: list[float]


@dataclass(frozen=True)
class Cell:
    k_value: int
    group_values: tuple[float, ...]
    fixed_pair: tuple[float, float]
    stale_pair: tuple[float, float]
    fresh_pair: tuple[float, float]
    fixed_all: list[float]
    stale_only: list[float]
    fresh_only: list[float]
    oracle_subset: list[float]
    oracle_single: list[float]
    guarded: GuardedScore
    cost_pct: float


def ci95(vals: Iterable[float]) -> float:
    rows = tuple(vals)
    if len(rows) < 2:
        return 0.0
    return 1.96 * statistics.stdev(rows) / math.sqrt(len(rows))


def arm_index(groups: tuple[tuple[float, object, float], ...], value: float) -> int:
    for idx, (arm, _update, _prior) in enumerate(groups):
        if abs(arm - value) < 1e-12:
            return idx
    raise ValueError(f"arm lr_multiplier={value} missing")


def score_single_arm(
    rows: list[e60.GroupRoundRow],
    pair: tuple[float, float],
    arm_value: float,
) -> float:
    alpha, beta = pair
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=rows[0].theta.device)
    raw = 0.0
    total = 0.0
    for row in rows:
        groups = e60.group_updates(row)
        update = groups[arm_index(groups, arm_value)][1]
        direction = (update + beta * velocity).detach()
        raw += e60.improvement(row, row.fedavg)
        total += e60.improvement(row, alpha * direction)
        velocity = direction
    return total / raw if raw > EPS else total


def tune_single_arm_pair(
    groups: list[list[e60.GroupRoundRow]],
    arm_value: float,
) -> tuple[float, float]:
    best_pair = (1.0, 0.0)
    best_score = -float("inf")
    for alpha in e60.ALPHAS:
        for beta in e60.BETAS:
            pair = (alpha, beta)
            score = mean(score_single_arm(rows, pair, arm_value) for rows in groups)
            if score > best_score:
                best_score = score
                best_pair = pair
    return best_pair


def probe_slices(row: e60.GroupRoundRow, action) -> list[float]:
    b = PROBE_BATCH
    chunks = [
        (row.select_x[:b], row.select_y[:b]),
        (row.select2_x[:b], row.select2_y[:b]),
    ]
    if PROBE_SLICES >= 4:
        chunks.extend(
            [
                (row.select_x[b : 2 * b], row.select_y[b : 2 * b]),
                (row.select2_x[b : 2 * b], row.select2_y[b : 2 * b]),
            ]
        )
    return [e30.loss_float(row.theta + action, x, y) for x, y in chunks[:PROBE_SLICES]]


def group_slice_losses(
    row: e60.GroupRoundRow,
    velocity,
    pair: tuple[float, float],
) -> list[list[float]]:
    alpha, beta = pair
    return [
        probe_slices(row, alpha * (update + beta * velocity))
        for _value, update, _prior in e60.group_updates(row)
    ]


def shuffle_by_label(losses: list[list[float]], row_idx: int, rep_idx: int) -> list[list[float]]:
    if not losses:
        return losses
    shift = (row_idx + 5 * rep_idx + 1) % len(losses)
    return losses[shift:] + losses[:shift]


def guarded_choice(
    row: e60.GroupRoundRow,
    velocity,
    pair: tuple[float, float],
    row_idx: int,
    rep_idx: int,
    shuffled: bool,
) -> tuple[int, float]:
    groups = e60.group_updates(row)
    losses = group_slice_losses(row, velocity, pair)
    if shuffled:
        losses = shuffle_by_label(losses, row_idx, rep_idx)
    stale_idx = arm_index(groups, STALE_DEFAULT)
    means = [mean(arm_losses) for arm_losses in losses]
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


def score_guarded(
    rows: list[e60.GroupRoundRow],
    pair: tuple[float, float],
    rep_idx: int,
    shuffled: bool,
) -> tuple[float, float, dict[float, float]]:
    alpha, beta = pair
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=rows[0].theta.device)
    raw = 0.0
    total = 0.0
    abandons = 0.0
    mass: dict[float, float] = {}
    for row_idx, row in enumerate(rows):
        groups = e60.group_updates(row)
        chosen_idx, abandoned = guarded_choice(row, velocity, pair, row_idx, rep_idx, shuffled)
        chosen_arm, update, _prior = groups[chosen_idx]
        direction = (update + beta * velocity).detach()
        raw += e60.improvement(row, row.fedavg)
        total += e60.improvement(row, alpha * direction)
        abandons += abandoned
        mass[chosen_arm] = mass.get(chosen_arm, 0.0) + 1.0
        velocity = direction
    rounds = max(len(rows), 1)
    return (
        total / raw if raw > EPS else total,
        abandons / rounds,
        {arm: count / rounds for arm, count in mass.items()},
    )


def score_guarded_groups(
    heldout_groups: list[list[e60.GroupRoundRow]],
    pair: tuple[float, float],
) -> GuardedScore:
    vals: list[float] = []
    shuf_vals: list[float] = []
    abandon: list[float] = []
    shuf_abandon: list[float] = []
    stale_mass: list[float] = []
    fresh_mass: list[float] = []
    high_mass: list[float] = []
    for rep_idx, rows in enumerate(heldout_groups):
        value, rate, mass = score_guarded(rows, pair, rep_idx, False)
        shuf, shuf_rate, _shuf_mass = score_guarded(rows, pair, rep_idx, True)
        vals.append(value)
        shuf_vals.append(shuf)
        abandon.append(rate)
        shuf_abandon.append(shuf_rate)
        stale_mass.append(mass.get(STALE_DEFAULT, 0.0))
        fresh_mass.append(mass.get(FRESH_DEFAULT, 0.0))
        high_arm = max(value for value, _update, _prior in e60.group_updates(rows[0]))
        high_mass.append(mass.get(high_arm, 0.0))
    return GuardedScore(vals, shuf_vals, abandon, shuf_abandon, stale_mass, fresh_mass, high_mass)


def cost_pct(group_count: int, k_value: int) -> float:
    return 100.0 * group_count * PROBE_SLICES * PROBE_BATCH / (
        e30.TRAIN_SIZE * 3.0 * N_WORKERS * k_value
    )


def capture(vals: list[float], base: list[float], oracle: list[float]) -> list[float]:
    out = []
    for val, bval, oval in zip(vals, base, oracle):
        gap = oval - bval
        out.append((val - bval) / gap if abs(gap) > EPS else 0.0)
    return out


def diagnose_cell(k_value: int, device: str) -> Cell:
    e30.ROUNDS = ROUNDS
    e60.SELECT_BATCH = SELECT_BATCH
    if PROBE_SLICES >= 4 and 2 * PROBE_BATCH > SELECT_BATCH:
        raise ValueError("SELECT_BATCH must cover two probe chunks from each select draw")
    pilot = e60.make_groups(k_value, N_WORKERS, SPREAD, PILOT_SEEDS, device)
    fixed_pair = e60.tune_fixed_pair(pilot)
    stale_pair = tune_single_arm_pair(pilot, STALE_DEFAULT)
    fresh_pair = tune_single_arm_pair(pilot, FRESH_DEFAULT)
    heldout = e60.make_groups(k_value, N_WORKERS, SPREAD, HELDOUT_SEEDS, device)
    fixed_vals = [e60.score_fixed_group(rows, fixed_pair) for rows in heldout]
    stale_vals = [score_single_arm(rows, stale_pair, STALE_DEFAULT) for rows in heldout]
    fresh_vals = [score_single_arm(rows, fresh_pair, FRESH_DEFAULT) for rows in heldout]
    oracle_subset: list[float] = []
    oracle_single: list[float] = []
    for rows in heldout:
        subset, single = e60.score_group_oracle(rows, stale_pair)
        oracle_subset.append(subset)
        oracle_single.append(single)
    groups = e60.group_updates(heldout[0][0])
    return Cell(
        k_value=k_value,
        group_values=tuple(value for value, _update, _prior in groups),
        fixed_pair=fixed_pair,
        stale_pair=stale_pair,
        fresh_pair=fresh_pair,
        fixed_all=fixed_vals,
        stale_only=stale_vals,
        fresh_only=fresh_vals,
        oracle_subset=oracle_subset,
        oracle_single=oracle_single,
        guarded=score_guarded_groups(heldout, stale_pair),
        cost_pct=cost_pct(len(groups), k_value),
    )


def decide(cells: list[Cell], compute_status: str) -> str:
    if compute_status != "REAL_GPU":
        return "NO_GPU_HARNESS"
    guarded_stale: list[float] = []
    guarded_shuffled: list[float] = []
    guarded_fresh: list[float] = []
    oracle_stale: list[float] = []
    for cell in cells:
        guarded_stale.extend(paired(cell.guarded.values, cell.stale_only))
        guarded_shuffled.extend(paired(cell.guarded.values, cell.guarded.shuffled))
        guarded_fresh.extend(paired(cell.guarded.values, cell.fresh_only))
        oracle_stale.extend(paired(cell.oracle_single, cell.stale_only))
    if (
        positive_ci(guarded_stale)
        and positive_ci(guarded_shuffled)
        and mean(cell_value for cell in cells for cell_value in cell.guarded.abandon_rate) >= 0.05
    ):
        if positive_ci(guarded_fresh):
            return "STALE_DEFAULT_GUARDED_PROBE_BEATS_FRESH_DEFAULT"
        return "STALE_DEFAULT_GUARDED_PROBE_LIVE"
    if positive_ci(oracle_stale):
        return "STALE_DEFAULT_ORACLE_HEADROOM_ONLY"
    return "STALE_DEFAULT_GUARDED_PROBE_NEGATIVE"


def no_gpu_artifacts(diag: e30.ComputeDiagnostic) -> tuple[str, str, str]:
    run_log = "\n".join(
        (
            "DFW-E085 guarded stale-default HP arm",
            f"host={platform.node()} python={sys.executable}",
            f"DECISION NO_GPU_HARNESS compute_status={diag.status} detail={diag.detail}",
        )
    )
    result = "# DFW-E085 Result - Guarded Stale-Default HP Arm\n\nDecision label: `NO_GPU_HARNESS`\n"
    legit = "# DFW-E085 Legitimacy Check\n\n- GPU workload: not run.\n"
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
        "DFW-E085 guarded stale-default HP arm",
        f"host={platform.node()} python={sys.executable} compute_status={diag.status} {diag.detail}",
        (
            f"model=torch_two_layer_binary_classifier dim={e30.DIM} N={N_WORKERS} "
            f"K={K_VALUES} rounds={ROUNDS} eta={e30.ETA}"
        ),
        (
            f"spread={spread_text(SPREAD)} stale_default={STALE_DEFAULT} fresh_default={FRESH_DEFAULT} "
            f"probe_batch={PROBE_BATCH} slices={PROBE_SLICES} guard_z={GUARD_Z} min_agree={MIN_AGREE}"
        ),
        f"pilot_seeds={len(PILOT_SEEDS)} heldout_seeds={len(HELDOUT_SEEDS)}",
        "",
    ]
    for cell in cells:
        run_lines.append(
            f"CELL K={cell.k_value} cost={cell.cost_pct:.2f}% "
            f"groups={','.join(f'{value:.3g}' for value in cell.group_values)} "
            f"fixed_pair=({cell.fixed_pair[0]:.1f},{cell.fixed_pair[1]:.2f}) "
            f"stale_pair=({cell.stale_pair[0]:.1f},{cell.stale_pair[1]:.2f}) "
            f"fresh_pair=({cell.fresh_pair[0]:.1f},{cell.fresh_pair[1]:.2f}) "
            f"fixed_all={fmt_ci(cell.fixed_all)} stale_only={fmt_ci(cell.stale_only)} "
            f"fresh_only={fmt_ci(cell.fresh_only)} oracle_single={fmt_ci(cell.oracle_single)} "
            f"oracle_subset={fmt_ci(cell.oracle_subset)} guarded={fmt_ci(cell.guarded.values)} "
            f"guarded-stale={fmt_ci(paired(cell.guarded.values, cell.stale_only))} "
            f"guarded-fresh={fmt_ci(paired(cell.guarded.values, cell.fresh_only))} "
            f"guarded-shuffled={fmt_ci(paired(cell.guarded.values, cell.guarded.shuffled))} "
            f"oracle_single-stale={fmt_ci(paired(cell.oracle_single, cell.stale_only))} "
            f"capture_vs_stale={fmt_ci(capture(cell.guarded.values, cell.stale_only, cell.oracle_single))} "
            f"abandon={fmt_ci(cell.guarded.abandon_rate)} shuffled_abandon={fmt_ci(cell.guarded.shuffled_abandon_rate)} "
            f"mass_stale={fmt_ci(cell.guarded.stale_mass)} mass_fresh={fmt_ci(cell.guarded.fresh_mass)} "
            f"mass_high={fmt_ci(cell.guarded.high_mass)}"
        )
    run_lines.append("")
    run_lines.append(f"DECISION {decision}")
    run_lines.append(f"compute_status={diag.status}")

    lines = [
        "# DFW-E085 Result - Guarded Stale-Default HP Arm",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{diag.status}` via PyTorch CUDA: `{diag.detail}`.",
        "",
        "## Held-Out Scores",
        "",
        "| K | cost | stale `(alpha,beta)` | fresh `(alpha,beta)` | stale only | fresh only | oracle single | guarded | guarded-stale | guarded-fresh | guarded-shuffled | abandon | stale mass | fresh mass | high mass | capture vs stale |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(cell.k_value),
                    f"{cell.cost_pct:.2f}%",
                    f"({cell.stale_pair[0]:.1f},{cell.stale_pair[1]:.2f})",
                    f"({cell.fresh_pair[0]:.1f},{cell.fresh_pair[1]:.2f})",
                    fmt_ci(cell.stale_only).strip(),
                    fmt_ci(cell.fresh_only).strip(),
                    fmt_ci(cell.oracle_single).strip(),
                    fmt_ci(cell.guarded.values).strip(),
                    fmt_ci(paired(cell.guarded.values, cell.stale_only)).strip(),
                    fmt_ci(paired(cell.guarded.values, cell.fresh_only)).strip(),
                    fmt_ci(paired(cell.guarded.values, cell.guarded.shuffled)).strip(),
                    fmt_ci(cell.guarded.abandon_rate).strip(),
                    fmt_ci(cell.guarded.stale_mass).strip(),
                    fmt_ci(cell.guarded.fresh_mass).strip(),
                    fmt_ci(cell.guarded.high_mass).strip(),
                    fmt_ci(capture(cell.guarded.values, cell.stale_only, cell.oracle_single)).strip(),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```", ""])
    legit = "\n".join(
        [
            "# DFW-E085 Legitimacy Check",
            "",
            f"- Compute: `{diag.status}` / `{diag.detail}`.",
            "- Models a stale/default-wrong LR arm as the fallback: lr_multiplier=0.25.",
            "- Same starting theta per round; worker groups differ only by inner LR multiplier.",
            "- Probe losses use select batches only; target batches score policies and oracle diagnostics.",
            "- Guard abandons the stale default only when the best probed arm beats it by a slice-disagreement margin and at least 3/4 slices agree.",
            "- Shuffled labels rotate probe losses across HP arms before the same guard is applied.",
            "- Reports fresh lr=1 default, oracle single/subset, abandonment rate, and selected-arm mass.",
            "",
        ]
    )
    return "\n".join(run_lines) + "\n", "\n".join(lines), legit + "\n"


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
