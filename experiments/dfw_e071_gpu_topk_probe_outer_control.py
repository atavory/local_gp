"""DFW-E071: top-k/shrunk probe selectors for real-path outer control.

E039 showed a precise failure mode: probe rankings often put the target-good
outer-control action in the top few candidates, but top-1 argmin still loses to
fixed/schedule controls.  This test keeps the same real PyTorch/CUDA local-SGD
path and the same probe candidate families, but replaces brittle top-1
selection with conservative top-k and soft selectors.
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
import dfw_e033_gpu_probe_design_real_path as e33


DATE = "20261005"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e071_gpu_topk_probe_outer_control_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e071_gpu_topk_probe_outer_control_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e071_gpu_topk_probe_outer_control_LEGITIMACY_CHECK_{DATE}.md"

K_VALUES = e32.env_int_tuple("DFW_E071_K_VALUES", (8, 32))
N_WORKERS = e32.env_int_tuple("DFW_E071_N_WORKERS", (8,))
SELECT_BATCHES = e32.env_int_tuple("DFW_E071_SELECT_BATCHES", (384,))
PILOT_SEEDS = e32.env_seed_tuple("DFW_E071_PILOT_SEEDS", tuple(range(950_000, 950_003)))
HELDOUT_SEEDS = e32.env_seed_tuple("DFW_E071_HELDOUT_SEEDS", tuple(range(951_000, 951_012)))
DESIGN_NAMES = tuple(
    part.strip()
    for part in os.environ.get("DFW_E071_DESIGNS", "old12,scaled,hybrid").split(",")
    if part.strip()
)
POLICIES = ("top1", "top3_mean", "top5_mean", "soft_std1", "soft_std2", "top3_guard_fixed")
EPS = 1e-12


@dataclass(frozen=True)
class PolicyScore:
    design: str
    policy: str
    points: int
    values: list[float]
    shuffled: list[float]


@dataclass(frozen=True)
class Cell:
    n_workers: int
    k_value: int
    select_batch: int
    fixed_pair: tuple[float, float]
    diloco_alpha: float
    fixed: list[float]
    diloco: list[float]
    schedule: list[float]
    oracle: list[float]
    design_oracle: dict[str, list[float]]
    policy_scores: tuple[PolicyScore, ...]


def mean(vals: Iterable[float]) -> float:
    rows = tuple(vals)
    return sum(rows) / len(rows) if rows else 0.0


def ci95(vals: Iterable[float]) -> float:
    rows = tuple(vals)
    if len(rows) < 2:
        return 0.0
    return 1.96 * statistics.stdev(rows) / math.sqrt(len(rows))


def mean_ci(vals: Iterable[float]) -> tuple[float, float]:
    rows = tuple(vals)
    return mean(rows), ci95(rows)


def fmt_ci(vals: Iterable[float], width: int = 7, prec: int = 3) -> str:
    rows = tuple(vals)
    return f"{mean(rows):{width}.{prec}f}+/-{ci95(rows):.{prec}f}"


def paired(a: Iterable[float], b: Iterable[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def positive_ci(vals: Iterable[float]) -> bool:
    m, hw = mean_ci(vals)
    return m - hw > 0.0


def capture(selector: list[float], fixed: list[float], oracle: list[float]) -> list[float]:
    out: list[float] = []
    for sel, fix, ora in zip(selector, fixed, oracle):
        gap = ora - fix
        out.append((sel - fix) / gap if abs(gap) > EPS else 0.0)
    return out


def best_control(cell: Cell) -> list[float]:
    return [max(f, d, s) for f, d, s in zip(cell.fixed, cell.diloco, cell.schedule)]


def pair_from_weighted_indices(
    design: e33.Design,
    indices: list[int],
    weights: list[float],
) -> tuple[float, float]:
    total_w = sum(weights)
    if total_w <= EPS:
        return design[indices[0]]
    alpha_bar = sum(weights[j] * design[idx][0] for j, idx in enumerate(indices)) / total_w
    alpha_beta_bar = sum(
        weights[j] * design[idx][0] * design[idx][1] for j, idx in enumerate(indices)
    ) / total_w
    beta_eff = alpha_beta_bar / alpha_bar if abs(alpha_bar) > EPS else 0.0
    return (alpha_bar, max(0.0, min(0.95, beta_eff)))


def probe_losses(row: e33.FastRoundRow, velocity, design: e33.Design) -> list[float]:
    losses: list[float] = []
    for alpha, beta in design:
        update = e33.update_from_pair(row, velocity, alpha, beta)
        losses.append(e30.loss_float(row.theta + update, row.select_x, row.select_y))
    return losses


def select_pair(
    row: e33.FastRoundRow,
    velocity,
    design: e33.Design,
    policy: str,
    fixed_pair: tuple[float, float],
    shuffle_shift: int | None,
) -> tuple[float, float]:
    losses = probe_losses(row, velocity, design)
    if shuffle_shift is not None:
        shift = shuffle_shift % len(losses)
        losses = losses[shift:] + losses[:shift]
    order = sorted(range(len(design)), key=lambda idx: losses[idx])
    if policy == "top1":
        return design[order[0]]
    if policy == "top3_mean":
        idxs = order[: min(3, len(order))]
        return pair_from_weighted_indices(design, idxs, [1.0] * len(idxs))
    if policy == "top5_mean":
        idxs = order[: min(5, len(order))]
        return pair_from_weighted_indices(design, idxs, [1.0] * len(idxs))
    if policy in ("soft_std1", "soft_std2"):
        tau_mult = 1.0 if policy == "soft_std1" else 2.0
        loss_mean = mean(losses)
        denom = statistics.pstdev(losses) * tau_mult
        if denom <= EPS:
            return pair_from_weighted_indices(design, list(range(len(design))), [1.0] * len(design))
        raw_weights = [math.exp(-(loss - min(losses)) / denom) for loss in losses]
        return pair_from_weighted_indices(design, list(range(len(design))), raw_weights)
    if policy == "top3_guard_fixed":
        fixed_loss = e30.loss_float(
            row.theta + e33.update_from_pair(row, velocity, fixed_pair[0], fixed_pair[1]),
            row.select_x,
            row.select_y,
        )
        if losses[order[0]] >= fixed_loss:
            return fixed_pair
        idxs = order[: min(3, len(order))]
        return pair_from_weighted_indices(design, idxs, [1.0] * len(idxs))
    raise ValueError(f"unknown policy: {policy}")


def target_design_pair(row: e33.FastRoundRow, velocity, design: e33.Design) -> tuple[float, float]:
    best_pair = design[0]
    best_loss = float("inf")
    for alpha, beta in design:
        update = e33.update_from_pair(row, velocity, alpha, beta)
        loss = e30.loss_float(row.theta + update, row.target_x, row.target_y)
        if loss < best_loss:
            best_loss = loss
            best_pair = (alpha, beta)
    return best_pair


def score_selector(
    rows: list[e33.FastRoundRow],
    design: e33.Design,
    policy: str,
    fixed_pair: tuple[float, float],
    shuffle_seed: int | None,
) -> float:
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=rows[0].theta.device)
    raw = 0.0
    total = 0.0
    for idx, row in enumerate(rows):
        shift = None if shuffle_seed is None else idx + 7 * shuffle_seed + 1
        alpha, beta = select_pair(row, velocity, design, policy, fixed_pair, shift)
        update = e33.update_from_pair(row, velocity, alpha, beta)
        raw += e33.improvement(row, row.fedavg)
        total += e33.improvement(row, update)
        velocity = (row.fedavg + beta * velocity).detach()
    return total / raw if raw > EPS else total


def score_cell(k_value: int, n_workers: int, select_batch: int, device: str) -> Cell:
    pilot = e33.make_groups(k_value, n_workers, select_batch, PILOT_SEEDS, device)
    fixed_pair = e33.tune_fixed_alpha_beta(pilot)
    diloco_alpha = e33.tune_diloco_alpha(pilot)
    schedule = e33.tune_schedule(pilot)
    heldout = e33.make_groups(k_value, n_workers, select_batch, HELDOUT_SEEDS, device)
    designs_all = e33.design_map(k_value)
    designs = {name: designs_all[name] for name in DESIGN_NAMES if name in designs_all}

    def schedule_pair(row, velocity, idx):
        del row, velocity
        return schedule[min(idx, len(schedule) - 1)]

    fixed_vals: list[float] = []
    diloco_vals: list[float] = []
    schedule_vals: list[float] = []
    oracle_vals: list[float] = []
    design_oracle: dict[str, list[float]] = {name: [] for name in designs}
    policy_vals: dict[tuple[str, str], list[float]] = {
        (name, policy): [] for name in designs for policy in POLICIES
    }
    shuffled_vals: dict[tuple[str, str], list[float]] = {
        (name, policy): [] for name in designs for policy in POLICIES
    }

    for rep_idx, rows in enumerate(heldout):
        fixed_vals.append(e33.score_policy(rows, lambda _row, _vel, _idx, p=fixed_pair: p))
        diloco_vals.append(e33.score_policy(rows, lambda _row, _vel, _idx, a=diloco_alpha: (a, 0.90)))
        schedule_vals.append(e33.score_policy(rows, schedule_pair))
        oracle_vals.append(e33.score_policy(rows, lambda row, vel, _idx: e33.oracle_pair(row, vel)))
        for name, design in designs.items():
            design_oracle[name].append(
                e33.score_policy(rows, lambda row, vel, _idx, d=design: target_design_pair(row, vel, d))
            )
            for policy in POLICIES:
                key = (name, policy)
                policy_vals[key].append(score_selector(rows, design, policy, fixed_pair, None))
                shuffled_vals[key].append(score_selector(rows, design, policy, fixed_pair, rep_idx))

    return Cell(
        n_workers=n_workers,
        k_value=k_value,
        select_batch=select_batch,
        fixed_pair=fixed_pair,
        diloco_alpha=diloco_alpha,
        fixed=fixed_vals,
        diloco=diloco_vals,
        schedule=schedule_vals,
        oracle=oracle_vals,
        design_oracle=design_oracle,
        policy_scores=tuple(
            PolicyScore(
                design=name,
                policy=policy,
                points=len(designs[name]),
                values=policy_vals[(name, policy)],
                shuffled=shuffled_vals[(name, policy)],
            )
            for name in designs
            for policy in POLICIES
        ),
    )


def decide(cells: list[Cell], compute_status: str) -> str:
    if compute_status != "REAL_GPU":
        return "NO_GPU_HARNESS"
    by_policy: dict[tuple[str, str], list[float]] = {}
    by_shuffle: dict[tuple[str, str], list[float]] = {}
    by_capture: dict[tuple[str, str], list[float]] = {}
    for cell in cells:
        controls = best_control(cell)
        for score in cell.policy_scores:
            key = (score.design, score.policy)
            by_policy.setdefault(key, []).extend(paired(score.values, controls))
            by_shuffle.setdefault(key, []).extend(paired(score.values, score.shuffled))
            by_capture.setdefault(key, []).extend(capture(score.values, cell.fixed, cell.oracle))
    for key, vals in by_policy.items():
        cap_mean, cap_hw = mean_ci(by_capture[key])
        if positive_ci(vals) and positive_ci(by_shuffle[key]) and cap_mean - cap_hw > 0.05:
            return "TOPK_PROBE_BEATS_TARGET_CONTROLS"
    for key, vals in by_policy.items():
        if positive_ci(vals):
            return "TOPK_PROBE_PARTIAL_CONTROL_WIN"
    return "TOPK_PROBE_STILL_LOSES"


def no_gpu_artifacts(diag: e30.ComputeDiagnostic) -> tuple[str, str, str]:
    run_log = "\n".join(
        (
            "DFW-E071 top-k probe outer control",
            f"host={platform.node()} python={sys.executable}",
            f"DECISION NO_GPU_HARNESS compute_status={diag.status} detail={diag.detail}",
        )
    )
    result = "\n".join(
        (
            "# DFW-E071 Result - Top-k Probe Outer Control",
            "",
            "Decision label: `NO_GPU_HARNESS`",
            "",
            f"Diagnostic: `{diag.detail}`.",
            "",
        )
    )
    legit = "# DFW-E071 Legitimacy Check\n\n- GPU workload: not run.\n"
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
        "DFW-E071 top-k probe outer control",
        f"host={platform.node()} python={sys.executable} compute_status={diag.status} {diag.detail}",
        (
            f"model=torch_two_layer_binary_classifier dim={e30.DIM} N={N_WORKERS} "
            f"K={K_VALUES} rounds={e30.ROUNDS} eta={e30.ETA}"
        ),
        f"select_batches={SELECT_BATCHES} designs={DESIGN_NAMES} policies={POLICIES}",
        f"pilot_seeds={len(PILOT_SEEDS)} heldout_seeds={len(HELDOUT_SEEDS)}",
        "",
    ]
    for select_batch in SELECT_BATCHES:
        for n_workers in N_WORKERS:
            for k_value in K_VALUES:
                cell = score_cell(k_value, n_workers, select_batch, diag.device)
                cells.append(cell)
                controls = best_control(cell)
                best = max(cell.policy_scores, key=lambda score: mean(paired(score.values, controls)))
                run_lines.append(
                    f"CELL select={select_batch} N={n_workers} K={k_value} "
                    f"fixed={cell.fixed_pair} schedule={fmt_ci(cell.schedule)} oracle={fmt_ci(cell.oracle)} "
                    f"best={best.design}/{best.policy} value={fmt_ci(best.values)} "
                    f"minus_control={fmt_ci(paired(best.values, controls))} "
                    f"minus_shuffled={fmt_ci(paired(best.values, best.shuffled))} "
                    f"capture={fmt_ci(capture(best.values, cell.fixed, cell.oracle))}"
                )
    decision = decide(cells, diag.status)
    run_lines.append("")
    run_lines.append(f"DECISION {decision}")
    run_lines.append(f"compute_status={diag.status}")

    lines = [
        "# DFW-E071 Result - Top-k Probe Outer Control",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{diag.status}` via PyTorch CUDA: `{diag.detail}`.",
        "",
        "## Metrics",
        "",
        "| select | N | K | design | policy | points | overhead | fixed | DiLoCo | schedule | oracle | design oracle | selector | selector minus best control | selector minus shuffled | capture |",
        "|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        controls = best_control(cell)
        for score in cell.policy_scores:
            overhead = score.points * cell.select_batch / e30.TRAIN_SIZE / (
                3.0 * cell.n_workers * cell.k_value
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{cell.select_batch}",
                        f"{cell.n_workers}",
                        f"{cell.k_value}",
                        score.design,
                        score.policy,
                        f"{score.points}",
                        f"{100.0 * overhead:.2f}%",
                        fmt_ci(cell.fixed).strip(),
                        fmt_ci(cell.diloco).strip(),
                        fmt_ci(cell.schedule).strip(),
                        fmt_ci(cell.oracle).strip(),
                        fmt_ci(cell.design_oracle[score.design]).strip(),
                        fmt_ci(score.values).strip(),
                        fmt_ci(paired(score.values, controls)).strip(),
                        fmt_ci(paired(score.values, score.shuffled)).strip(),
                        fmt_ci(capture(score.values, cell.fixed, cell.oracle)).strip(),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])
    legit = "\n".join(
        [
            "# DFW-E071 Legitimacy Check",
            "",
            f"- Compute: `{diag.status}` / `{diag.detail}`.",
            "- Selector uses select batches only; scoring uses target batches.",
            "- Controls include fixed, DiLoCo beta=.9, and target-retuned schedule.",
            "- Shuffled selector rotates probe labels as a noise control.",
            "- Top-k policies reuse the same probe evaluations as top-1.",
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
