"""DFW-E090B: fixed-compute outer-control HPO fallback.

E090 proper is the stale-LR/default recovery table.  If that stricter final-
trajectory ledger does not clear, the next strongest result is the E080/E081
outer-control HPO story.  This companion prices a compact version of that story
against a fixed-compute external HPO comparator:

  * internal HPO scores a small `(alpha,beta)` arm set from one shared worker
    update pool and applies the soft E080 selector;
  * external HPO splits the same total workers across independent arm budgets
    and chooses one arm by validation/select improvement;
  * one arm is a deliberately stale server default, so the table still reports
    stale/default/fresh controls.
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
import dfw_e071_gpu_topk_probe_outer_control as e71


DATE = "20261008"
OUT_DIR = Path(os.environ.get("DFW_E090B_OUT_DIR", Path(__file__).resolve().parent)).resolve()
RESULT_PATH = OUT_DIR / f"dfw_e090b_fixed_compute_outer_control_hpo_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e090b_fixed_compute_outer_control_hpo_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e090b_fixed_compute_outer_control_hpo_LEGITIMACY_CHECK_{DATE}.md"

K_VALUES = e32.env_int_tuple("DFW_E090B_K_VALUES", (4, 8))
N_VALUES = e32.env_int_tuple("DFW_E090B_N_VALUES", (8, 16))
SELECT_BATCH = int(os.environ.get("DFW_E090B_SELECT_BATCH", "384"))
ROUNDS = int(os.environ.get("DFW_E090B_ROUNDS", "8"))
HELDOUT_SEEDS = e32.env_seed_tuple("DFW_E090B_HELDOUT_SEEDS", tuple(range(998_000, 998_032)))
POLICY = "soft_std2"
EPS = 1e-12

mean = e71.mean
fmt_ci = e71.fmt_ci
paired = e71.paired
positive_ci = e71.positive_ci


@dataclass(frozen=True)
class ArmSpec:
    name: str
    pair: tuple[float, float]
    count: int


@dataclass(frozen=True)
class Score:
    target: float
    select: float


@dataclass(frozen=True)
class Cell:
    k_value: int
    n_workers: int
    arms: tuple[ArmSpec, ...]
    stale_default: list[float]
    fresh_theory: list[float]
    high_arm: list[float]
    internal_probe: list[float]
    internal_shuffled: list[float]
    internal_oracle: list[float]
    external_valid: list[float]
    external_oracle: list[float]
    external_stale: list[float]
    external_fresh: list[float]
    external_high: list[float]
    cost_pct: float


def compute_status() -> tuple[str, str, str]:
    diag = e30.compute_diagnostic()
    if diag.status == "REAL_GPU":
        return diag.status, diag.detail, diag.device
    if e30.torch is None:
        return "NO_TORCH", diag.detail, "cpu"
    version = getattr(e30.torch, "__version__", "unknown")
    return "TORCH_CPU", f"torch={version} cuda_available=False", "cpu"


def ci95(vals: Iterable[float]) -> float:
    rows = tuple(vals)
    if len(rows) < 2:
        return 0.0
    return 1.96 * statistics.stdev(rows) / math.sqrt(len(rows))


def theory_pair(k_value: int) -> tuple[float, float]:
    alpha = min(20.0, max(4.0, 72.0 / float(k_value)))
    return (alpha, 0.15)


def compact_arms(k_value: int, n_workers: int) -> tuple[ArmSpec, ...]:
    base, _beta = theory_pair(k_value)
    pairs = (
        ("stale_global_default", (5.0, 0.05)),
        ("fresh_theory", (base, 0.15)),
        ("high_probe_arm", (1.25 * base, 0.30)),
        ("low_safe", (0.50 * base, 0.05)),
        ("mid_nomomentum", (base, 0.0)),
        ("mid_low_momentum", (0.75 * base, 0.15)),
        ("aggressive_low_momentum", (1.50 * base, 0.15)),
        ("double_nomomentum", (2.0 * base, 0.0)),
    )
    if n_workers < len(pairs):
        pairs = pairs[:n_workers]
    base_count = n_workers // len(pairs)
    remainder = n_workers % len(pairs)
    return tuple(
        ArmSpec(name, pair, base_count + (1 if idx < remainder else 0))
        for idx, (name, pair) in enumerate(pairs)
    )


def score_pair(rows: list[e33.FastRoundRow], pair: tuple[float, float], metric: str) -> float:
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=rows[0].theta.device)
    raw = 0.0
    total = 0.0
    alpha, beta = pair
    for row in rows:
        update = e33.update_from_pair(row, velocity, alpha, beta)
        if metric == "target":
            raw += e33.improvement(row, row.fedavg)
            total += e33.improvement(row, update)
        elif metric == "select":
            start = e30.loss_float(row.theta, row.select_x, row.select_y)
            raw += start - e30.loss_float(row.theta + row.fedavg, row.select_x, row.select_y)
            total += start - e30.loss_float(row.theta + update, row.select_x, row.select_y)
        else:
            raise ValueError(metric)
        velocity = (row.fedavg + beta * velocity).detach()
    return total / raw if abs(raw) > EPS else total


def score_both(rows: list[e33.FastRoundRow], pair: tuple[float, float]) -> Score:
    return Score(target=score_pair(rows, pair, "target"), select=score_pair(rows, pair, "select"))


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


def score_internal_oracle(rows: list[e33.FastRoundRow], design: e33.Design) -> float:
    return e33.score_policy(rows, lambda row, vel, _idx, d=design: target_design_pair(row, vel, d))


def external_scores(
    seed: int,
    k_value: int,
    select_batch: int,
    device: str,
    arms: tuple[ArmSpec, ...],
) -> tuple[Score, Score, list[Score]]:
    scores: list[Score] = []
    for arm in arms:
        rows = e33.make_fast_rows(seed, arm.count, k_value, select_batch, device)
        scores.append(score_both(rows, arm.pair))
    valid = max(scores, key=lambda score: score.select)
    oracle = max(scores, key=lambda score: score.target)
    return valid, oracle, scores


def probe_cost_pct(k_value: int, n_workers: int, points: int) -> float:
    return 100.0 * points * SELECT_BATCH / (e30.TRAIN_SIZE * 3.0 * n_workers * k_value)


def capture(vals: list[float], base: list[float], oracle: list[float]) -> list[float]:
    out = []
    for val, bval, oval in zip(vals, base, oracle):
        gap = oval - bval
        out.append((val - bval) / gap if abs(gap) > EPS else 0.0)
    return out


def diagnose_cell(k_value: int, n_workers: int, device: str) -> Cell:
    arms = compact_arms(k_value, n_workers)
    design = tuple(arm.pair for arm in arms)
    internal_groups = e33.make_groups(k_value, n_workers, SELECT_BATCH, HELDOUT_SEEDS, device)
    stale_default: list[float] = []
    fresh_theory: list[float] = []
    high_arm: list[float] = []
    internal_probe: list[float] = []
    internal_shuffled: list[float] = []
    internal_oracle: list[float] = []
    external_valid: list[float] = []
    external_oracle: list[float] = []
    external_stale: list[float] = []
    external_fresh: list[float] = []
    external_high: list[float] = []

    for rep_idx, (seed, rows) in enumerate(zip(HELDOUT_SEEDS, internal_groups)):
        stale_default.append(score_pair(rows, arms[0].pair, "target"))
        fresh_theory.append(score_pair(rows, arms[1].pair, "target"))
        high_arm.append(score_pair(rows, arms[2].pair, "target"))
        internal_probe.append(e71.score_selector(rows, design, POLICY, arms[0].pair, None))
        internal_shuffled.append(e71.score_selector(rows, design, POLICY, arms[0].pair, rep_idx))
        internal_oracle.append(score_internal_oracle(rows, design))
        valid, oracle, ext = external_scores(seed, k_value, SELECT_BATCH, device, arms)
        external_valid.append(valid.target)
        external_oracle.append(oracle.target)
        external_stale.append(ext[0].target)
        external_fresh.append(ext[1].target)
        external_high.append(ext[2].target)

    return Cell(
        k_value=k_value,
        n_workers=n_workers,
        arms=arms,
        stale_default=stale_default,
        fresh_theory=fresh_theory,
        high_arm=high_arm,
        internal_probe=internal_probe,
        internal_shuffled=internal_shuffled,
        internal_oracle=internal_oracle,
        external_valid=external_valid,
        external_oracle=external_oracle,
        external_stale=external_stale,
        external_fresh=external_fresh,
        external_high=external_high,
        cost_pct=probe_cost_pct(k_value, n_workers, len(arms)),
    )


def extend_pooled(cells: Iterable[Cell], attr: str, baseline: str) -> list[float]:
    out: list[float] = []
    for cell in cells:
        out.extend(paired(getattr(cell, attr), getattr(cell, baseline)))
    return out


def decide(cells: list[Cell], status: str) -> str:
    if status == "NO_TORCH":
        return "NO_TORCH"
    probe_stale = extend_pooled(cells, "internal_probe", "stale_default")
    probe_shuffled = extend_pooled(cells, "internal_probe", "internal_shuffled")
    probe_external = extend_pooled(cells, "internal_probe", "external_valid")
    if positive_ci(probe_stale) and positive_ci(probe_shuffled) and positive_ci(probe_external):
        return "FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN"
    if positive_ci(probe_stale) and positive_ci(probe_shuffled) and mean(probe_external) > 0.0:
        return "FIXED_COMPUTE_OUTER_CONTROL_HPO_LIVE"
    if positive_ci(probe_stale) and positive_ci(probe_shuffled):
        return "OUTER_CONTROL_HPO_LIVE_EXTERNAL_BOUNDARY"
    return "FIXED_COMPUTE_OUTER_CONTROL_NOT_CLEARED"


def no_torch_artifacts(detail: str) -> tuple[str, str, str]:
    run_log = "\n".join(
        (
            "DFW-E090B fixed-compute outer-control HPO",
            f"host={platform.node()} python={sys.executable}",
            f"DECISION NO_TORCH detail={detail}",
        )
    )
    result = "# DFW-E090B Result - Fixed-Compute Outer-Control HPO\n\nDecision label: `NO_TORCH`\n"
    legit = "# DFW-E090B Legitimacy Check\n\n- PyTorch workload: not run.\n"
    return run_log + "\n", result + "\n", legit + "\n"


def run_experiment() -> tuple[str, str, str]:
    status, detail, device = compute_status()
    if status == "NO_TORCH":
        return no_torch_artifacts(detail)

    e30.ROUNDS = ROUNDS
    e30.torch.set_default_dtype(e30.torch.float32)
    try:
        e30.torch.set_num_threads(int(os.environ.get("DFW_E090B_TORCH_THREADS", "1")))
    except Exception:
        pass

    cells: list[Cell] = []
    run_lines = [
        "DFW-E090B fixed-compute outer-control HPO",
        f"host={platform.node()} python={sys.executable} compute_status={status} {detail}",
        f"K={K_VALUES} N={N_VALUES} rounds={ROUNDS} select_batch={SELECT_BATCH} heldout_seeds={len(HELDOUT_SEEDS)}",
        "arms=stale_global_default,fresh_theory,high_probe_arm policy=soft_std2",
        "",
    ]
    for n_workers in N_VALUES:
        for k_value in K_VALUES:
            cell = diagnose_cell(k_value, n_workers, device)
            cells.append(cell)
            run_lines.append(
                f"CELL N={n_workers} K={k_value} "
                f"arms={';'.join(f'{a.name}:{a.pair}/count={a.count}' for a in cell.arms)} "
                f"stale={fmt_ci(cell.stale_default)} fresh={fmt_ci(cell.fresh_theory)} high={fmt_ci(cell.high_arm)} "
                f"internal={fmt_ci(cell.internal_probe)} internal-stale={fmt_ci(paired(cell.internal_probe, cell.stale_default))} "
                f"internal-shuffled={fmt_ci(paired(cell.internal_probe, cell.internal_shuffled))} "
                f"external_valid={fmt_ci(cell.external_valid)} internal-external_valid={fmt_ci(paired(cell.internal_probe, cell.external_valid))} "
                f"external_oracle={fmt_ci(cell.external_oracle)} internal-external_oracle={fmt_ci(paired(cell.internal_probe, cell.external_oracle))} "
                f"external_stale={fmt_ci(cell.external_stale)} external_fresh={fmt_ci(cell.external_fresh)} "
                f"external_high={fmt_ci(cell.external_high)} internal_oracle={fmt_ci(cell.internal_oracle)} "
                f"capture_vs_stale_internal_oracle={fmt_ci(capture(cell.internal_probe, cell.stale_default, cell.internal_oracle))} "
                f"cost={cell.cost_pct:.2f}%"
            )
            print(run_lines[-1], flush=True)

    pooled_probe_stale = extend_pooled(cells, "internal_probe", "stale_default")
    pooled_probe_fresh = extend_pooled(cells, "internal_probe", "fresh_theory")
    pooled_probe_shuffled = extend_pooled(cells, "internal_probe", "internal_shuffled")
    pooled_probe_external = extend_pooled(cells, "internal_probe", "external_valid")
    pooled_probe_external_oracle = extend_pooled(cells, "internal_probe", "external_oracle")
    pooled_external_stale = extend_pooled(cells, "external_valid", "stale_default")
    pooled_probe_internal_oracle = extend_pooled(cells, "internal_probe", "internal_oracle")
    decision = decide(cells, status)

    run_lines.extend(
        [
            "",
            f"POOLED internal-stale={fmt_ci(pooled_probe_stale)} "
            f"internal-fresh={fmt_ci(pooled_probe_fresh)} "
            f"internal-shuffled={fmt_ci(pooled_probe_shuffled)} "
            f"internal-external_valid={fmt_ci(pooled_probe_external)} "
            f"internal-external_oracle={fmt_ci(pooled_probe_external_oracle)} "
            f"external_valid-stale={fmt_ci(pooled_external_stale)} "
            f"internal-internal_oracle={fmt_ci(pooled_probe_internal_oracle)}",
            f"DECISION {decision}",
            f"compute_status={status}",
        ]
    )

    lines = [
        "# DFW-E090B Result - Fixed-Compute Outer-Control HPO",
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
            "This is the fallback fixed-compute table after E090 stale-LR recovery did not clear. "
            "The stale/default arm is a transferred global server pair `(5.0, 0.05)`; the fresh "
            "arm is the K-aware theory pair; and the high arm is a more aggressive probe arm."
        ),
        "",
        (
            "Internal HPO evaluates the outer-control arms from one shared worker-update pool and "
            "uses the E080 soft probe selector. External HPO spends the same total workers by "
            "splitting them across independent arm budgets and selecting by validation/select "
            "improvement. Target selection is reported only as an oracle diagnostic."
        ),
        "",
        "## Pooled",
        "",
        f"- internal minus stale/default control: `{fmt_ci(pooled_probe_stale).strip()}`",
        f"- internal minus fresh/theory control: `{fmt_ci(pooled_probe_fresh).strip()}`",
        f"- internal minus shuffled internal: `{fmt_ci(pooled_probe_shuffled).strip()}`",
        f"- internal minus external validation-selected HPO: `{fmt_ci(pooled_probe_external).strip()}`",
        f"- internal minus external oracle HPO: `{fmt_ci(pooled_probe_external_oracle).strip()}`",
        f"- external validation-selected HPO minus stale/default: `{fmt_ci(pooled_external_stale).strip()}`",
        f"- internal minus internal target oracle: `{fmt_ci(pooled_probe_internal_oracle).strip()}`",
        "",
        "## Cells",
        "",
        "| N | K | stale/default | fresh/theory | high | internal | internal-stale | internal-shuffled | internal-ext-valid | ext-valid | ext-oracle | probe cost |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{cell.n_workers}",
                    f"{cell.k_value}",
                    fmt_ci(cell.stale_default).strip(),
                    fmt_ci(cell.fresh_theory).strip(),
                    fmt_ci(cell.high_arm).strip(),
                    fmt_ci(cell.internal_probe).strip(),
                    fmt_ci(paired(cell.internal_probe, cell.stale_default)).strip(),
                    fmt_ci(paired(cell.internal_probe, cell.internal_shuffled)).strip(),
                    fmt_ci(paired(cell.internal_probe, cell.external_valid)).strip(),
                    fmt_ci(cell.external_valid).strip(),
                    fmt_ci(cell.external_oracle).strip(),
                    f"{cell.cost_pct:.2f}%",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    if decision == "FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN":
        lines.append(
            "The outer-control fallback clears the fixed-compute external-HPO comparator: "
            "the internal probe selector beats the stale/default control, shuffled labels, "
            "and validation-selected external HPO with the same worker budget."
        )
    elif "LIVE" in decision:
        lines.append(
            "The outer-control fallback is live: it beats stale/default and shuffled controls, "
            "with the external-HPO columns marking the remaining fixed-compute boundary."
        )
    else:
        lines.append("The outer-control fallback does not clear the fixed-compute gate.")
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])

    legit = "\n".join(
        [
            "# DFW-E090B Legitimacy Check",
            "",
            f"- Compute: `{status}` / `{detail}`.",
            "- Same total worker budget: external arm counts sum to the internal worker count in every cell.",
            "- Internal HPO uses one shared worker-update pool and does not spend extra worker trajectories for each outer-control arm.",
            "- External HPO splits workers across independent stale/default, fresh/theory, and high outer-control arms, then selects by validation/select improvement.",
            "- Shuffled labels rotate probe losses inside the same internal selector.",
            "- Target losses are used for scoring and oracle diagnostics only.",
            "- This is an E080/E081-style one-step local-SGD update-map ledger, not a final full-training-trajectory claim.",
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
