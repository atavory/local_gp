"""DFW-E084: selected-arm trajectory diagnostic for the K<=8 controller.

E080/E081 made the deployment case for a frozen low-K scaled/soft_std2
probe controller.  This diagnostic asks what that controller is actually
doing over the round trajectory: pinning to design edges, drifting with the
training phase, or sitting in a stable interior mixture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import platform
import statistics
import sys
from typing import Iterable

import dfw_e071_gpu_topk_probe_outer_control as e71


DATE = "20261006"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e084_gpu_outer_control_trajectory_diagnostic_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e084_gpu_outer_control_trajectory_diagnostic_run_{DATE}.txt"
LEGITIMACY_PATH = (
    OUT_DIR / f"dfw_e084_gpu_outer_control_trajectory_diagnostic_LEGITIMACY_CHECK_{DATE}.md"
)

K_VALUES = (4, 8)
N_WORKERS = (8, 16)
SELECT_BATCHES = (384,)
PILOT_SEEDS = tuple(range(978_000, 978_002))
HELDOUT_SEEDS = tuple(range(979_000, 979_016))
DESIGN_NAME = "scaled"
POLICY = "soft_std2"
TAU_MULT = 2.0
EPS = 1e-12


@dataclass(frozen=True)
class RoundTrace:
    round_idx: int
    phase: str
    alpha: float
    beta: float
    top1_alpha: float
    top1_beta: float
    target_alpha: float
    target_beta: float
    entropy: float
    alpha_edge_weight: float
    beta_edge_weight: float
    top1_alpha_edge: float
    top1_beta_edge: float
    target_alpha_edge: float
    target_beta_edge: float


@dataclass(frozen=True)
class CellTrace:
    select_batch: int
    n_workers: int
    k_value: int
    fixed_pair: tuple[float, float]
    selector: list[float]
    shuffled: list[float]
    fixed: list[float]
    schedule: list[float]
    oracle: list[float]
    traces: list[RoundTrace]
    alpha_min: float
    alpha_max: float
    beta_min: float
    beta_max: float


def mean(vals: Iterable[float]) -> float:
    rows = tuple(vals)
    return sum(rows) / len(rows) if rows else 0.0


def ci95(vals: Iterable[float]) -> float:
    rows = tuple(vals)
    if len(rows) < 2:
        return 0.0
    return 1.96 * statistics.stdev(rows) / math.sqrt(len(rows))


def fmt(vals: Iterable[float], prec: int = 3) -> str:
    rows = tuple(vals)
    return f"{mean(rows):.{prec}f}+/-{ci95(rows):.{prec}f}"


def phase_for(idx: int, n_rounds: int) -> str:
    if idx < n_rounds / 3.0:
        return "early"
    if idx < 2.0 * n_rounds / 3.0:
        return "middle"
    return "late"


def normalized_entropy(weights: list[float]) -> float:
    total = sum(weights)
    if total <= EPS or len(weights) <= 1:
        return 0.0
    probs = [max(weight / total, EPS) for weight in weights]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    return entropy / math.log(len(weights))


def soft_std2_weights(losses: list[float]) -> list[float]:
    denom = statistics.pstdev(losses) * TAU_MULT
    if denom <= EPS:
        return [1.0 for _loss in losses]
    low = min(losses)
    return [math.exp(-(loss - low) / denom) for loss in losses]


def edge_mask(design: e71.e33.Design) -> tuple[list[bool], list[bool], float, float, float, float]:
    alphas = [point[0] for point in design]
    betas = [point[1] for point in design]
    alpha_min = min(alphas)
    alpha_max = max(alphas)
    beta_min = min(betas)
    beta_max = max(betas)
    alpha_edges = [alpha in (alpha_min, alpha_max) for alpha, _beta in design]
    beta_edges = [beta in (beta_min, beta_max) for _alpha, beta in design]
    return alpha_edges, beta_edges, alpha_min, alpha_max, beta_min, beta_max


def edge_weight(weights: list[float], mask: list[bool]) -> float:
    total = sum(weights)
    if total <= EPS:
        return 0.0
    return sum(weight for weight, is_edge in zip(weights, mask) if is_edge) / total


def is_edge(value: float, low: float, high: float) -> float:
    return float(abs(value - low) <= 1e-9 or abs(value - high) <= 1e-9)


def trace_group(
    rows: list[e71.e33.FastRoundRow],
    design: e71.e33.Design,
) -> list[RoundTrace]:
    alpha_edges, beta_edges, alpha_min, alpha_max, beta_min, beta_max = edge_mask(design)
    velocity = e71.e30.torch.zeros(
        e71.e30.DIM,
        dtype=e71.e30.torch.float32,
        device=rows[0].theta.device,
    )
    traces: list[RoundTrace] = []
    n_rounds = len(rows)
    for idx, row in enumerate(rows):
        losses = e71.probe_losses(row, velocity, design)
        weights = soft_std2_weights(losses)
        pair = e71.pair_from_weighted_indices(
            design,
            list(range(len(design))),
            weights,
        )
        top1_idx = min(range(len(design)), key=lambda point_idx: losses[point_idx])
        top1_pair = design[top1_idx]
        target_pair = e71.target_design_pair(row, velocity, design)
        traces.append(
            RoundTrace(
                round_idx=idx,
                phase=phase_for(idx, n_rounds),
                alpha=pair[0],
                beta=pair[1],
                top1_alpha=top1_pair[0],
                top1_beta=top1_pair[1],
                target_alpha=target_pair[0],
                target_beta=target_pair[1],
                entropy=normalized_entropy(weights),
                alpha_edge_weight=edge_weight(weights, alpha_edges),
                beta_edge_weight=edge_weight(weights, beta_edges),
                top1_alpha_edge=is_edge(top1_pair[0], alpha_min, alpha_max),
                top1_beta_edge=is_edge(top1_pair[1], beta_min, beta_max),
                target_alpha_edge=is_edge(target_pair[0], alpha_min, alpha_max),
                target_beta_edge=is_edge(target_pair[1], beta_min, beta_max),
            )
        )
        velocity = (row.fedavg + pair[1] * velocity).detach()
    return traces


def score_selector_and_trace(
    rows: list[e71.e33.FastRoundRow],
    design: e71.e33.Design,
    fixed_pair: tuple[float, float],
    shuffle_seed: int | None,
) -> tuple[float, list[RoundTrace]]:
    value = e71.score_selector(rows, design, POLICY, fixed_pair, shuffle_seed)
    if shuffle_seed is not None:
        return value, []
    return value, trace_group(rows, design)


def score_constant_pair(
    groups: list[list[e71.e33.FastRoundRow]],
    pair: tuple[float, float],
) -> list[float]:
    return [
        e71.e33.score_policy(rows, lambda _row, _velocity, _idx, p=pair: p)
        for rows in groups
    ]


def score_schedule(
    groups: list[list[e71.e33.FastRoundRow]],
    schedule: list[tuple[float, float]],
) -> list[float]:
    def choose(_row, _velocity, idx):
        return schedule[min(idx, len(schedule) - 1)]

    return [e71.e33.score_policy(rows, choose) for rows in groups]


def score_oracle(groups: list[list[e71.e33.FastRoundRow]]) -> list[float]:
    return [
        e71.e33.score_policy(rows, lambda row, velocity, _idx: e71.e33.oracle_pair(row, velocity))
        for rows in groups
    ]


def diagnose_cell(k_value: int, n_workers: int, select_batch: int, device: str) -> CellTrace:
    pilot = e71.e33.make_groups(k_value, n_workers, select_batch, PILOT_SEEDS, device)
    fixed_pair = e71.e33.tune_fixed_alpha_beta(pilot)
    schedule = e71.e33.tune_schedule(pilot)
    heldout = e71.e33.make_groups(k_value, n_workers, select_batch, HELDOUT_SEEDS, device)
    design = e71.e33.design_map(k_value)[DESIGN_NAME]
    _alpha_edges, _beta_edges, alpha_min, alpha_max, beta_min, beta_max = edge_mask(design)

    selector: list[float] = []
    shuffled: list[float] = []
    traces: list[RoundTrace] = []
    for rep_idx, rows in enumerate(heldout):
        value, group_trace = score_selector_and_trace(rows, design, fixed_pair, None)
        shuf_value, _empty = score_selector_and_trace(rows, design, fixed_pair, rep_idx)
        selector.append(value)
        shuffled.append(shuf_value)
        traces.extend(group_trace)

    return CellTrace(
        select_batch=select_batch,
        n_workers=n_workers,
        k_value=k_value,
        fixed_pair=fixed_pair,
        selector=selector,
        shuffled=shuffled,
        fixed=score_constant_pair(heldout, fixed_pair),
        schedule=score_schedule(heldout, schedule),
        oracle=score_oracle(heldout),
        traces=traces,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        beta_min=beta_min,
        beta_max=beta_max,
    )


def values(cell: CellTrace, attr: str, phase: str | None = None) -> list[float]:
    return [
        float(getattr(trace, attr))
        for trace in cell.traces
        if phase is None or trace.phase == phase
    ]


def phase_means(cell: CellTrace, attr: str) -> dict[str, float]:
    return {phase: mean(values(cell, attr, phase)) for phase in ("early", "middle", "late")}


def phase_span(cell: CellTrace, attr: str) -> float:
    by_phase = phase_means(cell, attr)
    return max(by_phase.values()) - min(by_phase.values())


def norm_span(span: float, low: float, high: float) -> float:
    width = high - low
    return span / width if width > EPS else 0.0


def classify_cell(cell: CellTrace) -> str:
    alpha_phase = norm_span(phase_span(cell, "alpha"), cell.alpha_min, cell.alpha_max)
    beta_phase = norm_span(phase_span(cell, "beta"), cell.beta_min, cell.beta_max)
    top1_alpha_edge = mean(values(cell, "top1_alpha_edge"))
    top1_beta_edge = mean(values(cell, "top1_beta_edge"))
    soft_alpha_edge = mean(values(cell, "alpha_edge_weight"))
    entropy = mean(values(cell, "entropy"))
    if top1_alpha_edge >= 0.70 and soft_alpha_edge >= 0.35:
        return "alpha_edge_pinned"
    if top1_beta_edge >= 0.80 and mean(values(cell, "top1_beta")) <= cell.beta_min + 1e-9:
        return "beta_low_boundary_pinned"
    if alpha_phase >= 0.10 or beta_phase >= 0.18:
        return "phase_moving"
    if entropy >= 0.55 and soft_alpha_edge < 0.35:
        return "stable_soft_interior"
    return "mixed_static"


def decide(cells: list[CellTrace], compute_status: str) -> str:
    if compute_status != "REAL_GPU":
        return "NO_GPU_HARNESS"
    labels = [classify_cell(cell) for cell in cells]
    if any(label == "phase_moving" for label in labels):
        return "OUTER_TRAJECTORY_PHASE_MOVING"
    if all(label in ("alpha_edge_pinned", "beta_low_boundary_pinned") for label in labels):
        return "OUTER_TRAJECTORY_EDGE_PINNED"
    if all(label == "stable_soft_interior" for label in labels):
        return "OUTER_TRAJECTORY_STABLE_SOFT_INTERIOR"
    if labels.count("beta_low_boundary_pinned") >= max(1, len(labels) // 2):
        return "OUTER_TRAJECTORY_BETA_LOW_BOUNDARY_STATIC"
    return "OUTER_TRAJECTORY_MIXED_STATIC"


def no_gpu_artifacts(diag: e71.e30.ComputeDiagnostic) -> tuple[str, str, str]:
    run_log = "\n".join(
        (
            "DFW-E084 outer-control trajectory diagnostic",
            f"host={platform.node()} python={sys.executable}",
            f"DECISION NO_GPU_HARNESS compute_status={diag.status} detail={diag.detail}",
        )
    )
    result = (
        "# DFW-E084 Result - Outer-Control Trajectory Diagnostic\n\n"
        "Decision label: `NO_GPU_HARNESS`\n"
    )
    legit = "# DFW-E084 Legitimacy Check\n\n- GPU workload: not run.\n"
    return run_log + "\n", result + "\n", legit + "\n"


def run_experiment() -> tuple[str, str, str]:
    e71.DESIGN_NAMES = (DESIGN_NAME,)
    e71.POLICIES = (POLICY,)
    e71.PILOT_SEEDS = PILOT_SEEDS
    e71.HELDOUT_SEEDS = HELDOUT_SEEDS

    diag = e71.e30.compute_diagnostic()
    if diag.status != "REAL_GPU":
        return no_gpu_artifacts(diag)
    e71.e30.torch.set_default_dtype(e71.e30.torch.float32)
    try:
        e71.e30.torch.set_num_threads(1)
    except Exception:
        pass

    cells = [
        diagnose_cell(k_value, n_workers, select_batch, diag.device)
        for select_batch in SELECT_BATCHES
        for n_workers in N_WORKERS
        for k_value in K_VALUES
    ]
    decision = decide(cells, diag.status)

    run_lines = [
        "DFW-E084 outer-control trajectory diagnostic",
        f"host={platform.node()} python={sys.executable} compute_status={diag.status} {diag.detail}",
        f"design={DESIGN_NAME} policy={POLICY} K={K_VALUES} N={N_WORKERS} select={SELECT_BATCHES}",
        f"pilot_seeds={len(PILOT_SEEDS)} heldout_seeds={len(HELDOUT_SEEDS)}",
        "",
    ]
    for cell in cells:
        alpha_phases = phase_means(cell, "alpha")
        beta_phases = phase_means(cell, "beta")
        label = classify_cell(cell)
        run_lines.append(
            f"CELL select={cell.select_batch} N={cell.n_workers} K={cell.k_value} "
            f"label={label} fixed_pair=({cell.fixed_pair[0]:.1f},{cell.fixed_pair[1]:.2f}) "
            f"selector={e71.fmt_ci(cell.selector)} shuffled={e71.fmt_ci(cell.shuffled)} "
            f"fixed={e71.fmt_ci(cell.fixed)} schedule={e71.fmt_ci(cell.schedule)} oracle={e71.fmt_ci(cell.oracle)} "
            f"alpha={fmt(values(cell, 'alpha'))} beta={fmt(values(cell, 'beta'))} "
            f"top1_alpha={fmt(values(cell, 'top1_alpha'))} top1_beta={fmt(values(cell, 'top1_beta'))} "
            f"target_alpha={fmt(values(cell, 'target_alpha'))} target_beta={fmt(values(cell, 'target_beta'))} "
            f"alpha_phase=early:{alpha_phases['early']:.3f},middle:{alpha_phases['middle']:.3f},late:{alpha_phases['late']:.3f} "
            f"beta_phase=early:{beta_phases['early']:.3f},middle:{beta_phases['middle']:.3f},late:{beta_phases['late']:.3f} "
            f"alpha_phase_norm={norm_span(phase_span(cell, 'alpha'), cell.alpha_min, cell.alpha_max):.3f} "
            f"beta_phase_norm={norm_span(phase_span(cell, 'beta'), cell.beta_min, cell.beta_max):.3f} "
            f"soft_alpha_edge={fmt(values(cell, 'alpha_edge_weight'))} "
            f"soft_beta_edge={fmt(values(cell, 'beta_edge_weight'))} "
            f"top1_alpha_edge={fmt(values(cell, 'top1_alpha_edge'))} "
            f"top1_beta_edge={fmt(values(cell, 'top1_beta_edge'))} "
            f"target_alpha_edge={fmt(values(cell, 'target_alpha_edge'))} "
            f"target_beta_edge={fmt(values(cell, 'target_beta_edge'))} "
            f"entropy={fmt(values(cell, 'entropy'))}"
        )
    run_lines.append("")
    run_lines.append(f"DECISION {decision}")
    run_lines.append(f"compute_status={diag.status}")

    lines = [
        "# DFW-E084 Result - Outer-Control Trajectory Diagnostic",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{diag.status}` via PyTorch CUDA: `{diag.detail}`.",
        "",
        "## Cell Trajectories",
        "",
        "| select | N | K | label | selector | selector-shuffled | alpha | beta | early alpha | middle alpha | late alpha | early beta | middle beta | late beta | alpha phase span | beta phase span | soft alpha edge | soft beta edge | top1 alpha edge | top1 beta edge | entropy |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        alpha_phases = phase_means(cell, "alpha")
        beta_phases = phase_means(cell, "beta")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{cell.select_batch}",
                    f"{cell.n_workers}",
                    f"{cell.k_value}",
                    classify_cell(cell),
                    e71.fmt_ci(cell.selector).strip(),
                    e71.fmt_ci(e71.paired(cell.selector, cell.shuffled)).strip(),
                    fmt(values(cell, "alpha")),
                    fmt(values(cell, "beta")),
                    f"{alpha_phases['early']:.3f}",
                    f"{alpha_phases['middle']:.3f}",
                    f"{alpha_phases['late']:.3f}",
                    f"{beta_phases['early']:.3f}",
                    f"{beta_phases['middle']:.3f}",
                    f"{beta_phases['late']:.3f}",
                    f"{norm_span(phase_span(cell, 'alpha'), cell.alpha_min, cell.alpha_max):.3f}",
                    f"{norm_span(phase_span(cell, 'beta'), cell.beta_min, cell.beta_max):.3f}",
                    fmt(values(cell, "alpha_edge_weight")),
                    fmt(values(cell, "beta_edge_weight")),
                    fmt(values(cell, "top1_alpha_edge")),
                    fmt(values(cell, "top1_beta_edge")),
                    fmt(values(cell, "entropy")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `top1 * edge` is the rate at which the lowest probe-loss grid point is on a design boundary.",
            "- `soft * edge` is the probability mass assigned to boundary grid points before the soft weighted pair is collapsed.",
            "- Phase spans are normalized by the scaled-design alpha or beta range within each K.",
            "",
            "## Run Log",
            "",
            "```text",
            "\n".join(run_lines),
            "```",
            "",
        ]
    )
    legit = "\n".join(
        [
            "# DFW-E084 Legitimacy Check",
            "",
            f"- Compute: `{diag.status}` / `{diag.detail}`.",
            "- Uses the E080/E081 frozen controller family: scaled design and soft_std2 policy.",
            "- Restricts live probe selection to K<=8 cells.",
            "- Pilot seeds tune only the fixed and schedule controls used for context.",
            "- Held-out target batches score selector and oracle context; trajectory summaries use select-batch probe losses.",
            "- Shuffled probe labels are retained as a content-control score.",
            "- Reports effective soft pairs, top-1 probe grid points, target-design grid points, edge mass, entropy, and phase movement.",
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
