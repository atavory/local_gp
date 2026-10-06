"""DFW-E081: deployment-rule audit for the replicated K<=8 probe controller.

E080 established that the frozen K<=8 probe controller beats shuffled labels
and fixed controls at high power.  This audit freezes the probe component and
tests deployable fallbacks for K>8 without importing per-K tuned fixed pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import sys

import dfw_e071_gpu_topk_probe_outer_control as e71


DATE = "20261006"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e081_gpu_deployment_rule_audit_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e081_gpu_deployment_rule_audit_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e081_gpu_deployment_rule_audit_LEGITIMACY_CHECK_{DATE}.md"


K_VALUES = (4, 8, 16, 32)
N_WORKERS = (8, 16)
SELECT_BATCHES = (384,)
PILOT_SEEDS = tuple(range(974_000, 974_002))
HELDOUT_SEEDS = tuple(range(975_000, 975_032))


@dataclass(frozen=True)
class CellEval:
    select_batch: int
    n_workers: int
    k_value: int
    fixed_pair: tuple[float, float]
    fixed: list[float]
    diloco: list[float]
    schedule: list[float]
    oracle: list[float]
    probe: list[float]
    shuffled_probe: list[float]
    global_fixed: list[float]
    k8_transfer: list[float]
    theory: list[float]


def nonnegative_ci(vals: list[float]) -> bool:
    mean, half_width = e71.mean_ci(vals)
    return mean + half_width >= 0.0


def theory_pair(k_value: int) -> tuple[float, float]:
    # Matches the center of the frozen scaled design family.  It uses only K,
    # which is a known optimizer control, and no pilot or held-out losses.
    alpha = min(20.0, max(4.0, 72.0 / float(k_value)))
    return (alpha, 0.15)


def tune_fixed_pair_for_specs(
    specs: list[tuple[int, int, int]],
    seeds: tuple[int, ...],
    device: str,
) -> tuple[float, float]:
    groups = []
    for k_value, n_workers, select_batch in specs:
        groups.extend(e71.e33.make_groups(k_value, n_workers, select_batch, seeds, device))
    return e71.e33.tune_fixed_alpha_beta(groups)


def score_constant_pair(groups, pair: tuple[float, float]) -> list[float]:
    return [
        e71.e33.score_policy(rows, lambda _row, _vel, _idx, p=pair: p)
        for rows in groups
    ]


def score_theory(groups, k_value: int) -> list[float]:
    pair = theory_pair(k_value)
    return [
        e71.e33.score_policy(rows, lambda _row, _vel, _idx, p=pair: p)
        for rows in groups
    ]


def score_schedule(groups, schedule: list[tuple[float, float]]) -> list[float]:
    def choose(_row, _velocity, idx):
        return schedule[min(idx, len(schedule) - 1)]

    return [e71.e33.score_policy(rows, choose) for rows in groups]


def score_probe(groups, design, fixed_pair: tuple[float, float]) -> tuple[list[float], list[float]]:
    values: list[float] = []
    shuffled: list[float] = []
    for rep_idx, rows in enumerate(groups):
        values.append(e71.score_selector(rows, design, "soft_std2", fixed_pair, None))
        shuffled.append(e71.score_selector(rows, design, "soft_std2", fixed_pair, rep_idx))
    return values, shuffled


def rule_values(
    cell: CellEval,
    fallback: str,
    *,
    shuffled: bool,
) -> list[float]:
    if cell.k_value <= 8:
        return cell.shuffled_probe if shuffled else cell.probe
    if fallback == "global_fixed":
        return cell.global_fixed
    if fallback == "k8_transfer":
        return cell.k8_transfer
    if fallback == "theory":
        return cell.theory
    raise ValueError(fallback)


def baseline_values(cell: CellEval, baseline: str) -> list[float]:
    if baseline == "global_fixed":
        return cell.global_fixed
    if baseline == "k8_transfer":
        return cell.k8_transfer
    if baseline == "theory":
        return cell.theory
    if baseline == "per_k_fixed":
        return cell.fixed
    if baseline == "schedule":
        return cell.schedule
    if baseline == "best_control_ceiling":
        return e71.best_control(
            e71.Cell(
                n_workers=cell.n_workers,
                k_value=cell.k_value,
                select_batch=cell.select_batch,
                fixed_pair=cell.fixed_pair,
                diloco_alpha=0.0,
                fixed=cell.fixed,
                diloco=cell.diloco,
                schedule=cell.schedule,
                oracle=cell.oracle,
                design_oracle={},
                policy_scores=(),
            )
        )
    raise ValueError(baseline)


def run_experiment() -> tuple[str, str, str]:
    e71.DESIGN_NAMES = ("scaled",)
    e71.POLICIES = ("soft_std2",)

    diag = e71.e30.compute_diagnostic()
    if diag.status != "REAL_GPU":
        run_log = "\n".join(
            (
                "DFW-E081 deployment-rule audit",
                f"host={platform.node()} python={sys.executable}",
                f"DECISION NO_GPU_HARNESS compute_status={diag.status} detail={diag.detail}",
            )
        )
        result = "# DFW-E081 Result - Deployment Rule Audit\n\nDecision label: `NO_GPU_HARNESS`\n"
        legit = "# DFW-E081 Legitimacy Check\n\n- GPU workload: not run.\n"
        return run_log + "\n", result + "\n", legit + "\n"

    e71.e30.torch.set_default_dtype(e71.e30.torch.float32)
    try:
        e71.e30.torch.set_num_threads(1)
    except Exception:
        pass

    specs = [
        (k_value, n_workers, select_batch)
        for select_batch in SELECT_BATCHES
        for n_workers in N_WORKERS
        for k_value in K_VALUES
    ]
    k8_specs = [spec for spec in specs if spec[0] == 8]
    global_pair = tune_fixed_pair_for_specs(specs, PILOT_SEEDS, diag.device)
    k8_pair = tune_fixed_pair_for_specs(k8_specs, PILOT_SEEDS, diag.device)

    run_lines = [
        "DFW-E081 deployment-rule audit",
        f"host={platform.node()} python={sys.executable} compute_status={diag.status} {diag.detail}",
        "rule=K<=8 scaled/soft_std2 probe; K>8 frozen no-probe fallback",
        f"K={K_VALUES} N={N_WORKERS} select={SELECT_BATCHES}",
        f"pilot_seeds={len(PILOT_SEEDS)} heldout_seeds={len(HELDOUT_SEEDS)}",
        f"global_pair={global_pair}",
        f"k8_transfer_pair={k8_pair}",
        "",
    ]

    cells: list[CellEval] = []
    for select_batch in SELECT_BATCHES:
        for n_workers in N_WORKERS:
            for k_value in K_VALUES:
                pilot = e71.e33.make_groups(k_value, n_workers, select_batch, PILOT_SEEDS, diag.device)
                fixed_pair = e71.e33.tune_fixed_alpha_beta(pilot)
                diloco_alpha = e71.e33.tune_diloco_alpha(pilot)
                schedule = e71.e33.tune_schedule(pilot)
                groups = e71.e33.make_groups(k_value, n_workers, select_batch, HELDOUT_SEEDS, diag.device)
                design = e71.e33.design_map(k_value)["scaled"]
                probe, shuffled_probe = score_probe(groups, design, fixed_pair)
                cell = CellEval(
                    select_batch=select_batch,
                    n_workers=n_workers,
                    k_value=k_value,
                    fixed_pair=fixed_pair,
                    fixed=score_constant_pair(groups, fixed_pair),
                    diloco=score_constant_pair(groups, (diloco_alpha, 0.90)),
                    schedule=score_schedule(groups, schedule),
                    oracle=[
                        e71.e33.score_policy(rows, lambda row, vel, _idx: e71.e33.oracle_pair(row, vel))
                        for rows in groups
                    ],
                    probe=probe,
                    shuffled_probe=shuffled_probe,
                    global_fixed=score_constant_pair(groups, global_pair),
                    k8_transfer=score_constant_pair(groups, k8_pair),
                    theory=score_theory(groups, k_value),
                )
                cells.append(cell)
                run_lines.append(
                    f"CELL select={select_batch} N={n_workers} K={k_value} "
                    f"fixed_pair={fixed_pair} theory_pair={theory_pair(k_value)} "
                    f"probe={e71.fmt_ci(cell.probe)} shuffled={e71.fmt_ci(cell.shuffled_probe)} "
                    f"global_fixed={e71.fmt_ci(cell.global_fixed)} "
                    f"k8_transfer={e71.fmt_ci(cell.k8_transfer)} theory={e71.fmt_ci(cell.theory)} "
                    f"per_k_fixed={e71.fmt_ci(cell.fixed)} schedule={e71.fmt_ci(cell.schedule)} "
                    f"best_control={e71.fmt_ci(baseline_values(cell, 'best_control_ceiling'))} "
                    f"oracle={e71.fmt_ci(cell.oracle)}"
                )

    fallbacks = ("global_fixed", "k8_transfer", "theory")
    baselines = (
        "global_fixed",
        "k8_transfer",
        "theory",
        "per_k_fixed",
        "schedule",
        "best_control_ceiling",
    )
    pooled: dict[str, dict[str, list[float]]] = {}
    pooled_low_k: dict[str, dict[str, list[float]]] = {}
    pooled_high_k: dict[str, dict[str, list[float]]] = {}
    for fallback in fallbacks:
        pooled[fallback] = {baseline: [] for baseline in baselines}
        pooled[fallback]["shuffled"] = []
        pooled_low_k[fallback] = {baseline: [] for baseline in baselines}
        pooled_low_k[fallback]["shuffled"] = []
        pooled_high_k[fallback] = {baseline: [] for baseline in baselines}
        pooled_high_k[fallback]["shuffled"] = []
        for cell in cells:
            vals = rule_values(cell, fallback, shuffled=False)
            shuffled_vals = rule_values(cell, fallback, shuffled=True)
            target = pooled_low_k if cell.k_value <= 8 else pooled_high_k
            for baseline in baselines:
                diff = e71.paired(vals, baseline_values(cell, baseline))
                pooled[fallback][baseline].extend(diff)
                target[fallback][baseline].extend(diff)
            shuffled_diff = e71.paired(vals, shuffled_vals)
            pooled[fallback]["shuffled"].extend(shuffled_diff)
            target[fallback]["shuffled"].extend(shuffled_diff)

    decision = "DEPLOYMENT_RULE_AUDIT_FAILS"
    for fallback in fallbacks:
        if (
            e71.positive_ci(pooled_low_k[fallback]["shuffled"])
            and e71.positive_ci(pooled[fallback][fallback])
            and nonnegative_ci(pooled[fallback]["per_k_fixed"])
        ):
            decision = f"DEPLOYMENT_RULE_LIVE_{fallback.upper()}"
            break
        if e71.positive_ci(pooled_low_k[fallback]["shuffled"]) and e71.positive_ci(
            pooled[fallback][fallback]
        ):
            decision = f"DEPLOYMENT_RULE_BASELINE_ONLY_{fallback.upper()}"
            break

    run_lines.append("")
    for fallback in fallbacks:
        run_lines.append(
            f"POOLED rule=probe_small_{fallback}_large "
            f"minus_own_baseline={e71.fmt_ci(pooled[fallback][fallback])} "
            f"minus_per_k_fixed={e71.fmt_ci(pooled[fallback]['per_k_fixed'])} "
            f"minus_schedule={e71.fmt_ci(pooled[fallback]['schedule'])} "
            f"minus_best_control_ceiling={e71.fmt_ci(pooled[fallback]['best_control_ceiling'])} "
            f"minus_shuffled_low_k={e71.fmt_ci(pooled_low_k[fallback]['shuffled'])} "
            f"minus_own_baseline_high_k={e71.fmt_ci(pooled_high_k[fallback][fallback])}"
        )
    run_lines.append(f"DECISION {decision}")
    run_lines.append(f"compute_status={diag.status}")

    lines = [
        "# DFW-E081 Result - Deployment Rule Audit",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{diag.status}` via PyTorch CUDA: `{diag.detail}`.",
        "",
        "## Frozen Pairs",
        "",
        f"- global fixed pair from all pilot cells: `{global_pair}`",
        f"- K=8 transferred pair from K=8 pilot cells: `{k8_pair}`",
        "- theory fallback: `(min(20, max(4, 72/K)), 0.15)`",
        "",
        "## Pooled Deployment Rules",
        "",
        "| rule | minus own no-probe baseline | minus per-K fixed | minus schedule | minus best-control ceiling | low-K minus shuffled | high-K minus own baseline |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fallback in fallbacks:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"probe_small_{fallback}_large",
                    e71.fmt_ci(pooled[fallback][fallback]).strip(),
                    e71.fmt_ci(pooled[fallback]["per_k_fixed"]).strip(),
                    e71.fmt_ci(pooled[fallback]["schedule"]).strip(),
                    e71.fmt_ci(pooled[fallback]["best_control_ceiling"]).strip(),
                    e71.fmt_ci(pooled_low_k[fallback]["shuffled"]).strip(),
                    e71.fmt_ci(pooled_high_k[fallback][fallback]).strip(),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Cell Metrics", ""])
    lines.append(
        "| select | N | K | probe | shuffled | global fixed | K8 transfer | theory | per-K fixed | schedule | best-control ceiling | oracle |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cell in cells:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{cell.select_batch}",
                    f"{cell.n_workers}",
                    f"{cell.k_value}",
                    e71.fmt_ci(cell.probe).strip(),
                    e71.fmt_ci(cell.shuffled_probe).strip(),
                    e71.fmt_ci(cell.global_fixed).strip(),
                    e71.fmt_ci(cell.k8_transfer).strip(),
                    e71.fmt_ci(cell.theory).strip(),
                    e71.fmt_ci(cell.fixed).strip(),
                    e71.fmt_ci(cell.schedule).strip(),
                    e71.fmt_ci(baseline_values(cell, "best_control_ceiling")).strip(),
                    e71.fmt_ci(cell.oracle).strip(),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])

    legit = "\n".join(
        [
            "# DFW-E081 Legitimacy Check",
            "",
            f"- Compute: `{diag.status}` / `{diag.detail}`.",
            "- K<=8 probe controller frozen from E080: scaled/soft_std2, select=384.",
            "- K>8 fallbacks do not use per-K held-out tuning.",
            "- Global and K=8 transfer fallbacks use pilot seeds only.",
            "- Theory fallback uses only known K.",
            "- Shuffled probe labels remain the low-K content control.",
            "- Held-out scoring uses fresh seeds relative to E078-E080.",
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
