"""DFW-E076: pilot-best control fallback for real-path probe control.

E075 showed that hard large-K fallback improved the pooled result, but fallback
to plain fixed still lost to the best fixed/DiLoCo/schedule comparator at large
K.  This run keeps the rule implementable: choose the large-K fallback from
pilot data among fixed, DiLoCo, and schedule, then use probes only for K<=8.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import sys

import dfw_e071_gpu_topk_probe_outer_control as e71


DATE = "20261006"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e076_gpu_pilot_best_control_fallback_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e076_gpu_pilot_best_control_fallback_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e076_gpu_pilot_best_control_fallback_LEGITIMACY_CHECK_{DATE}.md"


@dataclass(frozen=True)
class RuleScore:
    name: str
    fallback: str
    points: int
    values: list[float]
    shuffled: list[float]


def nonnegative_ci(vals: list[float]) -> bool:
    mean, hw = e71.mean_ci(vals)
    return mean + hw >= 0.0


def policy_score(cell: e71.Cell, policy: str) -> e71.PolicyScore:
    return next(score for score in cell.policy_scores if score.design == "scaled" and score.policy == policy)


def score_control_on_groups(groups, fixed_pair, diloco_alpha, schedule) -> dict[str, float]:
    def schedule_pair(_row, _velocity, idx):
        return schedule[min(idx, len(schedule) - 1)]

    scores = {"fixed": [], "diloco": [], "schedule": []}
    for rows in groups:
        scores["fixed"].append(e71.e33.score_policy(rows, lambda _row, _vel, _idx, p=fixed_pair: p))
        scores["diloco"].append(
            e71.e33.score_policy(rows, lambda _row, _vel, _idx, a=diloco_alpha: (a, 0.90))
        )
        scores["schedule"].append(e71.e33.score_policy(rows, schedule_pair))
    return {name: e71.mean(vals) for name, vals in scores.items()}


def pilot_best_control(k_value: int, n_workers: int, select_batch: int, device: str) -> str:
    pilot = e71.e33.make_groups(k_value, n_workers, select_batch, e71.PILOT_SEEDS, device)
    fixed_pair = e71.e33.tune_fixed_alpha_beta(pilot)
    diloco_alpha = e71.e33.tune_diloco_alpha(pilot)
    schedule = e71.e33.tune_schedule(pilot)
    scores = score_control_on_groups(pilot, fixed_pair, diloco_alpha, schedule)
    return max(scores, key=scores.__getitem__)


def control_values(cell: e71.Cell, name: str) -> list[float]:
    if name == "fixed":
        return cell.fixed
    if name == "diloco":
        return cell.diloco
    if name == "schedule":
        return cell.schedule
    raise ValueError(f"unknown control: {name}")


def rule_scores(cell: e71.Cell, fallback_name: str) -> list[RuleScore]:
    soft = policy_score(cell, "soft_std2")
    top3 = policy_score(cell, "top3_mean")

    if cell.k_value <= 8:
        return [
            RuleScore("e073_k_aware", "probe", soft.points, soft.values, soft.shuffled),
            RuleScore("pilot_best_large_k_fallback", "probe", soft.points, soft.values, soft.shuffled),
        ]

    fallback_values = control_values(cell, fallback_name)
    return [
        RuleScore("e073_k_aware", "probe", top3.points, top3.values, top3.shuffled),
        RuleScore("pilot_best_large_k_fallback", fallback_name, 0, fallback_values, fallback_values),
    ]


def decide(
    minus_control: dict[str, list[float]],
    minus_shuffled: dict[str, list[float]],
    large_k_minus_control: dict[str, list[float]],
    compute_status: str,
) -> str:
    if compute_status != "REAL_GPU":
        return "NO_GPU_HARNESS"
    name = "pilot_best_large_k_fallback"
    if (
        e71.positive_ci(minus_control[name])
        and nonnegative_ci(minus_shuffled[name])
        and nonnegative_ci(large_k_minus_control[name])
    ):
        return "PILOT_BEST_FALLBACK_LIVE"
    if e71.positive_ci(minus_control[name]):
        return "PILOT_BEST_FALLBACK_PARTIAL"
    return "PILOT_BEST_FALLBACK_FAILS"


def run_experiment() -> tuple[str, str, str]:
    e71.DESIGN_NAMES = ("scaled",)
    e71.POLICIES = ("top3_mean", "soft_std2")
    e71.K_VALUES = (4, 8, 16, 32)
    e71.N_WORKERS = (8, 16)
    e71.SELECT_BATCHES = (384,)
    e71.PILOT_SEEDS = tuple(range(964_000, 964_002))
    e71.HELDOUT_SEEDS = tuple(range(965_000, 965_012))

    diag = e71.e30.compute_diagnostic()
    if diag.status != "REAL_GPU":
        run_log = "\n".join(
            (
                "DFW-E076 pilot-best control fallback",
                f"host={platform.node()} python={sys.executable}",
                f"DECISION NO_GPU_HARNESS compute_status={diag.status} detail={diag.detail}",
            )
        )
        result = "# DFW-E076 Result - Pilot-Best Control Fallback\n\nDecision label: `NO_GPU_HARNESS`\n"
        legit = "# DFW-E076 Legitimacy Check\n\n- GPU workload: not run.\n"
        return run_log + "\n", result + "\n", legit + "\n"

    e71.e30.torch.set_default_dtype(e71.e30.torch.float32)
    try:
        e71.e30.torch.set_num_threads(1)
    except Exception:
        pass

    rows: list[tuple[e71.Cell, RuleScore]] = []
    fallback_by_cell: dict[tuple[int, int, int], str] = {}
    minus_control: dict[str, list[float]] = {}
    minus_shuffled: dict[str, list[float]] = {}
    capture: dict[str, list[float]] = {}
    large_k_minus_control: dict[str, list[float]] = {}
    run_lines = [
        "DFW-E076 pilot-best control fallback",
        f"host={platform.node()} python={sys.executable} compute_status={diag.status} {diag.detail}",
        "design=scaled policies=(top3_mean,soft_std2)",
        "rule=K<=8 probe soft_std2; K>8 pilot-best control fallback",
        f"K={e71.K_VALUES} N={e71.N_WORKERS} select={e71.SELECT_BATCHES}",
        f"pilot_seeds={len(e71.PILOT_SEEDS)} heldout_seeds={len(e71.HELDOUT_SEEDS)}",
        "",
    ]

    for select_batch in e71.SELECT_BATCHES:
        for n_workers in e71.N_WORKERS:
            for k_value in e71.K_VALUES:
                fallback = pilot_best_control(k_value, n_workers, select_batch, diag.device)
                cell = e71.score_cell(k_value, n_workers, select_batch, diag.device)
                fallback_by_cell[(select_batch, n_workers, k_value)] = fallback
                controls = e71.best_control(cell)
                for score in rule_scores(cell, fallback):
                    mc = e71.paired(score.values, controls)
                    ms = e71.paired(score.values, score.shuffled)
                    cap = e71.capture(score.values, cell.fixed, cell.oracle)
                    minus_control.setdefault(score.name, []).extend(mc)
                    minus_shuffled.setdefault(score.name, []).extend(ms)
                    capture.setdefault(score.name, []).extend(cap)
                    if k_value >= 16:
                        large_k_minus_control.setdefault(score.name, []).extend(mc)
                    rows.append((cell, score))
                    run_lines.append(
                        f"CELL select={select_batch} N={n_workers} K={k_value} rule={score.name} "
                        f"fallback={score.fallback} control={e71.fmt_ci(controls)} "
                        f"selector={e71.fmt_ci(score.values)} minus_control={e71.fmt_ci(mc)} "
                        f"minus_shuffled={e71.fmt_ci(ms)} capture={e71.fmt_ci(cap)}"
                    )

    decision = decide(minus_control, minus_shuffled, large_k_minus_control, diag.status)
    run_lines.append("")
    for name in sorted(minus_control):
        run_lines.append(
            f"POOLED rule={name} minus_control={e71.fmt_ci(minus_control[name])} "
            f"minus_shuffled={e71.fmt_ci(minus_shuffled[name])} "
            f"large_k_minus_control={e71.fmt_ci(large_k_minus_control.get(name, []))} "
            f"capture={e71.fmt_ci(capture[name])}"
        )
    run_lines.append(f"fallback_by_cell={fallback_by_cell}")
    run_lines.append(f"DECISION {decision}")
    run_lines.append(f"compute_status={diag.status}")

    lines = [
        "# DFW-E076 Result - Pilot-Best Control Fallback",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{diag.status}` via PyTorch CUDA: `{diag.detail}`.",
        "",
        "## Metrics",
        "",
        "| select | N | K | rule | fallback | points | overhead | fixed | DiLoCo | schedule | best control | oracle | selector | selector minus best control | selector minus shuffled | capture |",
        "|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell, score in rows:
        controls = e71.best_control(cell)
        overhead = score.points * cell.select_batch / e71.e30.TRAIN_SIZE / (
            3.0 * cell.n_workers * cell.k_value
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{cell.select_batch}",
                    f"{cell.n_workers}",
                    f"{cell.k_value}",
                    score.name,
                    score.fallback,
                    f"{score.points}",
                    f"{100.0 * overhead:.2f}%",
                    e71.fmt_ci(cell.fixed).strip(),
                    e71.fmt_ci(cell.diloco).strip(),
                    e71.fmt_ci(cell.schedule).strip(),
                    e71.fmt_ci(controls).strip(),
                    e71.fmt_ci(cell.oracle).strip(),
                    e71.fmt_ci(score.values).strip(),
                    e71.fmt_ci(e71.paired(score.values, controls)).strip(),
                    e71.fmt_ci(e71.paired(score.values, score.shuffled)).strip(),
                    e71.fmt_ci(e71.capture(score.values, cell.fixed, cell.oracle)).strip(),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Pooled", ""])
    for name in sorted(minus_control):
        lines.extend(
            [
                f"### {name}",
                "",
                f"- selector minus best control: `{e71.fmt_ci(minus_control[name]).strip()}`",
                f"- selector minus shuffled: `{e71.fmt_ci(minus_shuffled[name]).strip()}`",
                f"- large-K selector minus best control: `{e71.fmt_ci(large_k_minus_control.get(name, [])).strip()}`",
                f"- capture: `{e71.fmt_ci(capture[name]).strip()}`",
                "",
            ]
        )
    lines.extend(["## Pilot Fallbacks", ""])
    for key, fallback in sorted(fallback_by_cell.items()):
        lines.append(f"- select={key[0]} N={key[1]} K={key[2]}: `{fallback}`")
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])
    legit = "\n".join(
        [
            "# DFW-E076 Legitimacy Check",
            "",
            f"- Compute: `{diag.status}` / `{diag.detail}`.",
            "- Selector design frozen from E073/E074: scaled design only.",
            "- Large-K fallback selected from pilot data among fixed, DiLoCo, and schedule.",
            "- Pilot and held-out seeds are fresh relative to E071-E075.",
            "- Selector uses select batches only; scoring uses target batches.",
            "- Controls include fixed, DiLoCo beta=.9, and target-retuned schedule.",
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
