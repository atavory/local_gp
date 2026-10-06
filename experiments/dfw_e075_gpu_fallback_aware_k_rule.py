"""DFW-E075: fallback-aware K rule for real-path probe control.

E074 showed a small pooled K/N transfer win for the frozen K-aware selector, but
the gain came from K<=8.  At K>=16 the selector still beat shuffled probes yet
mostly tied or slightly lost to the best fixed/DiLoCo/schedule controls.  This
test keeps the candidate design frozen and asks whether making the large-K
fallback explicit removes that drag.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import sys

import dfw_e071_gpu_topk_probe_outer_control as e71


DATE = "20261006"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e075_gpu_fallback_aware_k_rule_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e075_gpu_fallback_aware_k_rule_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e075_gpu_fallback_aware_k_rule_LEGITIMACY_CHECK_{DATE}.md"


@dataclass(frozen=True)
class RuleScore:
    name: str
    points: int
    values: list[float]
    shuffled: list[float]


def nonnegative_ci(vals: list[float]) -> bool:
    mean, hw = e71.mean_ci(vals)
    return mean + hw >= 0.0


def policy_score(cell: e71.Cell, policy: str) -> e71.PolicyScore:
    return next(score for score in cell.policy_scores if score.design == "scaled" and score.policy == policy)


def rule_scores(cell: e71.Cell) -> list[RuleScore]:
    soft = policy_score(cell, "soft_std2")
    top3 = policy_score(cell, "top3_mean")
    guard = policy_score(cell, "top3_guard_fixed")

    if cell.k_value <= 8:
        return [
            RuleScore("e073_k_aware", soft.points, soft.values, soft.shuffled),
            RuleScore("hard_large_k_fallback", soft.points, soft.values, soft.shuffled),
            RuleScore("guard_large_k_fallback", soft.points, soft.values, soft.shuffled),
        ]

    return [
        RuleScore("e073_k_aware", top3.points, top3.values, top3.shuffled),
        RuleScore("hard_large_k_fallback", 0, cell.fixed, cell.fixed),
        RuleScore("guard_large_k_fallback", guard.points, guard.values, guard.shuffled),
    ]


def decide(
    minus_control: dict[str, list[float]],
    minus_shuffled: dict[str, list[float]],
    large_k_minus_control: dict[str, list[float]],
    compute_status: str,
) -> str:
    if compute_status != "REAL_GPU":
        return "NO_GPU_HARNESS"
    for name in ("hard_large_k_fallback", "guard_large_k_fallback"):
        if (
            e71.positive_ci(minus_control[name])
            and e71.positive_ci(minus_shuffled[name])
            and nonnegative_ci(large_k_minus_control[name])
        ):
            return "FALLBACK_AWARE_K_RULE_LIVE"
    if any(e71.positive_ci(vals) for vals in minus_control.values()):
        return "FALLBACK_AWARE_K_RULE_PARTIAL"
    return "FALLBACK_AWARE_K_RULE_FAILS"


def run_experiment() -> tuple[str, str, str]:
    e71.DESIGN_NAMES = ("scaled",)
    e71.POLICIES = ("top3_mean", "soft_std2", "top3_guard_fixed")
    e71.K_VALUES = (4, 8, 16, 32)
    e71.N_WORKERS = (8, 16)
    e71.SELECT_BATCHES = (384,)
    e71.PILOT_SEEDS = tuple(range(962_000, 962_002))
    e71.HELDOUT_SEEDS = tuple(range(963_000, 963_012))

    diag = e71.e30.compute_diagnostic()
    if diag.status != "REAL_GPU":
        run_log = "\n".join(
            (
                "DFW-E075 fallback-aware K rule",
                f"host={platform.node()} python={sys.executable}",
                f"DECISION NO_GPU_HARNESS compute_status={diag.status} detail={diag.detail}",
            )
        )
        result = "# DFW-E075 Result - Fallback-Aware K Rule\n\nDecision label: `NO_GPU_HARNESS`\n"
        legit = "# DFW-E075 Legitimacy Check\n\n- GPU workload: not run.\n"
        return run_log + "\n", result + "\n", legit + "\n"

    e71.e30.torch.set_default_dtype(e71.e30.torch.float32)
    try:
        e71.e30.torch.set_num_threads(1)
    except Exception:
        pass

    rows: list[tuple[e71.Cell, RuleScore]] = []
    minus_control: dict[str, list[float]] = {}
    minus_shuffled: dict[str, list[float]] = {}
    capture: dict[str, list[float]] = {}
    large_k_minus_control: dict[str, list[float]] = {}
    run_lines = [
        "DFW-E075 fallback-aware K rule",
        f"host={platform.node()} python={sys.executable} compute_status={diag.status} {diag.detail}",
        "design=scaled policies=(top3_mean,soft_std2,top3_guard_fixed)",
        "rules=e073_k_aware, hard_large_k_fallback, guard_large_k_fallback",
        f"K={e71.K_VALUES} N={e71.N_WORKERS} select={e71.SELECT_BATCHES}",
        f"pilot_seeds={len(e71.PILOT_SEEDS)} heldout_seeds={len(e71.HELDOUT_SEEDS)}",
        "",
    ]

    for select_batch in e71.SELECT_BATCHES:
        for n_workers in e71.N_WORKERS:
            for k_value in e71.K_VALUES:
                cell = e71.score_cell(k_value, n_workers, select_batch, diag.device)
                controls = e71.best_control(cell)
                for score in rule_scores(cell):
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
                        f"points={score.points} control={e71.fmt_ci(controls)} selector={e71.fmt_ci(score.values)} "
                        f"minus_control={e71.fmt_ci(mc)} minus_shuffled={e71.fmt_ci(ms)} "
                        f"capture={e71.fmt_ci(cap)}"
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
    run_lines.append(f"DECISION {decision}")
    run_lines.append(f"compute_status={diag.status}")

    lines = [
        "# DFW-E075 Result - Fallback-Aware K Rule",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{diag.status}` via PyTorch CUDA: `{diag.detail}`.",
        "",
        "## Metrics",
        "",
        "| select | N | K | rule | points | overhead | fixed | DiLoCo | schedule | best control | oracle | selector | selector minus best control | selector minus shuffled | capture |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
    lines.extend(
        [
            "",
            "## Pooled",
            "",
        ]
    )
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
    lines.extend(["## Run Log", "", "```text", "\n".join(run_lines), "```"])
    legit = "\n".join(
        [
            "# DFW-E075 Legitimacy Check",
            "",
            f"- Compute: `{diag.status}` / `{diag.detail}`.",
            "- Selector design frozen from E073/E074: scaled design only.",
            "- Fallback rules fixed before this run from the E074 readout.",
            "- Pilot and held-out seeds are fresh relative to E071-E074.",
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
