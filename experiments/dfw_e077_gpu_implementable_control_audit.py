"""DFW-E077: implementable-control audit for probe outer control.

E076 exposed a baseline problem: the "best control" comparator is the per-seed
max over fixed, DiLoCo, and schedule, so it is an oracle ceiling rather than a
deployable baseline.  This run keeps the same real-GPU path but reports probe
rules against implementable controls separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import sys

import dfw_e071_gpu_topk_probe_outer_control as e71


DATE = "20261006"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e077_gpu_implementable_control_audit_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e077_gpu_implementable_control_audit_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e077_gpu_implementable_control_audit_LEGITIMACY_CHECK_{DATE}.md"


@dataclass(frozen=True)
class RuleScore:
    name: str
    values: list[float]
    shuffled: list[float]
    points: int


def policy_score(cell: e71.Cell, policy: str) -> e71.PolicyScore:
    return next(score for score in cell.policy_scores if score.design == "scaled" and score.policy == policy)


def score_control_on_groups(groups, fixed_pair, diloco_alpha, schedule) -> dict[str, float]:
    def schedule_pair(_row, _velocity, idx):
        return schedule[min(idx, len(schedule) - 1)]

    out = {"fixed": [], "diloco": [], "schedule": []}
    for rows in groups:
        out["fixed"].append(e71.e33.score_policy(rows, lambda _row, _vel, _idx, p=fixed_pair: p))
        out["diloco"].append(
            e71.e33.score_policy(rows, lambda _row, _vel, _idx, a=diloco_alpha: (a, 0.90))
        )
        out["schedule"].append(e71.e33.score_policy(rows, schedule_pair))
    return {name: e71.mean(vals) for name, vals in out.items()}


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
    if name == "heldout_best":
        return e71.best_control(cell)
    raise ValueError(name)


def rule_scores(cell: e71.Cell, pilot_best: str) -> list[RuleScore]:
    soft = policy_score(cell, "soft_std2")
    top3 = policy_score(cell, "top3_mean")
    e073 = soft if cell.k_value <= 8 else top3
    hard_fixed = soft if cell.k_value <= 8 else None
    pilot_fallback = soft if cell.k_value <= 8 else None
    scores = [
        RuleScore("e073_k_aware", e073.values, e073.shuffled, e073.points),
        RuleScore(
            "small_k_probe_fixed_large_k",
            hard_fixed.values if hard_fixed is not None else cell.fixed,
            hard_fixed.shuffled if hard_fixed is not None else cell.fixed,
            hard_fixed.points if hard_fixed is not None else 0,
        ),
        RuleScore(
            "small_k_probe_pilot_large_k",
            pilot_fallback.values if pilot_fallback is not None else control_values(cell, pilot_best),
            pilot_fallback.shuffled if pilot_fallback is not None else control_values(cell, pilot_best),
            pilot_fallback.points if pilot_fallback is not None else 0,
        ),
    ]
    return scores


def run_experiment() -> tuple[str, str, str]:
    e71.DESIGN_NAMES = ("scaled",)
    e71.POLICIES = ("top3_mean", "soft_std2")
    e71.K_VALUES = (4, 8, 16, 32)
    e71.N_WORKERS = (8, 16)
    e71.SELECT_BATCHES = (384,)
    e71.PILOT_SEEDS = tuple(range(966_000, 966_002))
    e71.HELDOUT_SEEDS = tuple(range(967_000, 967_012))

    diag = e71.e30.compute_diagnostic()
    if diag.status != "REAL_GPU":
        run_log = "\n".join(
            (
                "DFW-E077 implementable-control audit",
                f"host={platform.node()} python={sys.executable}",
                f"DECISION NO_GPU_HARNESS compute_status={diag.status} detail={diag.detail}",
            )
        )
        result = "# DFW-E077 Result - Implementable-Control Audit\n\nDecision label: `NO_GPU_HARNESS`\n"
        legit = "# DFW-E077 Legitimacy Check\n\n- GPU workload: not run.\n"
        return run_log + "\n", result + "\n", legit + "\n"

    e71.e30.torch.set_default_dtype(e71.e30.torch.float32)
    try:
        e71.e30.torch.set_num_threads(1)
    except Exception:
        pass

    rows: list[tuple[e71.Cell, RuleScore, str]] = []
    pooled: dict[str, dict[str, list[float]]] = {}
    pilot_choices: dict[tuple[int, int, int], str] = {}
    run_lines = [
        "DFW-E077 implementable-control audit",
        f"host={platform.node()} python={sys.executable} compute_status={diag.status} {diag.detail}",
        "comparators=fixed,diloco,schedule,pilot_best,heldout_best_ceiling",
        f"K={e71.K_VALUES} N={e71.N_WORKERS} select={e71.SELECT_BATCHES}",
        f"pilot_seeds={len(e71.PILOT_SEEDS)} heldout_seeds={len(e71.HELDOUT_SEEDS)}",
        "",
    ]

    for select_batch in e71.SELECT_BATCHES:
        for n_workers in e71.N_WORKERS:
            for k_value in e71.K_VALUES:
                pilot_best = pilot_best_control(k_value, n_workers, select_batch, diag.device)
                cell = e71.score_cell(k_value, n_workers, select_batch, diag.device)
                pilot_choices[(select_batch, n_workers, k_value)] = pilot_best
                comps = {
                    "fixed": cell.fixed,
                    "diloco": cell.diloco,
                    "schedule": cell.schedule,
                    "pilot_best": control_values(cell, pilot_best),
                    "heldout_best_ceiling": e71.best_control(cell),
                }
                for score in rule_scores(cell, pilot_best):
                    rows.append((cell, score, pilot_best))
                    pooled.setdefault(score.name, {name: [] for name in comps})
                    pooled[score.name].setdefault("shuffled", [])
                    for comp_name, comp_vals in comps.items():
                        pooled[score.name][comp_name].extend(e71.paired(score.values, comp_vals))
                    pooled[score.name]["shuffled"].extend(e71.paired(score.values, score.shuffled))
                    run_lines.append(
                        f"CELL select={select_batch} N={n_workers} K={k_value} rule={score.name} "
                        f"pilot_best={pilot_best} selector={e71.fmt_ci(score.values)} "
                        f"minus_fixed={e71.fmt_ci(e71.paired(score.values, cell.fixed))} "
                        f"minus_pilot={e71.fmt_ci(e71.paired(score.values, comps['pilot_best']))} "
                        f"minus_ceiling={e71.fmt_ci(e71.paired(score.values, comps['heldout_best_ceiling']))} "
                        f"minus_shuffled={e71.fmt_ci(e71.paired(score.values, score.shuffled))}"
                    )

    # Decision: "live" means a rule beats fixed and pilot-best implementable
    # controls with positive CIs.  Heldout-best is reported as a ceiling only.
    decision = "IMPLEMENTABLE_CONTROL_AUDIT_FAILS"
    for rule_name, metrics in pooled.items():
        if e71.positive_ci(metrics["fixed"]) and e71.positive_ci(metrics["pilot_best"]):
            decision = "IMPLEMENTABLE_CONTROL_PROBE_LIVE"
            break
        if e71.positive_ci(metrics["fixed"]):
            decision = "IMPLEMENTABLE_CONTROL_FIXED_ONLY"

    run_lines.append("")
    for rule_name, metrics in sorted(pooled.items()):
        run_lines.append(
            f"POOLED rule={rule_name} "
            f"minus_fixed={e71.fmt_ci(metrics['fixed'])} "
            f"minus_diloco={e71.fmt_ci(metrics['diloco'])} "
            f"minus_schedule={e71.fmt_ci(metrics['schedule'])} "
            f"minus_pilot_best={e71.fmt_ci(metrics['pilot_best'])} "
            f"minus_heldout_best_ceiling={e71.fmt_ci(metrics['heldout_best_ceiling'])} "
            f"minus_shuffled={e71.fmt_ci(metrics['shuffled'])}"
        )
    run_lines.append(f"pilot_choices={pilot_choices}")
    run_lines.append(f"DECISION {decision}")
    run_lines.append(f"compute_status={diag.status}")

    lines = [
        "# DFW-E077 Result - Implementable-Control Audit",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{diag.status}` via PyTorch CUDA: `{diag.detail}`.",
        "",
        "## Pooled Comparisons",
        "",
        "| rule | minus fixed | minus DiLoCo | minus schedule | minus pilot-best | minus heldout-best ceiling | minus shuffled |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rule_name, metrics in sorted(pooled.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    rule_name,
                    e71.fmt_ci(metrics["fixed"]).strip(),
                    e71.fmt_ci(metrics["diloco"]).strip(),
                    e71.fmt_ci(metrics["schedule"]).strip(),
                    e71.fmt_ci(metrics["pilot_best"]).strip(),
                    e71.fmt_ci(metrics["heldout_best_ceiling"]).strip(),
                    e71.fmt_ci(metrics["shuffled"]).strip(),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Cell Metrics", ""])
    lines.append(
        "| select | N | K | rule | pilot-best | points | overhead | fixed | schedule | heldout-best ceiling | selector | minus fixed | minus pilot-best | minus heldout-best ceiling | minus shuffled |"
    )
    lines.append("|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cell, score, pilot_best in rows:
        overhead = score.points * cell.select_batch / e71.e30.TRAIN_SIZE / (
            3.0 * cell.n_workers * cell.k_value
        )
        comps = {
            "pilot_best": control_values(cell, pilot_best),
            "heldout_best_ceiling": e71.best_control(cell),
        }
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{cell.select_batch}",
                    f"{cell.n_workers}",
                    f"{cell.k_value}",
                    score.name,
                    pilot_best,
                    f"{score.points}",
                    f"{100.0 * overhead:.2f}%",
                    e71.fmt_ci(cell.fixed).strip(),
                    e71.fmt_ci(cell.schedule).strip(),
                    e71.fmt_ci(comps["heldout_best_ceiling"]).strip(),
                    e71.fmt_ci(score.values).strip(),
                    e71.fmt_ci(e71.paired(score.values, cell.fixed)).strip(),
                    e71.fmt_ci(e71.paired(score.values, comps["pilot_best"])).strip(),
                    e71.fmt_ci(e71.paired(score.values, comps["heldout_best_ceiling"])).strip(),
                    e71.fmt_ci(e71.paired(score.values, score.shuffled)).strip(),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Pilot Choices", ""])
    for key, choice in sorted(pilot_choices.items()):
        lines.append(f"- select={key[0]} N={key[1]} K={key[2]}: `{choice}`")
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])

    legit = "\n".join(
        [
            "# DFW-E077 Legitimacy Check",
            "",
            f"- Compute: `{diag.status}` / `{diag.detail}`.",
            "- Comparisons separate implementable controls from held-out oracle ceiling.",
            "- Selector design frozen from E073/E074: scaled design only.",
            "- Pilot and held-out seeds are fresh relative to E071-E076.",
            "- Selector uses select batches only; scoring uses target batches.",
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
