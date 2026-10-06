"""DFW-E079: K<=8 probe-content budget sweep.

E078 isolates the K<=8 cells where the frozen probe selector actually acts.
This companion sweep asks whether the selector-vs-shuffled contrast appears
when the probe batch is larger, or whether the content signal remains null.
"""

from __future__ import annotations

from pathlib import Path
import platform
import sys

import dfw_e071_gpu_topk_probe_outer_control as e71


DATE = "20261006"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e079_gpu_kle8_probe_budget_content_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e079_gpu_kle8_probe_budget_content_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e079_gpu_kle8_probe_budget_content_LEGITIMACY_CHECK_{DATE}.md"


def run_experiment() -> tuple[str, str, str]:
    e71.DESIGN_NAMES = ("scaled",)
    e71.POLICIES = ("soft_std2",)
    e71.K_VALUES = (4, 8)
    e71.N_WORKERS = (8, 16)
    e71.SELECT_BATCHES = (384, 768, 1536)
    e71.PILOT_SEEDS = tuple(range(970_000, 970_002))
    e71.HELDOUT_SEEDS = tuple(range(971_000, 971_013))

    diag = e71.e30.compute_diagnostic()
    if diag.status != "REAL_GPU":
        run_log = "\n".join(
            (
                "DFW-E079 K<=8 probe-content budget sweep",
                f"host={platform.node()} python={sys.executable}",
                f"DECISION NO_GPU_HARNESS compute_status={diag.status} detail={diag.detail}",
            )
        )
        result = "# DFW-E079 Result - K<=8 Probe-Content Budget Sweep\n\nDecision label: `NO_GPU_HARNESS`\n"
        legit = "# DFW-E079 Legitimacy Check\n\n- GPU workload: not run.\n"
        return run_log + "\n", result + "\n", legit + "\n"

    e71.e30.torch.set_default_dtype(e71.e30.torch.float32)
    try:
        e71.e30.torch.set_num_threads(1)
    except Exception:
        pass

    by_batch: dict[int, dict[str, list[float]]] = {}
    rows: list[tuple[e71.Cell, e71.PolicyScore]] = []
    run_lines = [
        "DFW-E079 K<=8 probe-content budget sweep",
        f"host={platform.node()} python={sys.executable} compute_status={diag.status} {diag.detail}",
        "design=scaled policy=soft_std2",
        f"K={e71.K_VALUES} N={e71.N_WORKERS} select={e71.SELECT_BATCHES}",
        f"pilot_seeds={len(e71.PILOT_SEEDS)} heldout_seeds={len(e71.HELDOUT_SEEDS)}",
        "",
    ]

    for select_batch in e71.SELECT_BATCHES:
        by_batch[select_batch] = {
            "minus_fixed": [],
            "minus_schedule": [],
            "minus_best_control": [],
            "minus_shuffled": [],
            "capture": [],
        }
        for n_workers in e71.N_WORKERS:
            for k_value in e71.K_VALUES:
                cell = e71.score_cell(k_value, n_workers, select_batch, diag.device)
                score = next(
                    s for s in cell.policy_scores if s.design == "scaled" and s.policy == "soft_std2"
                )
                rows.append((cell, score))
                best_control = e71.best_control(cell)
                metrics = by_batch[select_batch]
                metrics["minus_fixed"].extend(e71.paired(score.values, cell.fixed))
                metrics["minus_schedule"].extend(e71.paired(score.values, cell.schedule))
                metrics["minus_best_control"].extend(e71.paired(score.values, best_control))
                metrics["minus_shuffled"].extend(e71.paired(score.values, score.shuffled))
                metrics["capture"].extend(e71.capture(score.values, cell.fixed, cell.oracle))
                run_lines.append(
                    f"CELL select={select_batch} N={n_workers} K={k_value} "
                    f"selector={e71.fmt_ci(score.values)} fixed={e71.fmt_ci(cell.fixed)} "
                    f"schedule={e71.fmt_ci(cell.schedule)} best_control={e71.fmt_ci(best_control)} "
                    f"minus_fixed={e71.fmt_ci(e71.paired(score.values, cell.fixed))} "
                    f"minus_schedule={e71.fmt_ci(e71.paired(score.values, cell.schedule))} "
                    f"minus_best_control={e71.fmt_ci(e71.paired(score.values, best_control))} "
                    f"minus_shuffled={e71.fmt_ci(e71.paired(score.values, score.shuffled))} "
                    f"capture={e71.fmt_ci(e71.capture(score.values, cell.fixed, cell.oracle))}"
                )

    winning_batches = [
        batch
        for batch, metrics in by_batch.items()
        if e71.positive_ci(metrics["minus_shuffled"])
        and e71.mean_ci(metrics["minus_fixed"])[0] + e71.mean_ci(metrics["minus_fixed"])[1] >= 0.0
    ]
    if winning_batches:
        decision = "PROBE_CONTENT_BUDGET_LIVE"
    elif any(e71.positive_ci(metrics["minus_shuffled"]) for metrics in by_batch.values()):
        decision = "PROBE_CONTENT_BUDGET_SHUFFLED_ONLY"
    else:
        decision = "PROBE_CONTENT_BUDGET_NULL"

    run_lines.append("")
    for batch, metrics in by_batch.items():
        run_lines.append(
            f"POOLED select={batch} minus_fixed={e71.fmt_ci(metrics['minus_fixed'])} "
            f"minus_schedule={e71.fmt_ci(metrics['minus_schedule'])} "
            f"minus_best_control={e71.fmt_ci(metrics['minus_best_control'])} "
            f"minus_shuffled={e71.fmt_ci(metrics['minus_shuffled'])} "
            f"capture={e71.fmt_ci(metrics['capture'])}"
        )
    run_lines.append(f"DECISION {decision}")
    run_lines.append(f"compute_status={diag.status}")

    lines = [
        "# DFW-E079 Result - K<=8 Probe-Content Budget Sweep",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{diag.status}` via PyTorch CUDA: `{diag.detail}`.",
        "",
        "## Pooled By Probe Batch",
        "",
        "| select | minus fixed | minus schedule | minus best control | minus shuffled | capture |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for batch, metrics in by_batch.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{batch}",
                    e71.fmt_ci(metrics["minus_fixed"]).strip(),
                    e71.fmt_ci(metrics["minus_schedule"]).strip(),
                    e71.fmt_ci(metrics["minus_best_control"]).strip(),
                    e71.fmt_ci(metrics["minus_shuffled"]).strip(),
                    e71.fmt_ci(metrics["capture"]).strip(),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Cell Metrics", ""])
    lines.append(
        "| select | N | K | selector | minus fixed | minus schedule | minus best control | minus shuffled | capture |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cell, score in rows:
        best_control = e71.best_control(cell)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{cell.select_batch}",
                    f"{cell.n_workers}",
                    f"{cell.k_value}",
                    e71.fmt_ci(score.values).strip(),
                    e71.fmt_ci(e71.paired(score.values, cell.fixed)).strip(),
                    e71.fmt_ci(e71.paired(score.values, cell.schedule)).strip(),
                    e71.fmt_ci(e71.paired(score.values, best_control)).strip(),
                    e71.fmt_ci(e71.paired(score.values, score.shuffled)).strip(),
                    e71.fmt_ci(e71.capture(score.values, cell.fixed, cell.oracle)).strip(),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])

    legit = "\n".join(
        [
            "# DFW-E079 Legitimacy Check",
            "",
            f"- Compute: `{diag.status}` / `{diag.detail}`.",
            "- Isolates K<=8 probe-active cells.",
            "- Sweeps probe batch size with selector/policy frozen.",
            "- Pilot and held-out seeds are fresh relative to E071-E078.",
            "- Selector uses select batches only; scoring uses target batches.",
            "- Shuffled probe labels are the primary content control.",
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
