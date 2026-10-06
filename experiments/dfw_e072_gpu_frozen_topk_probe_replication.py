"""DFW-E072: frozen replication of the E071 top-k/soft probe selector.

E071 found a real-GPU positive for robust outer-control probe selection, but it
tested several selectors.  This file freezes the winning deployed selector
(`scaled` design with `soft_std2`) and repeats it on fresh seeds.
"""

from __future__ import annotations

from pathlib import Path
import platform
import sys

import dfw_e071_gpu_topk_probe_outer_control as e71


DATE = "20261005"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e072_gpu_frozen_topk_probe_replication_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e072_gpu_frozen_topk_probe_replication_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e072_gpu_frozen_topk_probe_replication_LEGITIMACY_CHECK_{DATE}.md"


def run_experiment() -> tuple[str, str, str]:
    # Freeze the policy family before calling E071 helpers.  These are the only
    # methodological choices copied from E071; all held-out seeds below are new.
    e71.DESIGN_NAMES = ("scaled",)
    e71.POLICIES = ("soft_std2",)
    e71.K_VALUES = (8, 32)
    e71.N_WORKERS = (8,)
    e71.SELECT_BATCHES = (384,)
    e71.PILOT_SEEDS = tuple(range(952_000, 952_003))
    e71.HELDOUT_SEEDS = tuple(range(953_000, 953_024))

    diag = e71.e30.compute_diagnostic()
    if diag.status != "REAL_GPU":
        run_log = "\n".join(
            (
                "DFW-E072 frozen top-k probe replication",
                f"host={platform.node()} python={sys.executable}",
                f"DECISION NO_GPU_HARNESS compute_status={diag.status} detail={diag.detail}",
            )
        )
        result = "\n".join(
            (
                "# DFW-E072 Result - Frozen Top-k Probe Replication",
                "",
                "Decision label: `NO_GPU_HARNESS`",
                "",
                f"Diagnostic: `{diag.detail}`.",
                "",
            )
        )
        legit = "# DFW-E072 Legitimacy Check\n\n- GPU workload: not run.\n"
        return run_log + "\n", result + "\n", legit

    e71.e30.torch.set_default_dtype(e71.e30.torch.float32)
    try:
        e71.e30.torch.set_num_threads(1)
    except Exception:
        pass

    cells: list[e71.Cell] = []
    run_lines = [
        "DFW-E072 frozen top-k probe replication",
        f"host={platform.node()} python={sys.executable} compute_status={diag.status} {diag.detail}",
        "frozen_selector=scaled/soft_std2",
        f"K={e71.K_VALUES} N={e71.N_WORKERS} select={e71.SELECT_BATCHES}",
        f"pilot_seeds={len(e71.PILOT_SEEDS)} heldout_seeds={len(e71.HELDOUT_SEEDS)}",
        "",
    ]

    for select_batch in e71.SELECT_BATCHES:
        for n_workers in e71.N_WORKERS:
            for k_value in e71.K_VALUES:
                cell = e71.score_cell(k_value, n_workers, select_batch, diag.device)
                cells.append(cell)
                score = cell.policy_scores[0]
                controls = e71.best_control(cell)
                run_lines.append(
                    f"CELL select={select_batch} N={n_workers} K={k_value} "
                    f"fixed={cell.fixed_pair} control={e71.fmt_ci(controls)} "
                    f"selector={e71.fmt_ci(score.values)} "
                    f"minus_control={e71.fmt_ci(e71.paired(score.values, controls))} "
                    f"minus_shuffled={e71.fmt_ci(e71.paired(score.values, score.shuffled))} "
                    f"capture={e71.fmt_ci(e71.capture(score.values, cell.fixed, cell.oracle))}"
                )

    decision = e71.decide(cells, diag.status)
    run_lines.append("")
    run_lines.append(f"DECISION {decision}")
    run_lines.append(f"compute_status={diag.status}")

    lines = [
        "# DFW-E072 Result - Frozen Top-k Probe Replication",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{diag.status}` via PyTorch CUDA: `{diag.detail}`.",
        "",
        "## Metrics",
        "",
        "| select | N | K | fixed | DiLoCo | schedule | best control | oracle | design oracle | selector | selector minus best control | selector minus shuffled | capture |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        score = cell.policy_scores[0]
        controls = e71.best_control(cell)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{cell.select_batch}",
                    f"{cell.n_workers}",
                    f"{cell.k_value}",
                    e71.fmt_ci(cell.fixed).strip(),
                    e71.fmt_ci(cell.diloco).strip(),
                    e71.fmt_ci(cell.schedule).strip(),
                    e71.fmt_ci(controls).strip(),
                    e71.fmt_ci(cell.oracle).strip(),
                    e71.fmt_ci(cell.design_oracle[score.design]).strip(),
                    e71.fmt_ci(score.values).strip(),
                    e71.fmt_ci(e71.paired(score.values, controls)).strip(),
                    e71.fmt_ci(e71.paired(score.values, score.shuffled)).strip(),
                    e71.fmt_ci(e71.capture(score.values, cell.fixed, cell.oracle)).strip(),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])
    legit = "\n".join(
        [
            "# DFW-E072 Legitimacy Check",
            "",
            f"- Compute: `{diag.status}` / `{diag.detail}`.",
            "- Selector family frozen from E071: scaled design, soft_std2 policy.",
            "- Pilot and held-out seeds are fresh relative to E071.",
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
