"""DFW-E073: frozen K-aware robust probe selector replication.

E072 froze a single selector across K and lost at K=32.  E071's best real-GPU
robust selector was K-dependent: `scaled/soft_std2` at K=8 and
`scaled/top3_mean` at K=32.  K is known to the outer optimizer, so this test
freezes that simple K-aware rule and repeats it on fresh seeds.
"""

from __future__ import annotations

from pathlib import Path
import platform
import sys

import dfw_e071_gpu_topk_probe_outer_control as e71


DATE = "20261005"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e073_gpu_k_aware_topk_probe_replication_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e073_gpu_k_aware_topk_probe_replication_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e073_gpu_k_aware_topk_probe_replication_LEGITIMACY_CHECK_{DATE}.md"


def policy_for_k(k_value: int) -> str:
    return "soft_std2" if k_value <= 8 else "top3_mean"


def run_experiment() -> tuple[str, str, str]:
    e71.DESIGN_NAMES = ("scaled",)
    e71.POLICIES = ("top3_mean", "soft_std2")
    e71.K_VALUES = (8, 32)
    e71.N_WORKERS = (8,)
    e71.SELECT_BATCHES = (384,)
    e71.PILOT_SEEDS = tuple(range(956_000, 956_003))
    e71.HELDOUT_SEEDS = tuple(range(957_000, 957_024))

    diag = e71.e30.compute_diagnostic()
    if diag.status != "REAL_GPU":
        run_log = "\n".join(
            (
                "DFW-E073 K-aware top-k probe replication",
                f"host={platform.node()} python={sys.executable}",
                f"DECISION NO_GPU_HARNESS compute_status={diag.status} detail={diag.detail}",
            )
        )
        result = "# DFW-E073 Result - K-aware Top-k Probe Replication\n\nDecision label: `NO_GPU_HARNESS`\n"
        legit = "# DFW-E073 Legitimacy Check\n\n- GPU workload: not run.\n"
        return run_log + "\n", result + "\n", legit

    e71.e30.torch.set_default_dtype(e71.e30.torch.float32)
    try:
        e71.e30.torch.set_num_threads(1)
    except Exception:
        pass

    cells: list[e71.Cell] = []
    chosen_scores: list[tuple[e71.Cell, e71.PolicyScore]] = []
    run_lines = [
        "DFW-E073 K-aware top-k probe replication",
        f"host={platform.node()} python={sys.executable} compute_status={diag.status} {diag.detail}",
        "frozen_rule=scaled/(soft_std2 if K<=8 else top3_mean)",
        f"K={e71.K_VALUES} N={e71.N_WORKERS} select={e71.SELECT_BATCHES}",
        f"pilot_seeds={len(e71.PILOT_SEEDS)} heldout_seeds={len(e71.HELDOUT_SEEDS)}",
        "",
    ]

    all_minus_control: list[float] = []
    all_minus_shuffled: list[float] = []
    for select_batch in e71.SELECT_BATCHES:
        for n_workers in e71.N_WORKERS:
            for k_value in e71.K_VALUES:
                cell = e71.score_cell(k_value, n_workers, select_batch, diag.device)
                cells.append(cell)
                wanted = policy_for_k(k_value)
                score = next(s for s in cell.policy_scores if s.design == "scaled" and s.policy == wanted)
                chosen_scores.append((cell, score))
                controls = e71.best_control(cell)
                minus_control = e71.paired(score.values, controls)
                minus_shuffled = e71.paired(score.values, score.shuffled)
                all_minus_control.extend(minus_control)
                all_minus_shuffled.extend(minus_shuffled)
                run_lines.append(
                    f"CELL select={select_batch} N={n_workers} K={k_value} policy={wanted} "
                    f"control={e71.fmt_ci(controls)} selector={e71.fmt_ci(score.values)} "
                    f"minus_control={e71.fmt_ci(minus_control)} "
                    f"minus_shuffled={e71.fmt_ci(minus_shuffled)} "
                    f"capture={e71.fmt_ci(e71.capture(score.values, cell.fixed, cell.oracle))}"
                )

    if e71.positive_ci(all_minus_control) and e71.positive_ci(all_minus_shuffled):
        decision = "K_AWARE_TOPK_PROBE_REPLICATES"
    elif e71.positive_ci(all_minus_control):
        decision = "K_AWARE_TOPK_CONTROL_WIN_ONLY"
    else:
        decision = "K_AWARE_TOPK_STILL_LOSES"
    run_lines.append("")
    run_lines.append(f"DECISION {decision}")
    run_lines.append(f"compute_status={diag.status}")

    lines = [
        "# DFW-E073 Result - K-aware Top-k Probe Replication",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{diag.status}` via PyTorch CUDA: `{diag.detail}`.",
        "",
        "## Metrics",
        "",
        "| select | N | K | policy | fixed | DiLoCo | schedule | best control | oracle | design oracle | selector | selector minus best control | selector minus shuffled | capture |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell, score in chosen_scores:
        controls = e71.best_control(cell)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{cell.select_batch}",
                    f"{cell.n_workers}",
                    f"{cell.k_value}",
                    score.policy,
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
            "# DFW-E073 Legitimacy Check",
            "",
            f"- Compute: `{diag.status}` / `{diag.detail}`.",
            "- Selector family frozen from E071: scaled design with a K-aware robust policy.",
            "- Pilot and held-out seeds are fresh relative to E071/E072.",
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
