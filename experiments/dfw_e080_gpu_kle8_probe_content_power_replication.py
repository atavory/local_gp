"""DFW-E080: high-power K<=8 probe-content replication.

E078 showed a positive selector-vs-shuffled contrast at K<=8. E079 swept probe
batch sizes on fewer held-out seeds and did not replicate that contrast, even
though the action family beat fixed controls. This run holds the cheapest
configuration fixed and spends seeds, not new ideas.
"""

from __future__ import annotations

from pathlib import Path
import platform
import sys

import dfw_e071_gpu_topk_probe_outer_control as e71


DATE = "20261006"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e080_gpu_kle8_probe_content_power_replication_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e080_gpu_kle8_probe_content_power_replication_run_{DATE}.txt"
LEGITIMACY_PATH = (
    OUT_DIR / f"dfw_e080_gpu_kle8_probe_content_power_replication_LEGITIMACY_CHECK_{DATE}.md"
)


def extend_metrics(
    metrics: dict[str, list[float]],
    cell: e71.Cell,
    score: e71.PolicyScore,
) -> None:
    best_control = e71.best_control(cell)
    metrics["minus_fixed"].extend(e71.paired(score.values, cell.fixed))
    metrics["minus_schedule"].extend(e71.paired(score.values, cell.schedule))
    metrics["minus_best_control"].extend(e71.paired(score.values, best_control))
    metrics["minus_shuffled"].extend(e71.paired(score.values, score.shuffled))
    metrics["capture"].extend(e71.capture(score.values, cell.fixed, cell.oracle))


def run_experiment() -> tuple[str, str, str]:
    e71.DESIGN_NAMES = ("scaled",)
    e71.POLICIES = ("soft_std2",)
    e71.K_VALUES = (4, 8)
    e71.N_WORKERS = (8, 16)
    e71.SELECT_BATCHES = (384,)
    e71.PILOT_SEEDS = tuple(range(972_000, 972_002))
    e71.HELDOUT_SEEDS = tuple(range(973_000, 973_064))

    diag = e71.e30.compute_diagnostic()
    if diag.status != "REAL_GPU":
        run_log = "\n".join(
            (
                "DFW-E080 K<=8 probe-content power replication",
                f"host={platform.node()} python={sys.executable}",
                f"DECISION NO_GPU_HARNESS compute_status={diag.status} detail={diag.detail}",
            )
        )
        result = (
            "# DFW-E080 Result - K<=8 Probe-Content Power Replication\n\n"
            "Decision label: `NO_GPU_HARNESS`\n"
        )
        legit = "# DFW-E080 Legitimacy Check\n\n- GPU workload: not run.\n"
        return run_log + "\n", result + "\n", legit + "\n"

    e71.e30.torch.set_default_dtype(e71.e30.torch.float32)
    try:
        e71.e30.torch.set_num_threads(1)
    except Exception:
        pass

    rows: list[tuple[e71.Cell, e71.PolicyScore]] = []
    pooled = {
        "minus_fixed": [],
        "minus_schedule": [],
        "minus_best_control": [],
        "minus_shuffled": [],
        "capture": [],
    }
    run_lines = [
        "DFW-E080 K<=8 probe-content power replication",
        f"host={platform.node()} python={sys.executable} compute_status={diag.status} {diag.detail}",
        "design=scaled policy=soft_std2",
        f"K={e71.K_VALUES} N={e71.N_WORKERS} select={e71.SELECT_BATCHES}",
        f"pilot_seeds={len(e71.PILOT_SEEDS)} heldout_seeds={len(e71.HELDOUT_SEEDS)}",
        "gate=selector_minus_shuffled_positive_ci AND selector_minus_fixed_nonnegative_ci",
        "",
    ]

    for select_batch in e71.SELECT_BATCHES:
        for n_workers in e71.N_WORKERS:
            for k_value in e71.K_VALUES:
                cell = e71.score_cell(k_value, n_workers, select_batch, diag.device)
                score = next(
                    s for s in cell.policy_scores if s.design == "scaled" and s.policy == "soft_std2"
                )
                rows.append((cell, score))
                extend_metrics(pooled, cell, score)
                best_control = e71.best_control(cell)
                run_lines.append(
                    f"CELL select={select_batch} N={n_workers} K={k_value} "
                    f"fixed={e71.fmt_ci(cell.fixed)} schedule={e71.fmt_ci(cell.schedule)} "
                    f"best_control={e71.fmt_ci(best_control)} oracle={e71.fmt_ci(cell.oracle)} "
                    f"selector={e71.fmt_ci(score.values)} "
                    f"minus_fixed={e71.fmt_ci(e71.paired(score.values, cell.fixed))} "
                    f"minus_schedule={e71.fmt_ci(e71.paired(score.values, cell.schedule))} "
                    f"minus_best_control={e71.fmt_ci(e71.paired(score.values, best_control))} "
                    f"minus_shuffled={e71.fmt_ci(e71.paired(score.values, score.shuffled))} "
                    f"capture={e71.fmt_ci(e71.capture(score.values, cell.fixed, cell.oracle))}"
                )

    fixed_mean, fixed_hw = e71.mean_ci(pooled["minus_fixed"])
    if e71.positive_ci(pooled["minus_shuffled"]) and fixed_mean + fixed_hw >= 0.0:
        decision = "PROBE_CONTENT_POWER_REPLICATES"
    elif e71.positive_ci(pooled["minus_shuffled"]):
        decision = "PROBE_CONTENT_POWER_SHUFFLED_ONLY"
    else:
        decision = "PROBE_CONTENT_POWER_NULL"

    run_lines.append("")
    run_lines.append(
        f"POOLED minus_fixed={e71.fmt_ci(pooled['minus_fixed'])} "
        f"minus_schedule={e71.fmt_ci(pooled['minus_schedule'])} "
        f"minus_best_control={e71.fmt_ci(pooled['minus_best_control'])} "
        f"minus_shuffled={e71.fmt_ci(pooled['minus_shuffled'])} "
        f"capture={e71.fmt_ci(pooled['capture'])}"
    )
    run_lines.append(f"DECISION {decision}")
    run_lines.append(f"compute_status={diag.status}")

    lines = [
        "# DFW-E080 Result - K<=8 Probe-Content Power Replication",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{diag.status}` via PyTorch CUDA: `{diag.detail}`.",
        "",
        "## Pooled",
        "",
        f"- selector minus fixed: `{e71.fmt_ci(pooled['minus_fixed']).strip()}`",
        f"- selector minus schedule: `{e71.fmt_ci(pooled['minus_schedule']).strip()}`",
        f"- selector minus best control: `{e71.fmt_ci(pooled['minus_best_control']).strip()}`",
        f"- selector minus shuffled: `{e71.fmt_ci(pooled['minus_shuffled']).strip()}`",
        f"- capture: `{e71.fmt_ci(pooled['capture']).strip()}`",
        "",
        "## Cell Metrics",
        "",
        "| select | N | K | fixed | schedule | best control | oracle | selector | selector minus fixed | selector minus schedule | selector minus best control | selector minus shuffled | capture |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell, score in rows:
        best_control = e71.best_control(cell)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{cell.select_batch}",
                    f"{cell.n_workers}",
                    f"{cell.k_value}",
                    e71.fmt_ci(cell.fixed).strip(),
                    e71.fmt_ci(cell.schedule).strip(),
                    e71.fmt_ci(best_control).strip(),
                    e71.fmt_ci(cell.oracle).strip(),
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
            "# DFW-E080 Legitimacy Check",
            "",
            f"- Compute: `{diag.status}` / `{diag.detail}`.",
            "- Isolates K<=8 probe-active cells.",
            "- Uses the cheapest E078/E079 probe batch: select=384.",
            "- Selector design and policy frozen: scaled/soft_std2.",
            "- Pilot and held-out seeds are fresh relative to E078/E079.",
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
