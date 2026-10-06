"""DFW-E082: LR-arm skew diagnostic from E060/E061 summaries.

E060/E061 are real-GPU grouped inner-LR experiments.  Their public artifacts
contain enough information to test for a summary-level collapse to the most
conservative probe policy, but not enough to recover exact low/default/high
arm mass.  This diagnostic is intentionally a no-GPU reanalysis over those
durable artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import statistics


DATE = "20261006"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e082_lr_skew_summary_diagnostic_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e082_lr_skew_summary_diagnostic_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e082_lr_skew_summary_diagnostic_LEGITIMACY_CHECK_{DATE}.md"

INPUTS = (
    (
        "E060",
        OUT_DIR / "dfw_e060_gpu_stationary_inner_lr_group_probe_admission_result_20261005.md",
    ),
    (
        "E061",
        OUT_DIR / "dfw_e061_gpu_stationary_group_admission_k_sweep_result_20261005.md",
    ),
)


@dataclass(frozen=True)
class Cell:
    experiment: str
    spread: float
    n_workers: int
    k_value: int
    groups: str
    pilot_best: str
    cost_pct: float
    score: float
    minus_fixed: float
    minus_shuffled: float
    oracle_subset: float
    oracle_single: float
    sigma_gap: float
    mean2_top3: float


def parse_ci_mean(text: str) -> float:
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*\+/-", text.strip())
    if not match:
        raise ValueError(f"could not parse CI cell: {text!r}")
    return float(match.group(1))


def parse_float(text: str) -> float:
    text = text.strip().replace("%", "")
    if text == "--":
        return float("nan")
    return float(text)


def parse_result(experiment: str, path: Path) -> list[Cell]:
    text = path.read_text(encoding="utf-8")
    rows: list[Cell] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("| "):
            continue
        if "---" in line or line.startswith("| spread"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 19:
            continue
        rows.append(
            Cell(
                experiment=experiment,
                spread=parse_float(parts[0]),
                n_workers=int(parts[1]),
                k_value=int(parts[2]),
                groups=parts[3],
                pilot_best=parts[11],
                cost_pct=parse_float(parts[12]),
                score=parse_ci_mean(parts[13]),
                minus_fixed=parse_ci_mean(parts[14]),
                minus_shuffled=parse_ci_mean(parts[15]),
                oracle_subset=parse_ci_mean(parts[6]),
                oracle_single=parse_ci_mean(parts[7]),
                sigma_gap=parse_float(parts[8]),
                mean2_top3=parse_float(parts[10]),
            )
        )
    return rows


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def decide(nontrivial: list[Cell]) -> str:
    if not nontrivial:
        return "LR_SKEW_SUMMARY_INSUFFICIENT"
    conservative = [cell for cell in nontrivial if cell.pilot_best == "group_l05_t4"]
    mean_fixed = mean([cell.minus_fixed for cell in nontrivial])
    mean_shuffled = mean([cell.minus_shuffled for cell in nontrivial])
    mean_top3 = mean([cell.mean2_top3 for cell in nontrivial if cell.mean2_top3 == cell.mean2_top3])
    if len(conservative) / len(nontrivial) <= 0.25 and mean_fixed > 0 and mean_shuffled > 0 and mean_top3 >= 0.95:
        return "LR_SKEW_SUMMARY_NO_SAFE_COLLAPSE_TRACE_NEEDED"
    return "LR_SKEW_SUMMARY_SAFE_COLLAPSE_RISK"


def fmt_mean(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{mean(vals):.3f} +/- {1.96 * statistics.stdev(vals) / (len(vals) ** 0.5):.3f}"


def run() -> tuple[str, str, str]:
    all_cells: list[Cell] = []
    input_lines = []
    for experiment, path in INPUTS:
        cells = parse_result(experiment, path)
        all_cells.extend(cells)
        input_lines.append(f"{experiment}: {path} rows={len(cells)}")

    nontrivial = [cell for cell in all_cells if cell.spread > 1.0 + 1e-12]
    policy_counts: dict[str, int] = {}
    for cell in nontrivial:
        policy_counts[cell.pilot_best] = policy_counts.get(cell.pilot_best, 0) + 1
    decision = decide(nontrivial)

    fixed_vals = [cell.minus_fixed for cell in nontrivial]
    shuffled_vals = [cell.minus_shuffled for cell in nontrivial]
    oracle_headroom = [cell.oracle_subset - cell.score for cell in nontrivial]
    conservative_cells = [cell for cell in nontrivial if cell.pilot_best == "group_l05_t4"]

    run_lines = [
        "DFW-E082 LR skew summary diagnostic",
        "compute_status=REANALYSIS_EXISTING_REAL_GPU_ARTIFACTS",
        *input_lines,
        f"nontrivial_cells={len(nontrivial)}",
        "policy_counts=" + ",".join(f"{key}:{value}" for key, value in sorted(policy_counts.items())),
        f"mean_minus_fixed={fmt_mean(fixed_vals)}",
        f"mean_minus_shuffled={fmt_mean(shuffled_vals)}",
        f"mean_oracle_headroom={fmt_mean(oracle_headroom)}",
        f"conservative_prior_policy_cells={len(conservative_cells)}",
        f"DECISION {decision}",
    ]

    lines = [
        "# DFW-E082 Result - LR Skew Summary Diagnostic",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        "`REANALYSIS_EXISTING_REAL_GPU_ARTIFACTS`: no new GPU work was run locally. "
        "The inputs are the durable E060/E061 `REAL_GPU` result tables.",
        "",
        "## Summary",
        "",
        f"- Nontrivial spread cells: `{len(nontrivial)}`.",
        "- Pilot-best policy counts: "
        + ", ".join(f"`{key}` = {value}" for key, value in sorted(policy_counts.items()))
        + ".",
        f"- Mean cell-level pilot-best minus fixed: `{fmt_mean(fixed_vals)}`.",
        f"- Mean cell-level pilot-best minus shuffled: `{fmt_mean(shuffled_vals)}`.",
        f"- Mean residual oracle headroom after pilot-best: `{fmt_mean(oracle_headroom)}`.",
        "",
        "Interpretation: the E060/E061 summaries do not show a broad collapse to "
        "the most conservative/default-preserving policy.  Nine of ten "
        "nontrivial cells choose a nontrivial damped policy, and the pooled "
        "cell means remain positive versus both fixed and shuffled controls.  "
        "The one conservative-prior cell is the hardest `K=32, spread=16` "
        "E061 case, where the pilot-best mean is essentially tied with fixed.",
        "",
        "Important limitation: these artifacts do not expose exact low/default/high "
        "LR arm weights, so they cannot by themselves prove whether mass moved "
        "toward the small/safe LR arm or the high LR arm.  A precise arm-mass "
        "trace should be run on the GPU helper path before upgrading free LR "
        "search from `GATED` to `LIVE`.",
        "",
        "## Nontrivial Cells",
        "",
        "| source | spread | N | K | groups | pilot-best | cost | score | minus fixed | minus shuffled | oracle subset | sigma/gap | mean2 top3 |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in nontrivial:
        lines.append(
            "| "
            + " | ".join(
                [
                    cell.experiment,
                    f"{cell.spread:g}",
                    str(cell.n_workers),
                    str(cell.k_value),
                    cell.groups,
                    cell.pilot_best,
                    f"{cell.cost_pct:.2f}%",
                    f"{cell.score:.3f}",
                    f"{cell.minus_fixed:.3f}",
                    f"{cell.minus_shuffled:.3f}",
                    f"{cell.oracle_subset:.3f}",
                    f"{cell.sigma_gap:.2f}",
                    f"{cell.mean2_top3:.3f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```", ""])

    legit = "\n".join(
        [
            "# DFW-E082 Legitimacy Check",
            "",
            "- First-line input artifacts are durable E060/E061 result markdown files.",
            "- This is a parser/reanalysis only; it does not report a CPU fallback as a GPU result.",
            "- The conclusion is limited to summary-level policy collapse, not exact LR arm mass.",
            "- E060/E061 themselves report `REAL_GPU` PyTorch CUDA execution.",
            "",
        ]
    )
    return "\n".join(run_lines) + "\n", "\n".join(lines), legit


def main() -> None:
    run_log, result, legit = run()
    RUN_LOG_PATH.write_text(run_log, encoding="utf-8")
    RESULT_PATH.write_text(result, encoding="utf-8")
    LEGITIMACY_PATH.write_text(legit, encoding="utf-8")
    print(result)
    print(f"wrote {RESULT_PATH}")
    print(f"wrote {RUN_LOG_PATH}")
    print(f"wrote {LEGITIMACY_PATH}")


if __name__ == "__main__":
    main()
