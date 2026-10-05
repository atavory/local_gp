"""DFW-E068: high-budget default-anchored probe diagnostic.

E067 tested a default-anchored grouped-LR probe rule at the E064 cheap budget
(`96 x 4`) and found a safety repair but no win over default-only.  This
diagnostic doubles the probe examples to the largest clean budget available in
the existing two-select-draw bed (`192 x 4`) without changing the decision rule.

If this still fails, the failure is not just that E067 was too cheap; the
same-round probe value signal is not extracting the oracle group gap in the
stationary/default-known regime.
"""

from __future__ import annotations

from pathlib import Path

import dfw_e067_gpu_default_anchored_probe_admission as e67


DATE = "20261005"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e068_gpu_high_budget_default_anchored_probe_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e068_gpu_high_budget_default_anchored_probe_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e068_gpu_high_budget_default_anchored_probe_LEGITIMACY_CHECK_{DATE}.md"


def retitle(text: str) -> str:
    return (
        text.replace("DFW-E067", "DFW-E068")
        .replace("E067 default-anchored probe admission", "E068 high-budget default-anchored probe admission")
        .replace(
            "Default-Anchored Probe Admission",
            "High-Budget Default-Anchored Probe",
        )
        .replace(
            "Default-anchored rule uses a one-hot default prior plus frozen E064 damping.",
            "Default-anchored rule uses a one-hot default prior plus frozen E064 damping at doubled probe batch.",
        )
    )


def main() -> None:
    e67.PROBE_BATCH = 192
    e67.PROBE_SLICES = 4
    e67.MAX_SELECT_BATCH = 384
    run_log, result, legit = e67.run_experiment()
    run_log = retitle(run_log)
    result = retitle(result)
    legit = retitle(legit)
    result = result.replace(
        "Decision label:",
        "Diagnostic: high-budget repeat of E067 (`probe_batch=192`, `slices=4`).\n\nDecision label:",
        1,
    )
    legit = legit.replace(
        "- Shuffled control rotates probe losses across LR groups for the anchored rule.",
        "- Shuffled control rotates probe losses across LR groups for the anchored rule.\n"
        "- Probe budget is doubled relative to E067: `probe_batch=192`, `slices=4`.",
    )
    RUN_LOG_PATH.write_text(run_log)
    RESULT_PATH.write_text(result)
    LEGITIMACY_PATH.write_text(legit)
    print(result)
    print(f"wrote {RESULT_PATH}")
    print(f"wrote {RUN_LOG_PATH}")
    print(f"wrote {LEGITIMACY_PATH}")


if __name__ == "__main__":
    main()
