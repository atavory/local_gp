"""DFW-E069: stale/default-wrong anchor probe diagnostic.

E066-E068 show that grouped probes do not beat a known strong default LR arm in
a stationary regime.  This test moves to the regime that still fits the thesis:
the default is stale or wrong.  We anchor the admission rule on the low-LR arm
(`lr_multiplier=0.25`) while leaving the better arms available, then ask whether
same-round probes can safely depart from that stale default.
"""

from __future__ import annotations

from pathlib import Path

import dfw_e067_gpu_default_anchored_probe_admission as e67


DATE = "20261005"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e069_gpu_stale_default_anchor_probe_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e069_gpu_stale_default_anchor_probe_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e069_gpu_stale_default_anchor_probe_LEGITIMACY_CHECK_{DATE}.md"

ANCHOR_VALUE = 0.25


def anchor_update(row: e67.e60.GroupRoundRow):
    for value, update, _prior in e67.e60.group_updates(row):
        if abs(value - ANCHOR_VALUE) < 1e-12:
            return update
    raise ValueError(f"anchor group lr_multiplier={ANCHOR_VALUE} missing")


def anchor_index(groups: tuple[tuple[float, object, float], ...]) -> int:
    for idx, (value, _update, _prior) in enumerate(groups):
        if abs(value - ANCHOR_VALUE) < 1e-12:
            return idx
    raise ValueError(f"anchor group lr_multiplier={ANCHOR_VALUE} missing")


def retitle(text: str) -> str:
    return (
        text.replace("DFW-E067", "DFW-E069")
        .replace("E067 default-anchored probe admission", "E069 stale-default anchored probe admission")
        .replace(
            "Default-Anchored Probe Admission",
            "Stale-Default Anchor Probe",
        )
        .replace("default only", "stale-anchor only")
        .replace("default_only", "stale_anchor_only")
        .replace("default pair", "stale-anchor pair")
        .replace("default_pair", "stale_anchor_pair")
    )


def main() -> None:
    e67.default_update = anchor_update
    e67.default_index = anchor_index
    e67.PROBE_BATCH = 96
    e67.PROBE_SLICES = 4
    e67.MAX_SELECT_BATCH = 384
    run_log, result, legit = e67.run_experiment()
    run_log = retitle(run_log)
    result = retitle(result)
    legit = retitle(legit)
    result = result.replace(
        "Decision label:",
        f"Diagnostic: stale/default-wrong anchor with `lr_multiplier={ANCHOR_VALUE}`.\n\nDecision label:",
        1,
    )
    legit = legit.replace(
        "- Default-only control uses only the lr_multiplier=1 group with its own tuned outer pair.",
        f"- Stale-anchor control uses only the lr_multiplier={ANCHOR_VALUE} group with its own tuned outer pair.",
    )
    legit = legit.replace(
        "- Default-anchored rule uses a one-hot default prior plus frozen E064 damping.",
        f"- Stale-anchor rule uses a one-hot lr_multiplier={ANCHOR_VALUE} prior plus frozen E064 damping.",
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
