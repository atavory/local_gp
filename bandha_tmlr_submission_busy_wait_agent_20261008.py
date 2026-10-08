#!/usr/bin/env python3
"""Guarded Codex turn for the TMLR-submission busy_wait loop."""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
from pathlib import Path


REPO = Path("/data/users/atavory/local_gp")
PROJECT = REPO / "bandha_tmlr_submission_busy_wait_20261008"
DONE = PROJECT / "TMLR_SUBMISSION_BUSY_DONE.md"
BLOCKED = PROJECT / "TMLR_SUBMISSION_BUSY_BLOCKED.md"
INNER_STDOUT = PROJECT / "last_inner_stdout.txt"
INNER_STDERR = PROJECT / "last_inner_stderr.txt"
INNER_TIMEOUT_SEC = 9000
START_HEAD = "985e0e0"

REQUIRED_DOCS = (
    REPO / "research_control_board.md",
    REPO / "bandha_internal_hpo_option_map_20261006.md",
    REPO / "bandha_parallel_hparam_probe_program_20261005.md",
    REPO / "bandha_hpo_aware_aggregation_paper_draft_20261007.md",
    REPO / "bandha_tmlr_submission_packet_20261008.md",
)

BAD_LABEL_PARTS = (
    "NO_TORCH",
    "INVALID",
    "NO_GPU_HARNESS",
    "ROUTE_FAILED",
    "WORKER_FAILED",
    "NOT_CLEARED",
    "NEGATIVE",
    "TIE",
)
GOOD_LABEL_PARTS = ("WIN", "POSITIVE", "CLEARED", "LIVE", "READY")


def first_line(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.readline().strip()


def text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def extract_label(path: Path) -> str:
    body = text(path)
    for pattern in (
        r"Decision label:\s*`([^`]+)`",
        r"Decision:\s*`([^`]+)`",
        r"DECISION\s+([A-Z0-9_]+)",
    ):
        match = re.search(pattern, body)
        if match:
            return match.group(1)
    return ""


def run_quiet(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"


def current_head() -> str:
    rc, out, _err = run_quiet(["git", "rev-parse", "--short", "HEAD"], 10)
    return out if rc == 0 else ""


def result_paths() -> list[Path]:
    e090 = sorted((REPO / "experiments").glob("dfw_e090*_result_*.md"))
    if e090:
        return e090
    return sorted((REPO / "experiments").glob("dfw_e09[1-9]*_result_*.md"))


def artifact_reasons(prefix: str = "dfw_e090") -> list[str]:
    reasons: list[str] = []
    scripts = sorted((REPO / "experiments").glob(f"{prefix}*.py"))
    results = sorted((REPO / "experiments").glob(f"{prefix}*_result_*.md"))
    runs = sorted((REPO / "experiments").glob(f"{prefix}*_run_*.txt"))
    legitimacy = sorted((REPO / "experiments").glob(f"{prefix}*_LEGITIMACY_CHECK_*.md"))
    if not scripts:
        reasons.append(f"missing {prefix} script")
    if not results:
        reasons.append(f"missing {prefix} result markdown")
    if not runs:
        reasons.append(f"missing {prefix} run log")
    if not legitimacy:
        reasons.append(f"missing {prefix} legitimacy check")
    return reasons


def good_result_found() -> tuple[bool, list[str]]:
    reasons: list[str] = []
    candidates = result_paths()
    if not candidates:
        reasons.extend(artifact_reasons("dfw_e090"))
        return False, reasons

    found_good = False
    for result in candidates:
        label = extract_label(result)
        body = text(result)
        upper_label = label.upper()
        lower_body = body.lower()
        if not label:
            reasons.append(f"missing decision label in {result.name}")
            continue
        if any(part in upper_label for part in BAD_LABEL_PARTS):
            reasons.append(f"non-completing label in {result.name}: {label}")
            continue
        if not any(part in upper_label for part in GOOD_LABEL_PARTS):
            reasons.append(f"label is not an explicit positive/live/win: {label}")
            continue
        required_terms = ("fixed-compute", "external", "internal", "stale")
        if not all(term in lower_body for term in required_terms):
            reasons.append(f"result lacks fixed-compute stale-HPO comparator text: {result.name}")
            continue
        found_good = True

    if not found_good:
        reasons.append("no good fixed-compute HPO result found")
    e090_artifact_reasons = artifact_reasons("dfw_e090")
    if e090_artifact_reasons and not found_good:
        reasons.extend(e090_artifact_reasons)
    return found_good, reasons


def docs_reasons() -> list[str]:
    reasons: list[str] = []
    for doc in REQUIRED_DOCS:
        body = text(doc)
        if not body:
            reasons.append(f"missing required doc: {doc.name}")
            continue
        if doc.name == "bandha_tmlr_submission_packet_20261008.md":
            for term in ("TMLR", "E090", "E089", "limitations", "submission"):
                if term.lower() not in body.lower():
                    reasons.append(f"submission packet lacks {term}: {doc.name}")
        elif "E090" not in body:
            reasons.append(f"required doc lacks E090 update: {doc.name}")
    return reasons


def done_is_valid() -> tuple[bool, str]:
    if first_line(DONE) != "TMLR_SUBMISSION_BUSY_DONE: done":
        return False, "done sentinel missing or first line is wrong"

    reasons: list[str] = []
    ok_result, result_reasons = good_result_found()
    if not ok_result:
        reasons.extend(result_reasons)
    reasons.extend(docs_reasons())

    done_text = text(DONE)
    head = current_head()
    if not head:
        reasons.append("could not resolve current HEAD")
    elif head == START_HEAD:
        reasons.append(f"HEAD has not advanced beyond start head {START_HEAD}")
    elif head not in done_text:
        reasons.append(f"done sentinel does not mention current HEAD {head}")
    if "pushed=yes" not in done_text:
        reasons.append("done sentinel lacks pushed=yes")

    if reasons:
        return False, "; ".join(reasons[:12])
    return True, f"TMLR submission gates validate at HEAD {head}"


def sentinel_status() -> tuple[str, str]:
    if BLOCKED.exists():
        return "blocked", "blocked sentinel exists"
    valid, reason = done_is_valid()
    if valid:
        return "complete", reason
    return "continue", reason


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: bandha_tmlr_submission_busy_wait_agent_20261008.py <prompt_file>")
        print("BUSY_WAIT_STATUS: continue")
        return 0

    prompt_file = Path(sys.argv[1])
    base_prompt = prompt_file.read_text(encoding="utf-8", errors="replace")
    status, reason = sentinel_status()
    if status != "continue":
        print(f"Sentinel status before turn: {status} ({reason})")
        print(f"BUSY_WAIT_STATUS: {status}")
        return 0

    guard = f"""
You are the inner worker for the guarded TMLR-submission busy_wait loop.

Hard stop condition:
- You may report `BUSY_WAIT_STATUS: complete` only after `{DONE}` exists with
  first line exactly `TMLR_SUBMISSION_BUSY_DONE: done`, and wrapper validation
  passes.
- If genuinely blocked on external human input or unavailable infrastructure,
  write `{BLOCKED}` with the exact blocker and end with
  `BUSY_WAIT_STATUS: blocked`.
- Otherwise keep working and end with `BUSY_WAIT_STATUS: continue`.

Current wrapper status:
{reason}

Mission:
1. Make the project TMLR-submission-ready, not merely draft-ready.
2. Main experiment target: E090 fixed-compute stale-default recovery.  This
   should turn E085's practical HPO story into a fixed-compute comparator table.
3. If E090 does not produce a positive/live/win result, continue to the next
   strongest fixed-compute HPO result, likely outer-control HPO from E080/E081.
4. Narrow the paper claims: E089 is a boundary/tie against external HPO, E087 is
   stochastic filtering under exchangeable fake labels, and the main positive
   claim is practical HPO-aware aggregation under designed nonexchangeability.
5. Update the durable docs and produce `bandha_tmlr_submission_packet_20261008.md`.
6. Commit and push relevant changes.  Use Mr. Gitties for git push/network.
7. Write `{DONE}` only after positive fixed-compute HPO evidence, docs, paper
   packet, commit, and push are done.

Execution guidance:
- Reuse E085 for stale-default guarded recovery and E088/E089 for fixed-compute
  external-HPO accounting.
- Keep same total worker budget between internal and external comparisons.
- Include shuffled controls, default/fresh/stale controls, validation-selected
  external HPO, external oracle diagnostic, probe cost, and legitimacy check.
- Prefer real GPU/Buck lanes where available, but TORCH_CPU is acceptable if
  clearly labeled and the experiment is still informative.
- Do not call the project GPU-blocked because a route is slow or a worker lane
  is pending.
- Do not use pip.
- The repo has unrelated dirty/untracked files.  Stage only files you create or
  intentionally edit for this loop.
- Do not end a turn with only a plan; run or write the next artifact.
"""

    full_prompt = guard + "\n\n" + base_prompt
    cmd = [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(REPO),
        "-",
    ]
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO,
            input=full_prompt,
            text=True,
            capture_output=True,
            timeout=INNER_TIMEOUT_SEC,
            check=False,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        rc = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stderr += f"\n[bandha wrapper] inner codex timed out after {INNER_TIMEOUT_SEC}s\n"
        rc = 124
        timed_out = True

    INNER_STDOUT.write_text(stdout, encoding="utf-8")
    INNER_STDERR.write_text(stderr, encoding="utf-8")

    status, reason = sentinel_status()
    finished = dt.datetime.now(dt.timezone.utc).isoformat()
    with (PROJECT / "log.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "",
                    f"## guarded TMLR worker - {finished}",
                    "",
                    f"- started: `{started}`",
                    f"- inner_returncode: `{rc}`",
                    f"- inner_timed_out: `{timed_out}`",
                    f"- sentinel_status: `{status}`",
                    f"- sentinel_reason: `{reason}`",
                    f"- inner_stdout: `{INNER_STDOUT}`",
                    f"- inner_stderr: `{INNER_STDERR}`",
                    "",
                ]
            )
        )

    print(stdout)
    if stderr:
        print("===== inner stderr =====")
        print(stderr)
    print(f"[bandha wrapper] inner_returncode={rc} timed_out={timed_out}")
    print(f"[bandha wrapper] sentinel_status={status} reason={reason}")
    print(f"BUSY_WAIT_STATUS: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
