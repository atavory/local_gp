#!/usr/bin/env python3
"""Guarded Codex turn for the HPO-strengthening busy_wait loop."""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
from pathlib import Path


REPO = Path("/data/users/atavory/local_gp")
PROJECT = REPO / "bandha_hpo_strengthening_busy_wait_20261008"
DONE = PROJECT / "HPO_STRENGTHENING_BUSY_DONE.md"
BLOCKED = PROJECT / "HPO_STRENGTHENING_BUSY_BLOCKED.md"
INNER_STDOUT = PROJECT / "last_inner_stdout.txt"
INNER_STDERR = PROJECT / "last_inner_stderr.txt"
INNER_TIMEOUT_SEC = 7200
START_HEAD = "9ccc4be"

DOCS = (
    REPO / "research_control_board.md",
    REPO / "bandha_internal_hpo_option_map_20261006.md",
    REPO / "bandha_parallel_hparam_probe_program_20261005.md",
    REPO / "bandha_hpo_aware_aggregation_paper_draft_20261007.md",
)

BAD_LABELS = {
    "NO_TORCH",
    "INVALID",
    "NO_GPU_HARNESS",
    "ROUTE_FAILED",
    "WORKER_FAILED",
}


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


def e089_result_paths() -> list[Path]:
    return sorted((REPO / "experiments").glob("dfw_e089*_result_*.md"))


def e089_script_paths() -> list[Path]:
    return sorted((REPO / "experiments").glob("dfw_e089*.py"))


def e089_run_paths() -> list[Path]:
    return sorted((REPO / "experiments").glob("dfw_e089*_run_*.txt"))


def e089_legitimacy_paths() -> list[Path]:
    return sorted((REPO / "experiments").glob("dfw_e089*_LEGITIMACY_CHECK_*.md"))


def result_has_fixed_compute_content(path: Path) -> bool:
    body = text(path).lower()
    required_terms = (
        "fixed-compute",
        "external",
        "parallel",
        "internal",
    )
    return all(term in body for term in required_terms)


def e089_missing_reasons() -> list[str]:
    reasons: list[str] = []
    scripts = e089_script_paths()
    results = e089_result_paths()
    runs = e089_run_paths()
    legitimacy = e089_legitimacy_paths()
    if not scripts:
        reasons.append("missing E089 script under experiments/dfw_e089*.py")
    if not results:
        reasons.append("missing E089 result markdown")
    if not runs:
        reasons.append("missing E089 run log")
    if not legitimacy:
        reasons.append("missing E089 legitimacy check")

    good_result = False
    for result in results:
        label = extract_label(result)
        if not label:
            reasons.append(f"E089 result label missing: {result.name}")
            continue
        if label in BAD_LABELS:
            reasons.append(f"E089 non-completing label in {result.name}: {label}")
            continue
        if not result_has_fixed_compute_content(result):
            reasons.append(f"E089 result lacks fixed-compute comparator text: {result.name}")
            continue
        good_result = True
    if results and not good_result:
        reasons.append("no valid E089 fixed-compute result found")
    return reasons


def docs_missing_reasons() -> list[str]:
    reasons: list[str] = []
    for doc in DOCS:
        body = text(doc)
        if not body:
            reasons.append(f"missing required doc: {doc}")
        elif "E089" not in body:
            reasons.append(f"required doc lacks E089 update: {doc.name}")
    return reasons


def done_is_valid() -> tuple[bool, str]:
    if first_line(DONE) != "HPO_STRENGTHENING_BUSY_DONE: done":
        return False, "done sentinel missing or first line is wrong"

    reasons: list[str] = []
    reasons.extend(e089_missing_reasons())
    reasons.extend(docs_missing_reasons())

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
        return False, "; ".join(reasons[:10])
    return True, f"all HPO-strengthening gates validate at HEAD {head}"


def sentinel_status() -> tuple[str, str]:
    if BLOCKED.exists():
        return "blocked", "blocked sentinel exists"
    valid, reason = done_is_valid()
    if valid:
        return "complete", reason
    return "continue", reason


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: bandha_hpo_strengthening_busy_wait_agent_20261008.py <prompt_file>")
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
You are the inner worker for the guarded HPO-strengthening busy_wait loop.

Hard stop condition:
- You may report `BUSY_WAIT_STATUS: complete` only after `{DONE}` exists with
  first line exactly `HPO_STRENGTHENING_BUSY_DONE: done`, and wrapper
  validation passes.
- If genuinely blocked on external human input or unavailable infrastructure,
  write `{BLOCKED}` with the exact blocker and end with
  `BUSY_WAIT_STATUS: blocked`.
- Otherwise keep working and end with `BUSY_WAIT_STATUS: continue`.

Current wrapper status:
{reason}

Science mission:
1. Strengthen HPO wins for the paper.  Focus on HPO-aware aggregation:
   distributed workers/candidates become hyperparameter arms, probes score
   them, and the server recombines/weights them into one shared trajectory.
2. Do not frame this as "no aggregation first" or as a GPU-blocked project.
   The useful contrast is internal HPO versus naive parallel HPO at fixed
   worker compute.
3. Preserve the IID nuance: E087 shows stochastic-update filtering under
   fake IID labels.  That is not a persistent HPO signal, but it matters for
   the null section.
4. Main immediate experiment: E089, a serious fixed-compute HPO comparison in
   the E063 grouped-LR headroom regime.  E088 ultra was a smoke and did not
   clear the baseline.
5. Update the durable markdown files after the result:
   `{REPO / "research_control_board.md"}`
   `{REPO / "bandha_internal_hpo_option_map_20261006.md"}`
   `{REPO / "bandha_parallel_hparam_probe_program_20261005.md"}`
   `{REPO / "bandha_hpo_aware_aggregation_paper_draft_20261007.md"}`
6. Commit and push relevant changes.  Use Mr. Gitties for git push/network.
7. Write `{DONE}` only after E089 artifacts, docs, commit, and push are done.

E089 design target:
- Reuse `experiments/dfw_e088_fixed_compute_parallel_hpo_baseline.py` where
  sensible.
- Move away from the tiny E088 settings.  Use more rounds/seeds and a
  grouped-LR arm spread with known headroom, preferably spread 16 from E063.
- Keep the fixed-compute accounting honest:
  internal HPO uses all workers in one trajectory as arm groups;
  external HPO splits the same total workers across separate trajectories and
  selects one by validation/probe loss;
  target-oracle selection is diagnostic only.
- Include default/full-worker, parallel-arm-average, shuffled-label, external
  validation HPO, external oracle, best full-worker arm, probe cost, and a
  legitimacy check.
- A scientifically valid negative result is acceptable, but it must be a real
  E089 result, not just a tiny smoke rerun.

Execution rules:
- Use existing local experiment helpers from E063/E080/E085/E088 rather than
  reinventing training code if practical.
- Prefer real GPU/Buck lanes when available, but TORCH_CPU is acceptable for
  algorithmic evidence if clearly labeled.
- Do not use pip.
- The repo has unrelated dirty/untracked files.  Stage only files you create or
  intentionally edit.
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
                    f"## guarded HPO-strengthening worker - {finished}",
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
