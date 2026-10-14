#!/usr/bin/env python3
"""Monitor DFW-032B and DFW-033 worker runs."""

from __future__ import annotations

import datetime as dt
import subprocess
import time
from pathlib import Path


JOBS = {
    "032b": {
        "subject": "talsarig__devvm49077",
        "needle": "032b_llm_content_boundary_vocab512",
        "run": "aai_research_tlv/tree/kalman_ddps/runs/talsarig__devvm49077/llm_content_boundary_vocab512_20261014",
    },
    "033": {
        "subject": "irisk__devvm47359",
        "needle": "033_llm_optimizer_boundary",
        "run": "aai_research_tlv/tree/kalman_ddps/runs/irisk__devvm47359/llm_optimizer_boundary_20261014",
    },
}


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ls(path: str) -> str:
    proc = subprocess.run(
        ["bash", "-lc", f"ROSETTA_DISABLE_AOT=1 manifold ls {path} 2>/dev/null || true"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    return proc.stdout


def main() -> int:
    log = Path("/data/users/atavory/local_gp/dfw_032_033_monitor_20261014.log")
    last = ""
    start = time.time()
    done: set[str] = set()
    while time.time() - start < 12 * 60 * 60:
        blocks: list[str] = []
        for key, job in JOBS.items():
            subject = job["subject"]
            needle = job["needle"]
            parts: list[str] = []
            for box in ["running", "failed", "from_workers"]:
                out = ls(f"aai_research_tlv/tree/double_fan/messages/{box}/{subject}")
                lines = [line for line in out.splitlines() if needle in line]
                if lines:
                    parts.append(f"{box}: " + " || ".join(lines[-5:]))
                    if box in {"failed", "from_workers"}:
                        done.add(key)
            run_lines = ls(job["run"]).splitlines()[-20:]
            if run_lines:
                parts.append("run: " + " || ".join(run_lines))
            if parts:
                blocks.append(f"{key} " + " ::: ".join(parts))
        text = "\n".join(blocks) or "no-visible-status"
        if text != last:
            with log.open("a", encoding="utf-8") as handle:
                handle.write(f"[{stamp()}]\n{text}\n")
            last = text
        if len(done) == len(JOBS):
            return 0
        time.sleep(120)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp()}]\ntimeout waiting for all jobs\n")
    return 124


if __name__ == "__main__":
    raise SystemExit(main())
