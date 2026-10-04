# Fresh-Agent Prompt: Beyond Averaging / Local GP

Use this prompt when context is cleared. It is intentionally compact. The
experiment files are the ledger; this file is the operational memory.

## Identity

Read `~/.agent_context.md` and `~/.agent_context.json` first. Use the prefix for
your agent. Codex is usually `bandha:`.

The project is:

```text
Beyond Averaging: Distributed Training as Sequential Experimental Design
```

Do not center the project on the old negative title "same-point worker clouds
are not momentum controllers." That is a frozen lesson from Double Fan, not the
project's identity.

The project view:

1. A single worker querying a stochastic objective creates a natural time
   series. Momentum and acceleration summarize that private time series.
2. Multiple workers are usually collapsed into one synthetic worker by
   averaging. That creates questions about plug-in momentum/acceleration,
   synchronization, and where estimators live.
3. We instead view the system as global optimization under noise. A worker
   should be allowed to act using all known previous information, not just its
   own private past.
4. Homogeneous synchronized workers are a special case, not an assumption.

The live scientific object is not a scalar momentum coefficient and not a full
GP over all parameters. It is a structured local update map:

```text
worker paths/probes/tags
  -> estimate a local K-step update map or tagged posterior
  -> shrink/reject unsafe corrections
  -> consolidate with a stochastic trust-region rule
```

## Repositories And Paper

GitHub repo:

```text
https://github.com/atavory/local_gp
/data/users/atavory/local_gp
```

Paper Overleaf checkout:

```text
/data/users/atavory/overleaf/6aa106bb39f566201f723916
main.tex
```

The paper should name the GitHub repo. The current paper title should align with
the project:

```text
Beyond Averaging: Distributed Training as Sequential Experimental Design
```

Use Mr. Gitties for external git and Overleaf network operations. Never embed an
Overleaf token in a command string. Read it from `~/.agent_context.json` and pass
it through the environment if cloning is required.

Minimal Mr. Gitties helper:

```bash
python3 - <<'PY'
import json, os, socket

def mrg(cmd, cwd=None, timeout=300, env=None):
    req = {"cmd": cmd, "timeout": timeout}
    if cwd:
        req["cwd"] = cwd
    if env:
        req["env"] = env
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(os.path.expanduser("~/.mrgitties/mrgitties.sock"))
    s.sendall(json.dumps(req).encode() + b"\n")
    s.shutdown(socket.SHUT_WR)
    buf = b""
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    print(buf.decode())

mrg("git pull --ff-only", cwd="/data/users/atavory/local_gp")
mrg("git push", cwd="/data/users/atavory/local_gp")
PY
```

For local commits, ordinary `git add` / `git commit` is fine. Use Mr. Gitties
for `git pull`, `git fetch`, and `git push`.

## Build And Runtime Rules

- Do not use `pip`.
- Prefer the repo's existing scripts and `buck2` when a build target exists.
- For GPU workers, do not conclude "CUDA is broken" from a bare local failure.
  Use the Buck2 GPU path, including `keep_gpu_sections` if that is how the
  worker harness is configured.
- Do not delete `__pycache__` directories unless the user explicitly asks.
- Unreturned worker prompts are not negative scientific results.

## Worker Control Plane

The current worker framework lives on Manifold. There are two buckets involved:

```text
aair_research_tlv_messaging/tree/project_worker/...
aai_research_tlv/tree/double_fan/...
```

The queue name `double_fan` is historical. It does not mean the science is the
old Double Fan project.

Bootstrap and scan:

```bash
mkdir -p /tmp/project_worker
env ROSETTA_DISABLE_AOT=1 manifold get --overwrite_local_path \
  aair_research_tlv_messaging/tree/project_worker/instructions/runner_controller.md \
  /tmp/project_worker/runner_controller.md
env ROSETTA_DISABLE_AOT=1 manifold get --overwrite_local_path \
  aair_research_tlv_messaging/tree/project_worker/project_workerctl.py \
  /tmp/project_worker/project_workerctl.py
chmod +x /tmp/project_worker/project_workerctl.py
python3 /tmp/project_worker/project_workerctl.py scan
```

Assign a client to the project:

```bash
python3 /tmp/project_worker/project_workerctl.py assign <mailbox> --project double_fan
```

Task queues:

```text
aai_research_tlv/tree/double_fan/messages/to_workers/<subject>/
aai_research_tlv/tree/double_fan/messages/running/<subject>/
aai_research_tlv/tree/double_fan/messages/done/<subject>/
aai_research_tlv/tree/double_fan/messages/failed/<subject>/
aai_research_tlv/tree/double_fan/runs/<subject>/<run_id>/
```

Upload a task:

```bash
env ROSETTA_DISABLE_AOT=1 manifold put \
  /tmp/task.md \
  aai_research_tlv/tree/double_fan/messages/to_workers/<subject>/<timestamp>_task.md
```

If a worker does not have the GitHub checkout or Mr. Gitties, give it a Manifold
source bundle. Current bundle convention:

```text
aai_research_tlv/tree/double_fan/source/local_gp_source_latest.tgz
```

Refresh it from the repo:

```bash
tar --exclude=.git --exclude='__pycache__' --exclude='*.pyc' \
  -czf /tmp/local_gp_source_latest.tgz \
  -C /data/users/atavory local_gp
env ROSETTA_DISABLE_AOT=1 manifold put \
  /tmp/local_gp_source_latest.tgz \
  aai_research_tlv/tree/double_fan/source/local_gp_source_latest.tgz
```

Framework maintenance:

```text
/tmp/project_worker/project_agent_loop.sh
aair_research_tlv_messaging/tree/project_worker/project_agent_loop.sh
```

If you patch the worker loop, upload it to the messaging bucket. Already-running
workers may not reload it until restarted, so put critical bootstrap
instructions directly in task prompts when needed.

## Known Clients From The Last Scan

These were the five clients discovered and assigned on 2026-10-04. Always scan
again before relying on the list.

```text
atavory__devvm1629   CPU 44  GPU 2  RAM 353.6 GiB
atavory__devvm43898  CPU 96  GPU 0  RAM 235.7 GiB
atavory__devvm44191  CPU 96  GPU 0  RAM 235.7 GiB
atavory__devvm52518  CPU 22  GPU 1  RAM 176.7 GiB
atavory__devvm60954  CPU 22  GPU 1  RAM 176.7 GiB
```

## Evidence State

Old Double Fan:

- Same-point worker variance is measurable.
- It did not yield a robust first-moment optimizer controller.
- Freeze it as a caution and boundary condition.

Local-map branch:

- Local path/probe geometry can beat server-side temporal secants on synthetic
  beds.
- The local-SGD K-step dynamics matter. A fresh inverse-Hessian correction
  double-applies curvature; the corrected operator is:

```text
[I - (I - eta Hhat)^K]^{-1} Delta_K
```

- Random probes plus residual shrinkage are live on CPU synthetic local-map
  beds.
- Dense low-rank-plus-diagonal geometry has headroom but is fragile.
- Same-round validation can make dense corrections safe on synthetic beds, but
  E026 says delayed gamma does not replicate as a reusable cross-round policy.

Global tagged-query branch:

- E027: one-step greedy global query policy lost to a fixed probe schedule.
  This is a policy failure, not a no-headroom result.
- E028: tagged heterogeneous async worker evidence is a clean positive.
  Tagged posterior tied homogeneous averaging when workers were homogeneous and
  beat equal averaging strongly under heterogeneous noise, bias, latency, and
  cost.

GPU/real-path branch:

- E029/E030/E031 were queued after the 2026-10-04 reset.
- E031 first failed because the worker lacked the checkout/Mr. Gitties; it was
  requeued with source-bundle bootstrap.
- Harvest or requeue through the Buck2 GPU path before drawing conclusions.

## How To Judge Experiments

Do not use "lost to an oracle tuned schedule" as an automatic death sentence.
That criterion would kill useful methods such as DiLoCo-style systems if applied
blindly.

Use oracle schedules as diagnostics:

- Is there headroom?
- Is the measured signal only rediscovering a per-regime scalar?
- Does the method reduce tuning or transfer across regimes?

Fair baselines:

- FedAvg / local SGD with tuned scalar scale;
- SlowMo / BMUF;
- DiLoCo-style outer optimization;
- FedAvgM / FedAdam / FedOpt;
- SCAFFOLD / FedProx when heterogeneity is active;
- gradient tracking, gossip, and EASGD variants;
- server-side temporal secants / quasi-Newton controls;
- best-worker or elitist fallback.

A useful result can be:

- matched-cost improvement;
- tuning reduction;
- transfer across K, N, heterogeneity, or workload;
- graceful collapse to averaging in the homogeneous case;
- a controlled negative that identifies a real impossibility.

Diagnostics without action are not enough.

## Next Actions

1. Scan workers and harvest E029/E030/E031 if available.
2. Requeue failed GPU work with the source bundle and Buck2 GPU instructions.
3. Expand E028's tagged posterior positive into stronger heterogeneous/async
   controls against DiLoCo, SlowMo/BMUF, FedOpt, and best-worker fallback.
4. For local-map experiments, stop polishing gamma if a hardcoded constant ties
   it. Work on representation, trust layer, and real-path transfer.
5. Keep the paper claim positive and stable: workers are global query processes;
   averaging is a special case, not the enemy.
