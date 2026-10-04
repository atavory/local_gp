# Local GP Progress Snapshot - 2026-10-03

This note is the shared read-ahead for ushmoo, wanger, Merkle, and future
workers. It is intentionally shorter than the full control board.

## Where To Read

Repository:

```text
https://github.com/atavory/local_gp
```

Main local files:

```text
README.md
research_control_board.md
model.md
algorithm_stack.md
event_triggered_multisecant_plan.md
double_fan_freeze_20261003.md
local_gp_progress_20261003.md
```

Manifold CPU dropbox:

```text
aai_research_tlv/tree/local_gp/cpu_pings/20261003/
```

Useful dispatch aliases in that dropbox:

```text
manifest.md
lgp_cpu_worker_manifest_20261003.md
CPU_1_NEXT.md
CPU_2_NEXT.md
local_gp_cpu_source_latest.tgz
```

GPU dropbox:

```text
aai_research_tlv/tree/local_gp/gpu_pings/20261003/
```

Useful dispatch aliases in that dropbox after the GPU requeue:

```text
manifest.md
lgp_gpu_worker_manifest_20261003.md
GPU_1_NEXT.md
GPU_2_NEXT.md
local_gp_gpu_requeue_source_20261003.tgz
```

## Reset

Double Fan / same-point cloud momentum is frozen as a negative optimizer line.
The freeze note is `double_fan_freeze_20261003.md`.

The lesson we keep is:

```text
alignment estimates noise; misalignment can estimate curvature.
```

Same-point worker variance is a real measurement, but it did not produce a
first-moment optimizer that survived tuned scalar, schedule, prior-only, or
frozen-R controls.

## New Thesis

The live project is not "GP over parameters" and not Optuna/TPE over a sync
schedule.

The live object is a local SGD event:

```text
workers/probes produce gradient pairs near x_t
  -> estimate a local K-step update map
  -> apply the dynamics-aware correction
  -> accept/damp/reject through residual or held-out trust
```

For K-step local GD on a quadratic,

```text
Delta_K = -H^-1 [I - (I - eta H)^K] g_0
```

so the correction operator is

```text
[I - (I - eta Hhat)^K]^-1 Delta_K
```

not a fresh Newton, BB, or L-BFGS step. Plain quasi-Newton consolidation
double-applies curvature already used by the local steps.

## What Looks Positive

E014:

- corrected dynamics-aware operator wins on an exact rotated-block quadratic;
- random/directed probes beat tuned scalar FedAvg scale.

E015:

- corrected probe-fed operator survives a nonlinear/heterogeneous CPU bed;
- random probe beats tuned scalar at N=8/32/128;
- path-only worker scatter fails.

E016:

- rotated nonlinear blocks do not kill the mechanism;
- random/directed probes again beat tuned scalar;
- best-worker fallback and server/time secants do not explain the gain.

E017:

- split-probe residual/risk shrinkage works on the current CPU bed;
- it replaces arbitrary target gamma tuning well enough to keep the mechanism
  alive;
- full gamma is destructive at small N, so shrinkage is doing real work.

Together: the CPU mechanism is real on controlled beds.

## What Looks Bad Or Unresolved

E020:

- active design v1 did not robustly reduce probe budget;
- full random probes remain strong;
- E023 later confirmed random probes around `2N` when `N >= 32`, with full
  `N*K` still likely needed for very small N.

E023:

- returned `TWO_N_RANDOM_CONFIRMED`;
- `2N` random probes beat tuned scalar and stayed close to full `N*K` random
  probes for N=32/128;
- `probe_reps=1` is usable when gradient-call budget binds, while
  `probe_reps=4` is a quality knob.

E021:

- dense correlated low-rank geometry breaks the current estimator;
- oracle dense-H correction has large headroom;
- diagonal, sketched low-rank, and combined estimates do not safely recover it;
- learned low-rank-plus-diagonal can be destructive under residual shrinkage.

Interpretation: dense geometry is not hopeless, but the current representation
needs a stricter held-out validation / trust layer before applying the combined
correction.

E022:

- returned `TRUST_LAYER_RECOVERS_DENSE`;
- held-out mini-loss validation-gamma prevents the E021 crash and beats tuned
  scalar at N=8/32/128;
- the recovery is modest versus oracle-H, so dense work now needs better
  subspace/model identification plus validation-budget accounting.

E024:

- returned `DELAYED_GAMMA_WEAK`;
- applying round `t`'s selected gamma to round `t+1` only clearly preserved a
  gain at N=32;
- N=8 was tiny and N=128 was null, so E022 is not yet a settled dense-method
  claim;
- the E022 validation grid costs roughly 17.23 full-forward equivalents, so
  validation cost must be priced against matched compute.

E025:

- returned `VALIDATION_COST_SURVIVES`;
- full 8-rep validation still beats tuned scalar and an `N+1` extra-worker
  scalar rival at N=8/32/128;
- extra fit probes do not explain the gain on the dense bed;
- one or two validation reps already recover most same-round benefit;
- this resolves the same-round cost objection, but E026 is still required
  because E024 weakened delayed/casual persistence.

## Current Worker Tasks

CPU-1:

```text
CPU_1_NEXT.md
experiments/lgp_cpu1_hold_after_e025_20261003.md
```

Status: E025 result landed and was imported. `CPU_1_NEXT.md` should now be a
hold note until the coordinator assigns another CPU-1 task; do not rerun E025
unless asked.

CPU-2:

```text
CPU_2_NEXT.md
experiments/lgp_e026_delayed_gamma_replication_prompt_20261003.md
```

Status: active E026 task. Replicate delayed/stale/shuffled gamma controls with
more held-out power.

GPU:

```text
lgp_e018_gpu_worker_prompt_20261003.md
lgp_e019_gpu_worker_prompt_20261003.md
```

Question: does the corrected operator and risk shrinkage survive real-path GPU
plumbing? The 2026-10-03 16:00 PDT liveness deadline passed with no GPU result
note, script, or run log in the dropbox, so this lane now needs worker
escalation or requeue through the Buck2 `keep_gpu_sections` path. Direct
`GPU_1_NEXT.md` and `GPU_2_NEXT.md` aliases now carry Buck2-specific requeue
instructions.

## What To Audit

For ushmoo:

- Is the operator story right?
- Does E021 kill the representation, the trust layer, or the whole mechanism?
- Does E024's delayed-gamma weakness mean E022 was same-round selection noise,
  or just underpowered causal validation?

For wanger:

- Is the algorithm-stack framing accurate: data-rich gradients, so import
  stochastic trust-region / TuRBO-radius / batching / async machinery, but not
  TPE or qEI over parameters?
- Are the controls still hard enough: tuned scalar FedAvg scale, server/time
  secants, best-worker fallback, hardcoded/full shrinkage?

For Merkle:

- CPU workers should read `CPU_1_NEXT.md` and `CPU_2_NEXT.md` directly from
  Manifold. CPU-1 is hold after E025; CPU-2 remains active on E026.
- Use `local_gp_cpu_source_latest.tgz` unless a dated source bundle is required.
- GPU workers should not be silently polled after the missed liveness deadline;
  they should use `GPU_1_NEXT.md` / `GPU_2_NEXT.md`, run through Buck2 with
  `keep_gpu_sections = True`, and preserve exact CUDA/Buck2 diagnostics if the
  result is `NO_GPU_HARNESS`.

## Current Claim Boundary

Claim we can make:

```text
local path/probe geometry contains usable curvature-action information on
controlled CPU beds, and the K-step dynamics-aware operator is the right
correction target.
```

Claim we cannot make yet:

```text
this is a working optimizer on real workloads.
```

The next decisive result is E026 for dense trust persistence, plus the real-path
GPU gate. E025 priced the same-round validation cost; E024 still says the
dense claim is not settled yet.
