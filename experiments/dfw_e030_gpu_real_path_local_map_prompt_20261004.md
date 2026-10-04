# DFW-E030 GPU Prompt - Real-Path Dynamics-Aware Local Map

## Mission

This is a Beyond Averaging / local_gp experiment queued through the `double_fan`
project-worker channel.

Test the corrected local-map operator on a real local-SGD path:

```text
[I - (I - eta Hhat)^K]^{-1} Delta_K
```

The point is not to run another Newton/L-BFGS consolidation. The workers have
already taken K local steps; the correction must respect the K-step dynamics.

## Repository

Use `/data/users/atavory/local_gp` if present. If not present, clone
`https://github.com/atavory/local_gp` through Mr. Gitties. Do not use `pip`.

Read:

```text
PROJECT.md
model.md
algorithm_stack.md
experiments/lgp_e014_dynamics_aware_correction_result_20261003.md
experiments/lgp_e015_dynamics_aware_nonlinear_probe_result_20261003.md
experiments/lgp_e016_rotated_nonlinear_probe_result_20261003.md
experiments/lgp_e017_risk_shrinkage_fallback_result_20261003.md
```

## Required Work

Create and run:

```text
experiments/dfw_e030_gpu_real_path_local_map.py
```

Write:

```text
experiments/dfw_e030_gpu_real_path_local_map_result_20261004.md
experiments/dfw_e030_gpu_real_path_local_map_run_20261004.txt
```

Also upload the script, result note, and run log to this run's Manifold artifact
directory.

## GPU / Buck2 Rule

Do not declare CUDA broken from bare `nvidia-smi`. Use the Buck2 GPU path if a
Buck target is needed:

```bash
buck2 run -c fbcode.enable_gpu_sections=true <target> -- <args>
```

or a `gpu_python_binary(..., keep_gpu_sections = True)` target. If no usable
PyTorch/CUDA path exists, return `NO_GPU_HARNESS` with exact diagnostics.

## Experiment

Use a small real network workload. Reuse the E018/E019 harness if possible.

Compare offline one-step validation improvements for:

- `fedavg`;
- `tuned_scalar_fedavg`;
- `server_time_secant`;
- `best_worker`;
- `local_map_unshrunk`;
- `local_map_residual_shrunk`;
- `local_map_validation_shrunk` if validation is cheap;
- `oracle_grid` diagnostic.

Probe budget:

- start with `2N` random/seeded probes if the harness supports explicit probes;
- otherwise use path-derived pairs and report that limitation;
- if runtime permits, include full `N*K` probes as an upper-bound diagnostic.

## Metrics

Report:

- local-map arm minus tuned scalar;
- local-map arm minus server/time secant;
- residual/validation shrinkage chosen;
- oracle headroom;
- cost accounting for probes/validation in gradient-call equivalents.

## Decision Labels

Choose exactly one:

- `REAL_LOCAL_MAP_LIVE`: residual- or validation-shrunk local map beats tuned
  scalar and server/time secant with paired CI excluding zero.
- `SCALAR_EXPLAINS_REAL_PATH`: tuned scalar matches/beats local-map arms.
- `SERVER_SECANT_EXPLAINS_REAL_PATH`: server/time secant matches/beats local
  map at lower cost.
- `PROBE_COST_UNDERWATER`: local map improves quality but loses matched-cost
  accounting.
- `NO_GPU_HARNESS`: GPU/Buck2/PyTorch harness cannot run; include diagnostics.
- `INVALID`: missing tuned scalar or server/time control, leakage, or run
  failure.

## Reporting Rules

If the method only works unshrunk or only with same-round validation, call that
out. Do not hide the cost.
