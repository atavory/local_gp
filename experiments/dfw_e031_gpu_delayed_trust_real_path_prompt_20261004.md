# DFW-E031 GPU Prompt - Real-Path Delayed Trust Gate

## Mission

This is a Beyond Averaging / local_gp experiment queued through the `double_fan`
project-worker channel.

Synthetic E026 says same-round validation can select dense corrections, but the
selected gamma does not persist cleanly to the next round. Test whether the same
selection-vs-policy split appears on a real local-SGD path.

## Repository

Use `/data/users/atavory/local_gp` if present. If not present, clone
`https://github.com/atavory/local_gp` through Mr. Gitties. Do not use `pip`.

Read:

```text
PROJECT.md
algorithm_stack.md
experiments/lgp_e022_dense_trust_validation_result_20261003.md
experiments/lgp_e024_delayed_trust_validation_result_20261003.md
experiments/lgp_e026_delayed_gamma_replication_result_20261003.md
```

## Required Work

Create and run:

```text
experiments/dfw_e031_gpu_delayed_trust_real_path.py
```

Write:

```text
experiments/dfw_e031_gpu_delayed_trust_real_path_result_20261004.md
experiments/dfw_e031_gpu_delayed_trust_real_path_run_20261004.txt
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

On a small real local-SGD workload, construct a finite grid of trust/shrinkage
values for a local-map or dense/path correction. Score:

- `scalar`;
- `same_round_validation_gamma`;
- `lag1_gamma`;
- `lag2_gamma`;
- `shuffled_gamma`;
- `zero_or_floor_gamma`;
- `oracle_same_round` diagnostic.

Use paired rounds/seeds. If full training is too expensive, do offline one-step
validation on logged local paths.

## Metrics

Report:

- same-round minus scalar;
- lag1 minus scalar;
- lag1 minus shuffled;
- same-round minus shuffled;
- gamma autocorrelation;
- mean absolute lag difference;
- validation/probe cost.

## Decision Labels

Choose exactly one:

- `REAL_DELAYED_TRUST_LIVE`: lag1 or lag2 beats scalar and shuffled with paired
  CI excluding zero.
- `REAL_SAME_ROUND_SELECTION_ONLY`: same-round beats scalar/shuffled, but lagged
  trust does not.
- `REAL_TRUST_NO_SIGNAL`: neither same-round nor delayed beats scalar.
- `NO_GPU_HARNESS`: GPU/Buck2/PyTorch harness cannot run; include diagnostics.
- `INVALID`: missing scalar/shuffled controls, leakage, or run failure.

## Reporting Rules

This experiment separates a same-round safety selector from a reusable optimizer
policy. Keep that distinction explicit.
