# DFW-E031 GPU Requeue Prompt - Real-Path Delayed Trust Gate

## Mission

This requeues DFW-E031 after a framework/bootstrap failure on `devvm60954`.
The previous attempt did not run science. It failed because the worker lacked
`/data/users/atavory/local_gp` and also lacked Mr. Gitties / agent context for a
GitHub clone.

Use the staged Manifold source bundle instead.

## Source Bundle

Fetch:

```text
aai_research_tlv/tree/double_fan/source/local_gp_source_latest.tgz
```

Unpack it under the local run directory or another writable worker-local
directory, for example:

```bash
mkdir -p "$RUN_DIR/src"
ROSETTA_DISABLE_AOT=1 manifold get --overwrite_local_path \
  aai_research_tlv/tree/double_fan/source/local_gp_source_latest.tgz \
  "$RUN_DIR/local_gp_source_latest.tgz"
tar -xzf "$RUN_DIR/local_gp_source_latest.tgz" -C "$RUN_DIR/src"
cd "$RUN_DIR/src/local_gp"
```

If `~/.agent_context.md`, `~/.agent_context.json`, or Mr. Gitties are missing,
do not fail solely for that reason. They are not needed for this requeue if the
Manifold source bundle fetch works. Do not use direct GitHub/network clone.

## Read First

After unpacking, read:

```text
PROJECT.md
algorithm_stack.md
experiments/lgp_e022_dense_trust_validation_result_20261003.md
experiments/lgp_e024_delayed_trust_validation_result_20261003.md
experiments/lgp_e026_delayed_gamma_replication_result_20261003.md
```

## Required Work

Create and run, inside the unpacked source tree:

```text
experiments/dfw_e031_gpu_delayed_trust_real_path.py
```

Write:

```text
experiments/dfw_e031_gpu_delayed_trust_real_path_result_20261004.md
experiments/dfw_e031_gpu_delayed_trust_real_path_run_20261004.txt
```

Upload the script, result note, run log, and legitimacy check to this run's
Manifold artifact directory.

## GPU / Buck2 Rule

Do not declare CUDA broken from bare `nvidia-smi`. Use the Buck2 GPU path if a
Buck target is needed:

```bash
buck2 run -c fbcode.enable_gpu_sections=true <target> -- <args>
```

or a `gpu_python_binary(..., keep_gpu_sections = True)` target. If no usable
PyTorch/CUDA path exists, return `NO_GPU_HARNESS` with exact diagnostics.

If the source-bundle copy is pure Python and can run a small real-path smoke
without Buck2, that is acceptable. Do not use `pip`.

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
- validation/probe cost;
- whether the result is real GPU, CPU fallback, or `NO_GPU_HARNESS`.

## Decision Labels

Choose exactly one:

- `REAL_DELAYED_TRUST_LIVE`: lag1 or lag2 beats scalar and shuffled with paired
  CI excluding zero.
- `REAL_SAME_ROUND_SELECTION_ONLY`: same-round beats scalar/shuffled, but lagged
  trust does not.
- `REAL_TRUST_NO_SIGNAL`: neither same-round nor delayed beats scalar.
- `NO_GPU_HARNESS`: GPU/Buck2/PyTorch harness cannot run; include diagnostics.
- `SOURCE_BUNDLE_BOOTSTRAP_FAILED`: the Manifold source bundle could not be
  fetched/unpacked or inspected.
- `INVALID`: missing scalar/shuffled controls, leakage, or run failure.

## Reporting Rules

This experiment separates a same-round safety selector from a reusable optimizer
policy. Keep that distinction explicit.
