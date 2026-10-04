# DFW-E029 GPU Prompt - Real-Path Headroom Gate

## Mission

This is a Beyond Averaging / local_gp experiment queued through the `double_fan`
project-worker channel.

Before testing a complicated controller on a real workload, quantify whether a
real local-SGD path has actionable headroom beyond tuned scalar FedAvg-scale.
If tuned scalar averaging already reaches the oracle over the action family, no
measurement can win.

## Repository

Use `/data/users/atavory/local_gp` if present. If not present, clone
`https://github.com/atavory/local_gp` through Mr. Gitties. Do not use `pip`.

Read:

```text
PROJECT.md
README.md
model.md
experiments/lgp_e018_gpu_worker_prompt_20261003.md
experiments/lgp_e019_gpu_worker_prompt_20261003.md
```

## Required Work

Create and run:

```text
experiments/dfw_e029_gpu_real_path_headroom.py
```

Write:

```text
experiments/dfw_e029_gpu_real_path_headroom_result_20261004.md
experiments/dfw_e029_gpu_real_path_headroom_run_20261004.txt
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

Use the smallest real workload that exercises local-SGD paths on GPU:

- MNIST or CIFAR-10 if already supported by the repo/harness;
- model small enough to finish quickly;
- `N in {8, 16}` virtual workers if practical;
- `K in {4, 8}`;
- paired source/target seeds.

Collect local worker endpoint displacements and enough held-out validation loss
to score one-step consolidation actions offline.

Action family:

- raw FedAvg;
- tuned scalar scale on FedAvg;
- best-worker / elitist fallback;
- server/time secant if cheap;
- dynamics-aware local-map correction using available/probe-fed estimates;
- oracle over the finite action grid, diagnostic only.

## Metrics

Report:

- oracle grid headroom over tuned scalar;
- best non-oracle arm over tuned scalar;
- best-worker over tuned scalar;
- whether local-map correction has room to matter;
- paired confidence intervals over seeds/rounds.

## Decision Labels

Choose exactly one:

- `REAL_PATH_HEADROOM_EXISTS`: oracle grid and at least one non-scalar family
  show headroom beyond tuned scalar.
- `SCALAR_REACHES_ORACLE`: tuned scalar matches oracle grid; no controller
  headroom on this workload.
- `LOCAL_MAP_PROMISING`: local-map correction beats tuned scalar with paired CI
  excluding zero.
- `NO_GPU_HARNESS`: GPU/Buck2/PyTorch harness cannot run; include diagnostics.
- `INVALID`: missing tuned scalar, missing oracle diagnostic, or unpaired seeds.

## Reporting Rules

Do not call a diagnostic difference a win. The main number is headroom over
tuned scalar FedAvg-scale.
