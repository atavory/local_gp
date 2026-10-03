# LGP-E004: Real-Path Optimizer Gate

You are an Assaf worker. This is project `local_gp`.

GitHub repository:

```text
https://github.com/atavory/local_gp
```

Use the repository's current `main`. The optimizer artifact to test is:

```text
local_gp.optimizer.LocalGPAggregator
```

## Purpose

Test whether the concrete local-GP aggregator built in E003 has value on a
fresh real local-SGD path ledger.

This is still a gate, not a broad closed-loop product claim.

## GPU Note

Do not use bare `nvidia-smi` as the availability gate. On `devvm23211`, GPU
execution was confirmed through Buck2 targets with GPU sections retained.

Use either:

```python
gpu_python_binary(..., keep_gpu_sections = True)
```

or:

```bash
buck2 run -c fbcode.enable_gpu_sections=true fbcode//aai/<package>:<target> -- <args>
```

If using CPU because the task is tiny, state that choice. If using CUDA, use the
Buck2 GPU-section path above.

## Data

Prefer CIFAR-10 ResNet if the existing harness can cheaply log full local paths.
Otherwise use MNIST/MLP. The requirement is full local trajectory logging, not
task glamour.

Required:

- K in `{4, 8}`;
- at least N=16 virtual workers if cheap;
- per-worker pre-step displacement vectors for every local step;
- per-worker gradients at those displacements;
- endpoint displacements;
- audit gradient or one-step validation proxy;
- source pilot split and held-out target split.

## Arms

1. `fedavg_source_tuned_scale`: FedAvg endpoint update with source-selected
   scalar scale.
2. `fedavg_target_oracle_scale`: diagnostic only, target-tuned scalar scale.
3. `local_gp_source_gamma`: `LocalGPAggregator` with gamma selected on source
   pilot only.
4. `local_gp_target_oracle_gamma`: diagnostic only.
5. `server_secant`: if cheap, free classical server-side secant/quasi-Newton
   control.
6. `shuffled_paths`: shuffled path/gradient pairing control if cheap.

## Primary Score

Use one-step gain or one-step validation proxy on held-out target seeds/rounds.
Report paired 95% CIs.

Primary fair comparison:

```text
local_gp_source_gamma - fedavg_source_tuned_scale
```

Required diagnostics:

- target oracle headroom;
- local GP versus target-oracle gamma;
- local GP versus server secant, if run;
- local GP versus shuffled paths, if run;
- selected gamma;
- mean coefficient, RMS coefficient, fallback fraction.

## Decision Labels

- `LOCAL_GP_REAL_PATH_LIVE`: source-gamma local GP beats source-tuned FedAvg on
  held-out target with CI excluding zero.
- `TUNED_GAMMA_ONLY`: target-oracle gamma helps but source-gamma does not.
- `FEDAVG_EXPLAINS`: source-tuned FedAvg matches or beats local GP.
- `NO_ORACLE_HEADROOM`: target oracle barely beats source-tuned FedAvg.
- `SECANT_EXPLAINS`: server secant matches or beats local GP.
- `SHUFFLE_EXPLAINS`: shuffled paths match local GP.
- `INVALID`: missing full path logs, target leakage, missing held-out target,
  missing source-only gamma selection, or unfair scale tuning.

## Required Outputs

- `lgp_e004_summary.md`
- `lgp_e004_decision.md`
- `lgp_e004_metrics.csv`
- `lgp_e004_selection.json`
- `lgp_e004_device_report.json`
- `lgp_e004_legitimacy_check.md`
- `final_message.txt`
