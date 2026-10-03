# LGP-E018/E019 GPU Liveness Note - 2026-10-03

Decision: `GPU_LIVENESS_ESCALATE_OR_REQUEUE`

Final liveness check: 2026-10-03 16:00 PDT.

The GPU dropbox was reachable at the liveness deadline, but no E018/E019
result note, script, or run log was visible. Do not adjudicate the real-path
GPU lane from absence. The lane now requires worker escalation or requeue
through the Buck2 GPU path with `keep_gpu_sections = True`.

## GPU Manifold State

Command:

```bash
ROSETTA_DISABLE_AOT=1 timeout 120s manifold ls aai_research_tlv/tree/local_gp/gpu_pings/20261003/
```

Result:

```text
        3915 lgp_e018_gpu_worker_prompt_20261003.md
        4834 lgp_e019_gpu_worker_prompt_20261003.md
      104736 local_gp_e018_source_20261003.tgz
      116159 local_gp_e019_source_20261003.tgz
        1987 manifest.md
```

Missing expected GPU artifacts:

```text
experiments/lgp_e018_real_path_dynamics_smoke.py
experiments/lgp_e018_real_path_dynamics_smoke_result_20261003.md
experiments/lgp_e018_real_path_dynamics_smoke_run_20261003.txt
experiments/lgp_e019_real_path_risk_shrinkage.py
experiments/lgp_e019_real_path_risk_shrinkage_result_20261003.md
experiments/lgp_e019_real_path_risk_shrinkage_run_20261003.txt
```

## CPU Manifold State

CPU was also checked first, per the overnight agenda. It was reachable and
still had no E025/E026 result note, script, or run log.

Command:

```bash
ROSETTA_DISABLE_AOT=1 timeout 120s manifold ls aai_research_tlv/tree/local_gp/cpu_pings/20261003/
```

Result:

```text
        3488 CPU_1_NEXT.md
        2761 CPU_2_NEXT.md
         438 lgp_cpu1_hold_after_e022_20261003.md
         439 lgp_cpu2_hold_after_e023_20261003.md
       10647 lgp_cpu_worker_manifest_20261003.md
        2769 lgp_e016_cpu_worker_prompt_20261003.md
       16127 lgp_e016_rotated_nonlinear_probe.py
        3259 lgp_e016_rotated_nonlinear_probe_result_20261003.md
        1018 lgp_e016_rotated_nonlinear_probe_run_20261003.txt
        2918 lgp_e017_cpu_worker_prompt_20261003.md
       10066 lgp_e017_risk_shrinkage_fallback.py
        3323 lgp_e017_risk_shrinkage_fallback_result_20261003.md
        3284 lgp_e020_cpu_worker_prompt_20261003.md
       18911 lgp_e020_probe_budget_active_design.py
        9382 lgp_e020_probe_budget_active_design_result_20261003.md
        6393 lgp_e020_probe_budget_active_design_run_20261003.txt
        3217 lgp_e021_cpu_worker_prompt_20261003.md
       22586 lgp_e021_dense_lowrank_correlated_stress.py
        5308 lgp_e021_dense_lowrank_correlated_stress_result_20261003.md
        1337 lgp_e021_dense_lowrank_correlated_stress_run_20261003.txt
        4520 lgp_e022_cpu_worker_prompt_20261003.md
       15074 lgp_e022_dense_trust_validation.py
        5552 lgp_e022_dense_trust_validation_result_20261003.md
        1908 lgp_e022_dense_trust_validation_run_20261003.txt
        3222 lgp_e023_cpu_worker_prompt_20261003.md
       16401 lgp_e023_random_probe_budget_confirm.py
        8588 lgp_e023_random_probe_budget_confirm_result_20261003.md
        5168 lgp_e023_random_probe_budget_confirm_run_20261003.txt
        8196 lgp_e024_delayed_trust_validation.py
        3798 lgp_e024_delayed_trust_validation_result_20261003.md
        1451 lgp_e024_delayed_trust_validation_run_20261003.txt
        3488 lgp_e025_validation_cost_pricing_prompt_20261003.md
        2761 lgp_e026_delayed_gamma_replication_prompt_20261003.md
      119490 local_gp_cpu_source_20261003.tgz
      145237 local_gp_cpu_source_20261003_e022_e023.tgz
      175602 local_gp_cpu_source_20261003_e024_e026.tgz
      175602 local_gp_cpu_source_latest.tgz
       10647 manifest.md
        4757 wanger_e022_verdict.md
```

## Next Action

Stop one-minute GPU heartbeat polling unless a worker signal appears. Requeue
or escalate E018/E019 with explicit Buck2 GPU instructions:

```text
gpu_python_binary(..., keep_gpu_sections = True)
```

or:

```bash
buck2 run -c fbcode.enable_gpu_sections=true ...
```

If the worker reports `NO_GPU_HARNESS`, preserve exact CUDA/Buck2 diagnostics.
CPU E025/E026 remain active and should still be checked first.
