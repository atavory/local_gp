# Local GP CPU Worker Manifest - 2026-10-03

Manifold directory:

```text
aai_research_tlv/tree/local_gp/cpu_pings/20261003/
```

Dispatch files in that directory:

```text
manifest.md
lgp_cpu_worker_manifest_20261003.md
CPU_1_NEXT.md
CPU_2_NEXT.md
local_gp_cpu_source_latest.tgz
local_gp_cpu_source_20261003_e022_e023.tgz
```

The `CPU_1_NEXT.md` and `CPU_2_NEXT.md` aliases are deliberate. Some workers
poll for direct next-task files rather than reading the long manifest.

Status checked 2026-10-03 13:44 PDT: E022 and E023 completed. E023 artifacts
and CPU-2 hold note were uploaded by the local loop; E022 script, result note,
and run log were visible in the CPU dropbox and imported. CPU aliases should
now be hold notes to avoid duplicate launches.

## Completed: CPU-1 / E022

Run E022:

```text
experiments/lgp_e022_cpu_worker_prompt_20261003.md
```

Purpose: dense trust-layer gate. Decide whether held-out validation can prevent
the destructive learned low-rank-plus-diagonal update from E021 and recover any
dense-geometry headroom.

Fetch:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e022_cpu_worker_prompt_20261003.md \
  /tmp/lgp_e022_cpu_worker_prompt_20261003.md
```

Completed artifacts:

```text
experiments/lgp_e022_dense_trust_validation.py
experiments/lgp_e022_dense_trust_validation_result_20261003.md
experiments/lgp_e022_dense_trust_validation_run_20261003.txt
```

Decision: `TRUST_LAYER_RECOVERS_DENSE`.

`CPU_1_NEXT.md` should now be a hold note rather than a runnable prompt, to
avoid duplicate E022 launches while GPU E018/E019 remain pending.

Fetch:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e022_dense_trust_validation.py \
  /tmp/lgp_e022_dense_trust_validation.py

ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e022_dense_trust_validation_result_20261003.md \
  /tmp/lgp_e022_dense_trust_validation_result_20261003.md

ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e022_dense_trust_validation_run_20261003.txt \
  /tmp/lgp_e022_dense_trust_validation_run_20261003.txt
```

## Completed: CPU-2 / E023

Run E023:

```text
experiments/lgp_e023_cpu_worker_prompt_20261003.md
```

Purpose: random-probe budget confirmation. Decide whether `2N` random probes
are enough for `N >= 32`, and whether `probe_reps=1` changes the cost story.

Fetch:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e023_cpu_worker_prompt_20261003.md \
  /tmp/lgp_e023_cpu_worker_prompt_20261003.md
```

Completed artifacts:

```text
experiments/lgp_e023_random_probe_budget_confirm.py
experiments/lgp_e023_random_probe_budget_confirm_result_20261003.md
experiments/lgp_e023_random_probe_budget_confirm_run_20261003.txt
```

Decision: `TWO_N_RANDOM_CONFIRMED`.

`CPU_2_NEXT.md` should now be a hold note rather than a runnable prompt, to
avoid duplicate E023 launches while CPU-1/E022 and the GPU gates are still
pending.

Fetch:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e023_random_probe_budget_confirm.py \
  /tmp/lgp_e023_random_probe_budget_confirm.py

ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e023_random_probe_budget_confirm_result_20261003.md \
  /tmp/lgp_e023_random_probe_budget_confirm_result_20261003.md

ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e023_random_probe_budget_confirm_run_20261003.txt \
  /tmp/lgp_e023_random_probe_budget_confirm_run_20261003.txt
```

## Completed: CPU-1 / E020

Run E020:

```text
experiments/lgp_e020_cpu_worker_prompt_20261003.md
```

Purpose: probe-budget / active-design gate. Decide whether value/uncertainty
probe selection reduces the probe budget versus random probes.

Fetch:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e020_cpu_worker_prompt_20261003.md \
  /tmp/lgp_e020_cpu_worker_prompt_20261003.md
```

Completed artifacts:

```text
experiments/lgp_e020_probe_budget_active_design.py
experiments/lgp_e020_probe_budget_active_design_result_20261003.md
experiments/lgp_e020_probe_budget_active_design_run_20261003.txt
```

Fetch:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e020_probe_budget_active_design.py \
  /tmp/lgp_e020_probe_budget_active_design.py

ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e020_probe_budget_active_design_result_20261003.md \
  /tmp/lgp_e020_probe_budget_active_design_result_20261003.md

ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e020_probe_budget_active_design_run_20261003.txt \
  /tmp/lgp_e020_probe_budget_active_design_run_20261003.txt
```

## Completed: CPU-2 / E021

Run E021:

```text
experiments/lgp_e021_cpu_worker_prompt_20261003.md
```

Purpose: dense / low-rank / correlated-noise stress. Decide whether the next
representation should be diagonal, block, low-rank/sketched, or combined.

Fetch:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e021_cpu_worker_prompt_20261003.md \
  /tmp/lgp_e021_cpu_worker_prompt_20261003.md
```

Completed artifacts:

```text
experiments/lgp_e021_dense_lowrank_correlated_stress.py
experiments/lgp_e021_dense_lowrank_correlated_stress_result_20261003.md
experiments/lgp_e021_dense_lowrank_correlated_stress_run_20261003.txt
```

Fetch:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e021_dense_lowrank_correlated_stress.py \
  /tmp/lgp_e021_dense_lowrank_correlated_stress.py

ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e021_dense_lowrank_correlated_stress_result_20261003.md \
  /tmp/lgp_e021_dense_lowrank_correlated_stress_result_20261003.md

ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e021_dense_lowrank_correlated_stress_run_20261003.txt \
  /tmp/lgp_e021_dense_lowrank_correlated_stress_run_20261003.txt
```

## Source Bundle

Fetch:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/local_gp_cpu_source_20261003_e022_e023.tgz \
  /tmp/local_gp_cpu_source_20261003_e022_e023.tgz
```

## Completed: CPU-1 / E016

Run E016:

```text
experiments/lgp_e016_cpu_worker_prompt_20261003.md
```

Purpose: rotated / dense nonlinear gate. Decide whether E015 was a coordinate
artifact.

Fetch:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e016_cpu_worker_prompt_20261003.md \
  /tmp/lgp_e016_cpu_worker_prompt_20261003.md
```

Completed artifacts:

```text
experiments/lgp_e016_rotated_nonlinear_probe.py
experiments/lgp_e016_rotated_nonlinear_probe_result_20261003.md
experiments/lgp_e016_rotated_nonlinear_probe_run_20261003.txt
```

Fetch:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e016_rotated_nonlinear_probe.py \
  /tmp/lgp_e016_rotated_nonlinear_probe.py

ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e016_rotated_nonlinear_probe_result_20261003.md \
  /tmp/lgp_e016_rotated_nonlinear_probe_result_20261003.md

ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e016_rotated_nonlinear_probe_run_20261003.txt \
  /tmp/lgp_e016_rotated_nonlinear_probe_run_20261003.txt
```

## Completed: CPU-2 / E017

Run E017:

```text
experiments/lgp_e017_cpu_worker_prompt_20261003.md
```

Purpose: risk-aware shrinkage and best-worker fallback. Decide whether tuned
gamma or the fallback explains the probe gain.

Fetch:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e017_cpu_worker_prompt_20261003.md \
  /tmp/lgp_e017_cpu_worker_prompt_20261003.md
```

Completed artifacts:

```text
experiments/lgp_e017_risk_shrinkage_fallback.py
experiments/lgp_e017_risk_shrinkage_fallback_result_20261003.md
```

Fetch:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e017_risk_shrinkage_fallback.py \
  /tmp/lgp_e017_risk_shrinkage_fallback.py

ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e017_risk_shrinkage_fallback_result_20261003.md \
  /tmp/lgp_e017_risk_shrinkage_fallback_result_20261003.md
```

## Fetch Everything

```bash
ROSETTA_DISABLE_AOT=1 manifold getr \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003 \
  /tmp/local_gp_cpu_pings_20261003
```

Both jobs are CPU-only.
