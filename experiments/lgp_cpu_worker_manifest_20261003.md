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
local_gp_cpu_source_20261003_e024_e026.tgz
local_gp_cpu_source_20261003_e022_e023.tgz
```

The `CPU_1_NEXT.md` and `CPU_2_NEXT.md` aliases are deliberate. Some workers
poll for direct next-task files rather than reading the long manifest.

Status checked 2026-10-03 22:10 PDT: E022 and E023 completed and were imported.
E024 was run locally and returned `DELAYED_GAMMA_WEAK`. E025 landed from CPU-1
and returned `VALIDATION_COST_SURVIVES`. CPU-2/E026 is still pending.

- `CPU_1_NEXT.md`: hold note after E025; do not rerun unless asked.
- `CPU_2_NEXT.md`: E026 delayed-gamma replication with stale/shuffled controls.

Do not rerun E022/E023 unless the coordinator explicitly asks for replication.

## Completed: CPU-1 / E025

Run E025:

```text
experiments/lgp_e025_validation_cost_pricing_prompt_20261003.md
```

Purpose: price E022's held-out mini-loss validation cost and compare the
validation-gamma gain against matched-compute alternatives.

Fetch:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/lgp_e025_validation_cost_pricing_prompt_20261003.md \
  /tmp/lgp_e025_validation_cost_pricing_prompt_20261003.md
```

Expected artifacts:

```text
experiments/lgp_e025_validation_cost_pricing.py
experiments/lgp_e025_validation_cost_pricing_result_20261003.md
experiments/lgp_e025_validation_cost_pricing_run_20261003.txt
```

Decision: `VALIDATION_COST_SURVIVES`.

## Active: CPU-2 / E026

Run E026:

```text
experiments/lgp_e026_delayed_gamma_replication_prompt_20261003.md
```

Purpose: resolve whether E022's same-round validation-gamma survives lagged
application, or whether same-round selection noise explains it.

Fetch:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/CPU_2_NEXT.md \
  /tmp/CPU_2_NEXT.md
```

Expected artifacts:

```text
experiments/lgp_e026_delayed_gamma_replication.py
experiments/lgp_e026_delayed_gamma_replication_result_20261003.md
experiments/lgp_e026_delayed_gamma_replication_run_20261003.txt
```

## Completed: local / E024

Run E024:

```text
experiments/lgp_e024_delayed_trust_validation.py
```

Purpose: cheap delayed-gamma control for E022 winner's-curse risk.

Completed artifacts:

```text
experiments/lgp_e024_delayed_trust_validation.py
experiments/lgp_e024_delayed_trust_validation_result_20261003.md
experiments/lgp_e024_delayed_trust_validation_run_20261003.txt
```

Decision: `DELAYED_GAMMA_WEAK`.

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

`lgp_cpu1_hold_after_e022_20261003.md` is now only an archival hold note. The
current `CPU_1_NEXT.md` alias should point to the E025 hold note; do not treat
the archived E022 hold note as current dispatch.

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

`lgp_cpu2_hold_after_e023_20261003.md` is now only an archival hold note. The
current `CPU_2_NEXT.md` alias points to E026; do not treat the archived hold
note as current dispatch.

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
