# Local GP CPU Worker Manifest - 2026-10-03

Manifold directory:

```text
aai_research_tlv/tree/local_gp/cpu_pings/20261003/
```

## CPU-1

Next run E020:

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

## CPU-2

Next run E021:

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
  aai_research_tlv/tree/local_gp/cpu_pings/20261003/local_gp_cpu_source_20261003.tgz \
  /tmp/local_gp_cpu_source_20261003.tgz
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
