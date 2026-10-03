# Local GP CPU Worker Manifest - 2026-10-03

Manifold directory:

```text
aai_research_tlv/tree/local_gp/cpu_pings/20261003/
```

## CPU-1

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

## CPU-2

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

## Fetch Everything

```bash
ROSETTA_DISABLE_AOT=1 manifold getr \
  aai_research_tlv/tree/local_gp/cpu_pings/20261003 \
  /tmp/local_gp_cpu_pings_20261003
```

Both jobs are CPU-only.
