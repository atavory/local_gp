# Local GP GPU Worker Manifest - 2026-10-03

Manifold directory:

```text
aai_research_tlv/tree/local_gp/gpu_pings/20261003/
```

## GPU-1

Run E018:

```text
experiments/lgp_e018_gpu_worker_prompt_20261003.md
```

Purpose: real-path readiness smoke for the corrected dynamics-aware local-map
operator. This is not a product claim while E016/E017 are still running on CPU.

Fetch prompt:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/gpu_pings/20261003/lgp_e018_gpu_worker_prompt_20261003.md \
  /tmp/lgp_e018_gpu_worker_prompt_20261003.md
```

Fetch source bundle:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/gpu_pings/20261003/local_gp_e018_source_20261003.tgz \
  /tmp/local_gp_e018_source_20261003.tgz
```

Fetch everything:

```bash
ROSETTA_DISABLE_AOT=1 manifold getr \
  aai_research_tlv/tree/local_gp/gpu_pings/20261003 \
  /tmp/local_gp_gpu_pings_20261003
```

## GPU-2

Run E019:

```text
experiments/lgp_e019_gpu_worker_prompt_20261003.md
```

Purpose: real-path risk-shrinkage gate. Test whether residual-shrunk
probe-fed dynamics-aware correction beats tuned scalar FedAvg scale on a real
workload, and how much probe budget is needed.

Fetch prompt:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/gpu_pings/20261003/lgp_e019_gpu_worker_prompt_20261003.md \
  /tmp/lgp_e019_gpu_worker_prompt_20261003.md
```

Fetch source bundle:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/gpu_pings/20261003/local_gp_e019_source_20261003.tgz \
  /tmp/local_gp_e019_source_20261003.tgz
```
