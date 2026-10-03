# Local GP GPU Worker Manifest - 2026-10-03

Manifold directory:

```text
aai_research_tlv/tree/local_gp/gpu_pings/20261003/
```

Status checked 2026-10-03 16:00 PDT: the GPU liveness deadline passed with no
E018/E019 result artifacts visible in the GPU dropbox. The prefix contained
only the E018/E019 prompts, source bundles, and manifest. Treat the GPU lane as
requiring worker escalation or requeue through the Buck2 `keep_gpu_sections`
path. Do not adjudicate GPU results until the worker uploads result notes and
run logs.

Liveness note:

```text
experiments/lgp_e018_e019_gpu_liveness_20261003.md
```

## GPU-1

Run E018:

```text
experiments/lgp_e018_gpu_worker_prompt_20261003.md
```

Purpose: real-path readiness smoke for the corrected dynamics-aware local-map
operator. This remains a plumbing gate, not a product claim; the CPU synthetic
mechanism is already positive after E016/E017/E022/E023, but real-path survival
is still unadjudicated.

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

## Liveness Escalation

The 2026-10-03 16:00 PDT liveness check found this exact GPU Manifold state:

```text
        3915 lgp_e018_gpu_worker_prompt_20261003.md
        4834 lgp_e019_gpu_worker_prompt_20261003.md
      104736 local_gp_e018_source_20261003.tgz
      116159 local_gp_e019_source_20261003.tgz
        1987 manifest.md
```

No E018/E019 result note, script, or run log was present. Requeue or escalate
with:

```text
gpu_python_binary(..., keep_gpu_sections = True)
```

or:

```bash
buck2 run -c fbcode.enable_gpu_sections=true ...
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
