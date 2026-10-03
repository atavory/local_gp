# LGP-E018 GPU Requeue Ping - Buck2 Real-Path Smoke

Status: `GPU_REQUEUE_REQUESTED`

This requeues E018 after the 2026-10-03 16:00 PDT liveness deadline passed with
no GPU result note, script, or run log in the dropbox. This is still a
readiness smoke, not a product claim.

## Required First Reads

Read:

```text
experiments/lgp_e018_gpu_worker_prompt_20261003.md
experiments/lgp_e018_e019_gpu_liveness_20261003.md
ops/devvm23211_gpu_buck2_result_20261003.md
```

The original E018 prompt defines the mechanism, arms, metrics, decision labels,
and expected result note. This requeue adds the operational requirement: use
the Buck2 GPU-section path before declaring the GPU harness unavailable.

## Fetch

Use the refreshed requeue source bundle when available:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/gpu_pings/20261003/local_gp_gpu_requeue_source_20261003.tgz \
  /tmp/local_gp_gpu_requeue_source_20261003.tgz
mkdir -p /tmp/local_gp_gpu_requeue
tar -xzf /tmp/local_gp_gpu_requeue_source_20261003.tgz -C /tmp/local_gp_gpu_requeue
cd /tmp/local_gp_gpu_requeue
```

If the refreshed bundle is not present, fall back to
`local_gp_e018_source_20261003.tgz`.

## GPU Harness Requirement

Do not use bare `nvidia-smi` as the availability gate. On `devvm23211`, CUDA was
confirmed through a Buck2 target using:

```python
load("@fbcode_macros//build_defs:gpu_python_binary.bzl", "gpu_python_binary")

gpu_python_binary(
    name = "run",
    srcs = ["run.py"],
    keep_gpu_sections = True,
    main_function = ".run.main",
    deps = ["fbsource//third-party/pypi/torch:torch"],
)
```

Run with either:

```bash
buck2 run fbcode//aai/<package>:run -- <task arguments>
```

or:

```bash
buck2 run -c fbcode.enable_gpu_sections=true fbcode//aai/<package>:<target> -- <task arguments>
```

If no usable Buck2/PyTorch CUDA path exists, return `NO_GPU_HARNESS` and include
the exact Buck2 target, command, stdout, stderr, CUDA/PyTorch diagnostics, and
why ordinary PyTorch CUDA or dataset fallback could not be used.

## Mission

Create and run:

```text
experiments/lgp_e018_real_path_dynamics_smoke.py
```

Evaluate the corrected local-map operator:

```text
[I - (I - eta Hhat)^K]^{-1} Delta_K
```

Use diagonal `Hhat` from noise-controlled probe pairs around the synchronized
state, compare against the strong scalar/best-worker controls from the original
prompt, and keep this as a plumbing/pilot gate.

## Required Output

Write:

```text
experiments/lgp_e018_real_path_dynamics_smoke_result_20261003.md
experiments/lgp_e018_real_path_dynamics_smoke_run_20261003.txt
```

Upload the result note, script, and run log to:

```text
aai_research_tlv/tree/local_gp/gpu_pings/20261003/
```

Use exactly one E018 decision label from the original prompt. Do not adjudicate
the GPU lane from missing artifacts or from a bare `nvidia-smi` failure.
