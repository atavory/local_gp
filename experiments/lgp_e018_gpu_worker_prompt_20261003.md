# LGP-E018 GPU Worker Ping - Real-Path Dynamics-Aware Smoke

You are running the GPU-side smoke for `local_gp`. This is not a product claim.
CPU-1/CPU-2 are deciding the synthetic method gates. Your job is to make sure
the corrected operator can be evaluated on a real local-SGD path as soon as the
CPU gates return.

## Source Bundle

Fetch:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/gpu_pings/20261003/local_gp_e018_source_20261003.tgz \
  /tmp/local_gp_e018_source_20261003.tgz
mkdir -p /tmp/local_gp_e018
tar -xzf /tmp/local_gp_e018_source_20261003.tgz -C /tmp/local_gp_e018
cd /tmp/local_gp_e018
```

Sanity:

```bash
python3 -m py_compile experiments/lgp_e014_dynamics_aware_correction.py
python3 -m py_compile experiments/lgp_e015_dynamics_aware_nonlinear_probe.py
```

## GPU Note

Do not use bare `nvidia-smi` as the CUDA availability gate. On Meta machines,
use the Buck2 GPU path if needed:

```text
gpu_python_binary(..., keep_gpu_sections = True)
```

or:

```bash
buck2 run -c fbcode.enable_gpu_sections=true ...
```

If you are outside fbsource and ordinary PyTorch CUDA works, that is fine too.
If no usable GPU path exists, report `NO_GPU_HARNESS` with exact logs.

## Mission

Build and run:

```text
experiments/lgp_e018_real_path_dynamics_smoke.py
```

Use a small real local-SGD workload. Preferred:

1. MNIST / Fashion-MNIST linear or tiny MLP, if data/deps are present;
2. CIFAR-10 tiny CNN, if already available;
3. an existing local-SGD ledger from the machine.

Do not spend time on data plumbing heroics. This is a smoke/pilot.

## Required Mechanism

Evaluate the corrected local-map operator, not plain L-BFGS:

```text
[I - (I - eta Hhat)^K]^{-1} Delta_K
```

For real paths, use a diagonal `Hhat` first. Estimate it from noise-controlled
probe pairs around the synchronized state:

```text
s = probe offset
y = grad(x + s; probe_batch) - grad(x; probe_batch)
lambda_hat_j = sum(s_j * y_j) / sum(s_j^2)
```

Then:

```text
c_j = 1 / (1 - (1 - eta * lambda_hat_j)^K)
candidate_update_j = fedavg_j + gamma * (c_j * fedavg_j - fedavg_j)
```

Clip invalid or unstable coefficients conservatively. Log fallback fractions.

## Required Arms

Use pilot seeds/rounds to select scalar constants, disjoint target seeds/rounds
for scoring.

- `fedavg_raw`;
- `fedavg_scale`: source-tuned scalar scale;
- `probe_tuned_gamma`: dynamics-aware probe correction with source-tuned gamma;
- `probe_full_gamma`: same with gamma=1;
- `probe_risk_gamma`: split-probe residual gamma if cheap;
- `path_only`: centered local trajectory estimate if cheap;
- `server_secant`: time/server secant if cheap;
- `best_worker_true`: oracle best worker endpoint by held-out/validation loss.

The strong controls are `fedavg_scale` and `best_worker_true`.

## Metrics

Primary: one-step held-out validation loss from the identical synchronized state
and identical local worker paths.

Secondary:

- accuracy;
- selected scale/gamma;
- coefficient distribution and fallback fraction;
- probe residual / risk gamma;
- wall-clock notes.

Use paired seed CIs. Do not report only means.

## Decision Labels

Use exactly one:

- `REAL_PATH_PLUMBING_VALID`: script runs, metrics are sane, no method claim.
- `PROBE_BEATS_SCALE_ON_REAL_PATH_PILOT`: probe correction beats tuned scalar
  scale with positive paired CI. This is still pilot-only.
- `SCALAR_OR_FALLBACK_EXPLAINS`: tuned scalar or best-worker fallback explains
  the gain.
- `NO_GPU_HARNESS`: no usable GPU/data path.
- `INVALID`: scoring or implementation bug.

## Required Output

Write a concise result note:

```text
experiments/lgp_e018_real_path_dynamics_smoke_result_20261003.md
```

Upload the result note to:

```text
aai_research_tlv/tree/local_gp/gpu_pings/20261003/
```

This is a readiness/pilot gate. Do not present it as the paper result unless it
beats both strong controls and CPU-1/CPU-2 also come back clean.
