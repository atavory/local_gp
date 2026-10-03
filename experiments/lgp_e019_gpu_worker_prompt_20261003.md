# LGP-E019 GPU Worker Ping - Real-Path Risk Shrinkage Gate

You are running the second GPU-side gate for `local_gp`.

GPU-1 / E018 is a plumbing smoke: can the corrected operator be evaluated on a
real local-SGD path? E019 is the complementary real-path method gate:

> Does risk-shrunk probe-fed dynamics-aware correction beat tuned scalar FedAvg
> scale on a real workload, and how much probe budget is needed?

This is still a pilot, not a final paper claim.

## Source Bundle

Fetch:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/local_gp/gpu_pings/20261003/local_gp_e019_source_20261003.tgz \
  /tmp/local_gp_e019_source_20261003.tgz
mkdir -p /tmp/local_gp_e019
tar -xzf /tmp/local_gp_e019_source_20261003.tgz -C /tmp/local_gp_e019
cd /tmp/local_gp_e019
```

Sanity:

```bash
python3 -m py_compile experiments/lgp_e015_dynamics_aware_nonlinear_probe.py
python3 -m py_compile experiments/lgp_e017_risk_shrinkage_fallback.py
```

## GPU Note

Do not reject the job because bare `nvidia-smi` fails. If you are in fbsource,
use the Buck2 GPU path:

```text
gpu_python_binary(..., keep_gpu_sections = True)
```

or:

```bash
buck2 run -c fbcode.enable_gpu_sections=true ...
```

If ordinary PyTorch CUDA works outside fbsource, that is fine. If no GPU/data
path exists, report `NO_GPU_HARNESS` with exact logs.

## Workload

Use the first available real local-SGD workload:

1. CIFAR-10 tiny CNN or ResNet-small, if data/deps are present;
2. MNIST/Fashion-MNIST tiny MLP or linear classifier;
3. an existing K>1 local-SGD ledger on the machine.

Prefer CIFAR if it is available without data-plumbing heroics. If CIFAR is not
available quickly, use MNIST. Do not spend the run debugging downloads.

Suggested settings:

```text
N = 8 workers
K = 4 and K = 8 if cheap; else K = 8 only
pilot seeds: at least 4
target seeds: at least 8
probe reps: 1, 2, 4
probe budget per round: N, 2N, and N*K if cheap
```

## Required Operator

Do not use a plain Newton / L-BFGS update. Use the K-step local-map correction:

```text
[I - (I - eta Hhat)^K]^{-1} Delta_K
```

For this real-path pilot, use diagonal `Hhat` first:

```text
s_j, y_j = probe offset and same-batch gradient difference
lambda_hat_j = sum(s_j * y_j) / sum(s_j^2)
c_j = 1 / (1 - (1 - eta * lambda_hat_j)^K)
candidate_j = fedavg_j + gamma * (c_j * fedavg_j - fedavg_j)
```

Clip invalid or unstable coefficients conservatively and report fallback rate.

## Probe Design

Use noise-controlled probes around the synchronized state:

- random Rademacher offsets, shared seed;
- same-batch gradient differences if possible;
- split probes into fit/validation halves for residual shrinkage.

If same-batch probing is too expensive, state that and use repeated minibatch
probes, but do not hide the substitution.

## Arms

Required:

- `fedavg_raw`;
- `fedavg_scale`: source-tuned scalar FedAvg scale;
- `probe_tuned_gamma`: corrected probe update with source-tuned gamma;
- `probe_risk_gamma`: corrected probe update with split-probe residual gamma;
- `probe_full_gamma`: gamma=1, mostly a safety/null arm;
- `best_worker_true`: oracle best worker endpoint by held-out/validation loss;
- `server_secant` if cheap.

Optional:

- `path_only`;
- Adam / SGD momentum baseline if already implemented locally.

## Metrics

Primary:

- one-step held-out validation loss from identical synchronized state and
  identical local worker paths.

Secondary:

- accuracy;
- selected scalar scale / tuned gamma;
- mean risk gamma;
- coefficient quantiles and fallback fraction;
- probe residual ratio;
- runtime / extra-gradient cost;
- results by probe budget.

Use paired seed CIs. Do not report only means.

## Decision Labels

Use exactly one:

- `REAL_PATH_RISK_PROBE_WINS`: risk-gamma probe beats tuned scalar scale with
  positive paired CI and is not explained by best-worker fallback.
- `TUNED_GAMMA_ONLY`: tuned-gamma probe wins but risk-gamma does not; shrinkage
  rule is not ready.
- `SCALAR_OR_FALLBACK_EXPLAINS`: tuned scalar scale or best-worker fallback
  matches/beats the probe arms.
- `PROBE_BUDGET_TOO_EXPENSIVE`: method only works at a probe budget too large
  to be plausible; include the budget table.
- `REAL_PATH_PLUMBING_ONLY`: script runs and metrics are sane, but no method
  claim is supported.
- `NO_GPU_HARNESS`: no usable GPU/data path.
- `INVALID`: scoring or implementation bug.

## Required Output

Write:

```text
experiments/lgp_e019_real_path_risk_shrinkage_result_20261003.md
```

Upload the result note and script to:

```text
aai_research_tlv/tree/local_gp/gpu_pings/20261003/
```

The final summary must say:

- workload used;
- whether same-batch probes were actually used;
- best decision label;
- whether risk-gamma beat tuned scalar;
- whether best-worker fallback explains the result;
- probe budget at which any win appears.
