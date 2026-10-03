# LGP-E007 Real-Path Auto-Gamma Gate

## Purpose

E004 was invalid because the worker could not access the repository / Mr.
Gitties path. This task is self-contained: use the Manifold source bundle below.

The research question is now narrow:

> Does scalar source-free auto-gamma reduce tuning cost on real local-SGD paths?

Do not attempt a full GP over parameters. The MVP is:

1. estimate the local-SGD correction shape from worker paths;
2. choose one scalar shrinkage gamma from split-half reliability;
3. apply the corrected FedAvg update.

## Source Bundle

Fetch and unpack:

```bash
ROSETTA_DISABLE_AOT=1 manifold get \
  aai_research_tlv/tree/double_fan/runs/local_gp/e007/local_gp_e007_bundle_20261003.tgz \
  /tmp/local_gp_e007_bundle_20261003.tgz
mkdir -p /tmp/local_gp_e007
tar -xzf /tmp/local_gp_e007_bundle_20261003.tgz -C /tmp/local_gp_e007
cd /tmp/local_gp_e007
```

First run the sanity checks:

```bash
python3 experiments/lgp_e005_auto_gamma_gate.py
python3 experiments/lgp_e006_group_gamma_gate.py
```

If these fail, stop and report `INVALID_BUNDLE`.

## Workload

Use a real local-SGD path source if available:

- first preference: an existing K>1 MNIST/CIFAR/local-SGD harness or ledger from
  the previous Double Fan experiments;
- second preference: a fresh small PyTorch MNIST/CIFAR/vision run if the data and
  deps are already present;
- last resort: report `NO_REAL_PATH_HARNESS` with exact missing dependency, but
  do not call this a method failure.

Do not require GitHub, Overleaf, or Mr. Gitties. The code to test is already in
the bundle.

GPU note: do not use bare `nvidia-smi` as an availability gate. If using Buck2,
use a `gpu_python_binary(..., keep_gpu_sections=True)` or
`buck2 run -c fbcode.enable_gpu_sections=true ...`.

## Required Arms

Use source/pilot seeds/rounds to select any tuned constants; use disjoint target
seeds/rounds for scoring.

1. `fedavg_scale`: FedAvg with source-tuned scalar scale.
2. `tuned_gamma`: `LocalGPAggregator` with gamma selected on source/pilot.
3. `auto_gamma`: `LocalGPAggregator(LocalGPConfig(gamma=None))`.
4. `full_gamma`: same aggregator with `gamma=1.0`.

Group-wise gamma is optional and secondary. Do not let it delay the scalar gate.

## Metrics

Report paired target-seed deltas:

- validation loss / accuracy if doing closed-loop training;
- one-step held-out objective proxy if using a ledger;
- selected source gamma;
- mean auto gamma;
- mean coefficient and fallback fraction;
- wall-clock or step-cost notes.

Important scoring convention:

- use paired 95% confidence intervals for all win/tie claims;
- if auto-gamma ties tuned-gamma while using no pilot grid, that is useful only
  if it beats `full_gamma`;
- if `full_gamma` ties the learned/tuned gamma arms, the gamma axis is flat and
  this is a null for gamma estimation;
- if auto-gamma beats tuned-gamma, that is an adaptation result;
- if tuned-gamma beats auto-gamma, the synthetic auto-gamma result does not
  transfer.

## Decision Labels

- `GAMMA_FLAT_ON_REAL_PATH`: full-gamma ties both tuned-gamma and auto-gamma,
  while at least one gamma arm beats FedAvg/source scale. This label takes
  precedence over all auto-gamma tie labels.
- `AUTO_GAMMA_REAL_TIES_TUNED`: auto-gamma statistically ties tuned-gamma,
  beats full-gamma with paired CI excluding zero, and beats FedAvg/source scale.
- `AUTO_GAMMA_REAL_WINS`: auto-gamma beats tuned-gamma, full-gamma, and
  FedAvg/source scale.
- `FULL_GAMMA_EXPLAINS`: full-gamma ties or beats auto-gamma and tuned-gamma.
  This is a null for gamma estimation.
- `TUNED_GAMMA_REQUIRED`: tuned-gamma beats auto-gamma.
- `FEDAVG_SCALE_EXPLAINS`: source-tuned FedAvg scale beats all gamma arms.
- `NO_REAL_PATH_HARNESS`: bundle works but no usable real local-SGD path source
  exists on the worker.
- `INVALID_BUNDLE`: bundle sanity checks fail.

## Output

Write a concise result note and upload it to:

`aai_research_tlv/tree/double_fan/runs/local_gp/e007/`

Also send the final summary to the worker outbox.
