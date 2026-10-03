# LGP-E007 real-path auto-gamma gate

Decision: `FEDAVG_SCALE_EXPLAINS`.

The source-tuned FedAvg scale has the lowest target mean one-step loss among the three primary arms.

## Protocol

- Real CUDA MNIST local-SGD paths: N=8, K=8, eta=0.2, 12 rounds per seed.
- Model: 14x14 average-pooled MNIST linear classifier (1,970 parameters), allowing the supplied exact pure-Python `LocalGPAggregator` to cover every coordinate.
- Pilot seeds [2100, 2101, 2102, 2103] select gamma from 41 values and FedAvg scale from 31 values on validation examples 50,000:55,000.
- Disjoint target seeds [3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111] are scored on held-out validation examples 55,000:60,000. Each arm is a one-step loss/accuracy proxy from the identical synchronized state and identical worker paths; the source trajectory advances with raw FedAvg.
- Selected source gamma: 0.050; selected source FedAvg scale: 1.900.

## Target results

| Arm | Mean one-step loss | Mean one-step accuracy |
|---|---:|---:|
| `fedavg_scale` | 0.39689761 | 0.893260 |
| `tuned_gamma` | 0.43057679 | 0.894921 |
| `auto_gamma` | 0.45685025 | 0.877885 |
| `full_gamma` | 0.50616432 | 0.856181 |
| `fedavg_raw` | 0.43140562 | 0.895275 |

Paired deltas are computed after averaging rounds within each target seed (negative loss favors the left arm; positive accuracy favors the left arm).

| Contrast | Loss delta (95% CI) | Accuracy delta (95% CI) |
|---|---:|---:|
| `auto_minus_tuned` | +0.02627346 [+0.01425404, +0.03829288] | -0.017036 [-0.023607, -0.010466] |
| `auto_minus_fedavg_scale` | +0.05995264 [+0.04719612, +0.07270915] | -0.015375 [-0.021869, -0.008881] |
| `tuned_minus_fedavg_scale` | +0.03367917 [+0.03173759, +0.03562076] | +0.001661 [+0.000846, +0.002476] |
| `full_minus_fedavg_scale` | +0.10926671 [+0.08304024, +0.13549317] | -0.037079 [-0.047229, -0.026929] |

## Auto-gamma and cost diagnostics

- Mean auto gamma: 0.284881.
- Mean auto coefficient: 1.569196.
- Mean fallback fraction: 0.000000.
- Auto-vs-tuned equivalence: False at absolute loss margin 0.001000.
- FedAvg identity max error: 1.49e-08; tuned affine identity max error: 1.49e-08.
- Target path generation: 1.82s; tuned/auto/full aggregation: 40.03s / 40.89s / 40.09s; arm evaluation: 0.37s.
- Device: NVIDIA PG509-210; PyTorch 2.15.0a0+fb; CUDA 13.3.0.

## Interpretation

This is a real-path one-step held-out objective gate, not a closed-loop optimizer claim. Pilot grid cost is paid by `tuned_gamma` and `fedavg_scale`; `auto_gamma` uses no pilot grid. The decision follows target mean loss, with the paired seed confidence intervals shown above and a preregistered 0.001 absolute-loss equivalence margin for the auto-vs-tuned tie label.

The synthetic auto-gamma result did not transfer: tuned gamma beat auto-gamma by 0.026273 loss (95% CI [0.014254, 0.038293]) and 1.704 accuracy points. The stronger primary result is that source-tuned FedAvg scale beat both gamma arms on loss, hence `FEDAVG_SCALE_EXPLAINS`. Tuned gamma was 0.166 accuracy points above the scale arm, so the label is specifically a validation-loss decision rather than dominance on every secondary metric.
