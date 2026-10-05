# DFW-E065 Result - LR x Momentum Group Probe

Decision label: `SECOND_DIM_GROUP_PROBE_BEATS_PARALLEL_AVERAGE`

## Compute Status

`REAL_GPU` via PyTorch CUDA: `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.

## Held-Out Scores

| K | arms | cost | fixed `(alpha,beta)` | default `(alpha,beta)` | fixed all | default only | oracle subset | oracle single | raw | raw minus fixed | raw minus default | raw minus shuffled | capture |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 9 | 0.89% | (4.0,0.20) | (3.0,0.35) | 1.924+/-0.736 | 1.686+/-1.893 | 9.599+/-3.385 | 7.741+/-2.445 | 2.157+/-0.682 | 0.233+/-0.280 | 0.471+/-1.906 | 0.283+/-0.293 | 0.036+/-0.038 |
| 16 | 9 | 0.45% | (2.0,0.00) | (1.0,0.00) | 1.127+/-0.220 | 1.296+/-1.131 | 9.366+/-3.923 | 6.737+/-2.677 | 1.445+/-0.282 | 0.318+/-0.274 | 0.150+/-1.215 | 0.400+/-0.316 | 0.041+/-0.026 |

## Run Log

```text
DFW-E065 LR x momentum group probe
host=devvm52518.cln0.facebook.com python=/usr/local/fbcode/platform010/bin/python3.12 compute_status=REAL_GPU torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210
model=torch_two_layer_binary_classifier dim=97 N=18 K=(8, 16) rounds=36 eta=0.04
lr_arms=(0.5, 1.0, 2.0) momentum_arms=(0.0, 0.5, 0.9) probe_batch=96 slices=4 lam=0.4 tau=2.0
pilot_seeds=3 heldout_seeds=8

CELL K=8 N=18 arms=9 cost=0.89% fixed_pair=(4.0,0.20) default_pair=(3.0,0.35) fixed_all=  1.924+/-0.736 default_only=  1.686+/-1.893 oracle_subset=  9.599+/-3.385 oracle_single=  7.741+/-2.445 raw=  2.157+/-0.682 raw-fixed=  0.233+/-0.280 raw-default=  0.471+/-1.906 raw-shuffled=  0.283+/-0.293 capture=  0.036+/-0.038
CELL K=16 N=18 arms=9 cost=0.45% fixed_pair=(2.0,0.00) default_pair=(1.0,0.00) fixed_all=  1.127+/-0.220 default_only=  1.296+/-1.131 oracle_subset=  9.366+/-3.923 oracle_single=  6.737+/-2.677 raw=  1.445+/-0.282 raw-fixed=  0.318+/-0.274 raw-default=  0.150+/-1.215 raw-shuffled=  0.400+/-0.316 capture=  0.041+/-0.026

DECISION SECOND_DIM_GROUP_PROBE_BEATS_PARALLEL_AVERAGE
compute_status=REAL_GPU
```
