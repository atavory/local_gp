# DFW-E066 Result - LR Group Default Control

Decision label: `LR_GROUP_PROBE_BEATS_PARALLEL_AVERAGE_ONLY`

## Compute Status

`REAL_GPU` via PyTorch CUDA: `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.

## Held-Out Scores

| K | cost | fixed `(alpha,beta)` | default `(alpha,beta)` | fixed all | default only | oracle subset | oracle single | raw | raw minus fixed | raw minus default | default minus fixed | raw minus shuffled | capture |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.67% | (4.0,0.25) | (6.0,0.45) | 3.482+/-0.596 | 5.313+/-1.185 | 7.056+/-1.290 | 6.773+/-1.309 | 3.719+/-0.487 | 0.238+/-0.160 | -1.593+/-1.415 | 1.831+/-1.472 | 0.467+/-0.415 | 0.072+/-0.035 |
| 16 | 0.33% | (2.0,0.10) | (4.0,0.35) | 1.451+/-0.755 | 5.884+/-4.853 | 7.283+/-6.143 | 7.158+/-6.134 | 1.763+/-0.469 | 0.312+/-0.295 | -4.122+/-5.293 | 4.433+/-5.570 | 0.798+/-1.016 | 0.070+/-0.027 |
| 32 | 0.17% | (1.0,0.00) | (3.0,0.20) | 1.000+/-0.000 | 4.362+/-4.419 | 4.020+/-2.004 | 3.980+/-1.995 | 1.100+/-0.041 | 0.100+/-0.041 | -3.262+/-4.428 | 3.362+/-4.419 | 0.337+/-0.219 | 0.049+/-0.020 |

## Run Log

```text
DFW-E066 LR group default-only control
host=devvm52518.cln0.facebook.com python=/usr/local/fbcode/platform010/bin/python3.12 compute_status=REAL_GPU torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210
model=torch_two_layer_binary_classifier dim=97 N=8 K=(8, 16, 32) rounds=36 eta=0.04
spread=16 probe_batch=96 slices=4 lam=0.4 tau=2.0
pilot_seeds=3 heldout_seeds=12

CELL K=8 cost=0.67% fixed_pair=(4.0,0.25) default_pair=(6.0,0.45) groups=0.25,1,4 fixed_all=  3.482+/-0.596 default_only=  5.313+/-1.185 oracle_subset=  7.056+/-1.290 oracle_single=  6.773+/-1.309 raw=  3.719+/-0.487 raw-fixed=  0.238+/-0.160 raw-default= -1.593+/-1.415 default-fixed=  1.831+/-1.472 raw-shuffled=  0.467+/-0.415 capture=  0.072+/-0.035
CELL K=16 cost=0.33% fixed_pair=(2.0,0.10) default_pair=(4.0,0.35) groups=0.25,1,4 fixed_all=  1.451+/-0.755 default_only=  5.884+/-4.853 oracle_subset=  7.283+/-6.143 oracle_single=  7.158+/-6.134 raw=  1.763+/-0.469 raw-fixed=  0.312+/-0.295 raw-default= -4.122+/-5.293 default-fixed=  4.433+/-5.570 raw-shuffled=  0.798+/-1.016 capture=  0.070+/-0.027
CELL K=32 cost=0.17% fixed_pair=(1.0,0.00) default_pair=(3.0,0.20) groups=0.25,1,4 fixed_all=  1.000+/-0.000 default_only=  4.362+/-4.419 oracle_subset=  4.020+/-2.004 oracle_single=  3.980+/-1.995 raw=  1.100+/-0.041 raw-fixed=  0.100+/-0.041 raw-default= -3.262+/-4.428 default-fixed=  3.362+/-4.419 raw-shuffled=  0.337+/-0.219 capture=  0.049+/-0.020

DECISION LR_GROUP_PROBE_BEATS_PARALLEL_AVERAGE_ONLY
compute_status=REAL_GPU
```
