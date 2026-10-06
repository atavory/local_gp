# DFW-E085 Result - Guarded Stale-Default HP Arm

Decision label: `STALE_DEFAULT_GUARDED_PROBE_LIVE`

## Compute Status

`REAL_GPU` via PyTorch CUDA: `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.

## Held-Out Scores

| K | cost | stale `(alpha,beta)` | fresh `(alpha,beta)` | stale only | fresh only | oracle single | guarded | guarded-stale | guarded-fresh | guarded-shuffled | abandon | stale mass | fresh mass | high mass | capture vs stale |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.34% | (18.0,0.45) | (6.0,0.35) | 3.170+/-1.910 | 2.877+/-2.290 | 6.506+/-1.301 | 4.167+/-1.546 | 0.997+/-0.739 | 1.290+/-2.179 | 98.628+/-51.240 | 0.222+/-0.073 | 0.778+/-0.073 | 0.208+/-0.078 | 0.014+/-0.013 | 0.179+/-0.252 |
| 16 | 0.67% | (11.0,0.30) | (3.0,0.20) | 1.988+/-1.989 | 1.696+/-1.448 | 5.100+/-1.224 | 2.383+/-1.598 | 0.395+/-0.654 | 0.687+/-2.200 | 76.452+/-44.321 | 0.197+/-0.086 | 0.803+/-0.086 | 0.188+/-0.089 | 0.009+/-0.010 | -0.167+/-0.385 |
| 32 | 0.33% | (6.0,0.20) | (2.0,0.00) | 1.075+/-1.937 | 1.198+/-1.594 | 4.083+/-1.932 | 1.802+/-1.765 | 0.727+/-1.114 | 0.604+/-1.618 | 46.815+/-21.597 | 0.208+/-0.121 | 0.792+/-0.121 | 0.188+/-0.107 | 0.021+/-0.025 | -0.247+/-0.714 |

## Run Log

```text
DFW-E085 guarded stale-default HP arm
host=devvm52518.cln0.facebook.com python=/usr/local/fbcode/platform010/bin/python3.12 compute_status=REAL_GPU torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210
model=torch_two_layer_binary_classifier dim=97 N=8 K=(8, 16, 32) rounds=36 eta=0.04
spread=16 stale_default=0.25 fresh_default=1.0 probe_batch=192 slices=4 guard_z=0.5 min_agree=0.75
pilot_seeds=3 heldout_seeds=12

CELL K=8 cost=1.34% groups=0.25,1,4 fixed_pair=(6.0,0.10) stale_pair=(18.0,0.45) fresh_pair=(6.0,0.35) fixed_all=  3.552+/-1.025 stale_only=  3.170+/-1.910 fresh_only=  2.877+/-2.290 oracle_single=  6.506+/-1.301 oracle_subset=  7.991+/-1.157 guarded=  4.167+/-1.546 guarded-stale=  0.997+/-0.739 guarded-fresh=  1.290+/-2.179 guarded-shuffled= 98.628+/-51.240 oracle_single-stale=  3.336+/-1.445 capture_vs_stale=  0.179+/-0.252 abandon=  0.222+/-0.073 shuffled_abandon=  0.597+/-0.039 mass_stale=  0.778+/-0.073 mass_fresh=  0.208+/-0.078 mass_high=  0.014+/-0.013
CELL K=16 cost=0.67% groups=0.25,1,4 fixed_pair=(3.0,0.05) stale_pair=(11.0,0.30) fresh_pair=(3.0,0.20) fixed_all=  2.021+/-0.407 stale_only=  1.988+/-1.989 fresh_only=  1.696+/-1.448 oracle_single=  5.100+/-1.224 oracle_subset=  6.744+/-1.206 guarded=  2.383+/-1.598 guarded-stale=  0.395+/-0.654 guarded-fresh=  0.687+/-2.200 guarded-shuffled= 76.452+/-44.321 oracle_single-stale=  3.112+/-1.728 capture_vs_stale= -0.167+/-0.385 abandon=  0.197+/-0.086 shuffled_abandon=  0.606+/-0.059 mass_stale=  0.803+/-0.086 mass_fresh=  0.188+/-0.089 mass_high=  0.009+/-0.010
CELL K=32 cost=0.33% groups=0.25,1,4 fixed_pair=(2.0,0.00) stale_pair=(6.0,0.20) fresh_pair=(2.0,0.00) fixed_all=  1.410+/-0.185 stale_only=  1.075+/-1.937 fresh_only=  1.198+/-1.594 oracle_single=  4.083+/-1.932 oracle_subset=  6.219+/-1.849 guarded=  1.802+/-1.765 guarded-stale=  0.727+/-1.114 guarded-fresh=  0.604+/-1.618 guarded-shuffled= 46.815+/-21.597 oracle_single-stale=  3.008+/-1.982 capture_vs_stale= -0.247+/-0.714 abandon=  0.208+/-0.121 shuffled_abandon=  0.581+/-0.036 mass_stale=  0.792+/-0.121 mass_fresh=  0.188+/-0.107 mass_high=  0.021+/-0.025

DECISION STALE_DEFAULT_GUARDED_PROBE_LIVE
compute_status=REAL_GPU
```
