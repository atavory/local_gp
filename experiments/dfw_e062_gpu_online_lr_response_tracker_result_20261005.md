# DFW-E062 Result - Online LR Response Tracker

Decision label: `FROZEN_GROUP_PROBE_LIVE_TRACKER_TIES`

## Compute Status

`REAL_GPU` via PyTorch CUDA: `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.

## Held-Out Scores

| spread | N | K | groups | fixed `(alpha,beta)` | fixed | oracle | raw frozen | raw minus fixed | raw minus shuffled | best tracker | tracker | tracker minus fixed | tracker minus shuffled | tracker minus raw | capture |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | 8 | 8 | 0.5,1,2 | (8.0,0.35) | 4.790+/-1.225 | 7.832+/-0.732 | 5.041+/-0.988 | 0.251+/-0.327 | 0.494+/-0.650 | quad_current_l40_t2 | 5.041+/-0.988 | 0.251+/-0.327 | 0.495+/-0.651 | 0.000+/-0.000 | 0.069+/-0.082 |
| 16 | 8 | 8 | 0.25,1,4 | (5.0,0.45) | 2.925+/-2.057 | 6.861+/-0.526 | 3.328+/-1.579 | 0.403+/-0.534 | 1.047+/-1.265 | quad_current_l40_t2 | 3.328+/-1.579 | 0.403+/-0.534 | 1.047+/-1.265 | -0.000+/-0.000 | 0.074+/-0.067 |
| 4 | 8 | 16 | 0.5,1,2 | (5.0,0.25) | 2.608+/-0.924 | 5.407+/-0.480 | 2.845+/-0.662 | 0.237+/-0.301 | 0.538+/-0.568 | quad_current_l40_t2 | 2.845+/-0.661 | 0.237+/-0.301 | 0.539+/-0.569 | 0.000+/-0.000 | 0.068+/-0.058 |
| 16 | 8 | 16 | 0.25,1,4 | (3.0,0.40) | 1.487+/-1.589 | 5.687+/-1.457 | 1.900+/-1.108 | 0.413+/-0.515 | 1.147+/-1.231 | quad_current_l40_t2 | 1.900+/-1.108 | 0.413+/-0.515 | 1.148+/-1.231 | -0.000+/-0.000 | 0.074+/-0.048 |
| 4 | 8 | 32 | 0.5,1,2 | (2.0,0.45) | 1.544+/-0.366 | 3.876+/-0.745 | 1.737+/-0.278 | 0.193+/-0.237 | 0.430+/-0.436 | quad_decay80_l40_t2 | 1.743+/-0.287 | 0.199+/-0.223 | 0.476+/-0.437 | 0.006+/-0.024 | 0.066+/-0.073 |
| 16 | 8 | 32 | 0.25,1,4 | (2.0,0.35) | 1.178+/-0.603 | 5.127+/-1.883 | 1.505+/-0.331 | 0.327+/-0.380 | 1.028+/-0.953 | quad_decay80_l40_t2 | 1.512+/-0.359 | 0.334+/-0.360 | 1.110+/-1.017 | 0.007+/-0.062 | 0.065+/-0.047 |

## Run Log

```text
DFW-E062 online LR response tracker
host=devvm52518.cln0.facebook.com python=/usr/local/fbcode/platform010/bin/python3.12 compute_status=REAL_GPU torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210
model=torch_two_layer_binary_classifier dim=97 N=(8,) K=(8, 16, 32) rounds=36 eta=0.04
pilot_seeds=2 heldout_seeds=6 spread_values=(4.0, 16.0) select_batch=384

CELL spread=4 N=8 K=8 fixed_pair=(8.0,0.35) groups=0.5,1,2 fixed=  4.790+/-1.225 oracle=  7.832+/-0.732 raw=  5.041+/-0.988 raw-fixed=  0.251+/-0.327 raw-shuffled=  0.494+/-0.650 best_tracker=quad_current_l40_t2 tracker=  5.041+/-0.988 tracker-fixed=  0.251+/-0.327 tracker-shuffled=  0.495+/-0.651 tracker-raw=  0.000+/-0.000
CELL spread=16 N=8 K=8 fixed_pair=(5.0,0.45) groups=0.25,1,4 fixed=  2.925+/-2.057 oracle=  6.861+/-0.526 raw=  3.328+/-1.579 raw-fixed=  0.403+/-0.534 raw-shuffled=  1.047+/-1.265 best_tracker=quad_current_l40_t2 tracker=  3.328+/-1.579 tracker-fixed=  0.403+/-0.534 tracker-shuffled=  1.047+/-1.265 tracker-raw= -0.000+/-0.000
CELL spread=4 N=8 K=16 fixed_pair=(5.0,0.25) groups=0.5,1,2 fixed=  2.608+/-0.924 oracle=  5.407+/-0.480 raw=  2.845+/-0.662 raw-fixed=  0.237+/-0.301 raw-shuffled=  0.538+/-0.568 best_tracker=quad_current_l40_t2 tracker=  2.845+/-0.661 tracker-fixed=  0.237+/-0.301 tracker-shuffled=  0.539+/-0.569 tracker-raw=  0.000+/-0.000
CELL spread=16 N=8 K=16 fixed_pair=(3.0,0.40) groups=0.25,1,4 fixed=  1.487+/-1.589 oracle=  5.687+/-1.457 raw=  1.900+/-1.108 raw-fixed=  0.413+/-0.515 raw-shuffled=  1.147+/-1.231 best_tracker=quad_current_l40_t2 tracker=  1.900+/-1.108 tracker-fixed=  0.413+/-0.515 tracker-shuffled=  1.148+/-1.231 tracker-raw= -0.000+/-0.000
CELL spread=4 N=8 K=32 fixed_pair=(2.0,0.45) groups=0.5,1,2 fixed=  1.544+/-0.366 oracle=  3.876+/-0.745 raw=  1.737+/-0.278 raw-fixed=  0.193+/-0.237 raw-shuffled=  0.430+/-0.436 best_tracker=quad_decay80_l40_t2 tracker=  1.743+/-0.287 tracker-fixed=  0.199+/-0.223 tracker-shuffled=  0.476+/-0.437 tracker-raw=  0.006+/-0.024
CELL spread=16 N=8 K=32 fixed_pair=(2.0,0.35) groups=0.25,1,4 fixed=  1.178+/-0.603 oracle=  5.127+/-1.883 raw=  1.505+/-0.331 raw-fixed=  0.327+/-0.380 raw-shuffled=  1.028+/-0.953 best_tracker=quad_decay80_l40_t2 tracker=  1.512+/-0.359 tracker-fixed=  0.334+/-0.360 tracker-shuffled=  1.110+/-1.017 tracker-raw=  0.007+/-0.062

DECISION FROZEN_GROUP_PROBE_LIVE_TRACKER_TIES
compute_status=REAL_GPU
```
