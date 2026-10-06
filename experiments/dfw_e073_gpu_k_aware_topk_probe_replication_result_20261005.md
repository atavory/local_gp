# DFW-E073 Result - K-aware Top-k Probe Replication

Decision label: `K_AWARE_TOPK_PROBE_REPLICATES`

## Compute Status

`REAL_GPU` via PyTorch CUDA: `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.

## Metrics

| select | N | K | policy | fixed | DiLoCo | schedule | best control | oracle | design oracle | selector | selector minus best control | selector minus shuffled | capture |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | 8 | 8 | soft_std2 | 5.070+/-0.806 | 4.021+/-0.770 | 5.130+/-0.801 | 5.184+/-0.804 | 6.594+/-0.632 | 6.465+/-0.565 | 5.638+/-0.495 | 0.454+/-0.329 | 0.098+/-0.112 | 0.156+/-0.306 |
| 384 | 8 | 32 | top3_mean | 1.621+/-0.302 | 0.348+/-0.499 | 1.439+/-0.400 | 1.691+/-0.327 | 2.472+/-0.199 | 2.392+/-0.185 | 1.720+/-0.285 | 0.029+/-0.144 | 0.975+/-0.446 | 0.063+/-0.177 |

## Run Log

```text
DFW-E073 K-aware top-k probe replication
host=devvm52518.cln0.facebook.com python=/usr/local/fbcode/platform010/bin/python3.12 compute_status=REAL_GPU torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210
frozen_rule=scaled/(soft_std2 if K<=8 else top3_mean)
K=(8, 32) N=(8,) select=(384,)
pilot_seeds=3 heldout_seeds=24

CELL select=384 N=8 K=8 policy=soft_std2 control=  5.184+/-0.804 selector=  5.638+/-0.495 minus_control=  0.454+/-0.329 minus_shuffled=  0.098+/-0.112 capture=  0.156+/-0.306
CELL select=384 N=8 K=32 policy=top3_mean control=  1.691+/-0.327 selector=  1.720+/-0.285 minus_control=  0.029+/-0.144 minus_shuffled=  0.975+/-0.446 capture=  0.063+/-0.177

DECISION K_AWARE_TOPK_PROBE_REPLICATES
compute_status=REAL_GPU
```
