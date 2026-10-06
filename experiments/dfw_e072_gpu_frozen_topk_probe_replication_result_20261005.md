# DFW-E072 Result - Frozen Top-k Probe Replication

Decision label: `TOPK_PROBE_STILL_LOSES`

## Compute Status

`REAL_GPU` via PyTorch CUDA: `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.

## Metrics

| select | N | K | fixed | DiLoCo | schedule | best control | oracle | design oracle | selector | selector minus best control | selector minus shuffled | capture |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | 8 | 8 | 4.677+/-1.471 | 3.617+/-1.375 | 4.502+/-1.544 | 4.882+/-1.433 | 6.846+/-0.806 | 6.645+/-0.653 | 5.425+/-0.781 | 0.543+/-0.688 | 0.093+/-0.157 | -0.150+/-0.440 |
| 384 | 8 | 32 | 1.085+/-1.326 | -0.600+/-2.058 | 0.795+/-1.469 | 1.152+/-1.340 | 2.995+/-0.950 | 2.666+/-0.520 | 1.134+/-1.200 | -0.017+/-0.195 | 0.387+/-0.534 | -0.044+/-0.114 |

## Run Log

```text
DFW-E072 frozen top-k probe replication
host=devvm52518.cln0.facebook.com python=/usr/local/fbcode/platform010/bin/python3.12 compute_status=REAL_GPU torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210
frozen_selector=scaled/soft_std2
K=(8, 32) N=(8,) select=(384,)
pilot_seeds=3 heldout_seeds=24

CELL select=384 N=8 K=8 fixed=(12.0, 0.15) control=  4.882+/-1.433 selector=  5.425+/-0.781 minus_control=  0.543+/-0.688 minus_shuffled=  0.093+/-0.157 capture= -0.150+/-0.440
CELL select=384 N=8 K=32 fixed=(4.0, 0.05) control=  1.152+/-1.340 selector=  1.134+/-1.200 minus_control= -0.017+/-0.195 minus_shuffled=  0.387+/-0.534 capture= -0.044+/-0.114

DECISION TOPK_PROBE_STILL_LOSES
compute_status=REAL_GPU
```
