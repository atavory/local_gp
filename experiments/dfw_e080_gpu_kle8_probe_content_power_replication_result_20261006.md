# DFW-E080 Result - K<=8 Probe-Content Power Replication

Decision label: `PROBE_CONTENT_POWER_REPLICATES`

## Compute Status

`REAL_GPU` via PyTorch CUDA: `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.

## Pooled

- selector minus fixed: `0.116+/-0.052`
- selector minus schedule: `0.164+/-0.057`
- selector minus best control: `0.034+/-0.048`
- selector minus shuffled: `0.177+/-0.049`
- capture: `-0.002+/-0.035`

## Cell Metrics

| select | N | K | fixed | schedule | best control | oracle | selector | selector minus fixed | selector minus schedule | selector minus best control | selector minus shuffled | capture |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | 8 | 4 | 11.031+/-0.590 | 11.042+/-0.555 | 11.158+/-0.571 | 12.831+/-0.692 | 11.201+/-0.581 | 0.169+/-0.100 | 0.159+/-0.127 | 0.043+/-0.107 | 0.212+/-0.118 | 0.057+/-0.054 |
| 384 | 8 | 8 | 5.867+/-0.324 | 5.841+/-0.332 | 5.907+/-0.322 | 7.054+/-0.434 | 5.937+/-0.318 | 0.070+/-0.064 | 0.096+/-0.074 | 0.031+/-0.065 | 0.124+/-0.068 | 0.013+/-0.059 |
| 384 | 16 | 4 | 10.663+/-0.744 | 10.573+/-0.735 | 10.748+/-0.723 | 12.475+/-0.801 | 10.822+/-0.676 | 0.160+/-0.136 | 0.249+/-0.143 | 0.075+/-0.123 | 0.245+/-0.121 | -0.006+/-0.072 |
| 384 | 16 | 8 | 5.647+/-0.447 | 5.559+/-0.435 | 5.723+/-0.408 | 6.748+/-0.443 | 5.712+/-0.374 | 0.066+/-0.102 | 0.153+/-0.099 | -0.010+/-0.076 | 0.128+/-0.069 | -0.072+/-0.086 |

## Run Log

```text
DFW-E080 K<=8 probe-content power replication
host=devvm52518.cln0.facebook.com python=/usr/local/fbcode/platform010/bin/python3.12 compute_status=REAL_GPU torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210
design=scaled policy=soft_std2
K=(4, 8) N=(8, 16) select=(384,)
pilot_seeds=2 heldout_seeds=64
gate=selector_minus_shuffled_positive_ci AND selector_minus_fixed_nonnegative_ci

CELL select=384 N=8 K=4 fixed= 11.031+/-0.590 schedule= 11.042+/-0.555 best_control= 11.158+/-0.571 oracle= 12.831+/-0.692 selector= 11.201+/-0.581 minus_fixed=  0.169+/-0.100 minus_schedule=  0.159+/-0.127 minus_best_control=  0.043+/-0.107 minus_shuffled=  0.212+/-0.118 capture=  0.057+/-0.054
CELL select=384 N=8 K=8 fixed=  5.867+/-0.324 schedule=  5.841+/-0.332 best_control=  5.907+/-0.322 oracle=  7.054+/-0.434 selector=  5.937+/-0.318 minus_fixed=  0.070+/-0.064 minus_schedule=  0.096+/-0.074 minus_best_control=  0.031+/-0.065 minus_shuffled=  0.124+/-0.068 capture=  0.013+/-0.059
CELL select=384 N=16 K=4 fixed= 10.663+/-0.744 schedule= 10.573+/-0.735 best_control= 10.748+/-0.723 oracle= 12.475+/-0.801 selector= 10.822+/-0.676 minus_fixed=  0.160+/-0.136 minus_schedule=  0.249+/-0.143 minus_best_control=  0.075+/-0.123 minus_shuffled=  0.245+/-0.121 capture= -0.006+/-0.072
CELL select=384 N=16 K=8 fixed=  5.647+/-0.447 schedule=  5.559+/-0.435 best_control=  5.723+/-0.408 oracle=  6.748+/-0.443 selector=  5.712+/-0.374 minus_fixed=  0.066+/-0.102 minus_schedule=  0.153+/-0.099 minus_best_control= -0.010+/-0.076 minus_shuffled=  0.128+/-0.069 capture= -0.072+/-0.086

POOLED minus_fixed=  0.116+/-0.052 minus_schedule=  0.164+/-0.057 minus_best_control=  0.034+/-0.048 minus_shuffled=  0.177+/-0.049 capture= -0.002+/-0.035
DECISION PROBE_CONTENT_POWER_REPLICATES
compute_status=REAL_GPU
```
