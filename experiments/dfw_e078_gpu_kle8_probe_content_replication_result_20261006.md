# DFW-E078 Result - K<=8 Probe-Content Replication

Decision label: `PROBE_CONTENT_REPLICATES`

## Compute Status

`REAL_GPU` via PyTorch CUDA: `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.

## Metrics

| select | N | K | fixed | schedule | best control | oracle | selector | selector minus fixed | selector minus schedule | selector minus best control | selector minus shuffled | capture |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | 8 | 4 | 10.937+/-0.711 | 10.921+/-0.697 | 11.043+/-0.708 | 13.318+/-1.169 | 11.308+/-0.862 | 0.372+/-0.209 | 0.387+/-0.217 | 0.265+/-0.211 | 0.243+/-0.203 | 0.008+/-0.211 |
| 384 | 8 | 8 | 5.739+/-0.389 | 5.698+/-0.439 | 5.807+/-0.411 | 7.209+/-0.711 | 5.934+/-0.492 | 0.196+/-0.130 | 0.236+/-0.106 | 0.127+/-0.111 | 0.168+/-0.127 | -0.018+/-0.177 |
| 384 | 16 | 4 | 11.147+/-0.974 | 10.839+/-1.073 | 11.347+/-0.881 | 13.483+/-1.110 | 11.424+/-0.952 | 0.276+/-0.141 | 0.585+/-0.250 | 0.076+/-0.286 | 0.353+/-0.186 | 0.102+/-0.055 |
| 384 | 16 | 8 | 5.916+/-0.652 | 5.718+/-0.682 | 6.058+/-0.554 | 7.423+/-0.678 | 6.057+/-0.553 | 0.141+/-0.141 | 0.339+/-0.189 | -0.001+/-0.230 | 0.204+/-0.112 | 0.007+/-0.096 |

## Pooled

- selector minus fixed: `0.246+/-0.080`
- selector minus schedule: `0.387+/-0.100`
- selector minus best control: `0.117+/-0.109`
- selector minus shuffled: `0.242+/-0.081`
- capture: `0.025+/-0.074`

## Run Log

```text
DFW-E078 K<=8 probe-content replication
host=devvm52518.cln0.facebook.com python=/usr/local/fbcode/platform010/bin/python3.12 compute_status=REAL_GPU torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210
design=scaled policy=soft_std2
K=(4, 8) N=(8, 16) select=(384,)
pilot_seeds=2 heldout_seeds=33
gate=selector_minus_shuffled_positive_ci AND selector_minus_fixed_nonnegative_ci

CELL select=384 N=8 K=4 fixed= 10.937+/-0.711 schedule= 10.921+/-0.697 best_control= 11.043+/-0.708 oracle= 13.318+/-1.169 selector= 11.308+/-0.862 minus_fixed=  0.372+/-0.209 minus_schedule=  0.387+/-0.217 minus_best_control=  0.265+/-0.211 minus_shuffled=  0.243+/-0.203 capture=  0.008+/-0.211
CELL select=384 N=8 K=8 fixed=  5.739+/-0.389 schedule=  5.698+/-0.439 best_control=  5.807+/-0.411 oracle=  7.209+/-0.711 selector=  5.934+/-0.492 minus_fixed=  0.196+/-0.130 minus_schedule=  0.236+/-0.106 minus_best_control=  0.127+/-0.111 minus_shuffled=  0.168+/-0.127 capture= -0.018+/-0.177
CELL select=384 N=16 K=4 fixed= 11.147+/-0.974 schedule= 10.839+/-1.073 best_control= 11.347+/-0.881 oracle= 13.483+/-1.110 selector= 11.424+/-0.952 minus_fixed=  0.276+/-0.141 minus_schedule=  0.585+/-0.250 minus_best_control=  0.076+/-0.286 minus_shuffled=  0.353+/-0.186 capture=  0.102+/-0.055
CELL select=384 N=16 K=8 fixed=  5.916+/-0.652 schedule=  5.718+/-0.682 best_control=  6.058+/-0.554 oracle=  7.423+/-0.678 selector=  6.057+/-0.553 minus_fixed=  0.141+/-0.141 minus_schedule=  0.339+/-0.189 minus_best_control= -0.001+/-0.230 minus_shuffled=  0.204+/-0.112 capture=  0.007+/-0.096

POOLED minus_fixed=  0.246+/-0.080 minus_schedule=  0.387+/-0.100 minus_best_control=  0.117+/-0.109 minus_shuffled=  0.242+/-0.081 capture=  0.025+/-0.074
DECISION PROBE_CONTENT_REPLICATES
compute_status=REAL_GPU
```
