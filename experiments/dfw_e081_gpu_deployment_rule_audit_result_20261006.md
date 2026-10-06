# DFW-E081 Result - Deployment Rule Audit

Decision label: `DEPLOYMENT_RULE_LIVE_GLOBAL_FIXED`

## Compute Status

`REAL_GPU` via PyTorch CUDA: `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.

## Frozen Pairs

- global fixed pair from all pilot cells: `(5.0, 0.05)`
- K=8 transferred pair from K=8 pilot cells: `(9.0, 0.05)`
- theory fallback: `(min(20, max(4, 72/K)), 0.15)`

## Pooled Deployment Rules

| rule | minus own no-probe baseline | minus per-K fixed | minus schedule | minus best-control ceiling | low-K minus shuffled | high-K minus own baseline |
|---|---:|---:|---:|---:|---:|---:|
| probe_small_global_fixed_large | 2.166+/-0.367 | 0.052+/-0.085 | 0.231+/-0.090 | -0.017+/-0.087 | 0.175+/-0.065 | 0.000+/-0.000 |
| probe_small_k8_transfer_large | 1.044+/-0.231 | -1.386+/-0.409 | -1.207+/-0.401 | -1.454+/-0.410 | 0.175+/-0.065 | 0.000+/-0.000 |
| probe_small_theory_large | 0.073+/-0.029 | 0.119+/-0.066 | 0.298+/-0.075 | 0.051+/-0.067 | 0.175+/-0.065 | 0.000+/-0.000 |

## Cell Metrics

| select | N | K | probe | shuffled | global fixed | K8 transfer | theory | per-K fixed | schedule | best-control ceiling | oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | 8 | 4 | 11.502+/-0.740 | 11.299+/-0.694 | 4.713+/-0.039 | 7.572+/-0.136 | 11.319+/-0.690 | 10.881+/-0.523 | 10.618+/-0.623 | 10.970+/-0.532 | 13.201+/-0.819 |
| 384 | 8 | 8 | 6.131+/-0.396 | 6.005+/-0.364 | 4.236+/-0.083 | 5.879+/-0.293 | 6.011+/-0.365 | 5.887+/-0.323 | 5.604+/-0.419 | 5.940+/-0.324 | 7.173+/-0.475 |
| 384 | 8 | 16 | 3.375+/-0.226 | 3.276+/-0.195 | 3.314+/-0.193 | 2.650+/-0.676 | 3.277+/-0.197 | 3.344+/-0.217 | 3.090+/-0.268 | 3.372+/-0.208 | 4.191+/-0.305 |
| 384 | 8 | 32 | 2.019+/-0.274 | 1.920+/-0.316 | 1.642+/-0.433 | -2.958+/-1.493 | 1.893+/-0.331 | 2.043+/-0.152 | 1.784+/-0.208 | 2.100+/-0.148 | 2.815+/-0.204 |
| 384 | 16 | 4 | 11.491+/-0.747 | 11.269+/-0.706 | 4.717+/-0.038 | 7.581+/-0.135 | 11.330+/-0.688 | 11.125+/-0.574 | 11.053+/-0.636 | 11.195+/-0.592 | 12.885+/-0.835 |
| 384 | 16 | 8 | 6.094+/-0.431 | 5.945+/-0.401 | 4.228+/-0.091 | 5.838+/-0.321 | 5.973+/-0.394 | 5.939+/-0.365 | 5.797+/-0.353 | 5.953+/-0.366 | 7.053+/-0.509 |
| 384 | 16 | 16 | 3.333+/-0.248 | 3.215+/-0.224 | 3.250+/-0.223 | 2.390+/-0.789 | 3.230+/-0.219 | 3.211+/-0.203 | 3.225+/-0.275 | 3.298+/-0.224 | 4.163+/-0.351 |
| 384 | 16 | 32 | 1.888+/-0.307 | 1.675+/-0.387 | 1.366+/-0.512 | -4.009+/-1.778 | 1.711+/-0.380 | 1.948+/-0.141 | 1.774+/-0.289 | 2.098+/-0.183 | 2.759+/-0.268 |

## Run Log

```text
DFW-E081 deployment-rule audit
host=devvm52518.cln0.facebook.com python=/usr/local/fbcode/platform010/bin/python3.12 compute_status=REAL_GPU torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210
rule=K<=8 scaled/soft_std2 probe; K>8 frozen no-probe fallback
K=(4, 8, 16, 32) N=(8, 16) select=(384,)
pilot_seeds=2 heldout_seeds=32
global_pair=(5.0, 0.05)
k8_transfer_pair=(9.0, 0.05)

CELL select=384 N=8 K=4 fixed_pair=(14.0, 0.25) theory_pair=(18.0, 0.15) probe= 11.502+/-0.740 shuffled= 11.299+/-0.694 global_fixed=  4.713+/-0.039 k8_transfer=  7.572+/-0.136 theory= 11.319+/-0.690 per_k_fixed= 10.881+/-0.523 schedule= 10.618+/-0.623 best_control= 10.970+/-0.532 oracle= 13.201+/-0.819
CELL select=384 N=8 K=8 fixed_pair=(8.0, 0.2) theory_pair=(9.0, 0.15) probe=  6.131+/-0.396 shuffled=  6.005+/-0.364 global_fixed=  4.236+/-0.083 k8_transfer=  5.879+/-0.293 theory=  6.011+/-0.365 per_k_fixed=  5.887+/-0.323 schedule=  5.604+/-0.419 best_control=  5.940+/-0.324 oracle=  7.173+/-0.475
CELL select=384 N=8 K=16 fixed_pair=(5.0, 0.1) theory_pair=(4.5, 0.15) probe=  3.375+/-0.226 shuffled=  3.276+/-0.195 global_fixed=  3.314+/-0.193 k8_transfer=  2.650+/-0.676 theory=  3.277+/-0.197 per_k_fixed=  3.344+/-0.217 schedule=  3.090+/-0.268 best_control=  3.372+/-0.208 oracle=  4.191+/-0.305
CELL select=384 N=8 K=32 fixed_pair=(3.0, 0.1) theory_pair=(4.0, 0.15) probe=  2.019+/-0.274 shuffled=  1.920+/-0.316 global_fixed=  1.642+/-0.433 k8_transfer= -2.958+/-1.493 theory=  1.893+/-0.331 per_k_fixed=  2.043+/-0.152 schedule=  1.784+/-0.208 best_control=  2.100+/-0.148 oracle=  2.815+/-0.204
CELL select=384 N=16 K=4 fixed_pair=(19.0, 0.0) theory_pair=(18.0, 0.15) probe= 11.491+/-0.747 shuffled= 11.269+/-0.706 global_fixed=  4.717+/-0.038 k8_transfer=  7.581+/-0.135 theory= 11.330+/-0.688 per_k_fixed= 11.125+/-0.574 schedule= 11.053+/-0.636 best_control= 11.195+/-0.592 oracle= 12.885+/-0.835
CELL select=384 N=16 K=8 fixed_pair=(10.0, 0.0) theory_pair=(9.0, 0.15) probe=  6.094+/-0.431 shuffled=  5.945+/-0.401 global_fixed=  4.228+/-0.091 k8_transfer=  5.838+/-0.321 theory=  5.973+/-0.394 per_k_fixed=  5.939+/-0.365 schedule=  5.797+/-0.353 best_control=  5.953+/-0.366 oracle=  7.053+/-0.509
CELL select=384 N=16 K=16 fixed_pair=(5.0, 0.0) theory_pair=(4.5, 0.15) probe=  3.333+/-0.248 shuffled=  3.215+/-0.224 global_fixed=  3.250+/-0.223 k8_transfer=  2.390+/-0.789 theory=  3.230+/-0.219 per_k_fixed=  3.211+/-0.203 schedule=  3.225+/-0.275 best_control=  3.298+/-0.224 oracle=  4.163+/-0.351
CELL select=384 N=16 K=32 fixed_pair=(3.0, 0.0) theory_pair=(4.0, 0.15) probe=  1.888+/-0.307 shuffled=  1.675+/-0.387 global_fixed=  1.366+/-0.512 k8_transfer= -4.009+/-1.778 theory=  1.711+/-0.380 per_k_fixed=  1.948+/-0.141 schedule=  1.774+/-0.289 best_control=  2.098+/-0.183 oracle=  2.759+/-0.268

POOLED rule=probe_small_global_fixed_large minus_own_baseline=  2.166+/-0.367 minus_per_k_fixed=  0.052+/-0.085 minus_schedule=  0.231+/-0.090 minus_best_control_ceiling= -0.017+/-0.087 minus_shuffled_low_k=  0.175+/-0.065 minus_own_baseline_high_k=  0.000+/-0.000
POOLED rule=probe_small_k8_transfer_large minus_own_baseline=  1.044+/-0.231 minus_per_k_fixed= -1.386+/-0.409 minus_schedule= -1.207+/-0.401 minus_best_control_ceiling= -1.454+/-0.410 minus_shuffled_low_k=  0.175+/-0.065 minus_own_baseline_high_k=  0.000+/-0.000
POOLED rule=probe_small_theory_large minus_own_baseline=  0.073+/-0.029 minus_per_k_fixed=  0.119+/-0.066 minus_schedule=  0.298+/-0.075 minus_best_control_ceiling=  0.051+/-0.067 minus_shuffled_low_k=  0.175+/-0.065 minus_own_baseline_high_k=  0.000+/-0.000
DECISION DEPLOYMENT_RULE_LIVE_GLOBAL_FIXED
compute_status=REAL_GPU
```
