# DFW-E075 Result - Fallback-Aware K Rule

Decision label: `FALLBACK_AWARE_K_RULE_PARTIAL`

## Compute Status

`REAL_GPU` via PyTorch CUDA: `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.

## Metrics

| select | N | K | rule | points | overhead | fixed | DiLoCo | schedule | best control | oracle | selector | selector minus best control | selector minus shuffled | capture |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | 8 | 4 | e073_k_aware | 23 | 10.27% | 9.342+/-0.723 | 7.780+/-0.619 | 8.986+/-0.957 | 9.419+/-0.765 | 11.368+/-1.185 | 9.939+/-1.037 | 0.520+/-0.389 | 0.221+/-0.274 | 0.208+/-0.188 |
| 384 | 8 | 4 | hard_large_k_fallback | 23 | 10.27% | 9.342+/-0.723 | 7.780+/-0.619 | 8.986+/-0.957 | 9.419+/-0.765 | 11.368+/-1.185 | 9.939+/-1.037 | 0.520+/-0.389 | 0.221+/-0.274 | 0.208+/-0.188 |
| 384 | 8 | 4 | guard_large_k_fallback | 23 | 10.27% | 9.342+/-0.723 | 7.780+/-0.619 | 8.986+/-0.957 | 9.419+/-0.765 | 11.368+/-1.185 | 9.939+/-1.037 | 0.520+/-0.389 | 0.221+/-0.274 | 0.208+/-0.188 |
| 384 | 8 | 8 | e073_k_aware | 23 | 5.13% | 5.001+/-0.531 | 3.718+/-0.489 | 4.625+/-0.579 | 5.009+/-0.535 | 6.168+/-0.754 | 5.229+/-0.642 | 0.220+/-0.204 | 0.178+/-0.168 | 0.145+/-0.169 |
| 384 | 8 | 8 | hard_large_k_fallback | 23 | 5.13% | 5.001+/-0.531 | 3.718+/-0.489 | 4.625+/-0.579 | 5.009+/-0.535 | 6.168+/-0.754 | 5.229+/-0.642 | 0.220+/-0.204 | 0.178+/-0.168 | 0.145+/-0.169 |
| 384 | 8 | 8 | guard_large_k_fallback | 23 | 5.13% | 5.001+/-0.531 | 3.718+/-0.489 | 4.625+/-0.579 | 5.009+/-0.535 | 6.168+/-0.754 | 5.229+/-0.642 | 0.220+/-0.204 | 0.178+/-0.168 | 0.145+/-0.169 |
| 384 | 8 | 16 | e073_k_aware | 23 | 2.57% | 2.671+/-0.292 | 1.653+/-0.327 | 2.513+/-0.323 | 2.703+/-0.308 | 3.627+/-0.484 | 2.706+/-0.492 | 0.003+/-0.332 | 0.517+/-0.521 | -0.031+/-0.494 |
| 384 | 8 | 16 | hard_large_k_fallback | 0 | 0.00% | 2.671+/-0.292 | 1.653+/-0.327 | 2.513+/-0.323 | 2.703+/-0.308 | 3.627+/-0.484 | 2.671+/-0.292 | -0.032+/-0.045 | 0.000+/-0.000 | 0.000+/-0.000 |
| 384 | 8 | 16 | guard_large_k_fallback | 23 | 2.57% | 2.671+/-0.292 | 1.653+/-0.327 | 2.513+/-0.323 | 2.703+/-0.308 | 3.627+/-0.484 | 2.703+/-0.493 | -0.000+/-0.333 | 0.509+/-0.517 | -0.035+/-0.494 |
| 384 | 8 | 32 | e073_k_aware | 23 | 1.28% | 1.629+/-0.176 | 0.000+/-0.000 | 0.879+/-0.662 | 1.671+/-0.190 | 2.411+/-0.290 | 1.551+/-0.413 | -0.119+/-0.322 | 1.240+/-1.043 | -0.082+/-0.445 |
| 384 | 8 | 32 | hard_large_k_fallback | 0 | 0.00% | 1.629+/-0.176 | 0.000+/-0.000 | 0.879+/-0.662 | 1.671+/-0.190 | 2.411+/-0.290 | 1.629+/-0.176 | -0.042+/-0.055 | 0.000+/-0.000 | 0.000+/-0.000 |
| 384 | 8 | 32 | guard_large_k_fallback | 23 | 1.28% | 1.629+/-0.176 | 0.000+/-0.000 | 0.879+/-0.662 | 1.671+/-0.190 | 2.411+/-0.290 | 1.551+/-0.413 | -0.119+/-0.322 | 0.986+/-0.936 | -0.082+/-0.445 |
| 384 | 16 | 4 | e073_k_aware | 23 | 5.13% | 9.529+/-0.876 | 7.973+/-1.066 | 9.567+/-0.994 | 9.670+/-0.932 | 11.289+/-1.271 | 9.881+/-1.371 | 0.211+/-0.474 | 0.191+/-0.327 | 0.011+/-0.349 |
| 384 | 16 | 4 | hard_large_k_fallback | 23 | 5.13% | 9.529+/-0.876 | 7.973+/-1.066 | 9.567+/-0.994 | 9.670+/-0.932 | 11.289+/-1.271 | 9.881+/-1.371 | 0.211+/-0.474 | 0.191+/-0.327 | 0.011+/-0.349 |
| 384 | 16 | 4 | guard_large_k_fallback | 23 | 5.13% | 9.529+/-0.876 | 7.973+/-1.066 | 9.567+/-0.994 | 9.670+/-0.932 | 11.289+/-1.271 | 9.881+/-1.371 | 0.211+/-0.474 | 0.191+/-0.327 | 0.011+/-0.349 |
| 384 | 16 | 8 | e073_k_aware | 23 | 2.57% | 4.968+/-0.450 | 3.842+/-0.680 | 4.856+/-0.434 | 5.006+/-0.421 | 6.070+/-0.789 | 5.158+/-0.752 | 0.153+/-0.343 | 0.068+/-0.147 | -0.033+/-0.338 |
| 384 | 16 | 8 | hard_large_k_fallback | 23 | 2.57% | 4.968+/-0.450 | 3.842+/-0.680 | 4.856+/-0.434 | 5.006+/-0.421 | 6.070+/-0.789 | 5.158+/-0.752 | 0.153+/-0.343 | 0.068+/-0.147 | -0.033+/-0.338 |
| 384 | 16 | 8 | guard_large_k_fallback | 23 | 2.57% | 4.968+/-0.450 | 3.842+/-0.680 | 4.856+/-0.434 | 5.006+/-0.421 | 6.070+/-0.789 | 5.158+/-0.752 | 0.153+/-0.343 | 0.068+/-0.147 | -0.033+/-0.338 |
| 384 | 16 | 16 | e073_k_aware | 23 | 1.28% | 2.665+/-0.235 | 1.690+/-0.508 | 2.506+/-0.240 | 2.684+/-0.217 | 3.532+/-0.540 | 2.568+/-0.516 | -0.116+/-0.344 | -0.003+/-0.218 | -0.490+/-0.616 |
| 384 | 16 | 16 | hard_large_k_fallback | 0 | 0.00% | 2.665+/-0.235 | 1.690+/-0.508 | 2.506+/-0.240 | 2.684+/-0.217 | 3.532+/-0.540 | 2.665+/-0.235 | -0.019+/-0.036 | 0.000+/-0.000 | 0.000+/-0.000 |
| 384 | 16 | 16 | guard_large_k_fallback | 23 | 1.28% | 2.665+/-0.235 | 1.690+/-0.508 | 2.506+/-0.240 | 2.684+/-0.217 | 3.532+/-0.540 | 2.568+/-0.516 | -0.116+/-0.344 | 0.004+/-0.227 | -0.490+/-0.616 |
| 384 | 16 | 32 | e073_k_aware | 23 | 0.64% | 1.536+/-0.113 | 0.000+/-0.000 | 1.529+/-0.227 | 1.622+/-0.154 | 2.446+/-0.396 | 1.509+/-0.476 | -0.112+/-0.368 | 1.913+/-1.421 | -0.400+/-0.697 |
| 384 | 16 | 32 | hard_large_k_fallback | 0 | 0.00% | 1.536+/-0.113 | 0.000+/-0.000 | 1.529+/-0.227 | 1.622+/-0.154 | 2.446+/-0.396 | 1.536+/-0.113 | -0.086+/-0.054 | 0.000+/-0.000 | 0.000+/-0.000 |
| 384 | 16 | 32 | guard_large_k_fallback | 23 | 0.64% | 1.536+/-0.113 | 0.000+/-0.000 | 1.529+/-0.227 | 1.622+/-0.154 | 2.446+/-0.396 | 1.522+/-0.465 | -0.100+/-0.361 | 1.818+/-1.373 | -0.404+/-0.703 |

## Pooled

### e073_k_aware

- selector minus best control: `0.095+/-0.128`
- selector minus shuffled: `0.541+/-0.263`
- large-K selector minus best control: `-0.086+/-0.166`
- capture: `-0.084+/-0.159`

### guard_large_k_fallback

- selector minus best control: `0.096+/-0.127`
- selector minus shuffled: `0.497+/-0.248`
- large-K selector minus best control: `-0.084+/-0.165`
- capture: `-0.085+/-0.160`

### hard_large_k_fallback

- selector minus best control: `0.116+/-0.097`
- selector minus shuffled: `0.082+/-0.061`
- large-K selector minus best control: `-0.045+/-0.024`
- capture: `0.041+/-0.068`

## Run Log

```text
DFW-E075 fallback-aware K rule
host=devvm52518.cln0.facebook.com python=/usr/local/fbcode/platform010/bin/python3.12 compute_status=REAL_GPU torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210
design=scaled policies=(top3_mean,soft_std2,top3_guard_fixed)
rules=e073_k_aware, hard_large_k_fallback, guard_large_k_fallback
K=(4, 8, 16, 32) N=(8, 16) select=(384,)
pilot_seeds=2 heldout_seeds=12

CELL select=384 N=8 K=4 rule=e073_k_aware points=23 control=  9.419+/-0.765 selector=  9.939+/-1.037 minus_control=  0.520+/-0.389 minus_shuffled=  0.221+/-0.274 capture=  0.208+/-0.188
CELL select=384 N=8 K=4 rule=hard_large_k_fallback points=23 control=  9.419+/-0.765 selector=  9.939+/-1.037 minus_control=  0.520+/-0.389 minus_shuffled=  0.221+/-0.274 capture=  0.208+/-0.188
CELL select=384 N=8 K=4 rule=guard_large_k_fallback points=23 control=  9.419+/-0.765 selector=  9.939+/-1.037 minus_control=  0.520+/-0.389 minus_shuffled=  0.221+/-0.274 capture=  0.208+/-0.188
CELL select=384 N=8 K=8 rule=e073_k_aware points=23 control=  5.009+/-0.535 selector=  5.229+/-0.642 minus_control=  0.220+/-0.204 minus_shuffled=  0.178+/-0.168 capture=  0.145+/-0.169
CELL select=384 N=8 K=8 rule=hard_large_k_fallback points=23 control=  5.009+/-0.535 selector=  5.229+/-0.642 minus_control=  0.220+/-0.204 minus_shuffled=  0.178+/-0.168 capture=  0.145+/-0.169
CELL select=384 N=8 K=8 rule=guard_large_k_fallback points=23 control=  5.009+/-0.535 selector=  5.229+/-0.642 minus_control=  0.220+/-0.204 minus_shuffled=  0.178+/-0.168 capture=  0.145+/-0.169
CELL select=384 N=8 K=16 rule=e073_k_aware points=23 control=  2.703+/-0.308 selector=  2.706+/-0.492 minus_control=  0.003+/-0.332 minus_shuffled=  0.517+/-0.521 capture= -0.031+/-0.494
CELL select=384 N=8 K=16 rule=hard_large_k_fallback points=0 control=  2.703+/-0.308 selector=  2.671+/-0.292 minus_control= -0.032+/-0.045 minus_shuffled=  0.000+/-0.000 capture=  0.000+/-0.000
CELL select=384 N=8 K=16 rule=guard_large_k_fallback points=23 control=  2.703+/-0.308 selector=  2.703+/-0.493 minus_control= -0.000+/-0.333 minus_shuffled=  0.509+/-0.517 capture= -0.035+/-0.494
CELL select=384 N=8 K=32 rule=e073_k_aware points=23 control=  1.671+/-0.190 selector=  1.551+/-0.413 minus_control= -0.119+/-0.322 minus_shuffled=  1.240+/-1.043 capture= -0.082+/-0.445
CELL select=384 N=8 K=32 rule=hard_large_k_fallback points=0 control=  1.671+/-0.190 selector=  1.629+/-0.176 minus_control= -0.042+/-0.055 minus_shuffled=  0.000+/-0.000 capture=  0.000+/-0.000
CELL select=384 N=8 K=32 rule=guard_large_k_fallback points=23 control=  1.671+/-0.190 selector=  1.551+/-0.413 minus_control= -0.119+/-0.322 minus_shuffled=  0.986+/-0.936 capture= -0.082+/-0.445
CELL select=384 N=16 K=4 rule=e073_k_aware points=23 control=  9.670+/-0.932 selector=  9.881+/-1.371 minus_control=  0.211+/-0.474 minus_shuffled=  0.191+/-0.327 capture=  0.011+/-0.349
CELL select=384 N=16 K=4 rule=hard_large_k_fallback points=23 control=  9.670+/-0.932 selector=  9.881+/-1.371 minus_control=  0.211+/-0.474 minus_shuffled=  0.191+/-0.327 capture=  0.011+/-0.349
CELL select=384 N=16 K=4 rule=guard_large_k_fallback points=23 control=  9.670+/-0.932 selector=  9.881+/-1.371 minus_control=  0.211+/-0.474 minus_shuffled=  0.191+/-0.327 capture=  0.011+/-0.349
CELL select=384 N=16 K=8 rule=e073_k_aware points=23 control=  5.006+/-0.421 selector=  5.158+/-0.752 minus_control=  0.153+/-0.343 minus_shuffled=  0.068+/-0.147 capture= -0.033+/-0.338
CELL select=384 N=16 K=8 rule=hard_large_k_fallback points=23 control=  5.006+/-0.421 selector=  5.158+/-0.752 minus_control=  0.153+/-0.343 minus_shuffled=  0.068+/-0.147 capture= -0.033+/-0.338
CELL select=384 N=16 K=8 rule=guard_large_k_fallback points=23 control=  5.006+/-0.421 selector=  5.158+/-0.752 minus_control=  0.153+/-0.343 minus_shuffled=  0.068+/-0.147 capture= -0.033+/-0.338
CELL select=384 N=16 K=16 rule=e073_k_aware points=23 control=  2.684+/-0.217 selector=  2.568+/-0.516 minus_control= -0.116+/-0.344 minus_shuffled= -0.003+/-0.218 capture= -0.490+/-0.616
CELL select=384 N=16 K=16 rule=hard_large_k_fallback points=0 control=  2.684+/-0.217 selector=  2.665+/-0.235 minus_control= -0.019+/-0.036 minus_shuffled=  0.000+/-0.000 capture=  0.000+/-0.000
CELL select=384 N=16 K=16 rule=guard_large_k_fallback points=23 control=  2.684+/-0.217 selector=  2.568+/-0.516 minus_control= -0.116+/-0.344 minus_shuffled=  0.004+/-0.227 capture= -0.490+/-0.616
CELL select=384 N=16 K=32 rule=e073_k_aware points=23 control=  1.622+/-0.154 selector=  1.509+/-0.476 minus_control= -0.112+/-0.368 minus_shuffled=  1.913+/-1.421 capture= -0.400+/-0.697
CELL select=384 N=16 K=32 rule=hard_large_k_fallback points=0 control=  1.622+/-0.154 selector=  1.536+/-0.113 minus_control= -0.086+/-0.054 minus_shuffled=  0.000+/-0.000 capture=  0.000+/-0.000
CELL select=384 N=16 K=32 rule=guard_large_k_fallback points=23 control=  1.622+/-0.154 selector=  1.522+/-0.465 minus_control= -0.100+/-0.361 minus_shuffled=  1.818+/-1.373 capture= -0.404+/-0.703

POOLED rule=e073_k_aware minus_control=  0.095+/-0.128 minus_shuffled=  0.541+/-0.263 large_k_minus_control= -0.086+/-0.166 capture= -0.084+/-0.159
POOLED rule=guard_large_k_fallback minus_control=  0.096+/-0.127 minus_shuffled=  0.497+/-0.248 large_k_minus_control= -0.084+/-0.165 capture= -0.085+/-0.160
POOLED rule=hard_large_k_fallback minus_control=  0.116+/-0.097 minus_shuffled=  0.082+/-0.061 large_k_minus_control= -0.045+/-0.024 capture=  0.041+/-0.068
DECISION FALLBACK_AWARE_K_RULE_PARTIAL
compute_status=REAL_GPU
```
