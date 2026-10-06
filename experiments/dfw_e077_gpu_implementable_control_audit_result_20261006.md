# DFW-E077 Result - Implementable-Control Audit

Decision label: `IMPLEMENTABLE_CONTROL_AUDIT_FAILS`

## Compute Status

`REAL_GPU` via PyTorch CUDA: `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.

## Pooled Comparisons

| rule | minus fixed | minus DiLoCo | minus schedule | minus pilot-best | minus heldout-best ceiling | minus shuffled |
|---|---:|---:|---:|---:|---:|---:|
| e073_k_aware | -0.127+/-0.196 | 1.593+/-0.269 | 0.165+/-0.181 | 0.165+/-0.181 | -0.205+/-0.202 | 0.216+/-0.306 |
| small_k_probe_fixed_large_k | 0.062+/-0.089 | 1.781+/-0.267 | 0.353+/-0.138 | 0.353+/-0.138 | -0.017+/-0.084 | 0.044+/-0.077 |
| small_k_probe_pilot_large_k | -0.091+/-0.130 | 1.628+/-0.236 | 0.200+/-0.115 | 0.200+/-0.115 | -0.170+/-0.119 | 0.044+/-0.077 |

## Cell Metrics

| select | N | K | rule | pilot-best | points | overhead | fixed | schedule | heldout-best ceiling | selector | minus fixed | minus pilot-best | minus heldout-best ceiling | minus shuffled |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | 8 | 4 | e073_k_aware | schedule | 23 | 10.27% | 11.197+/-1.766 | 10.985+/-1.698 | 11.256+/-1.685 | 11.202+/-1.741 | 0.006+/-0.235 | 0.217+/-0.312 | -0.054+/-0.269 | 0.113+/-0.354 |
| 384 | 8 | 4 | small_k_probe_fixed_large_k | schedule | 23 | 10.27% | 11.197+/-1.766 | 10.985+/-1.698 | 11.256+/-1.685 | 11.202+/-1.741 | 0.006+/-0.235 | 0.217+/-0.312 | -0.054+/-0.269 | 0.113+/-0.354 |
| 384 | 8 | 4 | small_k_probe_pilot_large_k | schedule | 23 | 10.27% | 11.197+/-1.766 | 10.985+/-1.698 | 11.256+/-1.685 | 11.202+/-1.741 | 0.006+/-0.235 | 0.217+/-0.312 | -0.054+/-0.269 | 0.113+/-0.354 |
| 384 | 8 | 8 | e073_k_aware | schedule | 23 | 5.13% | 5.963+/-0.998 | 5.853+/-1.028 | 5.972+/-0.999 | 5.918+/-0.937 | -0.046+/-0.168 | 0.065+/-0.222 | -0.055+/-0.169 | 0.016+/-0.274 |
| 384 | 8 | 8 | small_k_probe_fixed_large_k | schedule | 23 | 5.13% | 5.963+/-0.998 | 5.853+/-1.028 | 5.972+/-0.999 | 5.918+/-0.937 | -0.046+/-0.168 | 0.065+/-0.222 | -0.055+/-0.169 | 0.016+/-0.274 |
| 384 | 8 | 8 | small_k_probe_pilot_large_k | schedule | 23 | 5.13% | 5.963+/-0.998 | 5.853+/-1.028 | 5.972+/-0.999 | 5.918+/-0.937 | -0.046+/-0.168 | 0.065+/-0.222 | -0.055+/-0.169 | 0.016+/-0.274 |
| 384 | 8 | 16 | e073_k_aware | schedule | 23 | 2.57% | 3.110+/-0.832 | 3.031+/-0.756 | 3.258+/-0.669 | 2.650+/-1.494 | -0.460+/-0.705 | -0.381+/-0.854 | -0.607+/-0.896 | -0.056+/-1.070 |
| 384 | 8 | 16 | small_k_probe_fixed_large_k | schedule | 0 | 0.00% | 3.110+/-0.832 | 3.031+/-0.756 | 3.258+/-0.669 | 3.110+/-0.832 | 0.000+/-0.000 | 0.079+/-0.194 | -0.147+/-0.207 | 0.000+/-0.000 |
| 384 | 8 | 16 | small_k_probe_pilot_large_k | schedule | 0 | 0.00% | 3.110+/-0.832 | 3.031+/-0.756 | 3.258+/-0.669 | 3.031+/-0.756 | -0.079+/-0.194 | 0.000+/-0.000 | -0.226+/-0.115 | 0.000+/-0.000 |
| 384 | 8 | 32 | e073_k_aware | schedule | 23 | 1.28% | 1.565+/-0.923 | 1.069+/-1.536 | 1.605+/-0.933 | 0.955+/-1.972 | -0.610+/-1.068 | -0.114+/-0.499 | -0.650+/-1.062 | 1.103+/-1.869 |
| 384 | 8 | 32 | small_k_probe_fixed_large_k | schedule | 0 | 0.00% | 1.565+/-0.923 | 1.069+/-1.536 | 1.605+/-0.933 | 1.565+/-0.923 | 0.000+/-0.000 | 0.496+/-0.630 | -0.040+/-0.043 | 0.000+/-0.000 |
| 384 | 8 | 32 | small_k_probe_pilot_large_k | schedule | 0 | 0.00% | 1.565+/-0.923 | 1.069+/-1.536 | 1.605+/-0.933 | 1.069+/-1.536 | -0.496+/-0.630 | 0.000+/-0.000 | -0.536+/-0.617 | 0.000+/-0.000 |
| 384 | 16 | 4 | e073_k_aware | schedule | 23 | 5.13% | 10.551+/-2.056 | 10.227+/-1.937 | 10.841+/-1.832 | 10.938+/-1.617 | 0.387+/-0.532 | 0.711+/-0.582 | 0.097+/-0.421 | 0.184+/-0.372 |
| 384 | 16 | 4 | small_k_probe_fixed_large_k | schedule | 23 | 5.13% | 10.551+/-2.056 | 10.227+/-1.937 | 10.841+/-1.832 | 10.938+/-1.617 | 0.387+/-0.532 | 0.711+/-0.582 | 0.097+/-0.421 | 0.184+/-0.372 |
| 384 | 16 | 4 | small_k_probe_pilot_large_k | schedule | 23 | 5.13% | 10.551+/-2.056 | 10.227+/-1.937 | 10.841+/-1.832 | 10.938+/-1.617 | 0.387+/-0.532 | 0.711+/-0.582 | 0.097+/-0.421 | 0.184+/-0.372 |
| 384 | 16 | 8 | e073_k_aware | schedule | 23 | 2.57% | 5.613+/-1.253 | 5.154+/-1.330 | 5.623+/-1.251 | 5.760+/-0.937 | 0.148+/-0.355 | 0.607+/-0.464 | 0.137+/-0.352 | 0.041+/-0.249 |
| 384 | 16 | 8 | small_k_probe_fixed_large_k | schedule | 23 | 2.57% | 5.613+/-1.253 | 5.154+/-1.330 | 5.623+/-1.251 | 5.760+/-0.937 | 0.148+/-0.355 | 0.607+/-0.464 | 0.137+/-0.352 | 0.041+/-0.249 |
| 384 | 16 | 8 | small_k_probe_pilot_large_k | schedule | 23 | 2.57% | 5.613+/-1.253 | 5.154+/-1.330 | 5.623+/-1.251 | 5.760+/-0.937 | 0.148+/-0.355 | 0.607+/-0.464 | 0.137+/-0.352 | 0.041+/-0.249 |
| 384 | 16 | 16 | e073_k_aware | schedule | 23 | 1.28% | 3.016+/-0.892 | 2.700+/-0.879 | 3.085+/-0.821 | 2.891+/-1.006 | -0.125+/-0.328 | 0.191+/-0.395 | -0.194+/-0.337 | -0.100+/-0.697 |
| 384 | 16 | 16 | small_k_probe_fixed_large_k | schedule | 0 | 0.00% | 3.016+/-0.892 | 2.700+/-0.879 | 3.085+/-0.821 | 3.016+/-0.892 | 0.000+/-0.000 | 0.316+/-0.167 | -0.069+/-0.130 | 0.000+/-0.000 |
| 384 | 16 | 16 | small_k_probe_pilot_large_k | schedule | 0 | 0.00% | 3.016+/-0.892 | 2.700+/-0.879 | 3.085+/-0.821 | 2.700+/-0.879 | -0.316+/-0.167 | 0.000+/-0.000 | -0.385+/-0.169 | 0.000+/-0.000 |
| 384 | 16 | 32 | e073_k_aware | schedule | 23 | 0.64% | 1.797+/-0.677 | 1.462+/-0.749 | 1.800+/-0.678 | 1.483+/-0.995 | -0.314+/-0.383 | 0.021+/-0.351 | -0.317+/-0.382 | 0.425+/-0.749 |
| 384 | 16 | 32 | small_k_probe_fixed_large_k | schedule | 0 | 0.00% | 1.797+/-0.677 | 1.462+/-0.749 | 1.800+/-0.678 | 1.797+/-0.677 | 0.000+/-0.000 | 0.335+/-0.118 | -0.003+/-0.006 | 0.000+/-0.000 |
| 384 | 16 | 32 | small_k_probe_pilot_large_k | schedule | 0 | 0.00% | 1.797+/-0.677 | 1.462+/-0.749 | 1.800+/-0.678 | 1.462+/-0.749 | -0.335+/-0.118 | 0.000+/-0.000 | -0.338+/-0.115 | 0.000+/-0.000 |

## Pilot Choices

- select=384 N=8 K=4: `schedule`
- select=384 N=8 K=8: `schedule`
- select=384 N=8 K=16: `schedule`
- select=384 N=8 K=32: `schedule`
- select=384 N=16 K=4: `schedule`
- select=384 N=16 K=8: `schedule`
- select=384 N=16 K=16: `schedule`
- select=384 N=16 K=32: `schedule`

## Run Log

```text
DFW-E077 implementable-control audit
host=devvm52518.cln0.facebook.com python=/usr/local/fbcode/platform010/bin/python3.12 compute_status=REAL_GPU torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210
comparators=fixed,diloco,schedule,pilot_best,heldout_best_ceiling
K=(4, 8, 16, 32) N=(8, 16) select=(384,)
pilot_seeds=2 heldout_seeds=12

CELL select=384 N=8 K=4 rule=e073_k_aware pilot_best=schedule selector= 11.202+/-1.741 minus_fixed=  0.006+/-0.235 minus_pilot=  0.217+/-0.312 minus_ceiling= -0.054+/-0.269 minus_shuffled=  0.113+/-0.354
CELL select=384 N=8 K=4 rule=small_k_probe_fixed_large_k pilot_best=schedule selector= 11.202+/-1.741 minus_fixed=  0.006+/-0.235 minus_pilot=  0.217+/-0.312 minus_ceiling= -0.054+/-0.269 minus_shuffled=  0.113+/-0.354
CELL select=384 N=8 K=4 rule=small_k_probe_pilot_large_k pilot_best=schedule selector= 11.202+/-1.741 minus_fixed=  0.006+/-0.235 minus_pilot=  0.217+/-0.312 minus_ceiling= -0.054+/-0.269 minus_shuffled=  0.113+/-0.354
CELL select=384 N=8 K=8 rule=e073_k_aware pilot_best=schedule selector=  5.918+/-0.937 minus_fixed= -0.046+/-0.168 minus_pilot=  0.065+/-0.222 minus_ceiling= -0.055+/-0.169 minus_shuffled=  0.016+/-0.274
CELL select=384 N=8 K=8 rule=small_k_probe_fixed_large_k pilot_best=schedule selector=  5.918+/-0.937 minus_fixed= -0.046+/-0.168 minus_pilot=  0.065+/-0.222 minus_ceiling= -0.055+/-0.169 minus_shuffled=  0.016+/-0.274
CELL select=384 N=8 K=8 rule=small_k_probe_pilot_large_k pilot_best=schedule selector=  5.918+/-0.937 minus_fixed= -0.046+/-0.168 minus_pilot=  0.065+/-0.222 minus_ceiling= -0.055+/-0.169 minus_shuffled=  0.016+/-0.274
CELL select=384 N=8 K=16 rule=e073_k_aware pilot_best=schedule selector=  2.650+/-1.494 minus_fixed= -0.460+/-0.705 minus_pilot= -0.381+/-0.854 minus_ceiling= -0.607+/-0.896 minus_shuffled= -0.056+/-1.070
CELL select=384 N=8 K=16 rule=small_k_probe_fixed_large_k pilot_best=schedule selector=  3.110+/-0.832 minus_fixed=  0.000+/-0.000 minus_pilot=  0.079+/-0.194 minus_ceiling= -0.147+/-0.207 minus_shuffled=  0.000+/-0.000
CELL select=384 N=8 K=16 rule=small_k_probe_pilot_large_k pilot_best=schedule selector=  3.031+/-0.756 minus_fixed= -0.079+/-0.194 minus_pilot=  0.000+/-0.000 minus_ceiling= -0.226+/-0.115 minus_shuffled=  0.000+/-0.000
CELL select=384 N=8 K=32 rule=e073_k_aware pilot_best=schedule selector=  0.955+/-1.972 minus_fixed= -0.610+/-1.068 minus_pilot= -0.114+/-0.499 minus_ceiling= -0.650+/-1.062 minus_shuffled=  1.103+/-1.869
CELL select=384 N=8 K=32 rule=small_k_probe_fixed_large_k pilot_best=schedule selector=  1.565+/-0.923 minus_fixed=  0.000+/-0.000 minus_pilot=  0.496+/-0.630 minus_ceiling= -0.040+/-0.043 minus_shuffled=  0.000+/-0.000
CELL select=384 N=8 K=32 rule=small_k_probe_pilot_large_k pilot_best=schedule selector=  1.069+/-1.536 minus_fixed= -0.496+/-0.630 minus_pilot=  0.000+/-0.000 minus_ceiling= -0.536+/-0.617 minus_shuffled=  0.000+/-0.000
CELL select=384 N=16 K=4 rule=e073_k_aware pilot_best=schedule selector= 10.938+/-1.617 minus_fixed=  0.387+/-0.532 minus_pilot=  0.711+/-0.582 minus_ceiling=  0.097+/-0.421 minus_shuffled=  0.184+/-0.372
CELL select=384 N=16 K=4 rule=small_k_probe_fixed_large_k pilot_best=schedule selector= 10.938+/-1.617 minus_fixed=  0.387+/-0.532 minus_pilot=  0.711+/-0.582 minus_ceiling=  0.097+/-0.421 minus_shuffled=  0.184+/-0.372
CELL select=384 N=16 K=4 rule=small_k_probe_pilot_large_k pilot_best=schedule selector= 10.938+/-1.617 minus_fixed=  0.387+/-0.532 minus_pilot=  0.711+/-0.582 minus_ceiling=  0.097+/-0.421 minus_shuffled=  0.184+/-0.372
CELL select=384 N=16 K=8 rule=e073_k_aware pilot_best=schedule selector=  5.760+/-0.937 minus_fixed=  0.148+/-0.355 minus_pilot=  0.607+/-0.464 minus_ceiling=  0.137+/-0.352 minus_shuffled=  0.041+/-0.249
CELL select=384 N=16 K=8 rule=small_k_probe_fixed_large_k pilot_best=schedule selector=  5.760+/-0.937 minus_fixed=  0.148+/-0.355 minus_pilot=  0.607+/-0.464 minus_ceiling=  0.137+/-0.352 minus_shuffled=  0.041+/-0.249
CELL select=384 N=16 K=8 rule=small_k_probe_pilot_large_k pilot_best=schedule selector=  5.760+/-0.937 minus_fixed=  0.148+/-0.355 minus_pilot=  0.607+/-0.464 minus_ceiling=  0.137+/-0.352 minus_shuffled=  0.041+/-0.249
CELL select=384 N=16 K=16 rule=e073_k_aware pilot_best=schedule selector=  2.891+/-1.006 minus_fixed= -0.125+/-0.328 minus_pilot=  0.191+/-0.395 minus_ceiling= -0.194+/-0.337 minus_shuffled= -0.100+/-0.697
CELL select=384 N=16 K=16 rule=small_k_probe_fixed_large_k pilot_best=schedule selector=  3.016+/-0.892 minus_fixed=  0.000+/-0.000 minus_pilot=  0.316+/-0.167 minus_ceiling= -0.069+/-0.130 minus_shuffled=  0.000+/-0.000
CELL select=384 N=16 K=16 rule=small_k_probe_pilot_large_k pilot_best=schedule selector=  2.700+/-0.879 minus_fixed= -0.316+/-0.167 minus_pilot=  0.000+/-0.000 minus_ceiling= -0.385+/-0.169 minus_shuffled=  0.000+/-0.000
CELL select=384 N=16 K=32 rule=e073_k_aware pilot_best=schedule selector=  1.483+/-0.995 minus_fixed= -0.314+/-0.383 minus_pilot=  0.021+/-0.351 minus_ceiling= -0.317+/-0.382 minus_shuffled=  0.425+/-0.749
CELL select=384 N=16 K=32 rule=small_k_probe_fixed_large_k pilot_best=schedule selector=  1.797+/-0.677 minus_fixed=  0.000+/-0.000 minus_pilot=  0.335+/-0.118 minus_ceiling= -0.003+/-0.006 minus_shuffled=  0.000+/-0.000
CELL select=384 N=16 K=32 rule=small_k_probe_pilot_large_k pilot_best=schedule selector=  1.462+/-0.749 minus_fixed= -0.335+/-0.118 minus_pilot=  0.000+/-0.000 minus_ceiling= -0.338+/-0.115 minus_shuffled=  0.000+/-0.000

POOLED rule=e073_k_aware minus_fixed= -0.127+/-0.196 minus_diloco=  1.593+/-0.269 minus_schedule=  0.165+/-0.181 minus_pilot_best=  0.165+/-0.181 minus_heldout_best_ceiling= -0.205+/-0.202 minus_shuffled=  0.216+/-0.306
POOLED rule=small_k_probe_fixed_large_k minus_fixed=  0.062+/-0.089 minus_diloco=  1.781+/-0.267 minus_schedule=  0.353+/-0.138 minus_pilot_best=  0.353+/-0.138 minus_heldout_best_ceiling= -0.017+/-0.084 minus_shuffled=  0.044+/-0.077
POOLED rule=small_k_probe_pilot_large_k minus_fixed= -0.091+/-0.130 minus_diloco=  1.628+/-0.236 minus_schedule=  0.200+/-0.115 minus_pilot_best=  0.200+/-0.115 minus_heldout_best_ceiling= -0.170+/-0.119 minus_shuffled=  0.044+/-0.077
pilot_choices={(384, 8, 4): 'schedule', (384, 8, 8): 'schedule', (384, 8, 16): 'schedule', (384, 8, 32): 'schedule', (384, 16, 4): 'schedule', (384, 16, 8): 'schedule', (384, 16, 16): 'schedule', (384, 16, 32): 'schedule'}
DECISION IMPLEMENTABLE_CONTROL_AUDIT_FAILS
compute_status=REAL_GPU
```
