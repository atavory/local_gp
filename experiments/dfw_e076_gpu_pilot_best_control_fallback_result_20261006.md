# DFW-E076 Result - Pilot-Best Control Fallback

Decision label: `PILOT_BEST_FALLBACK_FAILS`

## Compute Status

`REAL_GPU` via PyTorch CUDA: `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.

## Metrics

| select | N | K | rule | fallback | points | overhead | fixed | DiLoCo | schedule | best control | oracle | selector | selector minus best control | selector minus shuffled | capture |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | 8 | 4 | e073_k_aware | probe | 23 | 10.27% | 11.902+/-1.544 | 10.021+/-1.394 | 11.667+/-1.635 | 12.139+/-1.515 | 14.029+/-2.406 | 11.936+/-1.030 | -0.202+/-0.532 | 0.206+/-0.207 | 0.043+/-0.212 |
| 384 | 8 | 4 | pilot_best_large_k_fallback | probe | 23 | 10.27% | 11.902+/-1.544 | 10.021+/-1.394 | 11.667+/-1.635 | 12.139+/-1.515 | 14.029+/-2.406 | 11.936+/-1.030 | -0.202+/-0.532 | 0.206+/-0.207 | 0.043+/-0.212 |
| 384 | 8 | 8 | e073_k_aware | probe | 23 | 5.13% | 6.394+/-0.891 | 5.055+/-0.807 | 6.006+/-1.058 | 6.464+/-0.899 | 7.695+/-1.376 | 6.326+/-0.544 | -0.138+/-0.372 | 0.094+/-0.138 | -0.047+/-0.228 |
| 384 | 8 | 8 | pilot_best_large_k_fallback | probe | 23 | 5.13% | 6.394+/-0.891 | 5.055+/-0.807 | 6.006+/-1.058 | 6.464+/-0.899 | 7.695+/-1.376 | 6.326+/-0.544 | -0.138+/-0.372 | 0.094+/-0.138 | -0.047+/-0.228 |
| 384 | 8 | 16 | e073_k_aware | probe | 23 | 2.57% | 3.658+/-0.520 | 2.149+/-0.858 | 3.391+/-0.639 | 3.682+/-0.537 | 4.550+/-0.912 | 3.418+/-0.502 | -0.264+/-0.182 | 0.193+/-0.321 | -0.388+/-0.344 |
| 384 | 8 | 16 | pilot_best_large_k_fallback | schedule | 0 | 0.00% | 3.658+/-0.520 | 2.149+/-0.858 | 3.391+/-0.639 | 3.682+/-0.537 | 4.550+/-0.912 | 3.391+/-0.639 | -0.291+/-0.181 | 0.000+/-0.000 | -0.592+/-0.384 |
| 384 | 8 | 32 | e073_k_aware | probe | 23 | 1.28% | 2.226+/-0.309 | 1.207+/-0.466 | 2.124+/-0.339 | 2.286+/-0.320 | 3.002+/-0.514 | 2.135+/-0.396 | -0.152+/-0.149 | 0.536+/-0.368 | -0.085+/-0.127 |
| 384 | 8 | 32 | pilot_best_large_k_fallback | schedule | 0 | 0.00% | 2.226+/-0.309 | 1.207+/-0.466 | 2.124+/-0.339 | 2.286+/-0.320 | 3.002+/-0.514 | 2.124+/-0.339 | -0.162+/-0.107 | 0.000+/-0.000 | -0.276+/-0.239 |
| 384 | 16 | 4 | e073_k_aware | probe | 23 | 5.13% | 10.100+/-2.400 | 8.008+/-2.355 | 9.891+/-2.403 | 10.184+/-2.393 | 12.647+/-2.563 | 10.827+/-1.503 | 0.642+/-0.929 | 0.128+/-0.239 | 0.186+/-0.323 |
| 384 | 16 | 4 | pilot_best_large_k_fallback | probe | 23 | 5.13% | 10.100+/-2.400 | 8.008+/-2.355 | 9.891+/-2.403 | 10.184+/-2.393 | 12.647+/-2.563 | 10.827+/-1.503 | 0.642+/-0.929 | 0.128+/-0.239 | 0.186+/-0.323 |
| 384 | 16 | 8 | e073_k_aware | probe | 23 | 2.57% | 5.269+/-1.378 | 3.774+/-1.297 | 5.160+/-1.465 | 5.311+/-1.421 | 6.805+/-1.304 | 5.718+/-0.759 | 0.407+/-0.674 | 0.049+/-0.126 | 0.031+/-0.476 |
| 384 | 16 | 8 | pilot_best_large_k_fallback | probe | 23 | 2.57% | 5.269+/-1.378 | 3.774+/-1.297 | 5.160+/-1.465 | 5.311+/-1.421 | 6.805+/-1.304 | 5.718+/-0.759 | 0.407+/-0.674 | 0.049+/-0.126 | 0.031+/-0.476 |
| 384 | 16 | 16 | e073_k_aware | probe | 23 | 1.28% | 2.972+/-0.794 | 0.795+/-1.343 | 2.929+/-0.777 | 3.019+/-0.779 | 3.936+/-0.718 | 2.992+/-0.643 | -0.027+/-0.354 | 0.046+/-0.384 | -0.208+/-0.667 |
| 384 | 16 | 16 | pilot_best_large_k_fallback | schedule | 0 | 0.00% | 2.972+/-0.794 | 0.795+/-1.343 | 2.929+/-0.777 | 3.019+/-0.779 | 3.936+/-0.718 | 2.929+/-0.777 | -0.090+/-0.071 | 0.000+/-0.000 | -0.240+/-0.318 |
| 384 | 16 | 32 | e073_k_aware | probe | 23 | 0.64% | 1.766+/-0.486 | 0.354+/-0.759 | 1.716+/-0.478 | 1.784+/-0.478 | 2.524+/-0.362 | 1.776+/-0.373 | -0.008+/-0.252 | 0.422+/-0.411 | -0.047+/-0.370 |
| 384 | 16 | 32 | pilot_best_large_k_fallback | schedule | 0 | 0.00% | 1.766+/-0.486 | 0.354+/-0.759 | 1.716+/-0.478 | 1.784+/-0.478 | 2.524+/-0.362 | 1.716+/-0.478 | -0.068+/-0.044 | 0.000+/-0.000 | -0.187+/-0.250 |

## Pooled

### e073_k_aware

- selector minus best control: `0.032+/-0.180`
- selector minus shuffled: `0.209+/-0.105`
- large-K selector minus best control: `-0.113+/-0.123`
- capture: `-0.064+/-0.133`

### pilot_best_large_k_fallback

- selector minus best control: `0.012+/-0.172`
- selector minus shuffled: `0.060+/-0.046`
- large-K selector minus best control: `-0.153+/-0.060`
- capture: `-0.135+/-0.117`

## Pilot Fallbacks

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
DFW-E076 pilot-best control fallback
host=devvm52518.cln0.facebook.com python=/usr/local/fbcode/platform010/bin/python3.12 compute_status=REAL_GPU torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210
design=scaled policies=(top3_mean,soft_std2)
rule=K<=8 probe soft_std2; K>8 pilot-best control fallback
K=(4, 8, 16, 32) N=(8, 16) select=(384,)
pilot_seeds=2 heldout_seeds=12

CELL select=384 N=8 K=4 rule=e073_k_aware fallback=probe control= 12.139+/-1.515 selector= 11.936+/-1.030 minus_control= -0.202+/-0.532 minus_shuffled=  0.206+/-0.207 capture=  0.043+/-0.212
CELL select=384 N=8 K=4 rule=pilot_best_large_k_fallback fallback=probe control= 12.139+/-1.515 selector= 11.936+/-1.030 minus_control= -0.202+/-0.532 minus_shuffled=  0.206+/-0.207 capture=  0.043+/-0.212
CELL select=384 N=8 K=8 rule=e073_k_aware fallback=probe control=  6.464+/-0.899 selector=  6.326+/-0.544 minus_control= -0.138+/-0.372 minus_shuffled=  0.094+/-0.138 capture= -0.047+/-0.228
CELL select=384 N=8 K=8 rule=pilot_best_large_k_fallback fallback=probe control=  6.464+/-0.899 selector=  6.326+/-0.544 minus_control= -0.138+/-0.372 minus_shuffled=  0.094+/-0.138 capture= -0.047+/-0.228
CELL select=384 N=8 K=16 rule=e073_k_aware fallback=probe control=  3.682+/-0.537 selector=  3.418+/-0.502 minus_control= -0.264+/-0.182 minus_shuffled=  0.193+/-0.321 capture= -0.388+/-0.344
CELL select=384 N=8 K=16 rule=pilot_best_large_k_fallback fallback=schedule control=  3.682+/-0.537 selector=  3.391+/-0.639 minus_control= -0.291+/-0.181 minus_shuffled=  0.000+/-0.000 capture= -0.592+/-0.384
CELL select=384 N=8 K=32 rule=e073_k_aware fallback=probe control=  2.286+/-0.320 selector=  2.135+/-0.396 minus_control= -0.152+/-0.149 minus_shuffled=  0.536+/-0.368 capture= -0.085+/-0.127
CELL select=384 N=8 K=32 rule=pilot_best_large_k_fallback fallback=schedule control=  2.286+/-0.320 selector=  2.124+/-0.339 minus_control= -0.162+/-0.107 minus_shuffled=  0.000+/-0.000 capture= -0.276+/-0.239
CELL select=384 N=16 K=4 rule=e073_k_aware fallback=probe control= 10.184+/-2.393 selector= 10.827+/-1.503 minus_control=  0.642+/-0.929 minus_shuffled=  0.128+/-0.239 capture=  0.186+/-0.323
CELL select=384 N=16 K=4 rule=pilot_best_large_k_fallback fallback=probe control= 10.184+/-2.393 selector= 10.827+/-1.503 minus_control=  0.642+/-0.929 minus_shuffled=  0.128+/-0.239 capture=  0.186+/-0.323
CELL select=384 N=16 K=8 rule=e073_k_aware fallback=probe control=  5.311+/-1.421 selector=  5.718+/-0.759 minus_control=  0.407+/-0.674 minus_shuffled=  0.049+/-0.126 capture=  0.031+/-0.476
CELL select=384 N=16 K=8 rule=pilot_best_large_k_fallback fallback=probe control=  5.311+/-1.421 selector=  5.718+/-0.759 minus_control=  0.407+/-0.674 minus_shuffled=  0.049+/-0.126 capture=  0.031+/-0.476
CELL select=384 N=16 K=16 rule=e073_k_aware fallback=probe control=  3.019+/-0.779 selector=  2.992+/-0.643 minus_control= -0.027+/-0.354 minus_shuffled=  0.046+/-0.384 capture= -0.208+/-0.667
CELL select=384 N=16 K=16 rule=pilot_best_large_k_fallback fallback=schedule control=  3.019+/-0.779 selector=  2.929+/-0.777 minus_control= -0.090+/-0.071 minus_shuffled=  0.000+/-0.000 capture= -0.240+/-0.318
CELL select=384 N=16 K=32 rule=e073_k_aware fallback=probe control=  1.784+/-0.478 selector=  1.776+/-0.373 minus_control= -0.008+/-0.252 minus_shuffled=  0.422+/-0.411 capture= -0.047+/-0.370
CELL select=384 N=16 K=32 rule=pilot_best_large_k_fallback fallback=schedule control=  1.784+/-0.478 selector=  1.716+/-0.478 minus_control= -0.068+/-0.044 minus_shuffled=  0.000+/-0.000 capture= -0.187+/-0.250

POOLED rule=e073_k_aware minus_control=  0.032+/-0.180 minus_shuffled=  0.209+/-0.105 large_k_minus_control= -0.113+/-0.123 capture= -0.064+/-0.133
POOLED rule=pilot_best_large_k_fallback minus_control=  0.012+/-0.172 minus_shuffled=  0.060+/-0.046 large_k_minus_control= -0.153+/-0.060 capture= -0.135+/-0.117
fallback_by_cell={(384, 8, 4): 'schedule', (384, 8, 8): 'schedule', (384, 8, 16): 'schedule', (384, 8, 32): 'schedule', (384, 16, 4): 'schedule', (384, 16, 8): 'schedule', (384, 16, 16): 'schedule', (384, 16, 32): 'schedule'}
DECISION PILOT_BEST_FALLBACK_FAILS
compute_status=REAL_GPU
```
