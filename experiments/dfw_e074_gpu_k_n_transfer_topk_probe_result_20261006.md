# DFW-E074 Result - K/N Transfer Top-k Probe

Decision label: `K_N_TRANSFER_REPLICATES`

## Compute Status

`REAL_GPU` via PyTorch CUDA: `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.

## Metrics

| select | N | K | policy | fixed | DiLoCo | schedule | best control | oracle | design oracle | selector | selector minus best control | selector minus shuffled | capture |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | 8 | 4 | soft_std2 | 10.510+/-0.973 | 8.287+/-1.207 | 10.511+/-1.009 | 10.595+/-0.977 | 12.530+/-1.470 | 12.368+/-1.455 | 10.917+/-1.408 | 0.322+/-0.458 | 0.267+/-0.280 | 0.084+/-0.295 |
| 384 | 8 | 8 | soft_std2 | 5.647+/-0.426 | 4.103+/-0.648 | 5.391+/-0.385 | 5.687+/-0.392 | 6.870+/-0.780 | 6.769+/-0.739 | 5.900+/-0.665 | 0.213+/-0.297 | 0.145+/-0.170 | 0.116+/-0.214 |
| 384 | 8 | 16 | top3_mean | 3.263+/-0.240 | 1.957+/-0.389 | 3.076+/-0.221 | 3.276+/-0.230 | 4.086+/-0.533 | 3.994+/-0.477 | 3.286+/-0.526 | 0.010+/-0.338 | 0.445+/-0.337 | -0.275+/-0.402 |
| 384 | 8 | 32 | top3_mean | 2.033+/-0.144 | 0.000+/-0.000 | 1.806+/-0.197 | 2.043+/-0.142 | 2.780+/-0.332 | 2.701+/-0.326 | 2.025+/-0.413 | -0.017+/-0.303 | 0.601+/-0.330 | -0.346+/-0.501 |
| 384 | 16 | 4 | soft_std2 | 10.632+/-1.061 | 8.536+/-0.992 | 10.117+/-1.292 | 10.735+/-1.044 | 12.640+/-1.807 | 12.502+/-1.790 | 11.322+/-1.627 | 0.587+/-0.617 | 0.585+/-0.324 | 0.167+/-0.267 |
| 384 | 16 | 8 | soft_std2 | 5.754+/-0.674 | 4.202+/-0.618 | 5.283+/-0.711 | 5.828+/-0.643 | 6.893+/-0.943 | 6.786+/-0.918 | 6.022+/-0.833 | 0.194+/-0.232 | 0.339+/-0.191 | 0.086+/-0.229 |
| 384 | 16 | 16 | top3_mean | 3.151+/-0.328 | 1.958+/-0.438 | 2.826+/-0.415 | 3.186+/-0.316 | 3.974+/-0.557 | 3.854+/-0.493 | 3.183+/-0.663 | -0.002+/-0.436 | 0.379+/-0.387 | -0.621+/-1.008 |
| 384 | 16 | 32 | top3_mean | 1.913+/-0.254 | 0.000+/-0.000 | 1.545+/-0.359 | 1.913+/-0.254 | 2.628+/-0.345 | 2.532+/-0.305 | 1.853+/-0.419 | -0.060+/-0.292 | 0.985+/-0.953 | -0.547+/-0.734 |

## Pooled

- selector minus best control: `0.156+/-0.139`
- selector minus shuffled: `0.468+/-0.157`
- capture: `-0.167+/-0.189`

## Run Log

```text
DFW-E074 K/N transfer top-k probe
host=devvm52518.cln0.facebook.com python=/usr/local/fbcode/platform010/bin/python3.12 compute_status=REAL_GPU torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210
frozen_rule=scaled/(soft_std2 if K<=8 else top3_mean)
K=(4, 8, 16, 32) N=(8, 16) select=(384,)
pilot_seeds=2 heldout_seeds=12

CELL select=384 N=8 K=4 policy=soft_std2 control= 10.595+/-0.977 selector= 10.917+/-1.408 minus_control=  0.322+/-0.458 minus_shuffled=  0.267+/-0.280 capture=  0.084+/-0.295
CELL select=384 N=8 K=8 policy=soft_std2 control=  5.687+/-0.392 selector=  5.900+/-0.665 minus_control=  0.213+/-0.297 minus_shuffled=  0.145+/-0.170 capture=  0.116+/-0.214
CELL select=384 N=8 K=16 policy=top3_mean control=  3.276+/-0.230 selector=  3.286+/-0.526 minus_control=  0.010+/-0.338 minus_shuffled=  0.445+/-0.337 capture= -0.275+/-0.402
CELL select=384 N=8 K=32 policy=top3_mean control=  2.043+/-0.142 selector=  2.025+/-0.413 minus_control= -0.017+/-0.303 minus_shuffled=  0.601+/-0.330 capture= -0.346+/-0.501
CELL select=384 N=16 K=4 policy=soft_std2 control= 10.735+/-1.044 selector= 11.322+/-1.627 minus_control=  0.587+/-0.617 minus_shuffled=  0.585+/-0.324 capture=  0.167+/-0.267
CELL select=384 N=16 K=8 policy=soft_std2 control=  5.828+/-0.643 selector=  6.022+/-0.833 minus_control=  0.194+/-0.232 minus_shuffled=  0.339+/-0.191 capture=  0.086+/-0.229
CELL select=384 N=16 K=16 policy=top3_mean control=  3.186+/-0.316 selector=  3.183+/-0.663 minus_control= -0.002+/-0.436 minus_shuffled=  0.379+/-0.387 capture= -0.621+/-1.008
CELL select=384 N=16 K=32 policy=top3_mean control=  1.913+/-0.254 selector=  1.853+/-0.419 minus_control= -0.060+/-0.292 minus_shuffled=  0.985+/-0.953 capture= -0.547+/-0.734

POOLED minus_control=  0.156+/-0.139 minus_shuffled=  0.468+/-0.157 capture= -0.167+/-0.189
DECISION K_N_TRANSFER_REPLICATES
compute_status=REAL_GPU
```
