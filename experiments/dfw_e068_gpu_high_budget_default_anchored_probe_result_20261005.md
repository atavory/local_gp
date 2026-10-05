# DFW-E068 Result - High-Budget Default-Anchored Probe

Diagnostic: high-budget repeat of E067 (`probe_batch=192`, `slices=4`).

Decision label: `DEFAULT_ANCHORED_PROBE_NEGATIVE`

## Compute Status

`REAL_GPU` via PyTorch CUDA: `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.

## Held-Out Scores

| K | cost | fixed `(alpha,beta)` | default `(alpha,beta)` | fixed all | default only | oracle subset | oracle single | all-prior raw | all-prior minus default | anchored | anchored minus fixed | anchored minus default | default minus fixed | anchored minus shuffled | capture vs default |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.34% | (5.0,0.00) | (6.0,0.35) | 0.569+/-3.941 | 2.854+/-2.343 | 9.616+/-3.408 | 8.714+/-3.324 | 1.277+/-3.066 | -1.577+/-4.125 | 2.235+/-2.305 | 1.665+/-2.940 | -0.619+/-2.026 | 2.284+/-4.894 | 2.467+/-3.376 | -0.024+/-0.179 |
| 16 | 0.67% | (2.0,0.05) | (4.0,0.10) | 0.935+/-1.194 | 1.331+/-1.423 | 7.677+/-4.715 | 6.881+/-4.513 | 1.149+/-0.981 | -0.183+/-1.196 | 1.344+/-1.555 | 0.409+/-0.781 | 0.013+/-0.774 | 0.396+/-1.295 | 1.420+/-2.553 | -0.224+/-0.341 |
| 32 | 0.33% | (2.0,0.00) | (2.0,0.10) | -0.764+/-3.492 | 11.280+/-19.975 | 18.058+/-23.580 | 17.473+/-23.360 | 0.559+/-1.185 | -10.721+/-21.081 | 8.851+/-14.270 | 9.615+/-17.729 | -2.429+/-5.725 | 12.044+/-23.425 | 2.961+/-5.590 | -0.273+/-0.371 |

## Run Log

```text
DFW-E068 default-anchored probe admission
host=devvm52518.cln0.facebook.com python=/usr/local/fbcode/platform010/bin/python3.12 compute_status=REAL_GPU torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210
model=torch_two_layer_binary_classifier dim=97 N=8 K=(8, 16, 32) rounds=36 eta=0.04
spread=16 probe_batch=192 slices=4 lam=0.4 tau=2.0
pilot_seeds=3 heldout_seeds=12

CELL K=8 cost=1.34% fixed_pair=(5.0,0.00) default_pair=(6.0,0.35) groups=0.25,1,4 fixed_all=  0.569+/-3.941 default_only=  2.854+/-2.343 oracle_subset=  9.616+/-3.408 oracle_single=  8.714+/-3.324 allprior_raw=  1.277+/-3.066 allprior-default= -1.577+/-4.125 anchored=  2.235+/-2.305 anchored-fixed=  1.665+/-2.940 anchored-default= -0.619+/-2.026 default-fixed=  2.284+/-4.894 anchored-shuffled=  2.467+/-3.376 capture_vs_default= -0.024+/-0.179
CELL K=16 cost=0.67% fixed_pair=(2.0,0.05) default_pair=(4.0,0.10) groups=0.25,1,4 fixed_all=  0.935+/-1.194 default_only=  1.331+/-1.423 oracle_subset=  7.677+/-4.715 oracle_single=  6.881+/-4.513 allprior_raw=  1.149+/-0.981 allprior-default= -0.183+/-1.196 anchored=  1.344+/-1.555 anchored-fixed=  0.409+/-0.781 anchored-default=  0.013+/-0.774 default-fixed=  0.396+/-1.295 anchored-shuffled=  1.420+/-2.553 capture_vs_default= -0.224+/-0.341
CELL K=32 cost=0.33% fixed_pair=(2.0,0.00) default_pair=(2.0,0.10) groups=0.25,1,4 fixed_all= -0.764+/-3.492 default_only= 11.280+/-19.975 oracle_subset= 18.058+/-23.580 oracle_single= 17.473+/-23.360 allprior_raw=  0.559+/-1.185 allprior-default=-10.721+/-21.081 anchored=  8.851+/-14.270 anchored-fixed=  9.615+/-17.729 anchored-default= -2.429+/-5.725 default-fixed= 12.044+/-23.425 anchored-shuffled=  2.961+/-5.590 capture_vs_default= -0.273+/-0.371

DECISION DEFAULT_ANCHORED_PROBE_NEGATIVE
compute_status=REAL_GPU
```
