# DFW-E083 Result - Batch Value-per-FLOP HPO

Decision label: `BATCH_VALUE_PER_FLOP_NEGATIVE`

## Compute Status

`REAL_GPU` via PyTorch CUDA: `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.

## Held-Out Scores

| K | fixed `(alpha,beta)` | default | equal mix | raw probe | value/FLOP probe | shuffled | oracle arm | oracle subset | vpf-default | vpf-shuffled | selected cost | mixed cost | probe overhead |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | (4.0,0.40) | 4.112+/-1.912 | 5.924+/-0.640 | 5.235+/-0.523 | 5.121+/-0.700 | 5.068+/-0.598 | 7.467+/-2.132 | 8.254+/-2.176 | 1.009+/-1.254 | 0.054+/-0.112 | 1.043+/-0.023 | 1.167x | 10.71% |
| 16 | (3.0,0.05) | 0.307+/-4.333 | 9.965+/-12.174 | 5.916+/-5.078 | 5.954+/-5.256 | 5.667+/-4.861 | 16.478+/-22.651 | 17.689+/-23.614 | 5.647+/-9.583 | 0.287+/-0.405 | 1.065+/-0.015 | 1.167x | 5.36% |

## Run Log

```text
DFW-E083 batch-size value-per-FLOP HPO
host=devvm52518.cln0.facebook.com python=/usr/local/fbcode/platform010/bin/python3.12 compute_status=REAL_GPU torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210
N=12 K=(8, 16) rounds=32 batch_arms=(8, 16, 32) default_batch=16
pilot_seeds=2 heldout_seeds=6 probe_batch=96 slices=2

CELL K=8 fixed_pair=(4.0,0.40) default=  4.112+/-1.912 equal_mix=  5.924+/-0.640 raw_probe=  5.235+/-0.523 vpf_probe=  5.121+/-0.700 vpf_shuffled=  5.068+/-0.598 oracle_arm=  7.467+/-2.132 oracle_subset=  8.254+/-2.176 vpf-default=  1.009+/-1.254 vpf-shuffled=  0.054+/-0.112 selected_cost=  1.043+/-0.023 mixed_cost=1.167 probe_overhead=10.71%
CELL K=16 fixed_pair=(3.0,0.05) default=  0.307+/-4.333 equal_mix=  9.965+/-12.174 raw_probe=  5.916+/-5.078 vpf_probe=  5.954+/-5.256 vpf_shuffled=  5.667+/-4.861 oracle_arm= 16.478+/-22.651 oracle_subset= 17.689+/-23.614 vpf-default=  5.647+/-9.583 vpf-shuffled=  0.287+/-0.405 selected_cost=  1.065+/-0.015 mixed_cost=1.167 probe_overhead=5.36%

DECISION BATCH_VALUE_PER_FLOP_NEGATIVE
compute_status=REAL_GPU
```
