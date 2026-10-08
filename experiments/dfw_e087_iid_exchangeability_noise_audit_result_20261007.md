# DFW-E087 Result - IID Exchangeability Noise Audit

Decision label: `IID_STOCHASTIC_FILTERING_POSITIVE`

## Compute Status

`TORCH_CPU`: `torch=2.15.0a0+fb cuda_available=False`.

## Pooled

- selector minus uniform: `0.621+/-0.320`
- selector minus shuffled: `0.476+/-0.222`
- target oracle minus uniform: `9.781+/-9.714`
- mean weight L1 from uniform prior: `0.137`
- mean weight entropy fraction: `0.991`
- mean probe/target Spearman: `0.406`
- top-1 probe hit rate: `0.438`

## Cells

| K | fixed `(alpha,beta)` | uniform | target oracle single | selector | selector minus uniform | selector minus shuffled | oracle minus uniform | weight L1 | entropy | probe/target rho | top1 hit | max label mass | cost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | (11.0,0.40) | -3.589+/-14.741 | 4.576+/-4.380 | -2.935+/-14.343 | 0.654+/-0.484 | 0.586+/-0.400 | 8.166+/-10.574 | 0.137 | 0.991 | 0.446 | 0.438 | 0.500 | 1.79% |
| 16 | (5.0,0.55) | -11.403+/-25.512 | -0.006+/-7.605 | -10.815+/-25.056 | 0.588+/-0.489 | 0.367+/-0.198 | 11.397+/-17.940 | 0.138 | 0.991 | 0.367 | 0.438 | 0.562 | 0.89% |

## Interpretation

Artificial labels create non-uniform realized probe weights without creating a durable hyperparameter signal.  A positive selector row here would be stochastic-update filtering and must be separated from the HPO claim.

## Run Log

```text
DFW-E087 IID exchangeability noise audit
host=devvm24439.ldc0.facebook.com python=/data/repos/fbsource/buck-out/v2/art/fbcode/ab2a2c0557db637b/aai/double_fan_gpu_runner/__run_e087_e088_static__/run_e087_e088_static#link-tree/runtime/bin/run_e087_e088_static#native-main#platform-runtime#python#py_version_3_12 compute_status=TORCH_CPU torch=2.15.0a0+fb cuda_available=False
N=8 fake_groups=4 K=(8, 16) rounds=12 select_batch=384
pilot_seeds=1 heldout_seeds=4 spec=iid_fake_group_mean2_l40_t2

CELL K=8 fixed_pair=(11.0,0.40) uniform= -3.589+/-14.741 oracle_single=  4.576+/-4.380 selector= -2.935+/-14.343 selector-uniform=  0.654+/-0.484 selector-shuffled=  0.586+/-0.400 target_oracle-uniform=  8.166+/-10.574 weight_l1=0.137 entropy=0.991 probe_target_spearman=0.446 top1_hit=0.438 max_label_mass=0.500 cost=1.79%
CELL K=16 fixed_pair=(5.0,0.55) uniform=-11.403+/-25.512 oracle_single= -0.006+/-7.605 selector=-10.815+/-25.056 selector-uniform=  0.588+/-0.489 selector-shuffled=  0.367+/-0.198 target_oracle-uniform= 11.397+/-17.940 weight_l1=0.138 entropy=0.991 probe_target_spearman=0.367 top1_hit=0.438 max_label_mass=0.562 cost=0.89%

POOLED selector-uniform=  0.621+/-0.320 selector-shuffled=  0.476+/-0.222 target_oracle-uniform=  9.781+/-9.714 weight_l1=0.137 entropy=0.991 probe_target_spearman=0.406 top1_hit=0.438
DECISION IID_STOCHASTIC_FILTERING_POSITIVE
compute_status=TORCH_CPU
```
