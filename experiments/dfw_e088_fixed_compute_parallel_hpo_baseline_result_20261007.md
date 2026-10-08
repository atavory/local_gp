# DFW-E088 Result - Fixed-Compute Parallel HPO Baseline

Decision label: `FIXED_COMPUTE_BASELINE_NOT_CLEARED`

## Compute Status

`TORCH_CPU`: `torch=2.15.0a0+fb cuda_available=False`.

## Pooled

- internal minus external validation-selected HPO: `-0.006+/-0.002`
- internal minus external oracle HPO: `-0.007+/-0.000`
- internal minus shuffled internal: `0.003+/-0.003`
- internal minus parallel-arm average: `0.000+/-0.001`
- internal minus default full-worker arm: `-0.006+/-0.001`

## Cells

| K | pair | stale | default | parallel arm average | internal probe | internal minus shuffled | internal minus ext-valid | internal minus ext-oracle | internal minus best full-worker arm | probe cost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | (4.00,0.35) | 0.081+/-0.036 | 0.092+/-0.045 | 0.086+/-0.047 | 0.086+/-0.046 | 0.003+/-0.003 | -0.006+/-0.002 | -0.007+/-0.000 | -0.007+/-0.006 | 1.34% |

## Interpretation

External HPO uses the same total workers by splitting them across separate LR-arm trajectories.  The internal rule uses those same workers as arms inside one trajectory and only pays probe overhead.

## Run Log

```text
DFW-E088 fixed-compute parallel HPO baseline
host=devvm14473.rva0.facebook.com python=/data/repos/fbsource/buck-out/v2/art/fbcode/ab2a2c0557db637b/aai/double_fan_gpu_runner/__run_e087_e088_static__/run_e087_e088_static#link-tree/runtime/bin/run_e087_e088_static#native-main#platform-runtime#python#py_version_3_12 compute_status=TORCH_CPU torch=2.15.0a0+fb cuda_available=False
N=8 arms=(0.25, 1.0, 4.0) counts=(2, 4, 2) K=(8,) rounds=8
pilot_seeds=1 heldout_seeds=2 alphas=(1.0, 2.0, 4.0) betas=(0.0, 0.35)

CELL K=8 pair=(4.00,0.35) stale=  0.081+/-0.036 default=  0.092+/-0.045 parallel_avg=  0.086+/-0.047 internal=  0.086+/-0.046 internal-shuffled=  0.003+/-0.003 internal-parallel_avg=  0.000+/-0.001 external_valid=  0.092+/-0.048 internal-external_valid= -0.006+/-0.002 external_oracle=  0.093+/-0.046 internal-external_oracle= -0.007+/-0.000 best_full=  0.093+/-0.041 internal-best_full= -0.007+/-0.006 cost=1.34%

POOLED internal-external_valid= -0.006+/-0.002 internal-external_oracle= -0.007+/-0.000 internal-shuffled=  0.003+/-0.003 internal-parallel_avg=  0.000+/-0.001 internal-default= -0.006+/-0.001
DECISION FIXED_COMPUTE_BASELINE_NOT_CLEARED
compute_status=TORCH_CPU
```
