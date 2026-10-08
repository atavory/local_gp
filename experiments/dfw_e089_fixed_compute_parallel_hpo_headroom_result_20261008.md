# DFW-E089 Result - Fixed-Compute Parallel HPO Headroom

Decision label: `INTERNAL_HPO_CONTENT_LIVE_EXTERNAL_HPO_TIE`

## Compute Status

`TORCH_CPU`: `torch=2.15.0a0+fb cuda_available=False`.

## Design

E089 reuses the E063 grouped-LR spread-16 headroom regime: `N=8`, arms `(0.25, 1.0, 4.0)`, counts `(2, 4, 2)`, `K=(8, 16, 32)`, and `36` rounds.

Internal HPO spends all workers in one shared trajectory and recombines grouped candidate updates. External HPO spends the same total workers on separate LR-arm trajectories and selects one by validation loss.

## Pooled

- internal minus external validation-selected HPO: `-0.006+/-0.003`
- internal minus external oracle HPO: `-0.007+/-0.003`
- internal minus shuffled internal: `0.004+/-0.002`
- internal minus parallel-arm average: `0.001+/-0.001`
- internal minus default full-worker arm: `-0.006+/-0.003`
- internal minus best full-worker arm: `-0.006+/-0.003`

## Cells

| K | pair | pair source | low full-worker | default full-worker | parallel arm average | internal probe | internal minus shuffled | internal minus ext-valid | internal minus ext-oracle | internal minus best full-worker arm | probe cost |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | (3.00,0.40) | E063 spread-16 fixed pair | 0.065+/-0.019 | 0.068+/-0.019 | 0.063+/-0.019 | 0.063+/-0.019 | 0.003+/-0.003 | -0.004+/-0.005 | -0.005+/-0.005 | -0.005+/-0.005 | 1.34% |
| 16 | (2.00,0.20) | E063 spread-16 fixed pair | 0.065+/-0.018 | 0.068+/-0.019 | 0.061+/-0.020 | 0.062+/-0.020 | 0.004+/-0.003 | -0.006+/-0.006 | -0.007+/-0.006 | -0.006+/-0.006 | 0.67% |
| 32 | (2.00,0.00) | E063 spread-16 fixed pair | 0.068+/-0.019 | 0.068+/-0.020 | 0.059+/-0.022 | 0.061+/-0.022 | 0.006+/-0.003 | -0.007+/-0.007 | -0.008+/-0.006 | -0.008+/-0.007 | 0.33% |

## Interpretation

The run shows internal HPO content against one or more internal controls, but the fixed-compute external-HPO comparison remains the binding check.

## Run Log

```text
DFW-E089 fixed-compute parallel HPO headroom
host=devvm44191.lla0.facebook.com python=/data/users/atavory/fbsource/buck-out/v2/art/fbcode/49da2d1f11ca4895/aai/double_fan_gpu_runner/__run_local_gp_script_static__/run_local_gp_script_static#link-tree/runtime/bin/run_local_gp_script_static#native-main#platform-runtime#python#py_version_3_12 compute_status=TORCH_CPU torch=2.15.0a0+fb cuda_available=False
N=8 spread=16 arms=(0.25, 1.0, 4.0) counts=(2, 4, 2) K=(8, 16, 32) rounds=36
pair_mode=e063_fixed pilot_seeds=3 heldout_seeds=8 select_batch=384
alphas=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0) betas=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6) spec=group_mean2_l40_t2

CELL K=8 pair=(3.00,0.40) source=E063 spread-16 fixed pair low_full=  0.065+/-0.019 default_full=  0.068+/-0.019 parallel_avg=  0.063+/-0.019 internal=  0.063+/-0.019 internal-shuffled=  0.003+/-0.003 internal-parallel_avg=  0.001+/-0.001 external_valid=  0.068+/-0.019 internal-external_valid= -0.004+/-0.005 external_oracle=  0.068+/-0.019 internal-external_oracle= -0.005+/-0.005 best_full=  0.069+/-0.019 internal-best_full= -0.005+/-0.005 cost=1.34%
CELL K=16 pair=(2.00,0.20) source=E063 spread-16 fixed pair low_full=  0.065+/-0.018 default_full=  0.068+/-0.019 parallel_avg=  0.061+/-0.020 internal=  0.062+/-0.020 internal-shuffled=  0.004+/-0.003 internal-parallel_avg=  0.001+/-0.001 external_valid=  0.068+/-0.018 internal-external_valid= -0.006+/-0.006 external_oracle=  0.069+/-0.018 internal-external_oracle= -0.007+/-0.006 best_full=  0.069+/-0.019 internal-best_full= -0.006+/-0.006 cost=0.67%
CELL K=32 pair=(2.00,0.00) source=E063 spread-16 fixed pair low_full=  0.068+/-0.019 default_full=  0.068+/-0.020 parallel_avg=  0.059+/-0.022 internal=  0.061+/-0.022 internal-shuffled=  0.006+/-0.003 internal-parallel_avg=  0.002+/-0.001 external_valid=  0.069+/-0.019 internal-external_valid= -0.007+/-0.007 external_oracle=  0.069+/-0.018 internal-external_oracle= -0.008+/-0.006 best_full=  0.069+/-0.019 internal-best_full= -0.008+/-0.007 cost=0.33%

POOLED internal-external_valid= -0.006+/-0.003 internal-external_oracle= -0.007+/-0.003 internal-shuffled=  0.004+/-0.002 internal-parallel_avg=  0.001+/-0.001 internal-default_full= -0.006+/-0.003 internal-best_full= -0.006+/-0.003
DECISION INTERNAL_HPO_CONTENT_LIVE_EXTERNAL_HPO_TIE
compute_status=TORCH_CPU
```
