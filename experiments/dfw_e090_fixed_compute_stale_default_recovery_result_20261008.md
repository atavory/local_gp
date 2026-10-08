# DFW-E090 Result - Fixed-Compute Stale-Default Recovery

Decision label: `FIXED_COMPUTE_STALE_DEFAULT_NOT_CLEARED`

## Compute Status

`TORCH_CPU`: `torch=2.15.0a0+fb cuda_available=False`.

## Design

E090 prices the E085 stale-default guard under fixed-compute accounting. The stale user default is LR multiplier `0.25`; the fresh/default candidate is `1`; and the high diagnostic arm is `4`.

Internal HPO spends all workers in one shared trajectory as grouped LR arms and abandons the stale fallback only when the same-round probe slices agree. External HPO spends the same total workers on independent stale/fresh/high trajectories with counts matching the internal arm allocation, then chooses one by validation loss.

## Pooled

- internal guarded minus stale full-worker default: `-0.008+/-0.002`
- internal guarded minus fresh/default full-worker control: `0.005+/-0.004`
- internal guarded minus shuffled guard: `6.117+/-1.841`
- internal guarded minus external validation-selected HPO: `-0.007+/-0.003`
- internal guarded minus external oracle HPO: `-0.008+/-0.003`
- external validation-selected HPO minus stale full-worker default: `-0.001+/-0.001`
- internal guarded minus best full-worker arm: `-0.008+/-0.002`

## Cells

| K | pair | stale full | fresh full | fresh tuned full | internal guarded | guarded-stale | guarded-shuffled | guarded-ext-valid | ext-valid | ext-oracle | abandon | stale mass | fresh mass | high mass | probe cost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | (18.00,0.45) | 0.056+/-0.013 | 0.046+/-0.014 | 0.055+/-0.014 | 0.051+/-0.015 | -0.005+/-0.003 | 6.695+/-2.778 | -0.004+/-0.003 | 0.055+/-0.014 | 0.058+/-0.014 | 0.220+/-0.070 | 0.780+/-0.070 | 0.220+/-0.070 | 0.000+/-0.000 | 1.34% |
| 16 | (11.00,0.30) | 0.057+/-0.013 | 0.044+/-0.011 | 0.056+/-0.013 | 0.046+/-0.016 | -0.010+/-0.005 | 5.779+/-2.547 | -0.010+/-0.006 | 0.056+/-0.013 | 0.057+/-0.013 | 0.188+/-0.055 | 0.812+/-0.055 | 0.188+/-0.055 | 0.000+/-0.000 | 0.67% |
| 32 | (6.00,0.20) | 0.056+/-0.013 | 0.039+/-0.014 | 0.055+/-0.013 | 0.048+/-0.013 | -0.008+/-0.004 | 5.877+/-4.241 | -0.007+/-0.005 | 0.054+/-0.014 | 0.056+/-0.013 | 0.176+/-0.062 | 0.824+/-0.062 | 0.174+/-0.062 | 0.002+/-0.005 | 0.33% |

## Interpretation

This run does not promote stale-default recovery beyond the fixed-compute baseline.

## Run Log

```text
DFW-E090 fixed-compute stale-default recovery
host=devvm44191.lla0.facebook.com python=/data/users/atavory/fbsource/buck-out/v2/art/fbcode/49da2d1f11ca4895/aai/double_fan_gpu_runner/__run_local_gp_script_static__/run_local_gp_script_static#link-tree/runtime/bin/run_local_gp_script_static#native-main#platform-runtime#python#py_version_3_12 compute_status=TORCH_CPU torch=2.15.0a0+fb cuda_available=False
N=8 arms=(0.25, 1.0, 4.0) counts=(2, 4, 2) K=(8, 16, 32) rounds=36
stale_default=0.25 fresh_default=1 pair_mode=e085_stale select_batch=384 probe_batch=192 slices=4 guard_z=0.5 min_agree=0.75
pilot_seeds=3 heldout_seeds=12

CELL K=8 pair=(18.00,0.45) source=E085 stale guarded pair stale_full=  0.056+/-0.013 fresh_full=  0.046+/-0.014 high_full= -5.481+/-2.269 fresh_tuned_full=  0.055+/-0.014 best_full=  0.056+/-0.013 internal_stale_arm=  0.030+/-0.019 internal_fresh_arm=  0.030+/-0.017 guarded=  0.051+/-0.015 guarded-stale_full= -0.005+/-0.003 guarded-fresh_full=  0.005+/-0.005 guarded-shuffled=  6.695+/-2.778 external_valid=  0.055+/-0.014 guarded-external_valid= -0.004+/-0.003 external_oracle=  0.058+/-0.014 guarded-external_oracle= -0.006+/-0.003 external_stale=  0.055+/-0.014 external_fresh=  0.050+/-0.015 external_high= -4.226+/-1.145 abandon=  0.220+/-0.070 shuffled_abandon=  0.701+/-0.018 mass_stale=  0.780+/-0.070 mass_fresh=  0.220+/-0.070 mass_high=  0.000+/-0.000 capture_vs_stale_external_oracle= -8.388+/-8.635 cost=1.34%
CELL K=16 pair=(11.00,0.30) source=E085 stale guarded pair stale_full=  0.057+/-0.013 fresh_full=  0.044+/-0.011 high_full= -6.626+/-3.376 fresh_tuned_full=  0.056+/-0.013 best_full=  0.057+/-0.013 internal_stale_arm=  0.030+/-0.019 internal_fresh_arm=  0.023+/-0.014 guarded=  0.046+/-0.016 guarded-stale_full= -0.010+/-0.005 guarded-fresh_full=  0.002+/-0.008 guarded-shuffled=  5.779+/-2.547 external_valid=  0.056+/-0.013 guarded-external_valid= -0.010+/-0.006 external_oracle=  0.057+/-0.013 guarded-external_oracle= -0.011+/-0.006 external_stale=  0.055+/-0.012 external_fresh=  0.048+/-0.017 external_high= -7.008+/-2.565 abandon=  0.188+/-0.055 shuffled_abandon=  0.678+/-0.034 mass_stale=  0.812+/-0.055 mass_fresh=  0.188+/-0.055 mass_high=  0.000+/-0.000 capture_vs_stale_external_oracle= -0.715+/-28.905 cost=0.67%
CELL K=32 pair=(6.00,0.20) source=E085 stale guarded pair stale_full=  0.056+/-0.013 fresh_full=  0.039+/-0.014 high_full= -5.333+/-1.399 fresh_tuned_full=  0.055+/-0.013 best_full=  0.056+/-0.013 internal_stale_arm=  0.031+/-0.018 internal_fresh_arm=  0.008+/-0.026 guarded=  0.048+/-0.013 guarded-stale_full= -0.008+/-0.004 guarded-fresh_full=  0.009+/-0.007 guarded-shuffled=  5.877+/-4.241 external_valid=  0.054+/-0.014 guarded-external_valid= -0.007+/-0.005 external_oracle=  0.056+/-0.013 guarded-external_oracle= -0.008+/-0.004 external_stale=  0.055+/-0.013 external_fresh=  0.041+/-0.012 external_high= -6.720+/-2.406 abandon=  0.176+/-0.062 shuffled_abandon=  0.697+/-0.052 mass_stale=  0.824+/-0.062 mass_fresh=  0.174+/-0.062 mass_high=  0.002+/-0.005 capture_vs_stale_external_oracle= -5.806+/-5.285 cost=0.33%

POOLED guarded-stale_full= -0.008+/-0.002 guarded-fresh_full=  0.005+/-0.004 guarded-shuffled=  6.117+/-1.841 guarded-external_valid= -0.007+/-0.003 guarded-external_oracle= -0.008+/-0.003 external_valid-stale_full= -0.001+/-0.001 guarded-best_full= -0.008+/-0.002
DECISION FIXED_COMPUTE_STALE_DEFAULT_NOT_CLEARED
compute_status=TORCH_CPU
```
