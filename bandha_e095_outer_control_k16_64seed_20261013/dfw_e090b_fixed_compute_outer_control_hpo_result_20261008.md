# DFW-E090B Result - Fixed-Compute Outer-Control HPO

Decision label: `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`

## Compute Status

`TORCH_CPU`: `torch=2.16.0a0+fb cuda_available=False`.

## Design

This is the fallback fixed-compute table after E090 stale-LR recovery did not clear. The stale/default arm is a transferred global server pair `(5.0, 0.05)`; the fresh arm is the K-aware theory pair; and the high arm is a more aggressive probe arm.

Internal HPO evaluates the outer-control arms from one shared worker-update pool and uses the E080 soft probe selector. External HPO spends the same total workers by splitting them across independent arm budgets and selecting by validation/select improvement. Target selection is reported only as an oracle diagnostic.

## Pooled

- internal minus stale/default control: `0.094+/-0.061`
- internal minus fresh/theory control: `0.115+/-0.056`
- internal minus shuffled internal: `0.075+/-0.048`
- internal minus external validation-selected HPO: `0.516+/-0.262`
- internal minus external oracle HPO: `-0.067+/-0.277`
- external validation-selected HPO minus stale/default: `-0.423+/-0.252`
- internal minus internal target oracle: `-0.968+/-0.259`

## Cells

| N | K | stale/default | fresh/theory | high | internal | internal-stale | internal-shuffled | internal-ext-valid | ext-valid | ext-oracle | probe cost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 3.096+/-0.534 | 3.078+/-0.555 | 2.682+/-1.339 | 3.198+/-0.601 | 0.101+/-0.082 | 0.096+/-0.080 | 0.504+/-0.419 | 2.693+/-0.419 | 3.264+/-0.226 | 0.89% |
| 16 | 16 | 3.235+/-0.257 | 3.210+/-0.259 | 2.967+/-0.620 | 3.321+/-0.343 | 0.086+/-0.091 | 0.054+/-0.055 | 0.528+/-0.317 | 2.793+/-0.451 | 3.389+/-0.256 | 0.45% |

## Interpretation

The outer-control fallback clears the fixed-compute external-HPO comparator: the internal probe selector beats the stale/default control, shuffled labels, and validation-selected external HPO with the same worker budget.

## Run Log

```text
DFW-E090B fixed-compute outer-control HPO
host=devvm44191.lla0.facebook.com python=/data/users/atavory/fbsource/buck-out/v2/art/fbcode/c95bb19b4474c2e4/aai/double_fan_gpu_runner/__run_local_gp_script_static__/run_local_gp_script_static#link-tree/runtime/bin/run_local_gp_script_static#native-main#platform-runtime#python#py_version_3_12 compute_status=TORCH_CPU torch=2.16.0a0+fb cuda_available=False
K=(16,) N=(8, 16) rounds=16 select_batch=384 heldout_seeds=64
arms=stale_global_default,fresh_theory,high_probe_arm policy=soft_std2

CELL N=8 K=16 arms=stale_global_default:(5.0, 0.05)/count=1;fresh_theory:(4.5, 0.15)/count=1;high_probe_arm:(5.625, 0.3)/count=1;low_safe:(2.25, 0.05)/count=1;mid_nomomentum:(4.5, 0.0)/count=1;mid_low_momentum:(3.375, 0.15)/count=1;aggressive_low_momentum:(6.75, 0.15)/count=1;double_nomomentum:(9.0, 0.0)/count=1 stale=  3.096+/-0.534 fresh=  3.078+/-0.555 high=  2.682+/-1.339 internal=  3.198+/-0.601 internal-stale=  0.101+/-0.082 internal-shuffled=  0.096+/-0.080 external_valid=  2.693+/-0.419 internal-external_valid=  0.504+/-0.419 external_oracle=  3.264+/-0.226 internal-external_oracle= -0.067+/-0.502 external_stale=  2.777+/-0.347 external_fresh=  2.845+/-0.331 external_high=  2.354+/-0.756 internal_oracle=  4.204+/-0.218 capture_vs_stale_internal_oracle=  0.120+/-0.062 cost=0.89%
CELL N=16 K=16 arms=stale_global_default:(5.0, 0.05)/count=2;fresh_theory:(4.5, 0.15)/count=2;high_probe_arm:(5.625, 0.3)/count=2;low_safe:(2.25, 0.05)/count=2;mid_nomomentum:(4.5, 0.0)/count=2;mid_low_momentum:(3.375, 0.15)/count=2;aggressive_low_momentum:(6.75, 0.15)/count=2;double_nomomentum:(9.0, 0.0)/count=2 stale=  3.235+/-0.257 fresh=  3.210+/-0.259 high=  2.967+/-0.620 internal=  3.321+/-0.343 internal-stale=  0.086+/-0.091 internal-shuffled=  0.054+/-0.055 external_valid=  2.793+/-0.451 internal-external_valid=  0.528+/-0.317 external_oracle=  3.389+/-0.256 internal-external_oracle= -0.068+/-0.238 external_stale=  2.917+/-0.259 external_fresh=  2.965+/-0.253 external_high=  2.557+/-0.583 internal_oracle=  4.250+/-0.234 capture_vs_stale_internal_oracle=  0.104+/-0.071 cost=0.45%

POOLED internal-stale=  0.094+/-0.061 internal-fresh=  0.115+/-0.056 internal-shuffled=  0.075+/-0.048 internal-external_valid=  0.516+/-0.262 internal-external_oracle= -0.067+/-0.277 external_valid-stale= -0.423+/-0.252 internal-internal_oracle= -0.968+/-0.259
DECISION FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN
compute_status=TORCH_CPU
```
