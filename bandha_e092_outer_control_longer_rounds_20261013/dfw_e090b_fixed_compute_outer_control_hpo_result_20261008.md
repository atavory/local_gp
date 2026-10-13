# DFW-E090B Result - Fixed-Compute Outer-Control HPO

Decision label: `FIXED_COMPUTE_OUTER_CONTROL_HPO_LIVE`

## Compute Status

`TORCH_CPU`: `torch=2.16.0a0+fb cuda_available=False`.

## Design

This is the fallback fixed-compute table after E090 stale-LR recovery did not clear. The stale/default arm is a transferred global server pair `(5.0, 0.05)`; the fresh arm is the K-aware theory pair; and the high arm is a more aggressive probe arm.

Internal HPO evaluates the outer-control arms from one shared worker-update pool and uses the E080 soft probe selector. External HPO spends the same total workers by splitting them across independent arm budgets and selecting by validation/select improvement. Target selection is reported only as an oracle diagnostic.

## Pooled

- internal minus stale/default control: `2.838+/-1.979`
- internal minus fresh/theory control: `0.619+/-0.472`
- internal minus shuffled internal: `0.771+/-0.632`
- internal minus external validation-selected HPO: `0.330+/-2.151`
- internal minus external oracle HPO: `-0.559+/-2.182`
- external validation-selected HPO minus stale/default: `2.508+/-0.642`
- internal minus internal target oracle: `-2.756+/-1.926`

## Cells

| N | K | stale/default | fresh/theory | high | internal | internal-stale | internal-shuffled | internal-ext-valid | ext-valid | ext-oracle | probe cost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 4 | 4.662+/-0.121 | 10.181+/-2.214 | 7.205+/-4.983 | 10.700+/-1.791 | 6.039+/-1.671 | 0.598+/-0.518 | 2.683+/-1.649 | 8.017+/-1.264 | 9.118+/-0.888 | 3.57% |
| 8 | 8 | 3.920+/-0.584 | 4.600+/-2.580 | 2.188+/-5.902 | 5.072+/-2.066 | 1.152+/-1.484 | 0.611+/-0.804 | 0.640+/-1.805 | 4.432+/-0.865 | 5.143+/-0.552 | 1.79% |
| 16 | 4 | 4.644+/-0.193 | 9.746+/-3.599 | 6.006+/-8.210 | 10.231+/-3.033 | 5.587+/-2.841 | 0.576+/-0.634 | 0.818+/-2.575 | 9.413+/-1.356 | 10.348+/-1.077 | 1.79% |
| 16 | 8 | 3.035+/-2.434 | 0.611+/-10.870 | -7.205+/-25.261 | 1.608+/-9.236 | -1.427+/-6.802 | 1.301+/-2.276 | -2.822+/-7.827 | 4.430+/-1.666 | 5.239+/-1.348 | 0.89% |

## Interpretation

The outer-control fallback is live: it beats stale/default and shuffled controls, with the external-HPO columns marking the remaining fixed-compute boundary.

## Run Log

```text
DFW-E090B fixed-compute outer-control HPO
host=devvm44191.lla0.facebook.com python=/data/users/atavory/fbsource/buck-out/v2/art/fbcode/c95bb19b4474c2e4/aai/double_fan_gpu_runner/__run_local_gp_script_static__/run_local_gp_script_static#link-tree/runtime/bin/run_local_gp_script_static#native-main#platform-runtime#python#py_version_3_12 compute_status=TORCH_CPU torch=2.16.0a0+fb cuda_available=False
K=(4, 8) N=(8, 16) rounds=16 select_batch=384 heldout_seeds=32
arms=stale_global_default,fresh_theory,high_probe_arm policy=soft_std2

CELL N=8 K=4 arms=stale_global_default:(5.0, 0.05)/count=1;fresh_theory:(18.0, 0.15)/count=1;high_probe_arm:(22.5, 0.3)/count=1;low_safe:(9.0, 0.05)/count=1;mid_nomomentum:(18.0, 0.0)/count=1;mid_low_momentum:(13.5, 0.15)/count=1;aggressive_low_momentum:(27.0, 0.15)/count=1;double_nomomentum:(36.0, 0.0)/count=1 stale=  4.662+/-0.121 fresh= 10.181+/-2.214 high=  7.205+/-4.983 internal= 10.700+/-1.791 internal-stale=  6.039+/-1.671 internal-shuffled=  0.598+/-0.518 external_valid=  8.017+/-1.264 internal-external_valid=  2.683+/-1.649 external_oracle=  9.118+/-0.888 internal-external_oracle=  1.583+/-1.421 external_stale=  4.479+/-0.128 external_fresh=  7.412+/-2.215 external_high=  2.625+/-4.758 internal_oracle= 12.636+/-1.114 capture_vs_stale_internal_oracle=  0.017+/-1.598 cost=3.57%
CELL N=8 K=8 arms=stale_global_default:(5.0, 0.05)/count=1;fresh_theory:(9.0, 0.15)/count=1;high_probe_arm:(11.25, 0.3)/count=1;low_safe:(4.5, 0.05)/count=1;mid_nomomentum:(9.0, 0.0)/count=1;mid_low_momentum:(6.75, 0.15)/count=1;aggressive_low_momentum:(13.5, 0.15)/count=1;double_nomomentum:(18.0, 0.0)/count=1 stale=  3.920+/-0.584 fresh=  4.600+/-2.580 high=  2.188+/-5.902 internal=  5.072+/-2.066 internal-stale=  1.152+/-1.484 internal-shuffled=  0.611+/-0.804 external_valid=  4.432+/-0.865 internal-external_valid=  0.640+/-1.805 external_oracle=  5.143+/-0.552 internal-external_oracle= -0.071+/-1.834 external_stale=  3.831+/-0.206 external_fresh=  4.418+/-0.887 external_high=  2.338+/-1.983 internal_oracle=  6.776+/-0.863 capture_vs_stale_internal_oracle=  0.217+/-0.768 cost=1.79%
CELL N=16 K=4 arms=stale_global_default:(5.0, 0.05)/count=2;fresh_theory:(18.0, 0.15)/count=2;high_probe_arm:(22.5, 0.3)/count=2;low_safe:(9.0, 0.05)/count=2;mid_nomomentum:(18.0, 0.0)/count=2;mid_low_momentum:(13.5, 0.15)/count=2;aggressive_low_momentum:(27.0, 0.15)/count=2;double_nomomentum:(36.0, 0.0)/count=2 stale=  4.644+/-0.193 fresh=  9.746+/-3.599 high=  6.006+/-8.210 internal= 10.231+/-3.033 internal-stale=  5.587+/-2.841 internal-shuffled=  0.576+/-0.634 external_valid=  9.413+/-1.356 internal-external_valid=  0.818+/-2.575 external_oracle= 10.348+/-1.077 internal-external_oracle= -0.117+/-2.527 external_stale=  4.567+/-0.135 external_fresh=  8.753+/-2.351 external_high=  4.812+/-5.174 internal_oracle= 12.926+/-1.156 capture_vs_stale_internal_oracle=  0.238+/-1.158 cost=1.79%
CELL N=16 K=8 arms=stale_global_default:(5.0, 0.05)/count=2;fresh_theory:(9.0, 0.15)/count=2;high_probe_arm:(11.25, 0.3)/count=2;low_safe:(4.5, 0.05)/count=2;mid_nomomentum:(9.0, 0.0)/count=2;mid_low_momentum:(6.75, 0.15)/count=2;aggressive_low_momentum:(13.5, 0.15)/count=2;double_nomomentum:(18.0, 0.0)/count=2 stale=  3.035+/-2.434 fresh=  0.611+/-10.870 high= -7.205+/-25.261 internal=  1.608+/-9.236 internal-stale= -1.427+/-6.802 internal-shuffled=  1.301+/-2.276 external_valid=  4.430+/-1.666 internal-external_valid= -2.822+/-7.827 external_oracle=  5.239+/-1.348 internal-external_oracle= -3.631+/-8.028 external_stale=  3.361+/-1.364 external_fresh=  2.231+/-6.000 external_high= -2.944+/-13.654 internal_oracle=  6.298+/-2.137 capture_vs_stale_internal_oracle=  0.256+/-0.760 cost=0.89%

POOLED internal-stale=  2.838+/-1.979 internal-fresh=  0.619+/-0.472 internal-shuffled=  0.771+/-0.632 internal-external_valid=  0.330+/-2.151 internal-external_oracle= -0.559+/-2.182 external_valid-stale=  2.508+/-0.642 internal-internal_oracle= -2.756+/-1.926
DECISION FIXED_COMPUTE_OUTER_CONTROL_HPO_LIVE
compute_status=TORCH_CPU
```
