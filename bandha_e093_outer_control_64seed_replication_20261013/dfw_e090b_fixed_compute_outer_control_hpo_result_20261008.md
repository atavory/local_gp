# DFW-E090B Result - Fixed-Compute Outer-Control HPO

Decision label: `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`

## Compute Status

`TORCH_CPU`: `torch=2.16.0a0+fb cuda_available=False`.

## Design

This is the fallback fixed-compute table after E090 stale-LR recovery did not clear. The stale/default arm is a transferred global server pair `(5.0, 0.05)`; the fresh arm is the K-aware theory pair; and the high arm is a more aggressive probe arm.

Internal HPO evaluates the outer-control arms from one shared worker-update pool and uses the E080 soft probe selector. External HPO spends the same total workers by splitting them across independent arm budgets and selecting by validation/select improvement. Target selection is reported only as an oracle diagnostic.

## Pooled

- internal minus stale/default control: `3.829+/-0.378`
- internal minus fresh/theory control: `0.095+/-0.057`
- internal minus shuffled internal: `0.140+/-0.058`
- internal minus external validation-selected HPO: `1.224+/-0.313`
- internal minus external oracle HPO: `0.343+/-0.255`
- external validation-selected HPO minus stale/default: `2.605+/-0.443`
- internal minus internal target oracle: `-1.092+/-0.106`

## Cells

| N | K | stale/default | fresh/theory | high | internal | internal-stale | internal-shuffled | internal-ext-valid | ext-valid | ext-oracle | probe cost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 4 | 4.675+/-0.035 | 10.607+/-0.636 | 8.443+/-1.392 | 10.737+/-0.696 | 6.061+/-0.662 | 0.215+/-0.137 | 1.863+/-0.733 | 8.873+/-0.883 | 9.808+/-0.729 | 3.57% |
| 8 | 8 | 4.141+/-0.081 | 5.596+/-0.356 | 4.567+/-0.803 | 5.636+/-0.408 | 1.495+/-0.329 | 0.072+/-0.081 | 0.847+/-0.336 | 4.789+/-0.495 | 5.460+/-0.371 | 1.79% |
| 16 | 4 | 4.685+/-0.033 | 10.758+/-0.607 | 8.721+/-1.340 | 10.883+/-0.646 | 6.197+/-0.614 | 0.183+/-0.141 | 1.510+/-0.821 | 9.373+/-1.066 | 10.548+/-0.743 | 1.79% |
| 16 | 8 | 4.152+/-0.080 | 5.629+/-0.357 | 4.619+/-0.817 | 5.716+/-0.389 | 1.564+/-0.311 | 0.088+/-0.088 | 0.675+/-0.450 | 5.040+/-0.606 | 5.785+/-0.401 | 0.89% |

## Interpretation

The outer-control fallback clears the fixed-compute external-HPO comparator: the internal probe selector beats the stale/default control, shuffled labels, and validation-selected external HPO with the same worker budget.

## Run Log

```text
DFW-E090B fixed-compute outer-control HPO
host=devvm44191.lla0.facebook.com python=/data/users/atavory/fbsource/buck-out/v2/art/fbcode/c95bb19b4474c2e4/aai/double_fan_gpu_runner/__run_local_gp_script_static__/run_local_gp_script_static#link-tree/runtime/bin/run_local_gp_script_static#native-main#platform-runtime#python#py_version_3_12 compute_status=TORCH_CPU torch=2.16.0a0+fb cuda_available=False
K=(4, 8) N=(8, 16) rounds=8 select_batch=384 heldout_seeds=64
arms=stale_global_default,fresh_theory,high_probe_arm policy=soft_std2

CELL N=8 K=4 arms=stale_global_default:(5.0, 0.05)/count=1;fresh_theory:(18.0, 0.15)/count=1;high_probe_arm:(22.5, 0.3)/count=1;low_safe:(9.0, 0.05)/count=1;mid_nomomentum:(18.0, 0.0)/count=1;mid_low_momentum:(13.5, 0.15)/count=1;aggressive_low_momentum:(27.0, 0.15)/count=1;double_nomomentum:(36.0, 0.0)/count=1 stale=  4.675+/-0.035 fresh= 10.607+/-0.636 high=  8.443+/-1.392 internal= 10.737+/-0.696 internal-stale=  6.061+/-0.662 internal-shuffled=  0.215+/-0.137 external_valid=  8.873+/-0.883 internal-external_valid=  1.863+/-0.733 external_oracle=  9.808+/-0.729 internal-external_oracle=  0.929+/-0.693 external_stale=  4.553+/-0.049 external_fresh=  8.798+/-0.880 external_high=  5.631+/-1.828 internal_oracle= 12.055+/-0.599 capture_vs_stale_internal_oracle=  0.759+/-0.102 cost=3.57%
CELL N=8 K=8 arms=stale_global_default:(5.0, 0.05)/count=1;fresh_theory:(9.0, 0.15)/count=1;high_probe_arm:(11.25, 0.3)/count=1;low_safe:(4.5, 0.05)/count=1;mid_nomomentum:(9.0, 0.0)/count=1;mid_low_momentum:(6.75, 0.15)/count=1;aggressive_low_momentum:(13.5, 0.15)/count=1;double_nomomentum:(18.0, 0.0)/count=1 stale=  4.141+/-0.081 fresh=  5.596+/-0.356 high=  4.567+/-0.803 internal=  5.636+/-0.408 internal-stale=  1.495+/-0.329 internal-shuffled=  0.072+/-0.081 external_valid=  4.789+/-0.495 internal-external_valid=  0.847+/-0.336 external_oracle=  5.460+/-0.371 internal-external_oracle=  0.176+/-0.341 external_stale=  3.982+/-0.095 external_fresh=  5.021+/-0.416 external_high=  3.633+/-0.932 internal_oracle=  6.483+/-0.328 capture_vs_stale_internal_oracle=  0.046+/-0.789 cost=1.79%
CELL N=16 K=4 arms=stale_global_default:(5.0, 0.05)/count=2;fresh_theory:(18.0, 0.15)/count=2;high_probe_arm:(22.5, 0.3)/count=2;low_safe:(9.0, 0.05)/count=2;mid_nomomentum:(18.0, 0.0)/count=2;mid_low_momentum:(13.5, 0.15)/count=2;aggressive_low_momentum:(27.0, 0.15)/count=2;double_nomomentum:(36.0, 0.0)/count=2 stale=  4.685+/-0.033 fresh= 10.758+/-0.607 high=  8.721+/-1.340 internal= 10.883+/-0.646 internal-stale=  6.197+/-0.614 internal-shuffled=  0.183+/-0.141 external_valid=  9.373+/-1.066 internal-external_valid=  1.510+/-0.821 external_oracle= 10.548+/-0.743 internal-external_oracle=  0.335+/-0.571 external_stale=  4.611+/-0.050 external_fresh=  9.628+/-0.902 external_high=  6.860+/-1.919 internal_oracle= 12.214+/-0.562 capture_vs_stale_internal_oracle=  0.789+/-0.058 cost=1.79%
CELL N=16 K=8 arms=stale_global_default:(5.0, 0.05)/count=2;fresh_theory:(9.0, 0.15)/count=2;high_probe_arm:(11.25, 0.3)/count=2;low_safe:(4.5, 0.05)/count=2;mid_nomomentum:(9.0, 0.0)/count=2;mid_low_momentum:(6.75, 0.15)/count=2;aggressive_low_momentum:(13.5, 0.15)/count=2;double_nomomentum:(18.0, 0.0)/count=2 stale=  4.152+/-0.080 fresh=  5.629+/-0.357 high=  4.619+/-0.817 internal=  5.716+/-0.389 internal-stale=  1.564+/-0.311 internal-shuffled=  0.088+/-0.088 external_valid=  5.040+/-0.606 internal-external_valid=  0.675+/-0.450 external_oracle=  5.785+/-0.401 internal-external_oracle= -0.069+/-0.312 external_stale=  4.055+/-0.100 external_fresh=  5.284+/-0.434 external_high=  4.045+/-0.963 internal_oracle=  6.587+/-0.325 capture_vs_stale_internal_oracle=  0.465+/-0.185 cost=0.89%

POOLED internal-stale=  3.829+/-0.378 internal-fresh=  0.095+/-0.057 internal-shuffled=  0.140+/-0.058 internal-external_valid=  1.224+/-0.313 internal-external_oracle=  0.343+/-0.255 external_valid-stale=  2.605+/-0.443 internal-internal_oracle= -1.092+/-0.106
DECISION FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN
compute_status=TORCH_CPU
```
