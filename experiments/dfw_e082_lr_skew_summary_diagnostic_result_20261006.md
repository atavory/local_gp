# DFW-E082 Result - LR Skew Summary Diagnostic

Decision label: `LR_SKEW_SUMMARY_NO_SAFE_COLLAPSE_TRACE_NEEDED`

## Compute Status

`REANALYSIS_EXISTING_REAL_GPU_ARTIFACTS`: no new GPU work was run locally. The inputs are the durable E060/E061 `REAL_GPU` result tables.

## Summary

- Nontrivial spread cells: `10`.
- Pilot-best policy counts: `group_l05_t4` = 1, `group_l20_t2` = 1, `group_mean2_l40_t2` = 8.
- Mean cell-level pilot-best minus fixed: `0.245 +/- 0.137`.
- Mean cell-level pilot-best minus shuffled: `0.447 +/- 0.231`.
- Mean residual oracle headroom after pilot-best: `2.830 +/- 0.597`.

Interpretation: the E060/E061 summaries do not show a broad collapse to the most conservative/default-preserving policy.  Nine of ten nontrivial cells choose a nontrivial damped policy, and the pooled cell means remain positive versus both fixed and shuffled controls.  The one conservative-prior cell is the hardest `K=32, spread=16` E061 case, where the pilot-best mean is essentially tied with fixed.

Important limitation: these artifacts do not expose exact low/default/high LR arm weights, so they cannot by themselves prove whether mass moved toward the small/safe LR arm or the high LR arm.  A precise arm-mass trace should be run on the GPU helper path before upgrading free LR search from `GATED` to `LIVE`.

## Nontrivial Cells

| source | spread | N | K | groups | pilot-best | cost | score | minus fixed | minus shuffled | oracle subset | sigma/gap | mean2 top3 |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| E060 | 4 | 8 | 8 | 0.5,1,2 | group_mean2_l40_t2 | 1.34% | 5.185 | 0.167 | 0.228 | 8.597 | 3.07 | 1.000 |
| E060 | 16 | 8 | 8 | 0.25,1,4 | group_mean2_l40_t2 | 1.34% | 3.531 | 0.304 | 0.691 | 7.644 | 3.60 | 1.000 |
| E061 | 4 | 8 | 4 | 0.5,1,2 | group_mean2_l40_t2 | 2.68% | 8.446 | 0.672 | 1.034 | 12.407 | 3.10 | 1.000 |
| E061 | 16 | 8 | 4 | 0.25,1,4 | group_mean2_l40_t2 | 2.68% | 6.143 | 0.597 | 1.103 | 9.756 | 4.01 | 1.000 |
| E061 | 4 | 8 | 8 | 0.5,1,2 | group_l20_t2 | 0.67% | 4.616 | 0.152 | 0.263 | 7.564 | 3.11 | 1.000 |
| E061 | 16 | 8 | 8 | 0.25,1,4 | group_mean2_l40_t2 | 1.34% | 3.441 | 0.231 | 0.460 | 5.826 | 3.07 | 1.000 |
| E061 | 4 | 8 | 16 | 0.5,1,2 | group_mean2_l40_t2 | 0.67% | 2.576 | 0.102 | 0.265 | 4.633 | 4.38 | 1.000 |
| E061 | 16 | 8 | 16 | 0.25,1,4 | group_mean2_l40_t2 | 0.67% | 1.214 | 0.156 | 0.237 | 2.706 | 4.88 | 1.000 |
| E061 | 4 | 8 | 32 | 0.5,1,2 | group_mean2_l40_t2 | 0.33% | 1.187 | 0.075 | 0.157 | 2.653 | 6.28 | 1.000 |
| E061 | 16 | 8 | 32 | 0.25,1,4 | group_l05_t4 | 0.17% | 1.332 | -0.002 | 0.029 | 4.185 | 3.38 | 1.000 |

## Run Log

```text
DFW-E082 LR skew summary diagnostic
compute_status=REANALYSIS_EXISTING_REAL_GPU_ARTIFACTS
E060: /data/users/atavory/local_gp/experiments/dfw_e060_gpu_stationary_inner_lr_group_probe_admission_result_20261005.md rows=3
E061: /data/users/atavory/local_gp/experiments/dfw_e061_gpu_stationary_group_admission_k_sweep_result_20261005.md rows=12
nontrivial_cells=10
policy_counts=group_l05_t4:1,group_l20_t2:1,group_mean2_l40_t2:8
mean_minus_fixed=0.245 +/- 0.137
mean_minus_shuffled=0.447 +/- 0.231
mean_oracle_headroom=2.830 +/- 0.597
conservative_prior_policy_cells=1
DECISION LR_SKEW_SUMMARY_NO_SAFE_COLLAPSE_TRACE_NEEDED
```
