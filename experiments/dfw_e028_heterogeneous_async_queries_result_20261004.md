# DFW-E028 Result - Heterogeneous Async Query Channels

Decision label: `TAGGED_GLOBAL_POSTERIOR_LIVE`

## Setup

Ran 40 paired seeds on 18-dimensional diagonal convex quadratics with 8 worker channels, wall-clock horizon 42.0, and query-cost budget 176.0.  Async controllers scale each event update by arrivals / worker_count, so identical workers collapse to the same update as a synchronous average.

Worker tags included reported noise scale/covariance shape, reported bias, latency, and cost.
`async_tagged_posterior` used those reported tags with staleness/downweighting; `oracle_tags` used true bias/noise plus exact quadratic transport from stale query points.

## Headline Metrics

- Homogeneous final objective: average=0.04634, tagged=0.04634, relative degrade=-0.000%.
- Heterogeneous final objective: average=0.33736, tagged=0.08990, oracle=0.05158.
- Heterogeneous tagged gain over homogeneous averaging: 72.072%; oracle gain: 85.739%; oracle headroom over tagged: 13.667%.

## Calibration

- homogeneous: tagged-vs-true reliability corr=nan, noise-only corr=nan, equal-weight L1 gap=0.000, tagged-weight L1 gap=0.000.
- heterogeneous: tagged-vs-true reliability corr=0.941, noise-only corr=0.487, equal-weight L1 gap=0.787, tagged-weight L1 gap=0.223.

## Interpretation

In the homogeneous regime, all tags are equal and arrivals are simultaneous, so the tagged posterior behaves like averaging rather than inventing a new action. The measured difference stays inside the predeclared tie band.

In the heterogeneous regime, equal averaging is pulled by fast biased workers and by high-variance channels.  Noise-only GLS helps only partially because it cannot remove client bias.  Dropping stale async reports wastes slow reliable channels. The tagged posterior uses reported bias/noise/latency/cost tags to assign trust, keeps useful asynchronous evidence, and beats equal averaging at the same ceilings.

The oracle remains better, so the experiment does not claim the tag model is optimal. It does support the narrower product claim: tagged async/global observations can beat homogeneous averaging under strong channel heterogeneity while degrading gracefully to averaging when homogeneity is true.

## Run Log

```text
DFW-E028 heterogeneous async query channels
d=18 workers=8 seeds=40 wall_clock=42.0 query_budget=176.0 eta=0.105 stale_limit=2

REGIME homogeneous
calibration tagged_corr=nan noise_corr=nan equal_l1=0.0000 tagged_l1=0.0000 oracle_weight_entropy=2.0794
method                                       final_loss            improve          cost        done     waste_frac    updates        gain_vs_avg   win_vs_avg
sync_homogeneous_average                0.0463+/-0.0114      0.996+/-0.001   176.0+/-0.0  176.0+/-0.0   0.000+/-0.000  22.0+/-0.0      0.000+/-0.000       0.000
sync_weighted_by_reported_noise         0.0463+/-0.0114      0.996+/-0.001   176.0+/-0.0  176.0+/-0.0   0.000+/-0.000  22.0+/-0.0      0.000+/-0.000       0.000
async_drop_stale                        0.0463+/-0.0114      0.996+/-0.001   176.0+/-0.0  176.0+/-0.0   0.000+/-0.000  22.0+/-0.0      0.000+/-0.000       0.000
async_tagged_posterior                  0.0463+/-0.0114      0.996+/-0.001   176.0+/-0.0  176.0+/-0.0   0.000+/-0.000  22.0+/-0.0      0.000+/-0.000       0.000
oracle_tags                             0.0449+/-0.0113      0.997+/-0.001   176.0+/-0.0  176.0+/-0.0   0.000+/-0.000  22.0+/-0.0      0.034+/-0.025       0.750

REGIME heterogeneous
calibration tagged_corr=0.9413 noise_corr=0.4869 equal_l1=0.7869 tagged_l1=0.2234 oracle_weight_entropy=1.6457
method                                       final_loss            improve          cost        done     waste_frac    updates        gain_vs_avg   win_vs_avg
sync_homogeneous_average                0.3374+/-0.0356      0.974+/-0.004   147.2+/-1.9  135.4+/-1.7   0.000+/-0.000  16.9+/-0.2      0.000+/-0.000       0.000
sync_weighted_by_reported_noise         0.1325+/-0.0266      0.990+/-0.002   147.2+/-1.9  135.4+/-1.7   0.000+/-0.000  16.9+/-0.2      0.636+/-0.047       1.000
async_drop_stale                        1.9084+/-0.1363      0.851+/-0.018   175.7+/-0.1  178.4+/-0.2   0.485+/-0.002  91.9+/-0.4     -4.922+/-0.316       0.000
async_tagged_posterior                  0.0899+/-0.0129      0.993+/-0.001   175.7+/-0.1  178.4+/-0.2   0.393+/-0.005 178.4+/-0.2      0.721+/-0.043       1.000
oracle_tags                             0.0516+/-0.0120      0.996+/-0.001   175.7+/-0.1  178.4+/-0.2   0.000+/-0.000 178.4+/-0.2      0.857+/-0.025       1.000
```
