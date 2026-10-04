# DFW-E028 CPU Prompt - Heterogeneous Async Query Channels

## Mission

This is a Beyond Averaging / local_gp experiment queued through the `double_fan`
project-worker channel.

Test the claim that worker homogeneity and synchronization are conveniences,
not assumptions. In the new framing, workers contribute tagged observations:

```text
worker_id, point/version, gradient, loss, noise, bias/staleness metadata, cost
```

The controller should decide how much each observation is worth.

## Repository

Use `/data/users/atavory/local_gp` if present. If not present, clone
`https://github.com/atavory/local_gp` through Mr. Gitties. Do not use `pip`.

Read:

```text
PROJECT.md
algorithm_stack.md
model.md
```

## Required Work

Create and run:

```text
experiments/dfw_e028_heterogeneous_async_queries.py
```

Write:

```text
experiments/dfw_e028_heterogeneous_async_queries_result_20261004.md
experiments/dfw_e028_heterogeneous_async_queries_run_20261004.txt
```

Also upload the script, result note, and run log to this run's Manifold artifact
directory.

## Experiment

Build a synthetic convex problem with multiple worker channels. Workers differ
in at least:

- gradient noise covariance/scale;
- optional additive worker bias/client heterogeneity;
- latency/staleness or cost per observation.

Compare at matched wall-clock/query budget:

- `sync_homogeneous_average`: pretend all workers are exchangeable and average;
- `sync_weighted_by_reported_noise`: GLS/precision-weighted observations;
- `async_drop_stale`: accept async observations but discard stale ones by age;
- `async_tagged_posterior`: use version/noise/bias tags in a global posterior
  or trust-weighted local model;
- `oracle_tags`: true noise/bias/staleness upper bound.

Include at least two regimes:

1. homogeneity true, where the fancy method should reduce to averaging;
2. heterogeneity strong, where averaging should be wrong.

## Metrics

Report:

- final objective or held-out one-step improvement;
- calibration of estimated worker weights versus true reliability;
- wasted-query fraction due to stale or downweighted observations;
- whether the method degrades gracefully in the homogeneous regime;
- oracle headroom.

## Decision Labels

Choose exactly one:

- `TAGGED_GLOBAL_POSTERIOR_LIVE`: tagged async/global method beats homogeneous
  averaging in the heterogeneous regime and ties it in the homogeneous regime.
- `HOMOGENEOUS_AVERAGING_SUFFICIENT`: averaging ties the tagged method even
  when heterogeneity is strong.
- `TAG_MODEL_TOO_FRAGILE`: tagged method helps one regime but hurts another
  enough to block the claim.
- `NO_HEADROOM`: oracle tags do not beat averaging.
- `INVALID`: missing controls, unpaired seeds, run failure, or oracle leakage.

## Reporting Rules

This is not an optimizer-product claim unless it beats averaging at matched
query cost. Detection of heterogeneity alone is not enough.
