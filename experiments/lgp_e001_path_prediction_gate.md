# LGP-E001: Local Path Prediction Gate

## Purpose

Test whether local worker paths contain held-out predictive information beyond
round-only schedules and simple path-norm baselines.

This is an offline gate. It is not a closed-loop optimizer experiment.

## Data

Use an existing K>1 local-SGD ledger if available. Acceptable sources:

- MNIST/MLP K>1 ledgers from the Double Fan campaign, if they contain worker
  path or proposal summaries;
- CIFAR-10 ResNet K>1 ledgers, if available;
- a new minimal K>1 run that logs worker paths and audit gradients.

Required per round/seed:

- synchronized starting point id or hash;
- worker endpoint proposal summaries;
- mean proposal;
- local losses or loss deltas if cheap;
- audit gradient or target oracle score;
- round index;
- setting metadata: K, batch size, worker count, model, seed.

Preferred additional data:

- shared-probe loss/logit changes;
- same-batch finite-difference probe;
- endpoint spread/covariance summaries.

## Train/Test Split

Use source-only fitting:

- source pilot seeds/rounds/settings for model selection;
- held-out source sanity split;
- held-out target split with at least one shift: K, batch/noise, phase, or task.

Do not tune on the held-out target split.

## Prediction Targets

Score at least one primary target:

1. oracle action from a finite grid;
2. audit-gradient alignment of each proposal;
3. validation-loss improvement proxy;
4. residual over the best round-only schedule.

If multiple targets are available, report all but pre-register one primary
before looking at target outcomes.

## Models

Baselines:

- round-only schedule;
- scalar source-selected baseline;
- mean proposal baseline;
- path-norm regression;
- shuffled-feature control.

Local-GP candidates:

- RBF-kernel ridge regression on standardized path/probe features;
- Matern-kernel ridge regression;
- additive round + path kernel;
- sparse/inducing approximation only if the exact kernel is too expensive.

Use GP/KRR first. Do not build an elaborate scalable GP until this small gate
passes.

## Metrics

For continuous targets:

- MSE;
- Spearman correlation;
- top-action regret versus oracle;
- paired 95% CI over held-out seeds.

For action targets:

- oracle-action accuracy;
- regret of selected action;
- top-1 and top-2 agreement;
- paired 95% CI over held-out seeds.

Always report target oracle headroom:

```text
headroom = loss(frozen source baseline) - loss(target oracle)
```

If headroom is near zero, the venue cannot show value.

## Decision Labels

- `PATH_SIGNAL_CONFIRMED`: GP/KRR beats round-only and path-norm baselines on
  the primary held-out target metric with CI excluding zero.
- `TRANSFER_CONFIRMED`: the GP/KRR advantage survives a source-to-target shift.
- `SCHEDULE_EXPLAINS`: round-only schedule matches GP/KRR.
- `PATH_NORM_EXPLAINS`: path-norm baseline matches GP/KRR.
- `NO_ORACLE_HEADROOM`: target oracle barely beats frozen source baseline.
- `NO_HELDOUT_SIGNAL`: source fit is positive but target held-out fails.
- `INVALID`: target leakage, missing oracle/headroom, missing held-out split, or
  absent baseline controls.

## Output Files

- `lgp_e001_summary.md`
- `lgp_e001_decision.md`
- `lgp_e001_metrics.csv`
- `lgp_e001_model_selection.json`
- `lgp_e001_legitimacy_check.md`

## Stop Rule

If E001 returns `SCHEDULE_EXPLAINS`, `PATH_NORM_EXPLAINS`,
`NO_ORACLE_HEADROOM`, or `NO_HELDOUT_SIGNAL`, do not run a closed-loop local-GP
optimizer. First decide whether the issue is data, features, or the whole idea.
