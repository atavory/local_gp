# LGP-E001 Worker Prompt: Local Path Prediction Gate

You are an Assaf worker. This is a new project, `local_gp`, not a Double Fan
optimizer-controller continuation.

Repository for notes/specs:

```text
https://github.com/atavory/local_gp
```

## Objective

Run the first offline local-GP/path-prediction gate.

Question:

> Do K>1 local-SGD worker paths contain held-out predictive information about
> proposal quality or oracle action beyond round-only and path-norm baselines?

This is not a closed-loop optimizer experiment. Do not run a controller. Do not
claim optimizer value.

## Execution Order

### Phase 1: Ledger Census

Inspect existing Double Fan artifacts first, especially K>1/local-SGD runs:

- E78;
- E81;
- E83;
- E88;
- E102;
- any later run with K>1 path/proposal summaries.

Look under:

```text
manifold://aai_research_tlv/tree/double_fan/runs/assafcohen/
```

Decide whether an existing ledger contains enough information for the gate:

- round index;
- seed;
- K;
- worker proposal/path summaries or endpoint summaries;
- mean proposal;
- audit gradient or proxy target;
- candidate action/oracle score or enough data to compute one.

If yes, run the offline prediction gate on that ledger.

### Phase 2: Fresh Minimal Ledger If Needed

If existing artifacts do not contain sufficient path/proposal data, generate a
minimal fresh ledger.

CPU is acceptable for this fallback. Prefer CUDA only if available.

Fallback task:

- MNIST or Fashion-MNIST if available locally;
- small MLP;
- K in `{2, 4, 8}`;
- N virtual workers in `{16, 32}` or larger if cheap;
- 80-120 synchronization rounds;
- at least 12 pilot seeds and 24 held-out seeds if cheap; otherwise state the
  reduced budget;
- log per-worker local endpoint proposal summaries and path features.

Required per round/seed:

- mean local proposal;
- proposal endpoint spread;
- worker proposal norm statistics;
- worker loss-delta statistics if cheap;
- current global gradient estimate;
- large audit gradient or deterministic validation-loss proxy;
- finite action grid score if feasible.

## Features

Start with cheap descriptors:

- round index;
- K;
- mean proposal norm;
- endpoint spread / proposal variance scalar;
- p10/p50/p90 proposal norms;
- cosine statistics between worker proposals and mean proposal;
- local loss-delta mean/spread;
- if available, shared-probe loss/logit deltas.

Standardize features using source/pilot statistics only.

## Prediction Targets

Use at least one primary target:

1. signed projection of mean proposal onto the negative audit gradient;
2. cosine of mean proposal with the negative audit gradient;
3. oracle action from a finite grid over proposal scale/trust;
4. residual over the best round-only schedule.

If generating a fresh ledger, include a finite action grid such as:

```text
scale in {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}
```

where action is `scale * mean_local_proposal`, scored against the audit gradient
or a one-step validation proxy.

## Models

Baselines:

- round-only mean predictor;
- scalar/source-selected constant predictor;
- path-norm regression;
- mean-proposal-only baseline;
- shuffled-feature control.

Local-GP/KRR models:

- RBF kernel ridge on standardized features;
- Matern-3/2 kernel ridge if easy;
- additive round + path kernel if easy.

Do not build a complex scalable GP. Exact KRR/GP on a small table is enough.

## Splits

Use source-only model selection:

- pilot/source split for choosing features/kernel/hyperparameters;
- held-out source sanity split;
- held-out target split with a shift if possible:
  - K shift, e.g. fit K=4 and test K=8;
  - phase shift, early to later rounds;
  - dataset/noise shift if available.

Do not tune on held-out target.

## Metrics

Report:

- target oracle headroom;
- MSE or regret;
- Spearman correlation;
- top-action regret if using action grid;
- paired 95% CIs over held-out seeds;
- comparison to round-only and path-norm baselines;
- shuffled-feature sanity.

## Decision Labels

- `PATH_SIGNAL_CONFIRMED`: GP/KRR beats round-only and path-norm baselines on
  the primary held-out target metric with CI excluding zero.
- `TRANSFER_CONFIRMED`: the GP/KRR advantage survives a source-to-target shift.
- `SCHEDULE_EXPLAINS`: round-only schedule matches GP/KRR.
- `PATH_NORM_EXPLAINS`: path-norm baseline matches GP/KRR.
- `NO_ORACLE_HEADROOM`: target oracle barely beats frozen source baseline.
- `NO_HELDOUT_SIGNAL`: source fit works but target held-out fails.
- `LEDGER_INSUFFICIENT`: existing ledgers lacked enough path/proposal data and
  the fallback could not generate a compliant ledger.
- `INVALID`: target leakage, missing held-out split, missing oracle/headroom, or
  no baseline controls.

## Required Outputs

Upload artifacts under a run directory and include:

- `lgp_e001_summary.md`
- `lgp_e001_decision.md`
- `lgp_e001_metrics.csv`
- `lgp_e001_model_selection.json`
- `lgp_e001_ledger_census.md`
- `lgp_e001_legitimacy_check.md`
- `final_message.txt`

The final message must include the decision label and one-paragraph
interpretation.
