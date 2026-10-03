# LGP-E024 Delayed Trust Validation Result - 2026-10-03

## Decision

`DELAYED_GAMMA_WEAK`

## Question

E022 selected `combo_val_gamma` row-by-row from a same-round held-out mini-loss
proxy, then scored the selected update on the same round's true objective. This
can suffer winner's curse: selecting the best of 8 noisy proxy candidates can
inflate the realized same-round improvement.

E024 tests the cheapest causal control: select gamma on round `t` and apply it
to the learned dense correction on round `t+1`. If E022's gate is measuring a
persistent local trust signal, the delayed arm should keep a positive edge over
tuned scalar. If the same-round result is mostly selection noise, delayed gamma
should collapse.

The bed and operator are unchanged from E022: dense low-rank correlated noise,
`K=8`, `eta=0.025`, and the dynamics-aware correction

```text
[I - (I - eta Hhat)^K]^-1 Delta_K
```

## Cost Row

The E022 validation proxy is not free. With
`LOSS_PROXY_REPS=8`, `coord_batch=24`, `lr_batch=4`, `D=96`, and `rank=8`,
E024 reports:

```text
per gamma = 2.154 full-forward equivalents
full 8-gamma grid = 17.231 full-forward equivalents
full 8-gamma grid = 5.744 full-gradient-call equivalents
```

The last conversion assumes one backward costs two forwards. This is only an
accounting row, not a matched-compute control.

## Main Table

Ratios are held-out loss improvement divided by raw FedAvg improvement. CIs are
paired across held-out replicate batches.

| N | scalar | same-round val-gamma | delayed gamma | delayed, drop first | exact gate | oracle-H |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.046 +/- 0.029 | 1.085 +/- 0.032 | 1.056 +/- 0.027 | 1.230 +/- 0.039 | 1.089 +/- 0.033 | 2.476 +/- 0.654 |
| 32 | 1.040 +/- 0.023 | 1.143 +/- 0.061 | 1.088 +/- 0.052 | 1.404 +/- 0.132 | 1.153 +/- 0.065 | 2.661 +/- 0.719 |
| 128 | 1.032 +/- 0.015 | 1.107 +/- 0.032 | 1.029 +/- 0.073 | 1.183 +/- 0.359 | 1.114 +/- 0.033 | 2.421 +/- 0.376 |

## Required Contrasts

| N | delayed - scalar | delayed - same-round | delayed - E021 combo | delayed - diag | best worker - delayed |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.010 +/- 0.012 | -0.028 +/- 0.005 | 7.084 +/- 3.855 | 0.272 +/- 0.396 | -0.146 +/- 0.055 |
| 32 | 0.048 +/- 0.036 | -0.055 +/- 0.047 | 6.552 +/- 8.650 | 0.224 +/- 0.156 | -0.122 +/- 0.070 |
| 128 | -0.003 +/- 0.080 | -0.078 +/- 0.096 | 2.822 +/- 2.323 | 0.433 +/- 0.272 | -0.031 +/- 0.081 |

The `delayed, drop first` row removes the first round of each replicate so the
initial `gamma=0` convention cannot decide the result. It is unstable and has
wide CIs, so it is diagnostic only.

## Interpretation

The delayed control weakens E022 materially. Same-round validation-gamma beat
scalar at all three worker counts; delayed gamma only gives a clear positive at
`N=32`, is tiny at `N=8`, and is null at `N=128`. It remains far above the
destructive E021 residual combo, so the trust layer still prevents the crash,
but the stronger E022 claim is no longer settled.

This does not prove E022 was pure winner's curse: the run has only four held-out
replicate batches, and the drop-first diagnostic is too noisy to read cleanly.
It does mean E022 should not be cited as a finalized dense-method positive until
a higher-rep delayed/stale validation check lands.

## Recommendation

Run a higher-rep E024b/E026 delayed-validation check with:

- more held-out reps;
- lag-1, lag-2, and shuffled-gamma controls;
- same-round, delayed, and delayed-drop-first contrasts on the same scoring
  subset;
- the same validation-cost accounting row.

Dense low-rank-plus-diagonal remains alive as a safety-gated direction, but the
winner's-curse question is now a real gate, not a footnote.

## Command

```bash
python3 -u experiments/lgp_e024_delayed_trust_validation.py
```
