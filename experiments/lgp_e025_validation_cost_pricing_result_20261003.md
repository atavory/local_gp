# LGP-E025 Validation Cost Pricing Result - 2026-10-03

## Decision

`VALIDATION_COST_SURVIVES`

## Question

E022 showed that same-round held-out mini-loss validation can keep the learned
low-rank-plus-diagonal dense correction from exploding and can beat tuned
scalar FedAvg scale. E024 weakened that claim with a delayed-gamma control.
E025 prices the E022 validation grid and asks whether the same-round gain is
explained by spending the same approximate compute elsewhere.

This result separates two claims:

- Quality: same-round validation improves the update on this synthetic bed.
- Efficiency: that improvement survives matched-compute scalar / extra-fit
  rivals.

The efficiency claim is conditional on the same-round gate. E024 still blocks
calling this a settled causal dense-method result.

## Setup

Same dense bed and K-step operator as E021/E022:

```text
[I - (I - eta Hhat)^K]^-1 Delta_K
```

Parameters: `D=96`, `rank=8`, `K=8`, `eta=0.025`, `rounds=24`,
`pilot_reps=3`, `heldout_reps=4`, `probe_reps=3`, `probe_budget_cap=128`.

Validation grid: `{0, .025, .05, .1, .2, .4, .7, 1}`.

Mini-loss proxy: `coord_batch=24`, `lr_batch=4`, with validation reps
`1, 2, 4, 8`.

Matched-compute rivals:

- `extra_fit_budget`: spend the full 8-rep validation budget on extra random
  fit probes. Nearest match is `+2` offsets, or 6 gradient calls.
- `extra_worker_scalar`: spend the same budget on a feasible scalar rival by
  using `N+1` workers. This costs 8 extra local gradients per round.

## Cost Ledger

`val_grad_b2` assumes backward = 2x forward, so one gradient call is three
forward equivalents. `val_grad_b1` assumes backward = 1x forward, so one
gradient call is two forward equivalents.

| arm | N | fit grad calls / round | val forward eq / round | val grad eq, bwd=2x | val grad eq, bwd=1x | total rel, bwd=2x | total rel, bwd=1x |
|---|---:|---:|---:|---:|---:|---:|---:|
| scalar | 8 | 0 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| diag / E021 combo | 8 | 192 | 0.000 | 0.000 | 0.000 | 4.000 | 4.000 |
| extra_fit_budget | 8 | 198 | 0.000 | 0.000 | 0.000 | 4.094 | 4.094 |
| extra_worker_scalar | 8 | 0 | 0.000 | 0.000 | 0.000 | 1.125 | 1.125 |
| combo_val_gamma_reps_1 | 8 | 192 | 2.154 | 0.718 | 1.077 | 4.011 | 4.017 |
| combo_val_gamma_reps_2 | 8 | 192 | 4.308 | 1.436 | 2.154 | 4.022 | 4.034 |
| combo_val_gamma_reps_4 | 8 | 192 | 8.615 | 2.872 | 4.308 | 4.045 | 4.067 |
| combo_val_gamma_reps_8 | 8 | 192 | 17.231 | 5.744 | 8.615 | 4.090 | 4.135 |
| scalar | 32 | 0 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| diag / E021 combo | 32 | 384 | 0.000 | 0.000 | 0.000 | 2.500 | 2.500 |
| extra_fit_budget | 32 | 390 | 0.000 | 0.000 | 0.000 | 2.523 | 2.523 |
| extra_worker_scalar | 32 | 0 | 0.000 | 0.000 | 0.000 | 1.031 | 1.031 |
| combo_val_gamma_reps_1 | 32 | 384 | 2.154 | 0.718 | 1.077 | 2.503 | 2.504 |
| combo_val_gamma_reps_2 | 32 | 384 | 4.308 | 1.436 | 2.154 | 2.506 | 2.508 |
| combo_val_gamma_reps_4 | 32 | 384 | 8.615 | 2.872 | 4.308 | 2.511 | 2.517 |
| combo_val_gamma_reps_8 | 32 | 384 | 17.231 | 5.744 | 8.615 | 2.522 | 2.534 |
| scalar | 128 | 0 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| diag / E021 combo | 128 | 384 | 0.000 | 0.000 | 0.000 | 1.375 | 1.375 |
| extra_fit_budget | 128 | 390 | 0.000 | 0.000 | 0.000 | 1.381 | 1.381 |
| extra_worker_scalar | 128 | 0 | 0.000 | 0.000 | 0.000 | 1.008 | 1.008 |
| combo_val_gamma_reps_1 | 128 | 384 | 2.154 | 0.718 | 1.077 | 1.376 | 1.376 |
| combo_val_gamma_reps_2 | 128 | 384 | 4.308 | 1.436 | 2.154 | 1.376 | 1.377 |
| combo_val_gamma_reps_4 | 128 | 384 | 8.615 | 2.872 | 4.308 | 1.378 | 1.379 |
| combo_val_gamma_reps_8 | 128 | 384 | 17.231 | 5.744 | 8.615 | 1.381 | 1.383 |

Matching notes for full 8-rep validation:

| assumption | full-grid validation grad eq | extra fit grad calls | extra fit delta | extra worker grad calls | extra worker delta |
|---|---:|---:|---:|---:|---:|
| backward = 2x forward | 5.744 | 6.000 | +0.256 | 8.000 | +2.256 |
| backward = 1x forward | 8.615 | 6.000 | -2.615 | 8.000 | -0.615 |

## Quality Table

Ratios are held-out loss improvement divided by raw FedAvg improvement. CIs are
paired over held-out replicate batches.

| N | scale* | extra-worker scale* | oracle* | scalar | extra worker scalar | val reps 1 | val reps 2 | val reps 4 | val reps 8 | extra fit budget | E021 combo | diag | exact gate | oracle-H | secant | best worker |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.30 | 1.30 | 0.85 | 1.055 +/- 0.031 | 1.056 +/- 0.029 | 1.071 +/- 0.039 | 1.095 +/- 0.042 | 1.096 +/- 0.036 | 1.105 +/- 0.038 | -3.300 +/- 2.548 | -8.961 +/- 8.666 | 0.800 +/- 0.187 | 1.110 +/- 0.040 | 2.678 +/- 0.456 | 0.601 +/- 0.246 | 0.894 +/- 0.061 |
| 32 | 1.25 | 1.25 | 1.00 | 1.067 +/- 0.030 | 1.067 +/- 0.030 | 1.169 +/- 0.053 | 1.179 +/- 0.049 | 1.180 +/- 0.051 | 1.183 +/- 0.052 | -0.675 +/- 1.860 | 0.221 +/- 0.463 | 0.949 +/- 0.162 | 1.194 +/- 0.050 | 3.165 +/- 0.611 | 0.773 +/- 0.095 | 0.965 +/- 0.013 |
| 128 | 1.40 | 1.40 | 1.00 | 1.024 +/- 0.034 | 1.024 +/- 0.034 | 1.151 +/- 0.054 | 1.153 +/- 0.050 | 1.157 +/- 0.047 | 1.158 +/- 0.053 | -0.789 +/- 1.576 | -1.114 +/- 1.697 | 0.787 +/- 0.208 | 1.169 +/- 0.054 | 2.621 +/- 0.506 | 0.865 +/- 0.136 | 1.002 +/- 0.009 |

## Required Contrasts

Primary priced arm: `combo_val_gamma_reps_8`, matching E022's validation
structure.

| N | val8 - scalar | val8 - extra fit | val8 - extra worker scalar | val8 - E021 combo | val8 - diag | best worker - val8 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.049 +/- 0.033 | 4.405 +/- 2.575 | 0.049 +/- 0.036 | 10.066 +/- 8.630 | 0.305 +/- 0.215 | -0.211 +/- 0.089 |
| 32 | 0.116 +/- 0.023 | 1.858 +/- 1.808 | 0.116 +/- 0.024 | 0.962 +/- 0.422 | 0.235 +/- 0.125 | -0.218 +/- 0.063 |
| 128 | 0.134 +/- 0.021 | 1.947 +/- 1.526 | 0.134 +/- 0.021 | 2.272 +/- 1.713 | 0.371 +/- 0.157 | -0.155 +/- 0.049 |

## Interpretation

The same-round validation quality result survives matched-compute pricing in
this run. Full 8-rep validation beats tuned scalar and the `N+1`
extra-worker scalar rival with paired CIs excluding zero at all tested worker
counts. Spending the validation budget on two additional fit-probe offsets does
not explain the gain; that arm is destructive on this dense bed.

The cost curve also says the full 8-rep grid is probably overkill for this
synthetic setting. One or two mini-loss proxy reps already capture most of the
same-round quality gain while costing only 0.718-1.436 gradient-call
equivalents under the backward=2x-forward assumption.

The result does not overturn E024. E024 showed delayed gamma is weak, so the
claim here is only that the same-round gate is not trivially explained by
validation compute being better spent on scalar workers or extra fit probes.
The stronger causal/product claim still needs the E026 delayed/stale/shuffled
replication.

Best-worker fallback does not explain the result, and oracle-H remains much
higher, so dense headroom still exists but is only partly recovered by the
current trust layer.

## Command

```bash
python3 -u experiments/lgp_e025_validation_cost_pricing.py | tee experiments/lgp_e025_validation_cost_pricing_run_20261003.txt
```
