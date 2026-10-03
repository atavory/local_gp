# LGP-E025 CPU Worker Prompt - Validation Cost Pricing

## Mission

Price the E022 trust gate. E022/E024 show that held-out mini-loss validation can
prevent the dense low-rank-plus-diagonal correction from exploding, but the
validation batches are not free. This experiment decides whether the E022 gain
survives honest matched-compute accounting.

## Context

Read these files first:

```text
README.md
research_control_board.md
experiments/lgp_e021_dense_lowrank_correlated_stress.py
experiments/lgp_e022_dense_trust_validation.py
experiments/lgp_e022_dense_trust_validation_result_20261003.md
experiments/lgp_e024_delayed_trust_validation_result_20261003.md
```

Use the same dense bed and K-step operator as E022:

```text
[I - (I - eta Hhat)^K]^-1 Delta_K
```

Do not use a plain Newton, BB, or L-BFGS step.

## Required Work

Create and run:

```text
experiments/lgp_e025_validation_cost_pricing.py
```

Write:

```text
experiments/lgp_e025_validation_cost_pricing_result_20261003.md
experiments/lgp_e025_validation_cost_pricing_run_20261003.txt
```

## Arms

At minimum, report:

1. `scalar`: the E022 source/pilot-tuned scalar FedAvg-scale baseline.
2. `combo_val_gamma`: E022 same-round validation-gamma.
3. `combo_val_gamma_reps_{1,2,4,8}`: validation-gamma with fewer held-out
   mini-loss proxy reps, so we get a cost/quality curve.
4. `extra_fit_budget`: spend the validation budget on additional fit probes
   instead of validation, then use the existing residual/risk shrinkage.
5. `extra_worker_scalar`: spend the same approximate gradient-call budget on
   more workers or larger effective batch for scalar FedAvg, using the nearest
   feasible worker counts. If exact matching is impossible, report the nearest
   under/over costs explicitly.

Keep E022 controls visible where cheap:

```text
E021 combo
diag
best_worker
secant
oracle-H
exact_gate
```

## Cost Ledger

The result note must include a cost table in these units:

- fit gradient calls / round;
- validation forward-equivalents / round;
- validation gradient-call-equivalents / round under two assumptions:
  `backward = 2x forward` and `backward = 1x forward`;
- total cost for each arm relative to raw FedAvg.

Use the actual E022 proxy structure:

```text
LOSS_PROXY_REPS
LOSS_PROXY_COORD_BATCH
LOSS_PROXY_LR_BATCH
D
RANK
len(VAL_GAMMAS)
```

Do not hide the full 8-gamma grid cost.

## Decision Labels

Choose exactly one:

- `VALIDATION_COST_SURVIVES`: validation-gamma still beats scalar after a
  matched-compute rival is armed, with paired CI excluding zero.
- `VALIDATION_COST_UNDERWATER`: the matched-compute rival explains or beats the
  validation-gamma gain.
- `VALIDATION_COST_INCONCLUSIVE`: cost can be priced, but CIs or infeasible
  matching prevent a clean efficiency verdict.
- `INVALID`: leakage, missing scalar control, missing cost ledger, or run
  failure.

## Reporting Rules

Use paired confidence intervals over held-out replicate batches. Separate:

- quality claim: validation improves the update;
- efficiency claim: validation is worth its extra compute.

E024 already weakened the same-round claim. Do not cite E022 as settled in this
result note; state whether this cost result is conditional on the delayed-gamma
gate.

## Command

Run from the repository root:

```bash
python3 -u experiments/lgp_e025_validation_cost_pricing.py | tee experiments/lgp_e025_validation_cost_pricing_run_20261003.txt
```

Then update the result note and leave all artifacts in `experiments/`.
