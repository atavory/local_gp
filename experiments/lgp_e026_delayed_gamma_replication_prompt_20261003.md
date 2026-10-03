# LGP-E026 CPU Worker Prompt - Delayed Gamma Replication

## Mission

Resolve the E024 ambiguity. E022's same-round validation-gamma looked positive,
but E024's causal delayed-gamma control only held clearly at `N=32` and was
weak/null at `N=8` and `N=128`. The drop-first diagnostic was noisy. This gate
reruns the delayed validation check with more held-out reps and explicit stale
and shuffled controls.

## Context

Read these files first:

```text
README.md
research_control_board.md
experiments/lgp_e022_dense_trust_validation.py
experiments/lgp_e022_dense_trust_validation_result_20261003.md
experiments/lgp_e024_delayed_trust_validation.py
experiments/lgp_e024_delayed_trust_validation_result_20261003.md
```

Use the same dense bed and dynamics-aware K-step operator as E022/E024.

## Required Work

Create and run:

```text
experiments/lgp_e026_delayed_gamma_replication.py
```

Write:

```text
experiments/lgp_e026_delayed_gamma_replication_result_20261003.md
experiments/lgp_e026_delayed_gamma_replication_run_20261003.txt
```

## Required Design

Use the E024 code as the starting point, but increase held-out power:

```text
HELDOUT_REPS >= 12
PILOT_REPS = 3 is OK unless runtime forces otherwise
N_WORKERS = (8, 32, 128)
```

Report all arms on the same scoring subsets:

1. `scalar`
2. `same_round_val_gamma`
3. `lag1_gamma`
4. `lag2_gamma`
5. `lag1_drop_first`
6. `lag2_drop_first`
7. `shuffled_gamma`: gamma values randomly permuted within replicate, preserving
   the marginal gamma distribution but destroying temporal alignment.
8. `zero_gamma` and/or `grid_floor`: safety floor.
9. `exact_gate` and `oracle-H` as diagnostics only.

The key paired contrasts are:

```text
lag1 - scalar
lag1 - same_round
lag1 - shuffled
lag2 - scalar
same_round - shuffled
```

Include mean gamma, gamma autocorrelation, and mean absolute lag difference.

## Decision Labels

Choose exactly one:

- `DELAYED_SIGNAL_REAL`: lag-1 or lag-2 beats scalar and shuffled with paired
  CIs excluding zero on the main scoring subset.
- `SAME_ROUND_SELECTION_EXPLAINS`: same-round beats scalar, but delayed does
  not beat scalar or shuffled.
- `DELAYED_GAMMA_INCONCLUSIVE`: mixed signs or CIs still too wide.
- `INVALID`: missing scalar/shuffled control, target leakage, or run failure.

## Reporting Rules

Do not let the first-round initialization convention decide the result. The
drop-first rows are mandatory. The result note must say plainly whether E022 can
still be cited as a dense-method positive.

## Command

Run from the repository root:

```bash
python3 -u experiments/lgp_e026_delayed_gamma_replication.py | tee experiments/lgp_e026_delayed_gamma_replication_run_20261003.txt
```

Then update the result note and leave all artifacts in `experiments/`.
