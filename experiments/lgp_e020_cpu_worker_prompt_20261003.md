# LGP-E020 CPU Worker Ping - Probe Budget / Active Design Gate

You are running the next CPU gate for `local_gp`. Use the GitHub repo
`https://github.com/atavory/local_gp`, or the Manifold source bundle listed in
the CPU manifest if GitHub access is inconvenient.

## Mission

E015-E017 show that probe-fed dynamics-aware local-map correction works on CPU
synthetic beds. E020 asks the product question:

> How much probe budget is actually needed, and can value/uncertainty-guided
> probe selection beat plain random probes at small budgets?

Build and run:

```text
experiments/lgp_e020_probe_budget_active_design.py
```

## Starting Point

Start from E016 or E017:

```text
experiments/lgp_e016_rotated_nonlinear_probe.py
experiments/lgp_e017_risk_shrinkage_fallback.py
```

Use the rotated nonlinear block bed from E016 as the primary bed. If time is
short, run only N=8 and N=32 first; include N=128 if cheap.

## Required Operator

Use the K-step dynamics-aware correction:

```text
[I - (I - eta Hhat)^K]^{-1} Delta_K
```

Do not use plain BB/L-BFGS/Newton consolidation.

## Probe Budgets

Compare at least:

```text
budget = N/2, N, 2N, N*K
```

If fractional `N/2` is awkward, use `max(4, N//2)`.

Keep probe reps fixed at the E016/E017 value first (`probe_reps=4`). If there is
time, add a second section with `probe_reps=1`.

## Probe Policies

Required:

- `random`: Rademacher / random offsets, as in E016/E017.
- `value_weighted`: probe directions sampled with probability proportional to
  predicted correction value:

```text
value(C) = (1 - 1/C)^2 / max(C, eps)
```

where `C = 1 / (1 - (1 - eta * lambda_prior)^K)` from a causal prior estimate.
If no prior exists on round 1, fall back to random.

- `uncertainty_value`: sample directions by

```text
value(C_prior) * posterior_or_bootstrap_uncertainty
```

Approximate uncertainty cheaply from split-probe disagreement or prior variance.
If this is too much for one pass, implement a simple two-stage policy:
use half the budget randomly, estimate uncertainty, spend the second half on
highest value*uncertainty blocks/directions.

Strong controls:

- tuned scalar FedAvg scale;
- risk-aware random probe from E017;
- oracle-H correction;
- best-worker true fallback.

## Shrinkage

Use residual shrinkage from E017 for the non-oracle probe arms. Do not tune an
arbitrary gamma except as a labeled upper-bound control.

## Decision Labels

Use exactly one:

- `SMALL_BUDGET_ACTIVE_WORKS`: value/uncertainty-guided probes beat tuned scalar
  scale and match the full `N*K` random-probe arm within ~10-15% at a smaller
  budget.
- `RANDOM_BUDGET_REQUIRED`: random probes work, but only at large budget; active
  policies do not reduce budget.
- `ACTIVE_POLICY_HURTS`: active policies are worse than random at matched budget.
- `SCALAR_OR_FALLBACK_EXPLAINS`: tuned scalar or best-worker fallback explains
  the gain.
- `INVALID`: implementation/scoring bug.

## Required Output

Write:

```text
experiments/lgp_e020_probe_budget_active_design_result_20261003.md
```

Upload script, result note, and run log to:

```text
aai_research_tlv/tree/local_gp/cpu_pings/20261003/
```

The result must include a table by `N x budget x policy` with paired CIs and a
short recommendation for the cheapest plausible probe policy.
