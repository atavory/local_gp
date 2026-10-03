# LGP-E023 CPU Worker Ping - Random Probe Budget Confirmation

You are running the next CPU budget confirmation for `local_gp`. Use the GitHub
repo `https://github.com/atavory/local_gp`, or the Manifold source bundle listed
in the CPU manifest if GitHub access is inconvenient.

## Mission

E020 returned `RANDOM_BUDGET_REQUIRED`: active designs did not robustly reduce
the probe budget, but random probes remained strong. E023 asks for a cleaner
confirmation of the cheapest random-probe policy, with more held-out reps and a
direct check of probe repeats.

Build and run:

```text
experiments/lgp_e023_random_probe_budget_confirm.py
```

## Starting Point

Start from:

```text
experiments/lgp_e020_probe_budget_active_design.py
experiments/lgp_e016_rotated_nonlinear_probe.py
experiments/lgp_e017_risk_shrinkage_fallback.py
```

Use the E016 rotated nonlinear block bed as the primary bed. If time allows,
add a second section on the E015 diagonal nonlinear/heterogeneous bed to check
that the recommendation is not rotation-specific.

## Required Operator

Use the K-step dynamics-aware correction:

```text
[I - (I - eta Hhat)^K]^-1 Delta_K
```

Do not use plain Newton, BB, or L-BFGS consolidation.

## Budget Grid

This gate is random-only. Drop active policies and spend the budget on more
replicates.

Required random-probe budgets:

```text
N, 2N, 4N, N*K
```

Also include `max(4, N//2)` for `N=8` if cheap.

Required worker counts:

```text
N = 8, 32, 128
```

Run at least `heldout_reps=8` if practical. Compare:

- `probe_reps=1`;
- `probe_reps=4` from E016/E020.

Use pilot rounds/reps only to choose the residual shrinkage rule
(`risk_reliability` versus `risk_sqrt`) and the tuned scalar FedAvg scale.
Do not tune arbitrary gamma except as a labeled upper-bound control.

## Controls

Required:

- raw FedAvg;
- tuned scalar FedAvg scale;
- random-probe dynamics-aware correction at each budget/repeat setting;
- full `N*K` random-probe arm;
- oracle-H dynamics-aware correction;
- server/time secant;
- best-worker true fallback.

Optional:

- tuned-gamma random-probe upper bound for `2N` and `N*K`;
- wall-clock/probe-count accounting if easy.

## Decision Labels

Use exactly one:

- `TWO_N_RANDOM_CONFIRMED`: `2N` random probes beat tuned scalar and are within
  roughly 10-15% of full `N*K` random probes for `N >= 32`; small `N=8` may
  still require full budget.
- `FOUR_N_RANDOM_REQUIRED`: `4N` is the cheapest reliable budget for `N >= 32`.
- `FULL_RANDOM_REQUIRED`: only full `N*K` random probes reliably clear tuned
  scalar.
- `PROBE_REPS_CRITICAL`: `probe_reps=1` materially fails while `probe_reps=4`
  works, changing the cost story.
- `SCALAR_OR_FALLBACK_EXPLAINS`: tuned scalar or best-worker fallback explains
  the gain.
- `INVALID`: implementation/scoring bug.

## Required Output

Write:

```text
experiments/lgp_e023_random_probe_budget_confirm_result_20261003.md
```

Upload script, result note, and run log to:

```text
aai_research_tlv/tree/local_gp/cpu_pings/20261003/
```

The result must include a compact table by `N x budget x probe_reps`, paired CIs
for budget minus tuned scalar and budget minus full random, and a recommendation
for the next default random-probe budget.
