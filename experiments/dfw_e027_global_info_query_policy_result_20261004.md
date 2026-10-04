# DFW-E027 Global Information Query Policy Result - 2026-10-04

## Decision

`FIXED_SCHEDULE_EXPLAINS`

## Question

This gate tested whether a worker should be treated as one query process in a
shared global information state.  The CPU bed let a controller choose worker
actions after each minibatch from:

- `exploit`: query a stochastic gradient at the current controller state;
- `probe`: query one seeded coordinate displacement to update a diagonal
  curvature posterior;
- `replicate`: repeat a current-state gradient query to reduce gradient/noise
  uncertainty.

The non-oracle global and round-robin controllers used only their shared
posterior over diagonal curvature and empirical noise.  The oracle arm used the
true diagonal Hessian and true noise as an upper-bound diagnostic.

## Setup

Synthetic objective:

```text
f(x) = 0.5 * sum_j lambda_j x_j^2
```

Parameters:

```text
d=160
workers=24
rounds=36
matched query budget=864 worker actions
pilot_reps=5
heldout_reps=24 paired seeds
lambda range=[0.04, 18.0] log-spaced and permuted
anisotropic stochastic gradient noise
probe_radius=0.22
```

Pilot-selected parameters:

```text
fedavg_fixed: eta=0.08
private_momentum: eta=0.055, beta=0.85
round_robin_probe: alpha=0.95
global_greedy_voI: alpha=0.8
oracle_policy: alpha=0.35
```

The progress ratio is reported for continuity with nearby gates, but it is
compressed because all policies remove almost all of the initial loss.  The
decision uses paired final-objective contrasts.

## Results

Lower final loss is better.  CIs are paired over 24 held-out seeds.

| policy | final loss | progress / FedAvg | exploit | probe | replicate | curvature posterior error | noise posterior error |
|---|---:|---:|---:|---:|---:|---:|---:|
| private_momentum | 0.4285 +/- 0.0310 | 1.0054 +/- 0.0008 | 1.000 | 0.000 | 0.000 | 0.9050 +/- 0.0000 | 1.2321 +/- 0.0000 |
| fedavg_fixed | 1.3790 +/- 0.0903 | 1.0000 +/- 0.0000 | 1.000 | 0.000 | 0.000 | 0.9050 +/- 0.0000 | 0.0544 +/- 0.0022 |
| round_robin_probe | 0.0360 +/- 0.0032 | 1.0076 +/- 0.0009 | 0.500 | 0.250 | 0.250 | 0.1910 +/- 0.0045 | 0.0621 +/- 0.0025 |
| global_greedy_voI | 0.0581 +/- 0.0038 | 1.0075 +/- 0.0009 | 0.175 | 0.492 | 0.333 | 0.4470 +/- 0.0071 | 0.0731 +/- 0.0029 |
| oracle_policy | 0.0175 +/- 0.0019 | 1.0077 +/- 0.0009 | 0.503 | 0.000 | 0.497 | 0.9050 +/- 0.0000 | 0.0546 +/- 0.0023 |

Paired final-objective contrasts, reported as loss saved by the first policy:

| contrast | loss saved |
|---|---:|
| global_greedy_voI beats fedavg_fixed | 1.3210 +/- 0.0899 |
| global_greedy_voI beats round_robin_probe | -0.0221 +/- 0.0049 |
| oracle_policy beats fedavg_fixed | 1.3616 +/- 0.0908 |
| oracle_policy beats global_greedy_voI | 0.0406 +/- 0.0045 |
| round_robin_probe beats fedavg_fixed | 1.3431 +/- 0.0901 |

## Interpretation

The global controller did not earn `GLOBAL_INFO_POLICY_LIVE`.  It beat fixed
FedAvg by a large paired final-loss margin, but it lost to the fixed
round-robin probe schedule by `0.0221 +/- 0.0049` final objective.  The fixed
schedule also learned a much better curvature posterior: relative curvature
RMSE `0.1910` versus `0.4470` for the greedy global controller.

The failure mode is clear: the greedy VoI rule over-spent on probes
(`49.2%` of all actions) and under-spent on ordinary exploit queries
(`17.5%`).  It did use the shared information state, but the one-step score
mainly rediscovered "probe a lot" and did not allocate probes as effectively as
the simple fixed 50/25/25 exploit/probe/replicate schedule.

There is still headroom.  The oracle final loss was `0.0175 +/- 0.0019`, better
than both round-robin and global, and oracle saved `1.3616 +/- 0.0908` loss
against FedAvg.  So the negative label is not `NO_HEADROOM`; it is that this
particular greedy global policy missed the headroom and a fixed schedule
explained the non-FedAvg gain.

## Command

```bash
python3 -u experiments/dfw_e027_global_info_query_policy.py | tee experiments/dfw_e027_global_info_query_policy_run_20261004.txt
```
