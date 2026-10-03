# LGP-E003 Optimizer Smoke Result - 2026-10-03

## Decision

`OPTIMIZER_PROTOTYPE_BUILT`

## What Was Built

`local_gp.optimizer.LocalGPAggregator` is now the first product-shaped optimizer
object in this repo.

It is a local-SGD server aggregator, not a standard PyTorch `Optimizer`, because
it needs worker-local trajectories before reduction.

Inputs:

- `N x K x d` local displacements before each worker step;
- `N x K x d` local gradients at those displacements;
- `N x d` final worker endpoint displacements.

Output:

- corrected server update;
- FedAvg update;
- per-coordinate coefficients;
- gain/fallback diagnostics.

Rule:

1. estimate diagonal local curvature from worker trajectories;
2. read out a finite posterior-ratio correction;
3. shrink it toward FedAvg with one scalar gamma;
4. cross-fit by default;
5. fall back to coefficient 1 for unstable coordinates.

## Smoke Protocol

Script: `experiments/lgp_e003_optimizer_smoke.py`.

- diagonal quadratic;
- d=120;
- 10 sharp directions;
- K=8;
- eta=0.05;
- N in {8, 32, 128};
- gamma selected on pilot reps;
- held-out reps scored separately;
- metric: one-step gain ratio over FedAvg.

## Result

| N | selected gamma | local_gp/base | oracle-C/base | mean coefficient |
|---:|---:|---:|---:|---:|
| 8 | 0.95 | 1.063 | 1.190 | 1.930 |
| 32 | 0.75 | 1.132 | 1.233 | 2.699 |
| 128 | 0.70 | 1.148 | 1.207 | 3.505 |

The public optimizer API reproduces a positive synthetic effect, though weaker
than E002's compact diagnostic because it estimates residual variance from data
instead of using the simulation's known noise level.

## Interpretation

This is the first actual optimizer artifact in `local_gp`.

It is not yet a real-data method claim:

- the bed is synthetic diagonal-quadratic;
- gamma is still source-tuned;
- E002 showed tuned gamma explains the current posterior readout;
- no real-network closed loop has run.

But it is now a concrete object that can be tested, criticized, and ported into
a Buck2/PyTorch worker harness.

## Next Gate

Run the same public aggregator through a richer real-data local-SGD ledger with
full worker paths or shared-probe features. It must beat:

- FedAvg/source-tuned scalar;
- tuned gamma/frozen prior controls;
- server-side secant/quasi-Newton controls.

If it only reproduces tuned gamma, call it a shrinkage-controlled curvature
aggregator, not a GP optimizer.
