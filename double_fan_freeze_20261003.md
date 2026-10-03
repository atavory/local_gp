# Double Fan Freeze - 2026-10-03

This file freezes the old Double Fan / cloud-momentum direction so the local-GP
paper does not accidentally inherit its claims.

## What Was Tried

The old line tried to use same-point worker statistics to choose a momentum or
trust coefficient. Workers evaluated independent minibatches at the same
parameter vector, and the controller attempted to use their centered variance,
transition residuals, or Bayesian shrinkage estimates to decide how much to
trust history versus the fresh gradient.

## What Survived

The measurements were real:

- same-point centered worker variance estimates observation noise;
- dense diagonal variance has within-tensor structure that scalar summaries miss;
- Riccati state propagation can differ from a memoryless trust rule;
- some schedules found by the controller improve over undertuned baselines.

These are diagnostics, not an optimizer product.

## Why The Approach Is Frozen

The action layer repeatedly failed.

- K=1 MNIST produced deterministic round-shape wins. Measured timing did not
  beat frozen schedules.
- Compensation rules coupled trust to step magnitude and created gain artifacts.
- Dense variance and EB shrinkage changed estimates, but optimizer gains were
  explained by scalar learning-rate or schedule controls.
- The validated centered variance channel did not by itself provide the missing
  state covariance \(P\) or transition uncertainty \(Q\). Without those, the
  Kalman gain has no principled action.
- At N=8 the likelihood was weak enough that many variants collapsed to the
  prior, a schedule, or Adam-like decisions.

The correct conclusion is not "the cloud has no information." It is: the old
paper's thesis, "same-point worker variance selects a useful first-moment
momentum controller," has no surviving optimizer claim.

## What We Carry Forward

Three lessons remain useful:

1. A measurement result is not a product. It must beat the deployable control,
   not just an undertuned baseline.
2. Any adaptive optimizer claim must include tuned scalar and frozen-schedule
   controls, because those controls repeatedly explained apparent wins.
3. Same-point alignment estimates noise. It does not estimate curvature.

The local-GP paper starts from the third lesson. In K>1 local SGD, workers no
longer sit at the same point. Their local paths and probe pairs can estimate a
local update map. That is a different observation and a different paper.

## New Boundary

Do not cite Double Fan as evidence for the new method except as a negative
motivation:

```text
alignment estimates noise; misalignment can estimate curvature.
```

The new claim must be established from local path/probe observations, the
dynamics-aware K-step correction, and controls against tuned scalar FedAvg,
server/time secants, best-worker fallback, and hardcoded shrinkage.
