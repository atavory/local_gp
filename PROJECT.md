# Beyond Averaging: Distributed Training as Sequential Experimental Design

This project is not "same-point worker clouds are not momentum controllers."
That is a lesson from the previous project, not this project's title, center, or
claim.

This project is about **distributed training as sequential experimental
design**. A single worker querying a stochastic objective naturally forms a time
series; momentum and acceleration are summaries of its past and predicted
future behavior. With many workers, the default move is to collapse the system
back to one synthetic time series by averaging. This project asks what happens
if we do not collapse so early.

After each minibatch or local path segment, a worker should be viewed as one
query in a shared global information state, not as an owner of a private
trajectory with special authority. The optimizer may use all known previous
information:

```text
points, gradients, losses, probes, accepted steps, rejected steps,
worker metadata, versions, noise estimates, and costs.
```

Then the question is global: choose the next worker actions and the next
consolidation step to maximize expected training progress under noise, compute,
wire, and trust-region constraints.

Local SGD is the first concrete implementation of this view. When workers take
`K > 1` local steps from a synchronized point, they do not only produce an
endpoint to average. They produce a small cloud of nearby points, gradients,
losses, and displacements. That cloud is data about the local update map. The
project asks whether an optimizer can use that data to choose a better
synchronization/consolidation step than plain averaging or a tuned scalar
rescaling of averaging.

The short version:

```text
local SGD worker paths/probes
  -> estimate a structured local K-step update map
  -> shrink or reject unsafe corrections with held-out evidence
  -> take a trust-region consolidation step
```

## The Boundary

The frozen Double Fan / cloud-momentum line used workers aligned at the same
point. In that regime, the worker cloud estimates gradient noise. It can
calibrate uncertainty, but it does not identify a first-moment optimizer action
well enough to survive tuned schedules, tuned learning-rate scale, or Adam-like
controls.

This project uses a different observation. With `K > 1`, workers evaluate
gradients at different nearby points. More generally, with asynchronous or
heterogeneous workers, every worker query contributes tagged evidence to the
shared information state. The object is no longer same-point noise around one
mean gradient. The object is a local field:

```text
(x_i,k, g_i,k)  for worker i and local step k.
```

That is why the old negative does not automatically transfer. Same-point
alignment estimates noise; local misalignment can estimate geometry. The second
statement is a hypothesis to test, not a slogan to cite.

Because the primitive is a shared information state, worker homogeneity and
full synchronization are not assumptions. They are special cases. A worker may
have different data, speed, batch size, noise, stale version, or assigned probe;
the controller should account for those tags when deciding how much the
observation is worth.

## The Actual Estimand

The estimand is not a scalar momentum coefficient and not a full GP over all
parameters. The useful object is the **local K-step update map** around the
current synchronized point.

For quadratic local dynamics with no momentum,

```text
Delta_K = x_K - x_0
        = -H^{-1} [I - (I - eta H)^K] g_0.
```

So the correction to a FedAvg displacement is not a fresh Newton step. The
workers have already applied part of the curvature correction during their
`K` local steps. The right quadratic operator is:

```text
[I - (I - eta Hhat)^K]^{-1} Delta_K.
```

Plain BB, L-BFGS, or Newton-style consolidation are important controls, but
they are not the product unless they are rewritten for the K-step dynamics.

## Why This Is Not Optuna

Optuna/TPE is for low-dimensional black-box search. This project is not that.
At one event, workers can provide many derivative observations. The system is
data-rich and high-dimensional.

The algorithmic primitives should therefore be:

- structured local models: diagonal, block, low-rank-plus-diagonal;
- empirical-Bayes or marginal-likelihood shrinkage for coefficients;
- held-out probe or loss validation to reject unsafe corrections;
- stochastic trust-region acceptance and radius adaptation;
- bounded-staleness / event-triggered execution if we care about wire count.

Optuna may still be useful as experiment infrastructure or for pruning bad
whole-run configurations. It should not be the sampler that "searches" the
high-dimensional optimizer action.

## What The Workers Communicate

The target architecture should avoid per-worker `d^2` objects and avoid sending
coordinate lists around as if bits were the cost center.

Useful payloads are additive sufficient statistics or full event payloads:

```text
mean endpoint displacement
probe/path secant statistics
held-out loss or validation scores
optional low-rank / block summaries
```

Under a pay-by-wire model, the important question is number of synchronization
events, not whether the event contains one vector or a few vectors. Random or
seeded probes can be generated locally from a shared seed. Dense statistics can
be all-reduced. The controller can be logically centralized or replicated; the
scientific object is the same.

## What Has Been Found So Far

The project has several durable findings, and they should be kept separate.

1. **The old same-point cloud-momentum direction is frozen negative.**
   Centered worker variance is real and measurable, but measurement alone did
   not create a surviving optimizer action.

2. **Intermediate local points contain information that server-side temporal
   secants miss.**
   The recurring gap is: server/time secants capture little, spatial/local
   path/probe information captures more, oracle curvature captures much more.

3. **The naive quasi-Newton version is wrong for local SGD.**
   It double-applies curvature already used by the K local steps. The K-step
   dynamics-aware operator is the corrected object.

4. **Noise-controlled explicit probes work on the CPU synthetic beds.**
   Random probes plus the K-step operator and residual shrinkage beat tuned
   scalar FedAvg-scale controls on the current rotated/nonlinear beds.

5. **Probe budget matters.**
   Active probe design has not yet earned its keep. The current default on the
   CPU bed is roughly `2N` random probes for `N >= 32`, with full `N*K` still
   safer for very small `N`.

6. **Dense geometry has headroom but is fragile.**
   Oracle dense curvature is very strong. The implementable low-rank-plus-
   diagonal estimator can crash without a trust layer.

7. **Same-round validation can make dense corrections safe on the synthetic
   dense bed.**
   E022 and E025 show that validation-gamma gating can beat scalar controls and
   survive matched-compute pricing in that same-round setting.

8. **That validation signal is not yet a reusable optimizer policy.**
   E026 says delayed gamma does not replicate as a causal cross-round trust
   signal. So the dense branch currently supports "same-round safety/selection
   layer," not "learned dense policy that transfers across rounds."

9. **The real-path/GPU gate has not returned.**
   E018/E019 are not negative results. They are unreturned worker-liveness
   problems until actual artifacts land.

## The Product Claim We Are Trying To Earn

The desired claim is not:

```text
we found a diagnostic different from what FedAvg sees
```

Diagnostics without action are not a product.

The desired claim is:

```text
local worker path/probe data lets a synchronization rule choose a safer or
better consolidation update than tuned scalar FedAvg-scale, server-side
secants, best-worker fallback, and validation-cost-matched controls.
```

The claim must survive:

- source/target separation;
- tuned scalar FedAvg-scale controls;
- server/time secant controls;
- best-worker or elitist fallback controls;
- cost-matched validation/probe budgets;
- delayed or held-out checks that prevent winner's-curse positives;
- real-path workload transfer.

## The Current Algorithm Sketch

Per event:

1. Start from synchronized state `x_t`.
2. Workers run local SGD and/or explicit seeded probes near `x_t`.
3. Collect or all-reduce sufficient statistics for the local update map.
4. Fit a structured `Hhat` or direct K-step map estimate.
5. Estimate residual risk with split probes or held-out validation.
6. Form the dynamics-aware correction:

   ```text
   corrected = [I - (I - eta Hhat)^K]^{-1} Delta_K
   ```

7. Shrink, reject, or validate the correction.
8. Apply a trust-region consolidation step.
9. Use an acceptance/ratio test to adapt future radius, probe budget, or model
   class.

This is closer to model-based stochastic trust-region optimization with
derivative observations than to black-box Bayesian optimization.

## What Would Kill It

The project should stop or reframe if the following all hold:

- dynamics-aware probe corrections lose to tuned scalar FedAvg-scale under
  source/target separation;
- validation/probe costs explain the wins;
- server/time secants or best-worker fallback explain the wins;
- real-path workloads show no headroom or no transfer;
- the only positives are same-round selection effects that vanish under delay
  or shuffled controls.

## What Would Make It A Paper

A paper exists if we can show:

1. a clear theory of why local SGD paths are derivative-rich observations of a
   K-step update map;
2. a structured estimator that is cheap enough for high dimension;
3. a trust layer that prevents overcorrection;
4. controlled synthetic evidence isolating the secant/path gap;
5. real-path evidence that the rule reduces tuning burden or improves
   synchronization at matched cost.

The honest title is something like:

```text
Local SGD as Parallel Local-Map Estimation
```

or:

```text
Worker Drift as Derivative-Rich Data for Local SGD Synchronization
```

Not a parody title about the last failure mode.
