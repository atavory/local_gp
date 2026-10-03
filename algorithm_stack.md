# Local GP Algorithm Stack

## Why Not Optuna

Optuna/TPE is a derivative-free density-ratio sampler for low-dimensional,
mixed or conditional hyperparameter spaces. That is the wrong primitive here.

In this project, each event can produce many gradient observations:

```text
(x_i, g_i),  probe pairs (s_i, y_i),  local update Delta_K.
```

The system is not data-poor black-box optimization. It is model-based stochastic
optimization with derivative observations. The expensive object is not "which
hyperparameter trial should I run next"; it is "how do I use noisy local
geometry without overcorrecting?"

This is the data-rich regime. BO exists for expensive black boxes where one
evaluation returns one scalar. Here, one event from `m` workers returns `m`
gradients, i.e. `m*d` derivative observations against `O(d)` diagonal or block
surrogate parameters. The acquisition function should not search parameter
space. The outer loop manages staleness, trust-region size, batching, and
asynchrony.

## Layer 1: Empirical Bayes / Type-II ML For Coefficients

The per-coordinate or block correction coefficients should be estimated by
maximizing a marginal likelihood, not by outer-loop search over training runs.

In GP language this is ARD / Type-II maximum likelihood:

```text
d/dtheta log p(y | theta)
  = 0.5 tr((alpha alpha^T - K^{-1}) dK/dtheta),
alpha = K^{-1} y.
```

Under the diagonal or block factorization used by the current CPU gates, the
problem splits into tiny independent fits. That is what makes the huge ambient
dimension tractable: no full GP over parameters, no search over `d` knobs.

For the current dynamics-aware correction, this layer means:

```text
probe pairs -> posterior / regularized estimate of Hhat or map coefficients
```

with shrinkage determined from the likelihood or held-out probe residuals.
E017 is the first crude version of this idea: split probes estimate residual
risk and turn it into a shrinkage gamma.

E020/E021 add two guardrails. First, active probe selection is not yet a reliable
budget reducer, so the default should remain random probes with an explicit
budget curve. Second, residual shrinkage alone can miss dense low-rank
correlated failure modes; low-rank-plus-diagonal fits need a held-out
validation layer that can zero or damp the combined correction before it reaches
the event action.

## Layer 2: Direct Differentiation Of The One-Step Surrogate

When the objective is the one-event improvement surrogate, the action
coefficients are differentiable. For a quadratic local model, one useful form is

```text
gain(c) = (g* dot (c * Delta))^2
          / (2 (c * Delta)^T H (c * Delta)).
```

So if we have a trustworthy local model, the coefficients can be optimized by
gradient ascent or closed-form/blockwise updates inside the event. Again: no
Optuna.

The live product should still use the K-step local-map operator:

```text
[I - (I - eta Hhat)^K]^{-1} Delta_K.
```

Plain Newton / L-BFGS consolidation is the wrong operator for local SGD because
the workers have already applied part of the curvature correction.

## Layer 3: Stochastic Trust Region For Events

The event controller should be a stochastic trust-region method, not HPO over
the sync threshold.

Closest named families:

- STORM / stochastic trust-region methods;
- ASTRO-DF style adaptive local modeling;
- TuRBO-style radius adaptation, but with gradients replacing black-box
  function-only GP fitting.

The intended import from BO-like systems is narrow:

- TuRBO-style radius adaptation: expand after accepted events, shrink after
  rejected events;
- batch/asynchronous bookkeeping: handle in-flight workers without changing the
  scientific object;
- pruners: reset or yank unpromising local trajectories.

The intended non-imports are equally important:

- no TPE over optimizer parameters;
- no qEI search in `d` dimensions;
- no SAASBO/ALEBO as the primary model. Random sketches may be used as
  compression, not as the explanation of the optimizer.

The controller maintains a trust region around the current synchronized state:

```text
fit local map -> propose corrected update -> compare predicted vs observed value
```

For the dense-geometry branch, the comparison should include a held-out
probe/loss proxy, not only fit residuals from the same representation. E021
showed that a combined low-rank-plus-diagonal update can look plausible enough
to get nonzero residual shrinkage while being destructive on held-out objective
decrease.

Then:

- accept and expand the radius on successful events;
- reject or damp and shrink the radius on failed events.

This replaces Optuna-over-delta with a standard ratio test.

## Layer 4: Trigger As Optimal Stopping

The trigger is a value-function comparison, not a searched hyperparameter.

Continue probing while the marginal value of more drift/probes exceeds wire or
compute cost. In the simple correction law, the relevant shape is:

```text
value(C) ~ (1 - 1/C)^2 / C.
```

This has an interior peak, so a norm-only drift budget is not the right
statistic. Norm is often dominated by flat motion, where curvature is hardest
to measure. The trigger should weight drift/probe information by predicted
correction value and model risk.

Under the current pay-by-wire cost model, directed probing is not a new wire
round. Assignments can be fused into the event broadcast/scatter. For random
or seeded probes, the controller broadcasts a seed and shared scale; each worker
generates its direction locally and the controller reconstructs it exactly. On
the uplink, additive sufficient statistics can be all-reduced:

```text
sum s, sum s^2, sum s*g, sum g.
```

This changes bits, not wires. The method should be judged on information value
and event count, not on whether a coordinate list was transmitted.

Pure async is also not automatically cheaper. Without collectives, each worker
report and controller reply becomes a separate wire use. The first practical
architecture is bounded-staleness semi-sync: quorum events, vintage-tagged
pairs, and late-pair integration only while the trust region still validates
them.

## Layer 5: If We Learn A Policy, Use Bilevel Gradients

If the controller policy itself has learnable parameters, use gradient-based
bilevel methods:

- implicit differentiation through the fixed point / training optimum;
- two-timescale stochastic bilevel optimization.

Avoid long unrolled hypergradients through full training trajectories. They are
high-variance and unstable in optimizer dynamics. This project already has two
timescales: fast worker/probe dynamics and a slow controller, which matches the
two-timescale bilevel setting.

## What Remains For Grid Search

Use a small grid only for discrete architecture choices:

- block size or sketch rank;
- L-BFGS memory if that branch returns;
- probe budget;
- model family choice.

These are few-candidate engineering choices. A coarse grid is more honest than
TPE here.

## Where Optuna Still Helps

Optuna can still provide infrastructure, not the core algorithm:

- ASHA / Hyperband style pruning of bad whole-run configurations;
- multi-objective bookkeeping for quality-vs-wire or quality-vs-compute
  frontiers;
- experiment tracking.

But the sampler should not be the scientific object. The scientific object is:

```text
empirical-Bayes local map + residual shrinkage + stochastic trust region.
```
