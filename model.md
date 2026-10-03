# Local GP Model

## Problem

At synchronization round `t`, all workers start from the same parameter vector
`x_t`. Each worker `i` then takes `K > 1` local steps on its own minibatches and
returns some proposal information:

```text
path_i,t = (x_t, x_i,t,1, ..., x_i,t,K)
proposal_i,t = x_i,t,K - x_t
```

The classical reducer only sees an average proposal or average gradient. That
throws away the geometry of the worker path cloud.

The local-GP idea is to model the local update field around `x_t`, using worker
paths as observations.

## Local-Map Correction

For K-step local SGD, the correction target is the local update map, not a
fresh Newton step. On a quadratic with no momentum,

```text
Delta_K = x_K - x_0 = -H^{-1} [I - (I - eta H)^K] g_0.
```

Therefore a curvature estimate should correct the FedAvg displacement by

```text
[I - (I - eta H)^K]^{-1} Delta_K
```

rather than applying `H^{-1} g` after the workers have already moved. This is
the operator tested in E014. Plain BB/L-BFGS consolidation is a useful null
control, but it is not the live product unless it is written against the
K-step dynamics.

## Latent Object

The latent is not a scalar momentum coefficient. It is a local function:

```text
F_t(z) = quality of an update/proposal/action with local descriptor z
```

Possible targets:

- audit-gradient alignment of a proposal;
- validation-loss improvement from taking a proposal;
- oracle trust action selected from a finite grid;
- pairwise preference between two candidate proposals;
- residual after subtracting a round-only schedule.

The most important target is held-out usefulness, not in-sample explanation.

## Inputs

A descriptor `z_i,t` may include:

- local path length and curvature proxies;
- endpoint displacement norm;
- proposal direction cosine with the mean proposal;
- worker local loss changes;
- spread of proposal endpoints;
- same-batch finite-difference probes, if available;
- shared-probe outputs, losses, logits, or representation changes.

The shared-probe route is attractive because it avoids high-dimensional
parameter communication. Workers can evaluate a fixed probe set and communicate
low-dimensional function-space summaries instead of dense parameter vectors.

## Kernel

Start simple. Do not invent a large GP system before a held-out signal exists.

Candidate kernels:

1. RBF on standardized path/probe features.
2. Matern-3/2 on standardized path/probe features.
3. Additive kernel:

```text
k(z, z') = k_round(t, t') + k_path(path, path') + k_probe(probe, probe')
```

4. Pairwise preference kernel for comparing candidate proposals.

## Baselines

Every gate must include:

- round-only schedule;
- scalar source-selected baseline;
- path-norm baseline;
- mean proposal baseline;
- random or shuffled-feature control;
- target oracle diagnostic, excluded from fair comparison.

## Product Hypothesis

A GP/kernel model is useful only if it transfers:

- fit on source seeds/rounds/settings;
- freeze model and hyperparameters;
- evaluate on held-out target seeds/rounds/settings;
- beat or tie the frozen baselines with less target tuning.

The project should not claim value for merely fitting the source setting.

## Stop Rule

If the GP does not beat round-only and path-norm baselines on held-out oracle
action or proposal-quality prediction, do not run a closed-loop optimizer.
