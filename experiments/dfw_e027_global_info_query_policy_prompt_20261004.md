# DFW-E027 CPU Prompt - Global Information Query Policy

## Mission

This is a Beyond Averaging / local_gp experiment queued through the `double_fan`
project-worker channel.

Test the core reframing directly:

> A worker is not the owner of a private trajectory. It is one query process in
> a shared global information state. The next action should be chosen from all
> known observations.

Build a small CPU synthetic experiment where a global controller can choose
worker query actions after each minibatch, and compare it against policies that
collapse workers into a single averaged trajectory.

## Repository

Use `/data/users/atavory/local_gp` if present. If not present, clone
`https://github.com/atavory/local_gp` through Mr. Gitties. Do not use `pip`.

Read:

```text
PROJECT.md
README.md
algorithm_stack.md
model.md
```

## Required Work

Create and run:

```text
experiments/dfw_e027_global_info_query_policy.py
```

Write:

```text
experiments/dfw_e027_global_info_query_policy_result_20261004.md
experiments/dfw_e027_global_info_query_policy_run_20261004.txt
```

Also upload the script, result note, and run log to this run's Manifold artifact
directory.

## Experiment

Use a convex quadratic or mildly nonlinear separable objective with known
ground truth:

```text
f(x) = 0.5 * x^T H x + optional small cubic/tanh perturbation
```

with `d >= 128`, log-spaced curvature, and stochastic gradients with controlled
anisotropic noise.

Simulate `m` workers. At each micro-step, a worker can choose one of at least
three actions:

1. `exploit`: gradient/local-SGD step from current controller state.
2. `probe`: symmetric finite-difference or seeded displacement query to improve
   the local map estimate.
3. `replicate`: repeat/average a high-noise direction to reduce uncertainty.

Compare these policies at matched query budget:

- `private_momentum`: each worker follows only its own momentum/time series;
- `fedavg_fixed`: fixed local SGD/FedAvg averaging;
- `round_robin_probe`: fixed schedule of exploit/probe/replicate;
- `global_greedy_voI`: uses the shared information state and chooses the next
  worker action by one-step expected improvement or uncertainty-weighted value
  of information;
- `oracle_policy`: uses true `H` and noise only as an upper-bound diagnostic.

The global policy can be simple. Do not spend the run on fancy BO. A diagonal
or block posterior over curvature/noise is enough. The important test is
whether using all known worker observations changes decisions and improves
held-out progress beyond fixed averaging/schedules.

## Metrics

Report paired results across seeds:

- final objective or one-step held-out improvement at fixed query budget;
- number/fraction of exploit/probe/replicate actions chosen;
- curvature/noise posterior error;
- improvement over `fedavg_fixed`;
- improvement over `round_robin_probe`;
- oracle headroom.

## Decision Labels

Choose exactly one:

- `GLOBAL_INFO_POLICY_LIVE`: global policy beats both fixed averaging and fixed
  round-robin with paired CI excluding zero.
- `FIXED_SCHEDULE_EXPLAINS`: global policy ties or loses to fixed round-robin
  or tuned fixed averaging.
- `NO_HEADROOM`: oracle policy does not beat fixed averaging enough to matter.
- `INVALID`: missing controls, unpaired seeds, run failure, or leaked oracle
  into non-oracle arms.

## Reporting Rules

Be blunt. If the global controller only discovers a schedule, say so. If the
headroom exists but the greedy policy misses it, separate those facts.
