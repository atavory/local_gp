# Stationary Probes By Creating Non-Exchangeability

Status: synthesis note, 2026-10-05.  Source input: ushmoo/double_fan.

The earlier stationary failures assume workers are exchangeable: same inner
optimizer, same `K`, same learning rate, same objective distribution up to
sampling noise.  In that case uniform averaging is close to optimal and the
worker-axis ceiling is nearly zero.

The way out is not to wait for drift.  It is to create or detect meaningful
non-exchangeability inside an aligned round.

## 1. Run Workers At Different Inner Hyperparameters

Use the parallel workers as a small population of candidate inner optimizers.
For example:

```text
worker groups use different inner learning rates
worker groups use different local K
worker groups use different momentum/weight decay/dropout knobs
```

The server still receives aligned updates from a common base point `theta_t`.
But the updates are no longer exchangeable.  The value spread is now caused by
real hyperparameter differences, not by minibatch luck.

That changes the probe problem:

```text
old gap: which worker's sampling noise happened to help?
new gap: which inner optimizer setting is good this round?
```

The second gap should be much larger.  The decisive cheap test is:

```text
stationary workload
fixed alpha,beta
spread worker inner LRs by about 4x
measure probe sigma / true candidate gap
```

If `sigma/gap` drops from the stationary value around `31.7` to below `1`, the
probe idea is alive on a non-drift stationary workload.  If it stays large, the
proxy is still too noisy for this use.

This is close to population-based training at round granularity, but with a
different cost model: PBT normally needs full evaluations and copying.  Here a
small probe set can evaluate candidate update families at sub-percent overhead
because the denominator is the distributed round cost, roughly `3*K*N`
forward-equivalents.

## 2. Admission, Not Fine Reweighting

The second stationary use is admission:

```text
keep normal averaging unless a worker/update is confidently harmful
```

This is easier than ranking all healthy workers.  Ranking asks for tiny
best-vs-second gaps.  Admission asks whether one update is bad enough to reject.
The signal is large exactly when the decision matters.

This connects to two earlier positives:

- E028: tagged admission/reliability made otherwise-toxic async updates usable.
- Sister-paper value gates: labels mostly acted as admission evidence, not as a
  subtle continuous optimizer.

Existing robust aggregation rules use indirect surrogates:

```text
norm
distance from the pack
coordinate median / Krum-style geometry
clipping
```

A probe measures the direct question:

```text
does this proposed action reduce the proxy objective?
```

A large update can be good; clipping cannot tell.  A normal-norm update can be
harmful after local overfitting; geometry may miss it, but a probe can see it.

## 3. Prediction Across K

This gives a complementary prediction to the outer-control branch.

Outer-control headroom shrinks with `K` because local SGD increasingly
preconditions the problem and the `rho` spread falls:

```text
outer-control gap decreases as K grows
```

Admission headroom should grow with `K` because local workers have more time to
overfit, diverge, or become locally stale:

```text
admission gain increases as K grows
```

So the project should test a crossover:

```text
small K  -> round-level outer control matters most
large K  -> probe admission matters most
```

If true, the branches are not competing explanations.  They cover different
parts of the distributed-local-SGD regime.

## 4. What Not To Center

Do not lead with batch-size or compute heterogeneity.  It is real, but the
standard solution is already obvious:

```text
weight by n_i / sum_j n_j
```

Do not lead with generic federated non-IID data either.  It is real and gives
large gaps, but it is crowded territory and overlaps the sister paper.  The
cleaner Double Fan claim is:

> probes add value when aligned workers are made non-exchangeable by inner
> optimizer choices, or when probes act as conservative admission tests for
> harmful updates.

## Results Now In Hand

The experiment sketch below has now partly run.

E059 spread inner learning rates across individual workers.  That created large
oracle headroom but did not extract it: per-worker probe noise remained about
four times the adjacent target gap.  This confirmed that designed
non-exchangeability is not enough if the action space is still `N` individual
workers.

E060 coarsened the action space to LR groups and probed group means.  This gave
the first clean stationary positive:

```text
spread=1:  exact collapse, +0.000 over fixed
spread=4:  +0.167+/-0.153 over fixed, +0.228+/-0.129 over shuffled
spread=16: +0.304+/-0.175 over fixed, +0.691+/-0.325 over shuffled
cost:      about 1.34% of the distributed round
```

E061 swept `K in {4,8,16,32}`.  The grouped-probe mechanism stayed positive at
`K=8` and `K=16`, with the cleanest rows at spread 16:

```text
K=8,  spread=16: +0.231+/-0.066 over fixed
K=16, spread=16: +0.156+/-0.027 over fixed
```

It also exposed the next failure mode: at `K=32, spread=16`, the pilot-chosen
rule missed (`-0.002+/-0.011`) even though the best policy sweep was still
positive (`+0.183+/-0.159`).  That means the mechanism is live but the extractor
is brittle.

The next version should not treat LR groups as a fixed handcrafted arm set.  It
should use the workers as an online hyperparameter experiment and fit a response
surface over `log(inner_lr)`, initially with a Bayesian quadratic and later
possibly with a small GP.

## Experiment Sketch

Minimal decisive test:

```text
E059: stationary inner-LR spread probe admission

fixed alpha,beta
aligned workers
inner LR multipliers, e.g. {0.5, 0.75, 1.0, 1.5, 2.0}
compare:
  uniform FedAvg over all workers
  oracle group admission/weights
  damped probe group weights
  shuffled probe control
measure:
  sigma/gap
  oracle headroom
  probe capture
  cost vs 3*K*N
```

Decision:

- If oracle headroom is weak, stationary averaging still explains this axis.
- If oracle headroom is strong but probes fail, proxy extraction is the blocker.
- If damped probes beat uniform and shuffled controls, stationary
  non-exchangeable probes are live.
