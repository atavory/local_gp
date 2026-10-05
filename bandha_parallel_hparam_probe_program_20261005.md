# Parallel Hyperparameter Probes

Status: live synthesis, 2026-10-05.

This note supersedes the narrow "rank workers with probes" framing.  The useful
object is not a set of exchangeable workers.  The useful object is a parallel
experiment over optimizer hyperparameters.

## One Sentence

Use aligned workers as a live experimental design over inner optimizer
hyperparameters, use cheap probe sets to estimate the value of the returned
candidate updates, and aggregate through a conservative posterior rule that
shrinks back to averaging when the evidence is weak.

## Why This Is Different

The failed stationary setup was:

```text
same theta_t
same K
same inner LR
same optimizer
N nearly exchangeable worker updates
probe tries to rank them
```

That is a bad statistical problem.  Uniform averaging is already close to the
right answer, and the gaps between workers are mostly minibatch luck.

The new setup is:

```text
same theta_t
aligned round
worker groups run different inner hyperparameters
probe scores the resulting group-mean candidate updates
server learns/uses a value surface over hyperparameters
```

The worker differences are now first-order design differences, not accidental
sampling noise.  The probe asks "which hyperparameter arm is valuable in this
training phase?" rather than "which worker got lucky?"

## Favorable Results In Hand

### E059: Individual LR Arms Create Headroom But Do Not Extract

E059 deliberately spread inner learning rates across individual workers.  It
created large oracle headroom, but the per-worker probe rule did not extract it:
the proxy noise was still roughly four times the adjacent candidate gap.

Interpretation: non-exchangeability is necessary but not sufficient.  Ranking
individual workers is still too noisy.

### E060: Grouped LR Arms Give A Real Stationary Positive

E060 grouped workers by inner LR and probed the group means.

Key held-out rows:

| spread | groups | fixed | oracle subset | probe policy | minus fixed | minus shuffled | cost |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `1` | 5.654 | 5.654 | 5.654 | 0.000+/-0.000 | 0.000+/-0.000 | 0.22% |
| 4 | `0.5,1,2` | 5.018 | 8.597 | 5.185 | 0.167+/-0.153 | 0.228+/-0.129 | 1.34% |
| 16 | `0.25,1,4` | 3.226 | 7.644 | 3.531 | 0.304+/-0.175 | 0.691+/-0.325 | 1.34% |

The important control is spread 1: the method collapses exactly to no effect
when the arms are identical.  It is not hallucinating gains under
exchangeability.

### E061: K Sweep Confirms The Mechanism, With Fragility

E061 repeats the grouped-LR admission across `K in {4,8,16,32}`.

Key held-out rows:

| K | spread | fixed | oracle subset | pilot-best | minus fixed | minus shuffled | cost |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 7.774 | 12.407 | 8.446 | 0.672+/-0.634 | 1.034+/-0.898 | 2.68% |
| 4 | 16 | 5.547 | 9.756 | 6.143 | 0.597+/-0.610 | 1.103+/-1.070 | 2.68% |
| 8 | 16 | 3.209 | 5.826 | 3.441 | 0.231+/-0.066 | 0.460+/-0.226 | 1.34% |
| 16 | 16 | 1.057 | 2.706 | 1.214 | 0.156+/-0.027 | 0.237+/-0.088 | 0.67% |
| 32 | 4 | 1.112 | 2.653 | 1.187 | 0.075+/-0.058 | 0.157+/-0.132 | 0.33% |
| 32 | 16 | 1.335 | 4.185 | 1.332 | -0.002+/-0.011 | 0.029+/-0.025 | 0.17% |

Read precisely:

- Positive at `K=8` and `K=16`, especially with spread 16.
- Correct zero effect when spread is 1.
- `K=32, spread=16` exposes policy fragility: the pilot-chosen damped rule
  fails even though the best sweep remains positive (`+0.183+/-0.159`).

So the mechanism is real but under-extracted.  This is not yet a polished
method.

### E062: Online Tracker Does Not Improve Raw Group Probes

E062 tested a quadratic/EMA response tracker over the same LR groups.  This was
the first attempt to turn the grouped arms into an online hyperparameter
surface model.

Decision: `FROZEN_GROUP_PROBE_LIVE_TRACKER_TIES`.

Readout:

- raw grouped probes remained positive;
- the tracker mostly selected the same `group_mean2_l40_t2` rule as raw;
- when it differed at `K=32`, the improvement over raw was tiny
  (`+0.006+/-0.024`, `+0.007+/-0.062`);
- therefore GP/response-surface machinery is not justified yet by these rows.

The important correction is that "fit a surface" is not automatically better
than "score three frozen group means."  The next model has to beat raw grouped
probes, not merely beat the fixed default arm.

### E063: Frozen Group Probes Replicate Across K

E063 froze the E060/E061 raw rule (`group_mean2_l40_t2`) and repeated it with
12 held-out seeds.  No pilot-selected policy was allowed to move.

Decision: `RAW_GROUP_PROBE_REPLICATES`.

| spread | K | fixed | oracle | raw | raw minus fixed | raw minus shuffled | capture | cost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 5.055 | 5.055 | 5.055 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.45% |
| 16 | 8 | 2.988 | 5.931 | 3.239 | 0.251+/-0.153 | 0.447+/-0.207 | 0.072+/-0.041 | 1.34% |
| 1 | 16 | 2.832 | 2.832 | 2.832 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.22% |
| 16 | 16 | 1.685 | 4.570 | 1.847 | 0.162+/-0.118 | 0.439+/-0.130 | 0.058+/-0.040 | 0.67% |
| 1 | 32 | 1.554 | 1.554 | 1.554 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.11% |
| 16 | 32 | 1.090 | 5.898 | 1.361 | 0.271+/-0.147 | 0.824+/-0.303 | 0.053+/-0.027 | 0.33% |

This is the cleanest favorable stationary result so far:

- exact zero effect under exchangeability (`spread=1`);
- positive held-out gains at `K=8,16,32` when the LR arms are deliberately
  non-exchangeable;
- positive shuffled-control gaps;
- sub-1.5% probe cost.

It is still a small-capture result, not a "smash everything" result.  The value
is repeatability and collapse behavior: the signal is real, conservative, and
not fabricated by noisy ranking.

## Current Interpretation

The favorable result is:

> Probes can extract stationary value from designed optimizer
> non-exchangeability when evidence is pooled at the hyperparameter-group
> level and shrinkage keeps the rule close to averaging.

The limitation is:

> Fixed damped rules capture only a small fraction of oracle headroom and are
> sensitive to K/regime.

E062 says the first online tracker does not solve that limitation.  So the
current live branch is narrower and firmer:

> use parallel workers to create optimizer-arm non-exchangeability, probe
> group-level candidate updates, and aggregate conservatively.

The next question is economic rather than philosophical: how cheap can the
probe be while retaining the E063 edge?

## New Program: Track The Hyperparameter Surface

Each worker group gets a hyperparameter setting:

```text
h_i = (inner_lr_multiplier, K, momentum, weight_decay, optimizer_type, ...)
```

All groups start from the same `theta_t`, run their local optimizer, and return
candidate updates:

```text
u_i = local_update(theta_t; h_i)
```

The probe set returns noisy values:

```text
y_i ~= V_t(h_i) = improvement(theta_t + alpha * u_i)
```

The server should not independently argmin one round's noisy values.  It should
fit a low-dimensional response model:

```text
V_t(h) = smooth function of log_lr, K, momentum, ...
```

First model to try:

```text
quadratic in log(inner_lr_multiplier)
with exponential forgetting over rounds
```

Then, if useful:

```text
small GP / Bayesian linear model over 1-3 hyperparameter dimensions
Thompson sampling or UCB to place next round's worker arms
```

The parallelism is the point.  Full-run HPO spends one whole training run per
point.  This method gets several same-base hyperparameter observations per
round and evaluates them with a tiny probe set.

## Why A GP Might Be Right, But Not First

A GP is attractive because the response surface over `log_lr` and `log_K` is
smooth but not necessarily quadratic.  However, the immediate bottleneck is not
expressiveness.  It is extraction under noisy probes.

Start with:

```text
Bayesian quadratic over log_lr
forgetting factor for drift
mandatory default arm
random exploration arms
conservative shrinkage toward uniform/FedAvg
```

Move to GP only after the quadratic beats:

- fixed grouped arms;
- shuffled probes;
- best fixed default arm;
- offline tuned schedule;
- oracle diagnostic gap.

## Next Experiments

### E062: Online Quadratic LR Tracker  [DONE]

Use aligned rounds.  Worker groups are placed at three or five
`log_lr_multiplier` points around a center.  Each round:

1. run group workers;
2. probe group-mean candidate updates;
3. update a Bayesian quadratic model of value versus `log_lr_multiplier`;
4. choose the next center/radius;
5. aggregate with damped weights around the model's posterior optimum.

Controls:

- uniform default LR;
- fixed LR group design from E060/E061;
- shuffled probes;
- oracle group choice;
- best fixed LR group chosen in hindsight.

Outcome:

- tracker tied raw grouped probes;
- GP/response-surface work is premature until it beats raw grouped probes.

### E063: Frozen Group-Probe Replication  [DONE]

Freeze `group_mean2_l40_t2`, repeat across `K in {8,16,32}`, and verify
collapse at spread 1.

Outcome: replicated.  This is now the baseline to beat.

### E064: Cost/Noise Sweep

Vary probe batch/slices:

```text
select_batch in {96, 192, 384, 768}
slices in {1, 2, 4}
```

Measure the capture/cost frontier.  The method only matters if it keeps a
positive edge below roughly 1-3% round overhead.

### E065: Add A Second Hyperparameter Dimension

After the cost/noise frontier:

```text
log_lr_multiplier x K_multiplier
```

or

```text
log_lr_multiplier x inner_momentum
```

This is where a GP becomes more natural.

### E066: Real Workload Path

Take the best E063/E064 setting to a real small network path.  Do not claim the
paper from toy rows alone.

## Paper Shape

The paper should not claim "Bayesian probes beat averaging" in full generality.

Better claim:

> Distributed training has parallel experimental bandwidth that standard
> averaging wastes.  If aligned workers are assigned diverse optimizer
> hyperparameters, cheap probe sets can track which local optimizer settings are
> valuable in the current training phase and can conservatively aggregate them,
> collapsing to averaging when the arms are exchangeable.

That claim fits the favorable evidence:

- E060/E061 show the stationary grouped-LR mechanism is live.
- E063 replicates the frozen group-probe rule across K with exact collapse.
- Spread-1 collapse shows the rule does not invent signal.
- E059 explains why individual-worker ranking failed.
- E062 shows online surface tracking is not yet useful beyond raw grouped
  probes.
- Earlier schedule-transfer failures motivate online tracking instead of
  offline retuning.
