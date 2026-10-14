# Deliberately Diverse Aggregation for Distributed Optimization

Status: TMLR submission packet scaffold, updated 2026-10-13.

Experiment state: E087/E088 were implemented and dispatched on 2026-10-07.
Finishable split runs landed on 2026-10-08.  E087 is a positive IID
stochastic-filtering null; E088 ultra is a negative fixed-compute sanity check.
E089 is the serious grouped-LR fixed-compute boundary against external HPO.
E090 is the stricter stale-LR final-trajectory audit and is negative. E090B is
the strongest fixed-compute demonstration of the same deliberately diverse
aggregation claim. E091 reruns E090B with fresh held-out seeds and preserves
the fixed-compute win. E093 repeats the main setting at 64 seeds and tightens
the external-HPO margin. E094 repeats the `rounds=16` stress run at 64 seeds
and clears the fixed-compute external-HPO comparator. E095 extends the
fixed-compute win to `K=16`. E096/E097 add lightweight tiny causal-LM tuning
results: probe labels matter, but the tiny-LM fixed-LR grid is simple enough
for external HPO to match the one-hot internal controller.

Target venue recommendation: **TMLR first**. The ICML version would need
another task family and a longer final-trajectory win beyond the current
local-SGD classifier/update-map bed.

Anonymous code and experiment artifacts:
https://anonymous.4open.science/r/local_gp-9F83/

## Working Title

Deliberately Diverse Aggregation for Distributed Optimization

Alternative:

- HPO-Aware Aggregation for Distributed Training
- Probe-Weighted Aggregation of Deliberately Diverse Updates
- Reinvesting Hyperparameter Exploration in Distributed Optimization
- Probe-Weighted Hyperparameter Arms in Local SGD

## One-Sentence Claim

Distributed training already pays for parallel worker updates; we deliberately
diversify those updates through optimizer/hyperparameter arms, use cheap probes
to score candidate next states, and aggregate the scored candidates back into
one shared trajectory.

## Abstract Skeleton

Hyperparameter tuning for distributed optimization is usually external to
training: run multiple jobs, select a winner, and discard the losing
trajectories.  Standard distributed aggregation, in contrast, treats worker
updates as exchangeable estimates of one optimizer setting.  We study
**deliberately diverse aggregation**: the training round itself creates a small
set of optimizer arms, cheap probes score the candidate next states, and the
server aggregates or selects the scored candidates into one shared trajectory.
This couples HPO with aggregation rather than placing HPO outside training.
Under IID identical arms, probe weights can still fluctuate and even filter
stochastic updates, but arm labels do not carry persistent hyperparameter
information.  The useful regime is deliberate diversity: worker LR arms,
server-side outer optimizer controls, and guarded stale-default recovery.
Across controlled local-SGD experiments, probe-weighted aggregation beats
shuffled probes and fixed/default controls, and in the outer-control setting it
beats validation-selected external HPO at fixed worker compute.

## Main Contributions

1. **Formulation.** Deliberately diverse aggregation: distributed rounds create
   optimizer-arm diversity and reinvest the resulting HPO evidence into the
   same update.
2. **Null model.** IID identical arms are exchangeable.  Realized probe weights
   can move, but any gain is stochastic-update filtering rather than HPO.
3. **Algorithm.** Group workers or server-side candidate updates into optimizer
   arms; score candidate next states on probe batches; apply conservative
   soft/shrunk weights or guarded fallback.
4. **Efficiency accounting.** Probe cost amortizes over `K * N` local work, so
   small validation probes cost sub-percent to low-single-digit overhead.
5. **Empirical HPO wins.** Deliberately diverse worker arms and server-side
   outer controls show label-content gains over shuffled probes and
   fixed/default baselines; E090B clears validation-selected external HPO at
   fixed worker compute, with a fresh-seed E091 replication.
6. **Boundary results.** Batch-size value/FLOP, naive validation-gradient
   scoring, free LR search, and exchangeable worker ranking are not promoted.

## Method

At round `t`, all workers start from `theta_t`.  Instead of assigning every
worker the same optimizer setting, define a small arm set
`h in H_t`.

Worker-arm version:

```text
worker/group i runs local optimizer h_i from theta_t
returns update u_i
probe scores theta_t + candidate(u_i)
server applies theta_{t+1} = theta_t + sum_i w_i u_i
```

Server-control version:

```text
workers return a common pool of updates
server forms candidate outer updates using (alpha, beta) arms
probe scores the candidate next states
server applies the selected/shrunk outer update
```

The second version answers the "fewer workers per parameter" objection: some
controls can be probed from the same collected worker updates without splitting
the training workers across arms.

### Algorithm Box Draft

Input: current parameters `theta_t`, worker count `N`, local steps `K`, arm set
`H_t`, probe batch `P_t`, prior weights `pi`, shrinkage `lambda`, fallback
policy `F`.

1. Assign workers or candidate server updates to arms `h_i in H_t`.
2. From common `theta_t`, obtain candidate updates `u_i`.
3. Form candidate next states `theta_t + A(u_i, h_i)`.
4. Evaluate probe losses `ell_i` on `P_t`.
5. Convert losses to posterior weights:

```text
q_i proportional to exp(-(ell_i - mean(ell)) / tau(ell))
w_i = (1 - lambda) pi_i + lambda q_i
```

6. If guard/fallback fails, use `F(theta_t)` or the prior/default arm.
7. Apply:

```text
theta_{t+1} = theta_t + sum_i w_i A(u_i, h_i)
```

For outer-control arms, `u_i` can be replaced by candidate server updates
constructed from the same worker-update pool.  For guarded stale-default
recovery, the soft weights become a binary abandon-or-fallback decision.

### Cost Accounting

Let each worker perform `K` local steps and let there be `N` workers.  If each
probe evaluation is a forward pass on `B_p` examples for `G` candidate groups
and `S` probe slices, the overhead is approximately:

```text
probe_overhead ~= G * S * B_p / (3 * K * N * train_examples_per_step)
```

The `3` is the rough forward/backward ratio used in the existing ledger.  The
important point is the amortization: the same probe cost is spread over `K*N`
worker-step work, which is why held-out measurement that would be expensive on
one machine becomes cheap in distributed training.

## Core Baselines

The main tables should use HPO columns, not only aggregation columns:

| baseline | purpose |
|---|---|
| default / stale default | what a user would run |
| best fixed arm | hindsight HPO target |
| tuned schedule | strong no-probe baseline |
| naive fixed-compute parallel HPO | same workers split across trajectories |
| shuffled probe | label-content control |
| internal HPO | our probe-weighted/guarded rule |
| oracle | headroom and gap closure |

Primary metrics:

- final target score or normalized improvement;
- internal HPO minus shuffled probe;
- internal HPO minus default/fixed/schedule;
- internal HPO minus naive parallel HPO at fixed worker budget;
- oracle gap closed;
- probe overhead percent.

## Experiment Package

### E063: Worker-Arm LR HPO

Frozen grouped LR arms with raw probe weighting.

Key evidence:

- spread 1: no persistent HPO gain;
- spread 16: raw probe beats fixed and shuffled at K=8,16,32;
- cost: `0.33%` to `1.34%`;
- capture is small, but label content is repeatable.

Use this as the clean worker-arm mechanism result.

### E080/E081: Outer-Control HPO

Low-K `(alpha,beta)` control with `scaled/soft_std2`.

Key evidence:

- E080 K<=8: selector minus shuffled `+0.177 +/- 0.049`;
- selector minus fixed `+0.116 +/- 0.052`;
- selector minus schedule `+0.164 +/- 0.057`;
- E081 packages this into a deployment rule with frozen large-K fallback.

Use this as the strongest low-dimensional server-side HPO result.

### E085: Stale-Default Recovery

Guarded stale LR default abandonment.

Key evidence:

- stale default `lr_multiplier=0.25`;
- guarded policy improves over stale-only at K=8/16/32;
- abandon rate around 20%;
- fallback remains the stale default unless probe slices agree.

Use this as the most legible practical HPO win.

### E087: IID Noise-Selection Null

Result: `IID_STOCHASTIC_FILTERING_POSITIVE` on Buck/PyTorch `TORCH_CPU`.

Purpose:

- fake arms with identical hyperparameters;
- measure weight motion under finite batch noise;
- report selector minus uniform and shuffled;
- separate stochastic-update filtering from HPO.

Expected paper role:

- strengthens the null section;
- call the positive result filtering, not HPO;
- forces precise language: identical IID arms do not imply literal uniform
  realized weights.

Key rows:

- selector minus uniform: `+0.621 +/- 0.320`;
- selector minus shuffled: `+0.476 +/- 0.222`;
- target oracle minus uniform: `+9.781 +/- 9.714`;
- mean weight L1 from uniform prior: `0.137`;
- entropy fraction: `0.991`;
- probe/target Spearman: `0.406`.

### E088: Fixed-Compute Parallel HPO Baseline

Result: ultra-reduced smoke returned `FIXED_COMPUTE_BASELINE_NOT_CLEARED` on
Buck/PyTorch `TORCH_CPU`.

Purpose:

- same total workers;
- external HPO splits them into separate LR-arm trajectories;
- internal HPO uses arms inside one shared trajectory and recombines;
- compare internal probe to external validation-selected and oracle HPO.

Expected paper role:

- directly answers "why not just run HPO in parallel?";
- this is the highest-value missing table for TMLR.

Ultra-smoke readout:

- `K=8`, `rounds=8`, `heldout=2`;
- internal minus external validation-selected HPO: `-0.006 +/- 0.002`;
- internal minus external oracle HPO: `-0.007 +/- 0.000`;
- internal minus shuffled internal: `+0.003 +/- 0.003`;
- internal minus parallel-arm average: `0.000 +/- 0.001`;
- cost: `1.34%`.

Interpretation: this does not support the fixed-compute claim yet.  It is a
sanity/negative smoke, not the final table.  The next fixed-compute run should
reuse a regime with demonstrated HPO headroom, likely E063 grouped LR with more
heldout seeds or E085 stale-default recovery.

### E089: Fixed-Compute Parallel HPO in E063 Headroom Regime

Result: serious grouped-LR run returned
`INTERNAL_HPO_CONTENT_LIVE_EXTERNAL_HPO_TIE` on Buck/PyTorch `TORCH_CPU`.

Purpose:

- same fixed-compute comparator as E088;
- E063 spread-16 grouped-LR headroom regime;
- internal HPO recombines grouped candidate updates into one shared trajectory;
- external HPO splits the same workers across separate LR-arm trajectories and
  selects one by validation loss.

Main readout:

- `K in {8,16,32}`, `rounds=36`, `heldout=8`;
- LR arms `(0.25,1.0,4.0)` with worker counts `(2,4,2)`;
- internal minus external validation-selected HPO:
  `-0.006 +/- 0.003`;
- internal minus external oracle HPO: `-0.007 +/- 0.003`;
- internal minus shuffled internal: `+0.004 +/- 0.002`;
- internal minus parallel-arm average: `+0.001 +/- 0.001`;
- probe costs: `1.34%`, `0.67%`, and `0.33%` for K=8/16/32.

Supplemental check: pilot-tuned parallel-average pairs reduced the validation
HPO gap to `-0.004 +/- 0.003`, but did not clear it.

Expected paper role:

- fixed-compute comparator is now real, not a tiny smoke;
- grouped-LR probes have label content against shuffled labels;
- do not claim this final-trajectory grouped-LR rule beats naive
  validation-selected parallel HPO;
- use E080/E081 and E085 for positive HPO wins, with E089 as the honest
  external-HPO boundary.

### E090: Fixed-Compute Stale-Default Recovery

Result: final-trajectory stale-LR recovery returned
`FIXED_COMPUTE_STALE_DEFAULT_NOT_CLEARED` on Buck/PyTorch `TORCH_CPU`.

Purpose:

- convert E085's guarded stale-default recovery into the same fixed-compute
  external-HPO ledger used by E089;
- include stale/fresh/high full-worker controls, shuffled guard, validation
  external HPO, external oracle, abandonment rate, arm mass, and probe cost;
- test whether the practical stale-default story survives a full-trajectory
  comparator rather than only E085's per-round recovery score.

Main readout:

- `K in {8,16,32}`, `rounds=36`, `heldout=12`;
- internal guarded minus stale full-worker default:
  `-0.008 +/- 0.002`;
- internal guarded minus fresh/default full-worker control:
  `+0.005 +/- 0.004`;
- internal guarded minus shuffled guard:
  `+6.117 +/- 1.841`;
- internal guarded minus external validation-selected HPO:
  `-0.007 +/- 0.003`;
- probe costs: `1.34%`, `0.67%`, and `0.33%`.

Expected paper role:

- keep E085 as a practical guarded-recovery result, but do not claim it clears
  the full stale/default final-trajectory comparator;
- use E090 as a limitation: a stale-default guard can have strong label-content
  and still trail a full-worker stale/default or external-HPO ledger when the
  fallback arm has more workers.

### E090B: Fixed-Compute Outer-Control HPO

Result: fixed-compute outer-control audit returned
`FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN` on Buck/PyTorch `TORCH_CPU`.

Purpose:

- price the E080/E081 low-K outer-control method against validation-selected
  external HPO at the same worker budget;
- use a stale/default transferred server pair, K-aware fresh/theory pair, and
  additional probe arms;
- compare internal soft probe selection, shuffled probes, validation-selected
  external HPO, and target-oracle diagnostics.

Main readout:

- `K in {4,8}`, `N in {8,16}`, `rounds=8`, `heldout=32`;
- internal minus stale/default control: `+3.789 +/- 0.515`;
- internal minus fresh/theory control: `+0.092 +/- 0.059`;
- internal minus shuffled internal: `+0.161 +/- 0.061`;
- internal minus external validation-selected HPO: `+1.354 +/- 0.463`;
- internal minus external oracle HPO: `+0.605 +/- 0.340`;
- external validation-selected HPO minus stale/default:
  `+2.434 +/- 0.562`;
- probe costs: `3.57%`, `1.79%`, `1.79%`, and `0.89%`.

Fresh-seed replication:

- E091 reran the same audit with held-out seeds `998100..998131`;
- decision: `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`;
- internal minus stale/default control: `+3.878 +/- 0.488`;
- internal minus shuffled internal: `+0.093 +/- 0.073`;
- internal minus external validation-selected HPO: `+1.148 +/- 0.511`;
- internal minus external oracle HPO: `+0.413 +/- 0.345`;
- probe costs remain `3.57%`, `1.79%`, `1.79%`, and `0.89%`.

64-seed replication:

- E093 reran the main `rounds=8` setting with held-out seeds `998300..998363`;
- decision: `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`;
- internal minus stale/default control: `+3.829 +/- 0.378`;
- internal minus shuffled internal: `+0.140 +/- 0.058`;
- internal minus external validation-selected HPO: `+1.224 +/- 0.313`;
- internal minus external oracle HPO: `+0.343 +/- 0.255`.

Longer-round stress run:

- E092 reran the audit with `rounds=16` and held-out seeds `998200..998231`;
- decision: `FIXED_COMPUTE_OUTER_CONTROL_HPO_LIVE`;
- internal minus stale/default control: `+2.838 +/- 1.979`;
- internal minus fresh/theory control: `+0.619 +/- 0.472`;
- internal minus shuffled internal: `+0.771 +/- 0.632`;
- internal minus external validation-selected HPO: `+0.330 +/- 2.151`;
- internal minus external oracle HPO: `-0.559 +/- 2.182`.

Longer-round 64-seed win:

- E094 reran the `rounds=16` audit with held-out seeds `998400..998463`;
- decision: `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`;
- internal minus stale/default control: `+4.403 +/- 0.417`;
- internal minus fresh/theory control: `+0.194 +/- 0.072`;
- internal minus shuffled internal: `+0.233 +/- 0.076`;
- internal minus external validation-selected HPO: `+1.275 +/- 0.239`;
- internal minus external oracle HPO: `+0.465 +/- 0.222`.

Large-K support:

- E095 reran the fixed-compute audit at `K=16`, `rounds=16`, with held-out
  seeds `998500..998563`;
- decision: `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`;
- internal minus stale/default control: `+0.094 +/- 0.061`;
- internal minus fresh/theory control: `+0.115 +/- 0.056`;
- internal minus shuffled internal: `+0.075 +/- 0.048`;
- internal minus external validation-selected HPO: `+0.516 +/- 0.262`;
- internal minus external oracle HPO: `-0.067 +/- 0.277`;
- probe cost falls to `0.89%` at `N=8` and `0.45%` at `N=16`.

Lightweight language-model tuning:

- E096 trains a tiny character-level causal Transformer on an embedded text
  corpus with LR arms `(0.5, 1, 2, 4)`, `rounds=12`, 16 held-out seeds, and a
  fixed-compute external HPO comparator;
- soft probe-weighted aggregation beats default fixed LR by
  `+0.202 +/- 0.011` and shuffled probes by `+0.033 +/- 0.021`, but trails
  validation-selected external HPO by `-0.096 +/- 0.015`;
- E097 switches the same LM audit to one-hot probe selection;
- one-hot internal selection beats default fixed LR by `+0.298 +/- 0.011`,
  beats shuffled probes by `+0.177 +/- 0.015`, and ties validation-selected
  external HPO (`-0.000 +/- 0.000`);
- interpretation: the LM result supports probe-label content and gives a
  language-model tuning sanity check, but it is not a full-scale LLM win.

Expected paper role:

- main positive fixed-compute table for deliberately diverse aggregation;
- cleanest demonstration that HPO-aware aggregation is not the same as
  ordinary parallel HPO: the internal method evaluates several server-control
  arms from the same worker-update pool, while external HPO must split the same
  workers across arm budgets;
- pairs naturally with E089/E090 as boundary results showing which forms of
  deliberate diversity are currently strong enough to beat external HPO.

## Null / Theory Section

State the null carefully:

> With identical IID arms, worker labels are exchangeable.  A probe controller
> may still produce non-uniform realized weights because both worker updates
> and probe batches are finite-sample random variables.  However, no arm label
> carries persistent hyperparameter information.  Any benefit in this regime is
> stochastic-update filtering and must be validated separately from the HPO
> claim.

Then:

- show symmetrization/Rao-Blackwell intuition;
- explain winner's curse under top-1 noisy probe selection;
- motivate shrinkage to the uniform/default prior;
- transition to designed nonexchangeability as the useful door.

## Introduction Draft Points

Opening paragraph:

Distributed optimization and hyperparameter optimization are usually treated as
separate loops.  The training system allocates many workers to estimate one
update, while HPO allocates many whole training runs to decide which optimizer
configuration should have been used.  This separation wastes information:
distributed workers already create parallel candidate computations from nearly
the same state, but standard training collapses them by uniform averaging and
standard HPO discards all but a winning trajectory.

Problem paragraph:

Naively combining the two loops is not enough.  If we simply run separate HPO
trials in parallel, each hyperparameter receives fewer workers per update and
the losing trials do not train the final model.  If we simply rank IID workers,
the labels are exchangeable and probe scores mostly select noise.  The central
move is to create **deliberate diversity** inside the distributed round:
workers or candidate server updates represent optimizer arms that are different
on purpose.

Thesis paragraph:

We propose deliberately diverse aggregation.  A training round becomes a paired
experiment over a small optimizer-control set.  Because all arms start at the
same `theta_t`, probe comparisons have far lower trajectory variance than
independent training runs.  Because the result is aggregated back into the
shared update, exploration contributes to the model rather than becoming
dead-end tuning work.

Boundary paragraph:

The boundary is part of the same claim.  Deliberate diversity is useful when
the arm choice changes same-round candidate quality enough for probes to
measure it.  When labels are exchangeable, probes may still filter stochastic
updates but do not create persistent HPO signal.  Controls whose value is
mostly cross-round, such as large changes in synchronization period, require a
sequential objective instead of a one-round aggregation decision.

## Related Work Hooks

- Population-based training: external online HPO, but trajectories branch and
  exploit by copying; our arms are paired at the same state and recombined.
- Hypergradient methods: differentiate through hyperparameters; our method uses
  held-out probe comparisons and does not require differentiating the training
  loop.
- D-Adaptation / Prodigy / schedule-free methods: reduce LR tuning burden; our
  setting uses distributed parallelism to cheaply test low-dimensional controls.
- FedAvg / local SGD / DiLoCo: standard aggregation treats worker updates as
  estimates of one optimizer setting; we intentionally create optimizer-arm
  variation and aggregate with probe evidence.
- Bandits / Bayesian optimization: standard BO spends trials; our feedback is
  paired within one round and acts through the shared update.

## Table Shells

Table 1: IID null / fake arms.

| K | selector-uniform | selector-shuffled | weight L1 | entropy | probe-target rho | top1 hit |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | `+0.654 +/- 0.484` | `+0.586 +/- 0.400` | `0.137` | `0.991` | `0.446` | `0.438` |
| 16 | `+0.588 +/- 0.489` | `+0.367 +/- 0.198` | `0.138` | `0.991` | `0.367` | `0.438` |

Table 2: Worker LR arms.

| K | fixed | shuffled | internal HPO | oracle | internal-fixed | internal-shuffled | cost |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | E063 | E063 | E063 | E063 | E063 | E063 | E063 |
| 16 | E063 | E063 | E063 | E063 | E063 | E063 | E063 |
| 32 | E063 | E063 | E063 | E063 | E063 | E063 | E063 |

Table 3: Low-K outer controls.

| setting | fixed | schedule | shuffled | internal HPO | internal-fixed | internal-schedule | internal-shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| E080 pooled | recorded | recorded | recorded | recorded | +0.116 | +0.164 | +0.177 |

Table 4: Stale-default recovery.

| K | stale default | fresh/default | shuffled guard | guarded internal HPO | guarded-stale | abandon rate |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | E085 | E085 | E085 | E085 | +0.997 | ~0.20 |
| 16 | E085 | E085 | E085 | E085 | +0.395 | ~0.20 |
| 32 | E085 | E085 | E085 | E085 | +0.727 | ~0.20 |

Table 5: Fixed-compute external HPO.

| setting | default | external HPO | external oracle | internal HPO | internal-external | internal-shuffled | cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 ultra | `0.092 +/- 0.045` | `0.092 +/- 0.048` | `0.093 +/- 0.046` | `0.086 +/- 0.046` | `-0.006 +/- 0.002` | `+0.003 +/- 0.003` | `1.34%` |
| 8 E089 | `0.068 +/- 0.019` | `0.068 +/- 0.019` | `0.068 +/- 0.019` | `0.063 +/- 0.019` | `-0.004 +/- 0.005` | `+0.003 +/- 0.003` | `1.34%` |
| 16 E089 | `0.068 +/- 0.019` | `0.068 +/- 0.018` | `0.069 +/- 0.018` | `0.062 +/- 0.020` | `-0.006 +/- 0.006` | `+0.004 +/- 0.003` | `0.67%` |
| 32 E089 | `0.068 +/- 0.020` | `0.069 +/- 0.019` | `0.069 +/- 0.018` | `0.061 +/- 0.022` | `-0.007 +/- 0.007` | `+0.006 +/- 0.003` | `0.33%` |
| E090 stale-LR pooled | stale full-worker | ext-valid | ext-oracle | guarded | `-0.007 +/- 0.003` | `+6.117 +/- 1.841` | `0.33%-1.34%` |
| E090B outer-control pooled | stale server pair | ext-valid | ext-oracle | soft probe | `+1.354 +/- 0.463` | `+0.161 +/- 0.061` | `0.89%-3.57%` |
| E091 outer-control fresh-seed replication | stale server pair | ext-valid | ext-oracle | soft probe | `+1.148 +/- 0.511` | `+0.093 +/- 0.073` | `0.89%-3.57%` |
| E093 outer-control 64-seed replication | stale server pair | ext-valid | ext-oracle | soft probe | `+1.224 +/- 0.313` | `+0.140 +/- 0.058` | `0.89%-3.57%` |
| E092 outer-control `rounds=16` stress | stale server pair | ext-valid | ext-oracle | soft probe | `+0.330 +/- 2.151` | `+0.771 +/- 0.632` | `0.89%-3.57%` |
| E094 outer-control `rounds=16` 64-seed | stale server pair | ext-valid | ext-oracle | soft probe | `+1.275 +/- 0.239` | `+0.233 +/- 0.076` | `0.89%-3.57%` |
| E095 outer-control `K=16`, `rounds=16` | stale server pair | ext-valid | ext-oracle | soft probe | `+0.516 +/- 0.262` | `+0.075 +/- 0.048` | `0.45%-0.89%` |
| E096 tiny causal-LM soft | fixed LR | ext-valid | ext-oracle | soft probe | `-0.096 +/- 0.015` | `+0.033 +/- 0.021` | `16.67%` |
| E097 tiny causal-LM argmin | fixed LR | ext-valid | ext-oracle | one-hot probe | `-0.000 +/- 0.000` | `+0.177 +/- 0.015` | `16.67%` |

## Claims To Avoid

- "Accidental worker differences are enough."
- "Probe weights should literally become uniform every IID round."
- "This solves large-K communication adaptation."
- "Batch size works" based on E083.
- "Validation-gradient marginal scoring works" until E086 has a real run.
- "GP/UCB is needed" before a new edge-pinning or regime-shift result.

## Venue Read

TMLR-ready shape:

- honest null model;
- HPO-first tables;
- careful cost accounting;
- unified claim plus boundary results.

ICML-ready shape:

- broader task family;
- broader fixed-compute HPO evidence;
- one clean algorithm box;
- fewer pending branches;
- likely closed-loop experiment beyond the current diagnostic bed.

Current recommendation: write the TMLR paper as **Deliberately Diverse
Aggregation for Distributed Optimization**.  E090B is the strongest
fixed-compute demonstration; E091 and E093 replicate it, with E093 tightening
the main fixed-compute external-HPO margin over 64 held-out seeds; E094 clears
the longer `rounds=16` stress ledger over 64 held-out seeds; E095 extends the
fixed-compute win to `K=16`; E096/E097 add lightweight LM tuning sanity rows
that should be framed as probe-label-content evidence, not as the main
fixed-compute win; E080/E081 replicate the core server-side outer-control
mechanism; E063 shows deliberately diverse worker LR arms; E089 and E090 are
boundary ledgers; E087 is stochastic filtering under exchangeable fake labels
rather than HPO.
