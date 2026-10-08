# Internal HPO Option Map

Status: live map, updated 2026-10-07.

Purpose: make the internal-HPO option space explicit.  Do not let the project
collapse back into "one clever adaptive method."  The research object is:

> aligned distributed workers create a cheap in-round experimental design over
> optimizer controls; probe sets decide which controls are worth using; the
> server reinvests the experiment into the same shared trajectory through
> probe-weighted aggregation, shrinkage, and fallback.

This is **HPO-aware aggregation**.  The comparison class is not only FedAvg and
not only external HPO.  It is the fixed-compute question: with the same workers,
is it better to fork arms into separate trajectories and keep one later, or to
probe the arms and recombine their useful signal into the model being trained
right now?

## Current Hard Facts

Actionable positive:

```text
E080, K<=8, select=384, scaled/soft_std2:
  selector - shuffled = +0.177 +/- 0.049
  selector - fixed    = +0.116 +/- 0.052
  selector - schedule = +0.164 +/- 0.057
```

Interpretation: probe labels carry real content in the low-K outer-control
setting.  This is not just the candidate family being good, because shuffled
labels lose.

Important null:

```text
E079 budget sweep, 13 heldout seeds:
  selector - shuffled = +0.040 to +0.090, CI crosses zero
```

Interpretation: small sweeps are underpowered for label-content claims.  Use
high-power replications for gates; use budget sweeps only after content is
established.

Main boundary:

```text
K<=8: probe content is live.
K>=16: outer-control headroom shrinks and fallback matters more than probing.
```

Deployment positive:

```text
E081:
  probe_small_global_fixed_large:
    minus schedule = +0.231 +/- 0.090
    minus per-K fixed = +0.052 +/- 0.085
    low-K minus shuffled = +0.175 +/- 0.065

  probe_small_theory_large:
    minus schedule = +0.298 +/- 0.075
    minus per-K fixed = +0.119 +/- 0.066
    minus best-control ceiling = +0.051 +/- 0.067
    low-K minus shuffled = +0.175 +/- 0.065
```

Interpretation: E080 was not merely a selector screen.  E081 gives a
deployment-shaped method: use the low-K probe controller where it is proven,
and use a frozen large-K fallback that does not import per-K tuning.

Busy-loop gate updates:

```text
E082:
  LR skew summary over E060/E061:
    nontrivial cells = 10
    pilot-best policy counts = group_mean2_l40_t2: 8, group_l20_t2: 1,
                               group_l05_t4: 1
    mean pilot-best - fixed = +0.245 +/- 0.137
    mean pilot-best - shuffled = +0.447 +/- 0.231

E083:
  batch-size/value-per-FLOP, REAL_GPU:
    decision = BATCH_VALUE_PER_FLOP_NEGATIVE
    K=8 value/FLOP - default = +1.009 +/- 1.254
    K=8 value/FLOP - shuffled = +0.054 +/- 0.112
    K=16 value/FLOP - default = +5.647 +/- 9.583
    K=16 value/FLOP - shuffled = +0.287 +/- 0.405

E084:
  selected-arm trajectory for E080/E081 scaled/soft_std2, REAL_GPU:
    decision = OUTER_TRAJECTORY_STABLE_SOFT_INTERIOR
    effective alpha ~= 19.7-19.9 at K=4 and ~= 9.95-10.03 at K=8
    effective beta ~= 0.116 in all cells
    normalized phase spans <= 0.008
    soft alpha-edge mass ~= 0.08-0.09
    entropy ~= 0.976

E085:
  guarded stale-default HP-arm, REAL_GPU:
    decision = STALE_DEFAULT_GUARDED_PROBE_LIVE
    stale default lr_multiplier = 0.25
    guard = abandon stale default only when 4-slice probe comparison clears
            margin and 3/4 slices agree
    guarded - stale = +0.997 +/- 0.739 at K=8
    guarded - stale = +0.395 +/- 0.654 at K=16
    guarded - stale = +0.727 +/- 1.114 at K=32
    abandon rate ~= 0.20 per round
```

Interpretation: E082 does not show a summary-level collapse to the small/safe
LR arm, but exact LR arm mass is not present in the old artifacts and still
needs a trajectory/arm-mass trace before free LR search becomes `LIVE`.  E083
finds batch-size oracle headroom, but the default-anchored soft value/FLOP
selector does not beat shuffled labels cleanly, so the simple batch-size arm is
not promoted.  E084 says the deployed outer-control selector is not secretly
edge-pinned and does not require a moving grid/GP yet: it is a high-entropy,
stable soft interior mixture centered near the theory-scaled design.  E085
rescues the intended stale-default customer case, but only after changing the
decision rule from soft mixing to a guarded abandonment test; that promotes
stale-default recovery as an internal-HPO use case without reopening free LR
search.

HPO table requirement for the writeup:

| column | reason |
|---|---|
| stale/default | what a user would have run |
| best fixed / best schedule | hindsight target and tuned baseline |
| naive fixed-compute HPO | the obvious "just run parallel HPO" alternative |
| shuffled probe | label-content control |
| probe-weighted internal HPO | our deployed rule |
| oracle | headroom and gap-closure denominator |

Do not report only "minus fixed" aggregation rows in the main text.  Report
regret to tuned HPO baselines, oracle gap closed, and improvement per percent
probe overhead.

## Status Labels

| label | meaning |
|---|---|
| `LIVE` | has replicated actionable signal |
| `AUDIT` | being tested as a deployable rule or cost ledger |
| `GATED` | plausible, but blocked behind a specific diagnostic |
| `DEFER` | possible later; no current evidence or too many degrees of freedom |
| `KILL` | controlled evidence says stop funding this branch |

## Sorting Rule

The main rule for deciding which HPO belongs inside the distributed round:

> Internal HPO works for hyperparameters whose **cost and payoff are both
> visible inside the round**.  It fails, or needs a different mechanism, for
> hyperparameters whose payoff is mostly cross-round.

Reason: the probe observes candidate updates from one shared base point
`theta_t`.  It can price "this candidate update is better per unit cost" very
well.  It cannot directly price "this choice will save communication over the
next thousand rounds" unless that long-horizon benefit is put into a separate
sequential objective.

This rule changes the priority order:

| tier | option | why |
|---|---|---|
| 1 | outer `(alpha,beta)` at `K<=8` | already replicated; direct one-round value |
| 1 | batch size / value per FLOP | cost and payoff are both in-round; E083 says the simple soft selector is not enough |
| 1 | inner LR multiplier around a schedule | visible in-round; E082 found no summary collapse but exact arm mass is still missing |
| 2 | fault veto / admission | one-round badness is visible; rare-event payoff |
| 3 | K / sync period | communication benefit is cross-round; needs sequential test, not worker-arm selection |
| 4 | weight decay / Adam internals | too slow to observe reliably in one round |

## Five Primary Option Families

| family | what is being internally tuned | current status | why it matters |
|---|---|---|---|
| A | outer optimizer controls | `LIVE` for K<=8 | strongest evidence, lowest dimensionality |
| B | in-round resource controls | `GATED` | batch size and value/FLOP are the cleanest untested HPO |
| C | inner worker hyperparameters | `GATED` | uses parallel workers as real HPO arms |
| D | synchronization / K / communication | `GATED` | valuable, but not an in-round worker-arm problem |
| E | admission / safety / fallback | `AUDIT` | converts probe signal into shippable behavior |
| F | measurement / scoring variants | `GATED` | may improve extraction without changing action space |

## Option Inventory

### A. Outer Optimizer Controls

These are the current center of gravity because they are low-dimensional and
already have replicated probe content.

| # | option | status | evidence | next gate |
|---:|---|---|---|---|
| A1 | choose outer scale `alpha_t` from frozen candidate set | `LIVE` | E078/E080 positive in K<=8 | keep in deployment rule |
| A2 | choose outer momentum `beta_t` jointly with `alpha_t` | `LIVE` | E078/E080 use `(alpha,beta)` pairs | keep joint, not separate |
| A3 | K-aware scaled candidate design | `LIVE` | E073/E078/E080; theory-motivated | freeze, do not retune per run |
| A4 | `soft_std2` shrinkage selector | `LIVE` | best replicated policy in E078/E080 | freeze as default selector |
| A5 | top-k mean selector | `DEFER` | screened earlier, weaker than soft policy | only revisit if soft fails |
| A6 | top-1 argmin selector | `KILL` | winner's curse / unstable rankings | do not use as headline |
| A7 | guard-to-fixed policy | `DEFER` | safety useful, not current best | test only in deployment audit |
| A8 | selected-arm trajectory diagnostic | `DONE` | E084: stable soft interior, no phase movement | no moving grid needed now |
| A9 | grid-edge expansion | `DEFER` | E084: low alpha-edge mass, high entropy | only revisit if a new regime pins edges |
| A10 | per-round quadratic surface fit | `KILL` | extrapolation / schedule issues | do not revive without trust region |
| A11 | sliding schedule fit over rounds | `KILL` for stationary | converges to tuned schedule | only possible under proven drift |
| A12 | hysteresis over selected `(alpha,beta)` | `GATED` | reduces churn; natural deployment control | test after frozen rule passes |

### B. In-Round Resource Controls

These are the cleanest candidates for internal HPO because the denominator is
inside the round.  The probe can score **value per unit compute**, not just raw
loss drop.

| # | option | status | evidence | next gate |
|---:|---|---|---|---|
| B1 | batch-size arms, scored value/FLOP | `GATED` | E083 negative for simple soft value/FLOP; oracle headroom remains | do not promote without better extractor |
| B2 | microbatch / accumulation arms | `DEFER` | same family as batch size | after B1 |
| B3 | local compute budget per worker | `GATED` | close to batch/K boundary | only if value/FLOP is explicit |
| B4 | adaptive repeat/replicate for noise | `GATED` | E027 oracle liked replicate | needs costed noise regime |
| B5 | probe budget allocation itself | `GATED` | E079 says naive bigger batch not enough | only after score improves |

Batch size had the right economics, but E083 blocks the simple implementation:

- a worker at `2B` spends about `2x` compute;
- the probe can score improvement per FLOP in the same round;
- critical batch size moves during training, so a static default can become
  stale;
- unlike K, the benefit is not hidden in future communication savings;
- nevertheless, the E083 value/FLOP selector tied shuffled labels, so this
  branch needs a stronger model or a different resource control before it is a
  method claim.

### C. Inner Worker Hyperparameters

These make workers deliberately non-exchangeable.  That is the right use of
parallelism: not "rank identical workers," but "run a small HPO design inside
the distributed round."

| # | option | status | evidence | next gate |
|---:|---|---|---|---|
| C1 | inner LR multiplier arms | `GATED` | E060/E061 positive; E082 no summary safe-collapse | exact arm-mass trace before free LR search |
| C2 | default-anchored LR arms | `AUDIT` | E067-E069 soft anchoring negative; E085 guarded rule positive | use guarded abandonment, not soft mixing |
| C3 | deliberately bad/stale default recovery | `LIVE` | E085 guarded stale-default test positive | preserve guard and shuffled control |
| C4 | log-spaced LR arm grid | `GATED` | plausible HPO design | only after C1/C3 pass |
| C5 | LR multiplier on a known schedule | `GATED` | E082 reduces myopia concern but old artifacts lack arm mass | test after selected-arm/arm-mass trace |
| C6 | inner momentum arms | `DEFER` | no headroom result yet | test only after LR/batch |
| C7 | AdamW beta1/beta2 arms | `DEFER` | high dimension, no evidence | wait for simpler axes |
| C8 | weight decay arms | `DEFER` | slow-timescale, hard to score per round | likely offline/slow posterior |
| C9 | gradient clipping arms | `GATED` for safety | may help bad-round/fault regimes | test only with injected bursts |
| C10 | optimizer-type arms, e.g. SGD vs AdamW | `DEFER` | too discrete/high-variance | only if phase-transition evidence |
| C11 | local LR schedule arms | `DEFER` | could be useful but many degrees | after selected-arm diagnostic |

LR myopia check: E082 reanalyzed E060/E061 and did not find a summary-level
collapse to the most conservative/default-preserving policy.  The caveat is
important: the old artifacts do not store exact low/default/high LR arm mass,
so free LR search stays gated.  E085 changes the viable customer story: do not
run free soft LR search from a strong known default; instead, keep a stale or
bad default as the fallback and abandon it only under a guarded same-round
probe comparison.

### D. Synchronization / Communication

These are valuable, but they are not clean in-round HPO.  K's benefit is fewer
communications in the future; a one-round probe sees mostly the cost side
(local drift / overfitting).  So K should not be tested as "workers try
different K and probes pick the best worker" unless the objective explicitly
includes communication cost.

The more plausible K mechanism is a sequential test across rounds:

```text
keep taking local steps while estimated marginal value of another local step
exceeds estimated communication/synchronization cost
```

That is closer to CUSUM/SPRT than to GP bandit HPO.

| # | option | status | evidence | next gate |
|---:|---|---|---|---|
| D1 | choose K / local steps online | `GATED` | valuable but wrong in-round mechanism | sequential marginal-value test |
| D2 | choose sync period from probe score | `GATED` | same as D1 | wall-clock/comm model |
| D3 | small-K probe controller, large-K fallback | `LIVE` | E080 content + E081 deployment | method core |
| D4 | K=8 transferred fallback | `KILL` | E081: large negative vs per-K fixed/schedule | drop |
| D5 | global fixed fallback | `LIVE` | E081 decision label | safe fallback |
| D6 | theory-derived K fallback | `LIVE` | E081: cleaner no-pilot result | promote as deployment candidate |
| D7 | per-K tuned fixed as baseline | `AUDIT` only | useful comparator but not free | cost separately |
| D8 | per-K tuned schedule as baseline | `AUDIT` only | can overfit pilots | cost separately |

### E. Admission, Safety, and Collapse

These are deployment controls: when to trust, when to fall back, and when to
veto.

| # | option | status | evidence | next gate |
|---:|---|---|---|---|
| E1 | exchangeability null when evidence has no persistent arm signal | `LIVE` | spread-1 no-gain rows; needs explicit IID noise audit | keep as invariant and measure realized weight motion |
| E2 | shuffled-label control | `LIVE` | E078/E080 decisive | keep in every content gate |
| E3 | fault veto / bad round rejection | `GATED` | easier than ranking healthy workers | inject bad-worker/burst test |
| E4 | conservative fallback outside proven regime | `LIVE` | E081 | freeze rule |
| E5 | tag/group reliability posterior | `GATED` | E028-like mechanism, side info helps | real tags only |
| E6 | stale update admission | `DEFER` | async confounded earlier | require cost ledger |
| E7 | rollback/retry trigger | `DEFER` | rare-event payoff, no test yet | after E3 |
| E8 | early-stop / restart trigger | `DEFER` | O(1) decision, pooled evidence | later systems claim |

### F. Measurement / Score Variants

These change how probe evidence is read.  They should be tested inside the
current proven action family before adding a GP or larger arm space.

| # | option | status | evidence | next gate |
|---:|---|---|---|---|
| F1 | forward-difference probe loss | `LIVE` | E078/E080 | current default |
| F2 | larger probe batch | `KILL` as simple fix | E079: no clear gain 384→1536 | do not spend here blindly |
| F3 | validation-gradient marginal score `<g_V,u>` | `PENDING_RUN` | E070 simple default-anchored version failed; E086 current-rule raw-group comparison implemented; first worker-route attempts did not produce a real run | run E086 on the correct GPU lane |
| F4 | response model over candidate features | `GATED` | may improve extraction | test after deployment rule |
| F5 | split select/eval probe shards | `GATED` | selection-bias audit | use in real workload version |
| F6 | rotating probe shards | `GATED` | avoids validation overfit | required for long runs |
| F7 | training-loss self-report | `DEFER` | biased but free | only as auxiliary feature |
| F8 | pooled round-quality score | `GATED` | higher SNR than per-worker ranking | combine with outer control |
| F9 | Bayesian shrinkage over probe scores | `LIVE` conceptually | `soft_std2` is empirical shrinkage | formalize after E081 |
| F10 | GP / UCB / Thompson over controls | `DEFER` | E084 shows no edge pinning or phase movement | require new regime-shift evidence |

### G. Killed Or Frozen Branches

These stay in the paper as boundaries, not as active engineering work.

| # | option | status | evidence | note |
|---:|---|---|---|---|
| G1 | rank exchangeable workers per round | `KILL` | exchangeability / low ceiling | do not revive without non-exchangeability |
| G2 | dense local map | `KILL` | rate/stationarity/cost | use only as oracle diagnostic |
| G3 | server/time secants | `KILL` | trajectory destroys spectral diversity | E020/B3 negative |
| G4 | active direction targeting | `KILL` | lost to random/fixed | no active-probe claim |
| G5 | entropy/D-opt acquisition | `KILL` | wrong objective | Bayes-risk only if revived |
| G6 | async throughput headline | `KILL` as main claim | cost-accounting confound | keep only as separate systems branch |

## What To Do Next

Immediate:

1. Write the frozen deployment rule as the current method.
2. Preserve both E081 variants in the paper: global-fixed fallback as the
   conservative decision-label winner, theory fallback as the cleaner no-pilot
   deployment candidate.
3. Do not run another same-family selector sweep for the E080/E081 controller:
   E084 shows a stable soft interior mixture, not edge pinning.
4. Add the fixed-compute HPO baseline and IID exchangeability audit before
   calling the experimental package submission-ready.

Then:

5. E082 LR-arm skew summary is done: no summary safe-collapse, but exact arm
   mass is still missing.
6. E083 batch-size/value-per-FLOP is done: simple soft value/FLOP is negative
   against shuffled labels despite oracle headroom.
7. E084 selected-arm trajectory is done: the frozen grid sits in a stable,
   high-entropy interior mixture, so moving-arm/GP machinery is not justified
   for the current outer-control rule.
8. E085 bad/stale-default HP-arm test is done: guarded abandonment from a
   stale LR default is live, while earlier soft default anchoring remains
   negative.
9. E086 marginal-value score in the current grouped-rule comparison is
   implemented but not scientifically run.  The first worker-route attempts
   lacked `torch` or did not expose CUDA device nodes inside the worker process;
   do not record that as a project-level GPU blockage.
10. Only consider GP/UCB/Thompson after a new regime-shift result shows frozen
   controls failing while probe labels still beat shuffled labels.

## Current Best Method Statement

> In aligned distributed training, use a frozen K-aware candidate set over
> outer `(alpha,beta)` controls.  At `K<=8`, score candidates with a small
> probe set and select by conservative soft shrinkage.  Outside the proven
> low-K regime, fall back to a frozen no-probe rule, preferably the
> theory-derived K rule unless a deployment demands the more conservative
> global fixed fallback.

That is the piece with replicated probe content.  Everything else in this file
is an option with a gate, not a claim.

## Explicit Research Queue

| priority | experiment | promotes which option | pass criterion |
|---:|---|---|---|
| 1 | write frozen deployment method | A/D/E frozen method | E080/E081 as core evidence |
| 2 | LR skew check on E060/E061 | C1/C5 | DONE E082: no summary safe-collapse; arm-mass trace still needed |
| 3 | batch-size arm | B1 | DONE E083: simple value/FLOP selector negative vs shuffled |
| 4 | selected-arm trajectory | A8/A9/F10 | DONE E084: stable soft interior; moving grid not triggered |
| 5 | bad/stale default HP-arm test | C2/C3 | DONE E085: guarded stale-default recovery live |
| 6 | marginal-value score in current rule | F3 | PENDING E086: implemented; needs the correct real-GPU run for raw-vs-gradient result |
| 7 | IID exchangeability/noise-selection audit | E1/G1 | DONE E087: fake IID arms give stochastic-update filtering, not HPO |
| 8 | fixed-compute naive-HPO baseline | paper comparator | DONE E089: serious grouped-LR table run; internal label content live, external validation-HPO not cleared |
| 9 | sequential K test prototype | D1/D2 | marginal local-step value predicts when to sync |

## Latest Result Addendum - 2026-10-08

E087 resolves the IID nuance.  Fake labels over identical workers do not create
persistent hyperparameter signal, but the realized probe weights are not
trivial: selector minus uniform was `+0.621 +/- 0.320` and selector minus
shuffled was `+0.476 +/- 0.222`, with mean L1 from prior `0.137` and entropy
`0.991`.  This belongs in the null section as **stochastic-update filtering**.
Do not use it as an internal-HPO win.

E088 ultra is the first fixed-compute external-HPO sanity check, and it did
not clear the baseline: internal minus external validation-selected HPO was
`-0.006 +/- 0.002`; internal minus shuffled was only `+0.003 +/- 0.003`.
Because this was intentionally tiny (`K=8`, `rounds=8`, `heldout=2`), it is a
negative smoke/sanity result, not the final fixed-compute table.  The full
claim still needs a high-power fixed-compute comparison in a regime where E063
or E085 has meaningful HPO headroom.

E089 supplies that higher-power fixed-compute table in the E063 grouped-LR
spread-16 regime.  It used `K in {8,16,32}`, 36 rounds, eight held-out seeds,
LR arms `(0.25,1.0,4.0)`, and worker counts `(2,4,2)` under Buck/PyTorch
`TORCH_CPU`.  Decision:
`INTERNAL_HPO_CONTENT_LIVE_EXTERNAL_HPO_TIE`.  Pooled internal minus external
validation-selected HPO was `-0.006 +/- 0.003`, internal minus external oracle
was `-0.007 +/- 0.003`, internal minus shuffled was `+0.004 +/- 0.002`, and
internal minus parallel-arm average was `+0.001 +/- 0.001`.  A supplemental
pilot-tuned-pair branch reduced the external gap to `-0.004 +/- 0.003` but did
not clear it.  This should now be cited as the serious fixed-compute
negative/partial result: the grouped-LR probe has real label content, but
validation-selected naive parallel HPO remains a stronger final-trajectory
baseline at the same worker budget.
