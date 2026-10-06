# Internal HPO Option Map

Status: live map, 2026-10-06.

Purpose: make the internal-HPO option space explicit.  Do not let the project
collapse back into "one clever adaptive method."  The research object is:

> aligned distributed workers create a cheap in-round experimental design over
> optimizer controls; probe sets decide which controls are worth using, with
> shrinkage and fallback when evidence is weak.

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
| 1 | batch size / value per FLOP | cost and payoff are both in-round; untested but excellent fit |
| 1 | inner LR multiplier around a schedule | visible in-round, but myopia must be checked |
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
| A8 | selected-arm trajectory diagnostic | `GATED` | needed to know if grid is enough | run after E081 |
| A9 | grid-edge expansion | `GATED` | only justified if selected arms pin edges | trigger from A8, not by taste |
| A10 | per-round quadratic surface fit | `KILL` | extrapolation / schedule issues | do not revive without trust region |
| A11 | sliding schedule fit over rounds | `KILL` for stationary | converges to tuned schedule | only possible under proven drift |
| A12 | hysteresis over selected `(alpha,beta)` | `GATED` | reduces churn; natural deployment control | test after frozen rule passes |

### B. In-Round Resource Controls

These are the cleanest candidates for internal HPO because the denominator is
inside the round.  The probe can score **value per unit compute**, not just raw
loss drop.

| # | option | status | evidence | next gate |
|---:|---|---|---|---|
| B1 | batch-size arms, scored value/FLOP | `GATED` | untested; best myopia profile | run batch-size arm |
| B2 | microbatch / accumulation arms | `DEFER` | same family as batch size | after B1 |
| B3 | local compute budget per worker | `GATED` | close to batch/K boundary | only if value/FLOP is explicit |
| B4 | adaptive repeat/replicate for noise | `GATED` | E027 oracle liked replicate | needs costed noise regime |
| B5 | probe budget allocation itself | `GATED` | E079 says naive bigger batch not enough | only after score improves |

Batch size gets promoted because it has the right economics:

- a worker at `2B` spends about `2x` compute;
- the probe can score improvement per FLOP in the same round;
- critical batch size moves during training, so a static default can become
  stale;
- unlike K, the benefit is not hidden in future communication savings.

### C. Inner Worker Hyperparameters

These make workers deliberately non-exchangeable.  That is the right use of
parallelism: not "rank identical workers," but "run a small HPO design inside
the distributed round."

| # | option | status | evidence | next gate |
|---:|---|---|---|---|
| C1 | inner LR multiplier arms | `GATED` | E060/E061 positive but default-known issue | stale/default-shift test |
| C2 | default-anchored LR arms | `GATED` | E066-E068 say anchoring is required | combine with better score |
| C3 | deliberately bad/stale default recovery | `GATED` | the customer setting; not settled | run bad-default HP-arm test |
| C4 | log-spaced LR arm grid | `GATED` | plausible HPO design | only after C1/C3 pass |
| C5 | LR multiplier on a known schedule | `GATED` | mitigates one-round myopia | skew check first |
| C6 | inner momentum arms | `DEFER` | no headroom result yet | test only after LR/batch |
| C7 | AdamW beta1/beta2 arms | `DEFER` | high dimension, no evidence | wait for simpler axes |
| C8 | weight decay arms | `DEFER` | slow-timescale, hard to score per round | likely offline/slow posterior |
| C9 | gradient clipping arms | `GATED` for safety | may help bad-round/fault regimes | test only with injected bursts |
| C10 | optimizer-type arms, e.g. SGD vs AdamW | `DEFER` | too discrete/high-variance | only if phase-transition evidence |
| C11 | local LR schedule arms | `DEFER` | could be useful but many degrees | after selected-arm diagnostic |

Risk to test immediately: LR myopia.  In earlier spread experiments the arms
included small and large LR multipliers.  If the selector systematically
prefers the small/safe arm, the one-round probe is too myopic for free LR HPO
and the right action is schedule-constrained multipliers, not open LR search.

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
| E1 | exact collapse when evidence is uninformative | `LIVE` | spread-1 collapse | keep as invariant |
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
| F3 | validation-gradient marginal score `<g_V,u>` | `GATED` | prior simple version failed in E070 | only with current outer-control rule |
| F4 | response model over candidate features | `GATED` | may improve extraction | test after deployment rule |
| F5 | split select/eval probe shards | `GATED` | selection-bias audit | use in real workload version |
| F6 | rotating probe shards | `GATED` | avoids validation overfit | required for long runs |
| F7 | training-loss self-report | `DEFER` | biased but free | only as auxiliary feature |
| F8 | pooled round-quality score | `GATED` | higher SNR than per-worker ranking | combine with outer control |
| F9 | Bayesian shrinkage over probe scores | `LIVE` conceptually | `soft_std2` is empirical shrinkage | formalize after E081 |
| F10 | GP / UCB / Thompson over controls | `DEFER` | no need until frozen selection hits edge | trigger: A8 grid-edge diagnostic |

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
3. Do not run another same-family selector sweep unless the selected-arm
   trajectory diagnostic shows edge pinning.

Then:

4. Run LR-arm skew check on existing E060/E061 data.  If the selector prefers
   the small/safe LR arm, free LR search is myopic and should be schedule-bound.
5. Run the batch-size arm.  This is the best untested fit to the in-round HPO
   rule because the score is value per FLOP.
6. Run selected-arm trajectory diagnostic.  This decides whether a moving grid,
   GP, or Thompson sampler is needed for outer controls.
7. Run the bad/stale-default HP-arm test.  This decides whether inner LR arms
   are useful in the intended customer setting.
8. Only then consider GP/UCB/Thompson.  Trigger condition: frozen grid either
   pins to edges or fails under a regime shift while probe labels still beat
   shuffled labels.

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
| 2 | LR skew check on E060/E061 | C1/C5 | no pathological skew toward small/safe LR |
| 3 | batch-size arm | B1 | value/FLOP beats shuffled and default |
| 4 | selected-arm trajectory | A8/A9/F10 | edge pinning or phase movement justifies moving arms |
| 5 | bad/stale default HP-arm test | C2/C3 | probes recover from stale default without losing collapse |
| 6 | marginal-value score in current rule | F3 | improves shuffled contrast or capture at matched cost |
| 7 | sequential K test prototype | D1/D2 | marginal local-step value predicts when to sync |
