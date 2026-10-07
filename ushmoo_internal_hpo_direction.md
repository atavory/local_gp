# Which HPO Belongs Inside the Round — Direction Note

2026-10-06, after E080. Supersedes the scattered branch-by-branch reasoning.

## The organizing principle

Internal HPO means: workers start from a shared `theta_t`, run **different**
hyperparameters `h_i`, and a probe scores the resulting updates on a common
validation shard.  HPO-aware aggregation is the stronger object: the server
uses those probe scores to select or weight the candidate updates, so the
exploration becomes the update of the shared model instead of an external
tuning run.  Two structural advantages over ordinary HPO:

1. **Paired, same-base.** All arms share `theta_t`, so differences are
   attributable to `h`, not to trajectory divergence. Large variance reduction
   over comparing independent runs.
2. **Already-paid parallelism.** The `N` workers exist for distributed
   training; the question is whether some of that parallelism can be used as an
   experimental design without throwing away the non-winning arms.

The fixed-compute baseline is therefore crucial: separate parallel HPO spends
the same workers on separate trajectories and then keeps one trajectory.  Our
method tries to recover exploration value inside a single trajectory by
recombining the probed arms.

The probe cost amortises over `3KN` (T6), which is the only reason held-out
measurement is affordable at all.

**The selection rule that decides which hyperparameters fit:**

> Internal HPO works for hyperparameters whose **resource cost is consumed
> within the round**, and fails for those whose **benefit is realised across
> rounds**.

Because the probe sees one round, it can only price what one round pays for. A
hyperparameter whose payoff is cross-round will be systematically mis-selected
no matter how good the extractor is.

## Criteria

| | criterion |
|---|---|
| C1 | within-round observability — does one round at `h` produce a measurable validation difference? |
| C2 | effect size vs probe noise (`sigma/gap`) |
| C3 | **myopia safety** — does greedy one-round selection have a known pathology? |
| C4 | non-stationarity — does `h*` actually move during training? If not, offline tuning wins |
| C5 | transfer brittleness — does the tuned value break when K/model/batch/data change? |
| C6 | can it vary across workers without breaking the round? |
| C7 | cost of being wrong (cliff vs flat region) |

C4 and C5 together are the economics: the probe earns only where `h*` moves or
where retuning is forced.

## The candidates

| candidate | C1 obs | C3 myopia | C4 moves | C5 brittle | verdict |
|---|---|---|---|---|---|
| **inner LR multiplier** | strong | **risky** | yes | yes | **Tier 1, with a schedule constraint** |
| **batch size (value/FLOP)** | strong | **best of any** | yes | yes | **Tier 1, untested** |
| outer `(alpha, beta)` | strong | low | **no — schedules capture 96-99%** | yes | Tier 2: "avoid retuning" only |
| gradient clip threshold | yes | low | yes | yes | Tier 2: episodic effect, needs extreme-value statistics |
| **K / sync period** | partial | **fatal** | yes | yes | **Tier 3 — wrong mechanism** |
| weight decay | **no** | fatal | yes | yes | unsuitable |
| Adam `beta_2`, `eps` | no | bad | — | — | unsuitable |

### Inner LR — Tier 1, with one unresolved risk

Measured: E060/E061 spread sweep turns signal on (1 -> 0.000, 4 -> noisy,
16 -> clean); E080 beats shuffled/fixed/schedule at K<=8 with CIs excluding
zero. Observability and non-stationarity are both strong.

**The risk is C3.** Greedy one-round LR selection is the classic myopia trap:
large LR pays off over many rounds, small LR looks safe for one. Held-out
scoring may not carry the classical training-loss bias in the same direction,
but that is a hypothesis, not a measurement.

Mitigation: have the probe select a **multiplier on a known decay schedule**,
not a free LR. That bounds the damage and uses the one thing already proven
about this problem — schedules capture 96-99% of a smooth control surface.

**Free check, never run:** in E060/E061 the arms were {0.25, 1, 4}. Does the
selector skew toward 0.25? If yes, myopia is already in the positive result.

### Batch size — Tier 1, untested, best myopia profile

Not yet tried, and it may be the best fit of any candidate:

- **The objective is value per FLOP**, and both terms are consumed inside the
  round. A worker at batch `2B` does `2x` the work; divide its measured value
  by 2. Local, honest, no cross-round accounting needed.
- **Critical batch size is known to grow during training** (gradient noise
  scale). So `h*` genuinely moves, and nobody adapts it online.
- **Low myopia risk** precisely because the resource cost is in-round. The
  optimum is the noise-dominated/curvature-dominated transition, which is a
  local property.
- Workers can trivially run different batch sizes.
- Batch size is expensive to tune offline and trades directly against compute,
  so the deployment value is high.

### K — Tier 3, and this is a correction

K is the one everyone wants, and it is the **worst** internal-HPO candidate.

Its benefit is **communication saved**, which is not consumed inside a round
and therefore is invisible to a within-round probe. Comparing workers at
different K measures only the *cost* side (local drift, overfitting) and never
the *benefit* side. Greedy selection will systematically pick small K.

This does **not** mean probe-controlled K is dead. It means the mechanism is
different: K wants a **sequential test across rounds** — "is the marginal value
of another local step still positive?" — accumulating evidence with a CUSUM/SPRT,
not a cross-worker comparison inside one round. Those are two different
machines and should not be conflated.

Also note the scope tension: E080's positive is confined to K<=8, while
DiLoCo's value proposition is large H. The current result points **away** from
the communication axis.

## What is working

1. **Probe content is real.** E080: `+0.177 +/- 0.049` over shuffled,
   `+0.116 +/- 0.052` over fixed, `+0.164 +/- 0.057` over schedule. First time
   anything has beaten `fixed`.
2. **The exchangeability null behaves.** When arms are identical, there is no
   persistent HPO signal.  Realized probe weights can still fluctuate because
   update/probe noise is finite, so the right claim is not "weights are always
   uniform"; it is "any gain is stochastic-update filtering, not HPO, and must
   beat shuffled/held-out controls."
3. **Manufactured heterogeneity is a working mechanism with a dial** (spread
   1/4/16), which is what the drift axis never produced.
4. **The paired same-base comparison** is a genuine structural advantage over
   full-run HPO.
5. **The theory predicts.** It called the cross-K schedule transfer failure
   before the run, and the E077 -> E080 dilution effect before the re-score.
6. **Cost closes** at ~1%, and the total-compute-including-tuning accounting is
   roughly 6x.

## What is not working

1. **Capture is 5-12%** against oracles in every branch that reported it.
   Extraction is the common blocker and has never been attacked directly.
2. **K<=8 only.** Points away from the communication-efficiency story.
3. **Myopia has never been tested** — and it is the textbook failure mode of
   exactly this design.
4. **The marginal-value score is implemented but not scientifically run**:
   E086 packages the current-rule raw-group comparison, but the first worker
   routes were bad routes and did not produce real CUDA science rows.
5. **Policy-level selection keeps biting**: E061 (K=32 pilot policy missed),
   E076 (pilot-best chose schedule everywhere).
6. **One bed.** No real model, no language, no scale.

## What to do, in order

**Now, on existing data (hours):**

- **Skew check** on E060/E061: does the selector prefer the 0.25 arm? Decides
  whether myopia is already inside the E080 positive.
- **K<=8 oracle ceiling**: where does `+0.116` sit? Decides whether remaining
  headroom justifies more extraction work.
- **IID exchangeability/noise-selection audit**: measure realized probe-weight
  motion, selected-arm identity persistence, target/probe correlation, and
  selector minus uniform/shuffled when worker arms are truly identical.
- **Fixed-compute HPO baseline**: compare separate parallel HPO trajectories
  against probe-weighted recombination on one shared trajectory with the same
  worker budget.

**Next, one run each:**

- **Marginal-value score** — E086 now implements this as a raw-group comparison:
  one validation backward, then `-<g_V, u_i>` for all grouped candidates at
  matched cost.  It remains pending until run on the correct real-GPU lane.
- **Batch-size arm** — the untested HPO with the best myopia profile and an
  in-round resource denominator.

**Then:**

- E081 deployment-rule freeze, with the claim stated as **total compute
  including tuning**.

## Framing

Call this **HPO-aware aggregation**: internal HPO whose exploration is
reinvested into the same distributed update.  That keeps the aggregation
mechanism central while making the scientific claim an HPO win, not a generic
promise to out-average FedAvg on exchangeable workers.  The comparison class is
PBT, hypergradient descent, D-Adaptation, Prodigy, Schedule-Free, and parallel
HPO under a fixed worker budget.  Our differentiator is structural: a held-out
probe costs 300% on one machine and about 1% here because of `3KN`, and the
losing arms still contribute through the shared update instead of being thrown
away.
