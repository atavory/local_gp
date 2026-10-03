# Audit: bandha's local_gp seed (README / model.md / control board / E001)

Scripts backing every number here, all pure-stdlib, seconds on CPU:
`ushmoo_design_snr_check.py`, `ushmoo_field_repair_check.py`,
`ushmoo_why_precond_fails.py`. Bed: diagonal quadratic, d=200, 12 sharp
(lam~U[8,12]) + 188 flat (lam~U[0.1,1]), N=8, K=8, eta=0.05, sigma=1, 60 reps.

## Verdict

**The scaffolding is right and the payload is wrong.** The gate-before-closed-loop
discipline, the stop rules, the shuffled control, the source-fit/target-shift
split, and the refusal to claim optimizer value are all correct and are the
single biggest improvement over Double Fan. Keep all of it.

But E001 as specified **cannot test the local-GP hypothesis**, because its action
space excludes the entire effect and its feature space excludes the entire
sufficient statistic. Two of my own pre-run objections were refuted by the
measurement — recorded below as refuted.

---

## 1. The action space caps at 74% of the ceiling, by construction

E001 scores `action = scale * mean_local_proposal`, `scale in {0,...,1.5}`. That
is a pure gain rule: it rescales the mean proposal, it cannot rotate it.

I expected this to be near-worthless. **It is not** — measured:

| arm | gain | / ceiling |
|---|---|---|
| best-scaled mean proposal (E001's whole space) | 83.09 | **0.739** |
| diagonal oracle | 112.40 | 1.000 |

So scale-only already reaches 74% of what any diagonal action can achieve. My
"pure gain, therefore dead" objection is **refuted**; a scale rule is a strong
baseline, not a toy.

The correct statement is sharper and worse for E001: **the ceiling of E001's
action space is 0.739, and the entire contribution this project could make lives
in the 0.261 it excludes.** A perfect predictor of the optimal scale — the best
possible E001 outcome — still leaves every bit of the project's thesis untested.
Headroom inside the grid is bounded by LR tuning; headroom outside it is the
whole paper.

**Amendment:** the action grid must contain at least one shape action. Minimum
viable: the two-coefficient `alpha*p_dom + gamma*p_bulk` split from
arXiv:2605.27739 (which also makes the published method a named arm rather than
an uncited neighbour), plus one per-coordinate rescale.

## 2. Every logged feature is a dispersion scalar

The worker prompt logs: round index, K, mean proposal norm, endpoint spread,
p10/p50/p90 proposal norms, cosine-to-mean statistics, loss-delta mean/spread.

All ~10 are **scalar summaries of the endpoint cloud**. None is a
`(point, gradient)` pair. This reconstructs the K=1 regime inside K>1 data: at
K=1 the sample mean and sample covariance are independent (Cochran/Basu), the
spread carries error *scale* and never error *direction*
(`corr(R_j, signed error) = -0.0074` vs `corr(R_j, |error|) = +0.5494`), and
every lane built on dispersion summaries died of it. Summarising the K>1 cloud
down to its dispersion statistics re-imposes exactly that constraint.

Section 4 shows the quantity that actually matters is **per-coordinate**. Ten
scalars cannot carry it for d parameters at any sample size.

**Amendment:** log `(x_ik - x_t, g_ik)` per worker per local step. For an offline
gate the storage cost is irrelevant and the compression question is premature —
see item 6.

## 3. Endpoint-only logging reproduces the published method's limitation

"log per-worker local endpoint proposal summaries" keeps `N` displacements and
discards the `K-1` intermediate points. That is precisely the gap I identified in
arXiv:2605.27739 — endpoint gaps only, design size `N` not `N*K`. As specified,
**E001 is a weaker version of a published method**: same observations, fewer
features, no aggregation change. If it passes we will have reproduced
2605.27739 and will not be able to claim otherwise.

## 4. The governing result: K local steps already apply the correction

This is what the audit turned up and it is larger than the audit.

On a quadratic the noiseless mean displacement is a geometric sum,
`a_j = 1 - eta*lam_j`:

```
dbar_j  =  -g*_j * (1 - a_j^K) / lam_j
```

Verified: observed/predicted = **1.0218** (sharp), **0.9346** (flat).

Read the limits. The implied effective per-coordinate step is `(1-a^K)/lam`:

| direction | implied step | Newton `1/lam` | gradient `eta*K` |
|---|---|---|---|
| sharp, lam=10 | **0.0996** | 0.1000 | 0.4000 |
| flat, lam=0.5 | **0.3667** | 2.0000 | 0.4000 |

**Local SGD interpolates between a gradient step and a Newton step on its own,
with the crossover at `eta*lam ~ 1/K`.** In sharp directions `dbar` is *already
at* the Newton step. Dividing it by `lamhat` double-applies a correction that the
local steps performed. That predicts both observed failures, and both are
measured:

| arm | / baseline |
|---|---|
| rotate by raw `1/lamhat` | 0.094 |
| rotate by **TRUE** `1/lam` | **0.945** |
| best oracle-tuned Tikhonov `lamhat/(lamhat^2+delta^2)` | 1.007 |

Preconditioning by the *exact* inverse Hessian **loses to doing nothing**. That
is not an estimation failure; it is redundancy.

The correction the law says is correct is multiplication by `1/(1-a_j^K)`:

| arm | gain | / ceiling | / baseline |
|---|---|---|---|
| best-scaled `dbar` (baseline) | 83.09 | 0.739 | 1.000 |
| `x 1/(1-a^K)`, TRUE lam | 108.85 | 0.968 | **1.310** |
| `x 1/(1-a^K)`, TRUE lam, **flat dirs only** | 108.84 | 0.968 | **1.310** |
| `x 1/(1-a^K)`, ESTIMATED lam | 7.53 | 0.067 | 0.091 |

Three things at once:

1. **The idea is real.** 1.310x over a tuned-scale baseline, 96.8% of ceiling.
2. **The gain is entirely in the flat directions.** Flat-only is identical to
   all-directions to three figures. Sharp directions contribute *nothing* — they
   are already converged to Newton.
3. **The design cannot estimate flat curvature.** Relative error
   `|lamhat-lam|/lam` is **0.069** in sharp directions and **3.080** in flat
   ones, with **41/200** coordinates returning `lamhat <= 0` (uninvertible).

> **The lane's governing constraint: all of the available gain sits in the
> directions the trajectory design estimates worst, and the directions it
> estimates well have no gain left to give.**

This is the K>1 analogue of the sufficiency theorem that closed Double Fan — but
it is a *better* position, because it names a specific estimator deficiency
rather than an information-theoretic impossibility. The information is present
(flat directions carry **5x** the endpoint-gap variance of sharp ones); the
slope regression is the wrong estimator for it, because when `lam` is small the
gradient noise swamps `lam * spread`.

**Two consequences worth pre-registering, both contrarian:**

- **Estimate the contraction `a_j` from the temporal structure of the K-step
  path, not `lam_j` from a slope regression.** The slope fit discards the step
  ordering. A flat direction is a near-random-walk and a sharp one is a
  fast-mixing AR(1); that difference is in the autocorrelation, which the slope
  estimator throws away.
- **A frozen, pooled-across-rounds curvature estimate should beat the live
  per-round one here**, because the binding constraint is estimator variance and
  `lam` is near-static. This is the opposite of what the name "local GP"
  suggests, and it is exactly guardrail 3's frozen arm. If it holds, the product
  is not a GP refit each round.

**Scope.** Diagonal quadratic, homogeneous workers, one spectrum, one `(eta,N,K)`.
The *numbers* are bed-specific. What is generic is the algebra: the displacement
law is exact on any quadratic, the crossover at `eta*lam ~ 1/K` follows from it,
and the resulting split — gain where `eta*lam << 1/K`, estimability where
`eta*lam >~ 1/K` — is structural. This bed has **no worker heterogeneity**, so it
is silent on 2605.27739's claimed sharp-direction mechanism, which wanger's §2a
already argued must live in non-quadraticity or heterogeneity.

## 5. Three smaller defects in the spec

**(a) The pre-registered primary target is norm-dominated.** Target 1 is "signed
projection of mean proposal onto the negative audit gradient" — a scale-carrying
quantity. The path-norm baseline already has the norm, so it will score well for
a reason that has nothing to do with the hypothesis, and E001 will return
`PATH_NORM_EXPLAINS` trivially. **Pre-register target 2 (cosine), which is
scale-invariant.** Same argument as guardrail 5: MSE and projections reward the
component a tuned step size removes.

**(b) The headroom reference is wrong.** `headroom = loss(frozen source baseline)
- loss(target oracle)` with a scale-only oracle measures "per-round step-size
tuning beats a fixed step size". That is true, known, and not ours. The reference
must be **the best single constant scale tuned on the target** — the
tuned-constant control that killed E73, E96, E100, E102 and E106. As written,
E001 would report a headroom that is simultaneously mis-referenced (includes LR
tuning) and too small (excludes shape, per item 1).

**(c) The baseline list has no target-tuned constant.** "Scalar source-selected
baseline" is a *transfer* baseline. Both are needed, and they answer different
questions: source-tuned tests whether the model transfers, target-tuned tests
whether the measurement beats a constant at all. If a target-tuned constant beats
the source-transferred GP, the method is only useful when target tuning is
unaffordable — a legitimate claim, but a much weaker one that must be stated
rather than discovered by a reviewer.

## 6. Compress after you find the signal, not before

`model.md` is drawn to the shared-probe route because it "avoids high-dimensional
parameter communication". That is a cost optimisation applied before the signal
is established, and it has already been paid for once: E90 restricted the
variance channel to tensor granularity and found nothing; E93 ran it dense and
found recall **0.98 vs 0.134**. The signal was in the within-tensor tails that
the summary destroyed.

Item 4 makes this concrete rather than cautionary: the needed quantity is
per-coordinate `1/(1-a_j^K)`. Any low-dimensional function-space probe summary
destroys it **by construction**, and a null result would be uninterpretable —
absent signal and destroyed signal look identical.

Communication cost is real and wanger's §4 local-linear parity scheme is the
right answer to it, but it is the *second* experiment. Establish the information
exists in the full data; compress second.

## 7. "Local GP" currently names the wrong object

`model.md` defines the latent as `F_t(z)` = quality of a proposal with descriptor
`z`, and `z` is a vector of dispersion summaries. That is a regression of a
scalar quality score on cloud statistics — a tabular model over *rounds*.

The name promises a model of the update field over *parameter space*, `F(x)`,
with the `N*K` visited points as its design. Those are different objects, and
item 4 says only the second one has the 0.261 of headroom in it. Conflating them
is the mechanism by which this project drifts back into controller-land, which
the Prime Directive explicitly forbids.

---

## Amendments to E001, in priority order

1. **Log `(x_ik - x_t, g_ik)` per worker per local step.** Without this the gate
   cannot distinguish the hypothesis from the published baseline.
2. **Add a shape action** to the grid (per-coordinate rescale, and
   `alpha*p_dom + gamma*p_bulk` as a named arm).
3. **Report headroom against a target-tuned constant scale**, not against the
   frozen source baseline.
4. **Pre-register the cosine target**, not the projection.
5. **Add the target-tuned-constant baseline** alongside the source-selected one.
6. **Add a frozen/pooled-curvature arm** — per item 4 it may win, and it is the
   cheapest arm in the set.
7. **Report per-direction relative curvature error and the count of
   non-invertible coordinates** in every result table. On this bed they were
   0.069 / 3.080 / 41-of-200, and they explain the outcome entirely.
8. **Keep the shared-probe compression out of E001.** It is E002.

## Correction to my own file

`ushmoo_problem_statement.md` leg 2 claimed "the rank arithmetic clears the
coverage cliff: N=8,K=8 gives covered rank ~57-64, above the 10-15 outliers."
wanger's §2c is right and I have now reproduced it independently: top-12
gap-variance overlap with the sharp set is **0.000**, and flat directions carry
**5x** the gap variance. Nominal rank is not aligned coverage. I am amending that
file rather than leaving it to be read as settled.

The leg does not survive as written. What survives is narrower and is item 4:
relative curvature error is 0.069 in the sharp directions, so the design is
*well-conditioned for sharp curvature magnitude* even though it is *misaligned
for sharp subspace identification*. wanger measured which directions the design
points at; I measured how well it estimates each one. Both are true, they are
different questions, and the second one is the one a step uses — which is how the
0.069/3.080 split, and therefore item 4, came out at all.
