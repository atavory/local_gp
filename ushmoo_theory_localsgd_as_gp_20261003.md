# Local SGD is Gaussian-process inference with a misspecified prior

Theory note. Every number is produced by a script in this repo, pure stdlib,
seconds on CPU. Where a prediction of mine failed, the failure is reported in
place rather than dropped — three of them did.

---

## 0. The result, first

On a diagonal quadratic (d=120, 10 sharp + 110 flat, K=8, eta=0.05), scored as
mean one-step gain on a real FedAvg trajectory, all ratios to a **tuned-scale**
FedAvg baseline at the same N:

| N | within-round, cross-fitted | across-round secant (free, classical) | true curvature (headroom) | Newton (ceiling) |
|---|---|---|---|---|
| 8 | **1.119** | 1.007 | 1.187 | 1.360 |
| 32 | **1.130** | 0.988 | 1.205 | 1.263 |
| 128 | **1.159** | 1.003 | 1.207 | 1.227 |

Three things at once:

1. **There is headroom over a tuned scalar: ~20%.** Not a step-size effect —
   the baseline's scale is optimised.
2. **The intermediate local points capture ~94% of it** (1.159 / 1.207).
3. **The free classical alternative captures none of it.** Server-side secants
   `gbar_{t+1} - gbar_t = H(x_{t+1} - x_t)` — what L-BFGS on the server would
   use, requiring no extra logging — sit at 1.00. The intermediate points are
   doing work nothing else does.

**Why it is not a same-sample artifact.** The curvature fit shares noise with
the `Delta-bar` it corrects, which can manufacture gain. Cross-fitting across
worker halves (fit on half A, apply to half B's displacement, and vice versa) is
noise-independent and **preserves the full effect** (1.119 vs 1.122 same-round
at N=8).

> ### But do not read row 1 as a working method yet
>
> §6 establishes that the 1.12-1.16 is **not** driven by the estimator's
> accuracy. A strictly causal estimator with *more* data and *provably lower*
> error (|rel err| 0.647 vs 1.011 in the flat directions) scores **1.00**. What
> the winning estimator has is a large upward **bias** in `lamhat` — exactly in
> the flat, high-`C` directions — which shrinks the correction toward 1 where
> the correction is least trustworthy. It is an accidental regulariser that
> happens to be well-shaped.
>
> **The control that decides this has not been run**: a deliberate shrinkage
> `c_j = 1 + gamma(C_j - 1)` with `gamma` a tuned constant. On this programme's
> record a measured controller has lost to its own tuned constant five times
> (E73, E96, E100, E102, E106). Until that arm exists, row 1 is a measured
> effect with an identified mechanism, not a method.
>
> What is *not* contingent on that control: the ~20% headroom over a tuned
> scalar (`oracleC`, true curvature), and the fact that the free classical
> alternative captures none of it.

---

## 1. The dictionary

Put a GP prior on the loss, `f ~ GP(0, k)`. The gradient of a GP is a GP, with
matrix kernel `d^2 k / dx dx'` — standard derivative-observation machinery. Local
SGD's worker trajectories are then **derivative observations of a latent GP**, and
the server's aggregation rule is a **posterior read-out**.

**Proposition 1 (FedAvg is a degenerate posterior).** Let the prior have
infinite lengthscale, so `grad f` is almost surely constant. Given gradient
observations `{g_ik}` with iid noise, the posterior mean of `grad f` is the
unweighted mean `gbar`, and the Bayes step under a fixed quadratic trust term is
proportional to `-gbar ∝ Delta-bar`.

> **Averaging the worker updates is exactly Bayes-optimal under the assumption
> that the gradient does not vary over the region the workers explored.**

That is the whole content of the averaging rule. Every proposal to improve on
FedAvg aggregation is, in this language, the claim that the lengthscale is
finite — and the lengthscale is estimable from the data the workers already
produced. This reframes a large empirical literature (filtered updates, weighted
averaging, "averaging is probably not the optimum") as **prior misspecification**
rather than as a collection of heuristics, and it makes the averaging baseline a
special case rather than a rival.

---

## 2. The displacement law

For `f` quadratic with diagonal curvature, `a_j = 1 - eta*lam_j`, the noiseless
mean displacement after K local steps is a geometric sum:

```
Delta-bar_j  =  -g*_j (1 - a_j^K) / lam_j
```

Verified: observed/predicted **1.0218** (sharp), **0.9346** (flat)
(`ushmoo_why_precond_fails.py`). Define the **correction owed**:

```
C_j  :=  1 / (1 - (1 - eta lam_j)^K)
```

so that `Delta-bar_j * C_j` is exactly the Newton step.

**Proposition 2.** `C_j -> 1` as `eta lam_j K >> 1`, and `C_j -> 1/(eta lam_j K)`
as `eta lam_j K << 1`.

> **K local steps already interpolate between a gradient step and a Newton step,
> with the crossover at `eta lam_j K ~ 1`.**

Measured implied per-coordinate step: sharp (lam=10) **0.0996** against Newton
`1/lam = 0.1000` and gradient `eta*K = 0.4`; flat (lam=0.5) **0.3667** against
Newton `2.0`. Sharp directions arrive at the Newton step unaided.

This has an immediate corollary that cost two experiments to learn empirically:
**preconditioning `Delta-bar` by the inverse Hessian double-applies a correction
the local steps already performed.** With the *true* `1/lam` it scores **0.945x**
the tuned-scale baseline — worse than doing nothing, with perfect knowledge.

---

## 3. The governing dimensionless group

Everything is controlled by

```
rho_j  :=  eta * lam_j * K
```

**Benefit.** Applying `C_j` recovers the fraction `b_j = (1 - 1/C_j)^2` of
direction `j`'s remaining loss. Zero at `rho >> 1`, one at `rho << 1`.

**Cost.** `C_j` must be estimated. Two branches for the design spread:

- *drift-dominated* (`|g_j| >> sigma sqrt(12/K)`):
  `eps_j = sd(lamhat_j)/lam_j ~ sqrt(12) sigma / (sqrt(N) |g_j| eta lam_j K^{3/2})`
- *worker-noise-dominated* (the flat directions): `eps_j ~ C_j / sqrt(c' N)`

**In both branches `eps_j` is proportional to `C_j`.** The relative error of the
estimate and the size of the correction owed scale identically.

> **Duality: the directions with gain to give are exactly the directions you
> cannot measure.**

Measured (`ushmoo_theory_predictions_check.py`, N=32):

| bucket | mean lam | eps | C_true | eps/C |
|---|---|---|---|---|
| sharp | 10.253 | 0.032 | 1.004 | 0.032 |
| flat, low | 0.266 | 2.473 | 11.678 | 0.212 |
| flat, mid | 0.567 | 0.927 | 4.965 | 0.187 |
| flat, high | 0.856 | 0.564 | 3.421 | 0.165 |

`eps/C` is constant to within 30% across the three flat buckets — the
worker-noise branch, as derived. The sharp row sits 6x lower because it is in
the *other* branch. **Prediction P1 holds within a branch, not across; I stated
it across and was wrong.**

**The sweet spot.** Precision `1/eps_j ∝ 1/C_j` falls while benefit
`(1-1/C_j)^2` rises, so their product `(1-1/C)^2 / C` has an interior maximum at
`C = 3`, i.e. `rho ≈ 0.405`. Headroom measured against spectrum centre:

| rho | 0.040 | 0.100 | **0.240** | 0.600 | 1.600 |
|---|---|---|---|---|---|
| headroom (oracle/base) | 0.890 | 1.133 | **1.154** | 1.097 | 1.031 |

Interior peak confirmed, at `rho ≈ 0.24` against a predicted 0.405 — right
ballpark on a broad curve. **Flatter is not monotonically better**, which is the
opposite of the naive reading and the opposite of what I expected.

Note the `rho = 0.04` entry is **below 1**: the true-curvature correction losing
to plain rescaling. That is the second obstruction.

---

## 4. Two obstructions, and which binds

**(a) Estimation noise in `lamhat`.** Covered above, `eps_j ∝ C_j`.

**(b) Amplification of `Delta-bar`'s own noise.** `Delta-bar_j` has standard error
`nu_j = sd_i(dev_ij)/sqrt(N)`. Multiplying by `C_j` scales signal and noise
together, so per-coordinate SNR is unchanged — but it **reweights the step toward
the lowest-SNR coordinates**, because a flat direction's displacement is small
precisely by the factor `1/C_j`. The correction systematically up-weights the
noisiest components.

The Bayes-optimal coefficient is therefore the correction times a Wiener
shrinkage:

```
c_j  =  [ 1 / (1 - (1-eta lam_j)^K) ]  x  [ SNR_j^2 / (1 + SNR_j^2) ]
        \______ debias local steps ______/  \____ shrink Delta-bar noise ____/
```

with `SNR_j^2 = max(0, Delta-bar_j^2 / nu_j^2 - 1)` estimable **from the worker
spread alone**. Note the resulting shape, `C_j/(1 + C_j^2/A^2)`, is the Tikhonov
form that `ushmoo_field_repair_check.py` had to tune on the test set. The
posterior supplies it for free, with the knob set by measured noise.

**Which binds?** Measured on a real FedAvg trajectory
(`ushmoo_bayes_step_check.py`, N=16):

| arm | /base |
|---|---|
| true `C`, no shrinkage | 1.249 |
| true `C` x Wiener shrink | 1.235 |
| plug-in `C(lamhat)`, no regularisation | **0.144** |

Obstruction (b) costs 1.4 points — **not binding**. Obstruction (a) is
catastrophic when unregularised. The lane is an estimation problem.

### The worker cloud finally has a job

The previous programme spent 112 experiments looking for a use for the
unreduced worker cloud and found only that it carries error *scale*, never error
*direction* (`corr(R_j, signed error) = -0.0074`; `corr(R_j, |error|) = +0.5494`).

Scale is **exactly** what the shrinkage factor in (b) needs. The cloud does not
tell you where to step. It tells you how far to trust the curvature correction.
That is a real role for the quantity, and it is the first one the programme has
found — though note it is the *minor* of the two obstructions, so this is a
footnote, not a headline.

---

## 5. The posterior read-out that works

Three modelling choices, each of which is standard GP practice and each of
which turned out to be load-bearing.

**(i) Positivity.** A Gaussian prior on `lam` lets the posterior mean reach zero,
and `C = 1/(1-(1-eta lam)^K)` is unbounded there. One blown-up coordinate out of
120 captures the whole step. Parameterising in `log lam` — what every GP package
does for lengthscales — is the fix, and it moved the arm from **0.144 to 0.972**.

**(ii) Read out the step, not the correction.** `E[C]` is infinite under any
posterior with mass near `lam = 0`. But the Bayes step is
`delta_j = -E[g_j]/E[lam_j]`, and the displacement law gives
`g*_j = -Delta-bar_j lam_j C_j` exactly, so

```
delta_j  =  Delta-bar_j  *  E[lam_j C_j] / E[lam_j]
```

**`lam*C(lam)` is bounded and smooth** — it tends to `1/(eta K)` as `lam -> 0` and
to `lam` as `lam` grows — while `C` alone is not. The ratio of posterior
expectations is finite where the plug-in is not. No clipping, no floor, no
Tikhonov knob: the regularisation comes from the prior, which is what a GP is
for.

**(iii) Cross-fitting.** Fit on half the workers, apply to the other half's
displacement, symmetrise. Costs nothing (the halves are already independent) and
removes any same-sample coupling between `lamhat` and `Delta-bar` from the
inference. Here it changed nothing — which is the point: it is what licenses
reading the 1.12-1.16 as curvature rather than correlation.

---

## 6. The causal anomaly: the gain is not coming from accuracy

| arm | N=8 | N=32 | N=128 |
|---|---|---|---|
| cross-fitted, uses round t, N/2 workers | 1.119 | 1.130 | 1.159 |
| strictly causal, rounds `< t`, all N workers | 1.010 | 0.999 | 1.028 |
| causal with prior variance forced small | 1.022 | 1.015 | 1.031 |

`lam` is **exactly constant across rounds** in this simulation, so the causal
estimator has strictly more data about the same target. It still loses. Three
explanations tested:

- *Same-sample correlation* — **refuted**, cross-fitting preserves the gain.
- *Miscalibrated shrinkage* (my conjecture: more data weakens shrinkage, and
  weakening it in the high-`C` directions is exactly wrong) — **refuted**, row
  3. Forcing `s^2 = 0.25` moves causal by 0.01.
- *The gain is not from `lamhat`'s accuracy* — **this one**.

Direct measurement (`ushmoo_causal_anomaly_diag.py`, N=32, rounds>=5):

| bucket | lam | C_true | \|rel err\| causal | \|rel err\| split | bias causal | bias split |
|---|---|---|---|---|---|---|
| sharp | 10.138 | 1.004 | 0.016 | 0.020 | +0.004 | +0.008 |
| flat, low | 0.312 | 10.238 | **0.647** | **1.011** | **+0.366** | **+0.767** |
| flat, mid | 0.635 | 4.435 | 0.247 | 0.402 | +0.135 | +0.296 |
| flat, high | 0.882 | 3.316 | 0.172 | 0.276 | +0.099 | +0.218 |

**The losing estimator is the more accurate one, in every bucket.** What
separates them is bias: `lamhat` is biased *upward*, the bias lives in the flat
directions, and the cross-fitted estimator has roughly **twice** the bias of the
causal one — consistent with an `O(1/N)` bias and half the workers.

The mechanism is then immediate. `lamhat` too large means `C(lamhat)` too small,
so the correction is pulled toward 1. The bias scales with `C` (+0.767 at
C=10.2, +0.218 at C=3.3), so it shrinks the correction hardest exactly where
the correction is least trustworthy. **It is an accidental regulariser with
almost the right shape.**

Source of the bias: within a round, `s_{i,k}` for `k>=1` contains the noise from
steps `k' < k`, and the round-centred `gbar` contains that same noise, so the
centred regressor and centred response share terms. The correlation is `O(1/N)`
and does not vanish by pooling rounds — it is a within-round bias averaged over
rounds.

### What this forces

Replacing an accident with a deliberate, well-shaped shrinkage is the obvious
move, and it immediately introduces a knob. So the decisive arm is:

```
c_j = 1 + gamma * (C_j - 1),     gamma a TUNED CONSTANT
```

This still varies per coordinate (because `C_j` does), so it is a shape change
and a learning rate cannot absorb it — it is a legitimate rival, not a trivial
baseline. If tuned-constant `gamma` on a crude curvature proxy matches the
measured estimator, the trajectory fit is decoration. Five arms in the
predecessor project died exactly here.

**Status: the effect is measured, its mechanism is identified, and the control
that adjudicates it is specified and unrun.**

**I also retract prediction P2.** I derived a design rule `N >~ 2.5 C_j^2` and
predicted the corrected arm would cross the baseline near N=75. It does not: the
realised arm is essentially **flat in N from 8 to 128**. The rule was derived for
a *typical* coordinate, but the step is dominated by the *worst* coordinate, and
`min_j lam_j` sets the requirement. Real Hessians have near-zero and negative
eigenvalues, so no N makes an unregularised correction safe. Regularisation is
not an optimisation, it is the enabling condition.

---

## 7. What GP theory actually imports

Beyond the dictionary, five concrete things.

**(1) Marginal likelihood replaces test-set tuning.** Our shrinkage knob was
swept on the test set and found at `delta = 2.0`. Type-II ML sets it from the
evidence, on the training fold. This is guardrail-compliant by construction and
removes a knob the tuned-constant control would otherwise eat.

**(2) ARD = per-direction trust.** The flat/sharp estimability split *is* the ARD
lengthscale posterior. The machinery for "which coordinates do I have resolution
in" already exists and is exactly what §3 needs.

**(3) Posterior variance is a derived trust region.** Minimising
`m(x) + beta*s(x)` instead of `m(x)` prevents the extrapolation blow-ups by
construction, rather than by a clipped `1/lamhat`. Relevant because this
programme previously measured that extrapolating 2x past the cloud mean costs
12x variance.

**(4) The step is an interpolation, not an extrapolation.** The quantity we need
— the field at the mean endpoint — lies **inside the convex hull of the design**.
GP error bounds are strong there (fill-distance rates) and weak outside. This is
a genuinely favourable structural fact about the local-SGD design and it is
worth stating in any theorem.

**(5) Distributed GPs are the communication-parity theory.** wanger's local-linear
scheme — each worker fits `(a_i, diag(B_i))` on its own `K+1` pairs and sends
`2d` floats, server combines — is a **product-of-experts / Bayesian Committee
Machine**. That literature has the consistency analysis and the known failure
mode (plain BCM is inconsistent; robust/generalised PoE corrects it). The FL
communication-efficiency problem and the distributed-GP problem are the same
problem, and nobody appears to have connected them.

**Heterogeneity, for later.** Clients with different data are a **multi-task GP**:
`f_i = f + delta_i`. Under that model SCAFFOLD's control variate is the posterior
mean of `grad delta_i` with an *infinite lengthscale on `delta_i`* (held constant
between rounds) — the same misspecification as Proposition 1, one level up. The
fix is the same: give `delta_i` a finite lengthscale. This bed has homogeneous
workers and says nothing about it.

---

## 8. The convergence-theory import

Standard local-SGD bounds decompose as optimisation + `variance/sqrt(NKT)` +
**client drift**, with drift `O(eta^2 K^2 G^2)`, derived from

```
|| grad f(x_ik) - grad f(x_t) ||  <=  L || x_ik - x_t ||
```

Under the GP model, replace the Lipschitz step with the posterior field error:

```
|| grad f(x_ik) - m(x_ik) ||  <=  beta * s(x_ik)
```

where `s` is the posterior sd at a point **inside the design hull**, controlled by
the fill distance of the trajectory design in each direction. The drift term is
then multiplied by the unexplained field fraction rather than bounded by `L`.

**The non-obvious consequence, now with an exponent.** Standard theory says
larger `K` is monotonically worse. In the drift-dominated branch
`eps_j ∝ K^{-3/2}` — longer trajectories give both more samples *and* a wider
lever arm for the regression, the classical "growing design range" effect. The
two terms compete and `K*` shifts **upward**. The competing term is measured: the
probe residual degrades 0.587 -> 0.762 from K=2 to K=8.

Caveat: `rho_j = eta lam_j K` also grows with `K`, which *reduces* the benefit
`(1-1/C_j)^2`. So the `K` story is the §3 sweet-spot curve, not a monotone win,
and the theorem must be stated in `rho` not in `K`.

---

## 9. Prior art, and what is actually new

**Cited from memory; verify before any of this is written up.** The first item
is the one that most threatens novelty and should be read first.

- **Hennig & Kiefel, "Quasi-Newton Methods: A New Direction", JMLR 2013.** Shows
  BFGS/DFP are posterior means of Gaussian regression on secant pairs. This is
  "GP posterior over the Hessian from (point, gradient) pairs" and it is 2013.
  **The mechanism is not new.**
- Hennig, "Probabilistic interpretation of linear solvers", SIOPT 2015.
- Mahsereci & Hennig, "Probabilistic Line Searches for Stochastic Optimization",
  NeurIPS 2015 / JMLR 2017 — GP on function + gradient along a line to choose the
  step. Closest in spirit; one line, one worker.
- Hennig, Osborne & Kersting, *Probabilistic Numerics*, CUP 2022.
- Solak et al., "Derivative observations in GP models of dynamic systems",
  NIPS 2002; Rasmussen & Williams §9.4.
- Wu et al., "Bayesian Optimization with Gradients", NeurIPS 2017.
- Eriksson et al., "Scaling GPs with Derivative Information", NeurIPS 2018 —
  the `O(n^3 d^3)` problem, which our per-coordinate factorisation sidesteps.
- Tresp, BCM 2000; Deisenroth & Ng, "Distributed Gaussian Processes", ICML 2015.
- Bonilla et al., multi-task GP, NeurIPS 2007.
- Local SGD: Stich 1805.09767; Khaled et al. 1909.04746; Woodworth et al.
  2002.07839; SCAFFOLD 1910.06378; DANE 1312.7853; Field Guide 2107.06917.
- arXiv:2605.27739, worker disagreement / filtered updates.

**So what is new.** Not the probabilistic-numerics machinery. Three things:

1. **The design.** Quasi-Newton accumulates secants *sequentially along one
   path*; local SGD generates them **in parallel from a common point**, `N`
   independent trajectories per round. That is a different and better-conditioned
   design, it is already paid for, and treating the FedAvg worker fan-out as a
   designed experiment for probabilistic numerics appears to be unoccupied.
2. **The duality (§3).** `eps_j ∝ C_j` in both branches, with the interior
   optimum at `rho ≈ 0.4`. This explains why the obvious thing fails and predicts
   where it can work. I have not seen it stated.
3. **The negative control (§0).** That server-side secants — the classical,
   free alternative — capture **none** of the available headroom while the
   intermediate points capture 94%. Without this the result would be
   "reinvented L-BFGS."

---

## 10. Open items, in priority order

1. **Run the tuned-constant `gamma` control (§6).** Nothing else should be built
   until it exists. Three arms, same bed: measured `lamhat`; tuned-constant
   `gamma` on `C(lamhat)`; tuned-constant `gamma` on a crude curvature proxy
   (e.g. a single pooled `lam` for all coordinates). If the third matches the
   first, the trajectory fit is decoration.
2. **Debias `lamhat`.** The `O(1/N)` within-round bias is analytically
   identifiable — the shared-noise terms between the centred regressor and
   centred response can be subtracted. A debiased estimator would separate
   "accuracy helps" from "bias helps" cleanly, and it is the right experiment
   whichever way the control in item 1 goes.
3. **Verify Hennig & Kiefel 2013** and check whether any probabilistic-numerics
   work already treats a parallel common-origin design.
3. **Break the quadratic.** The displacement law is exact algebra on a quadratic
   and everything here inherits that. Non-quadraticity is also where
   arXiv:2605.27739's sharp-direction mechanism must live (wanger's §2a), and
   this bed cannot see it.
4. **Heterogeneous clients** (multi-task GP, §7). This bed has none.
5. **Beyond diagonal.** Per-coordinate factorisation is what makes this tractable
   at `d` in the millions; E93 supports diagonal-dense over per-tensor, but the
   block case is untested here.
6. **Communication parity.** Local-linear / BCM head-to-head at matched bytes
   against the endpoint-filtering rule, per wanger §4.

## Errata in this note's own history

- **P1** ("`eps/C` constant across the spectrum") — holds within a branch only.
- **P2** (design rule `N >~ 2.5 C^2`) — **refuted**; the arm is flat in N.
- **"miscalibrated shrinkage explains the causal anomaly"** — **refuted** by its
  own one-line test (§6, row 3). The cause is estimator bias, and the biased
  estimator is the one that wins.
- **P3** ("rounds substitute for workers", 45x) — **artifact**. The sim drew a
  fresh `x_t ~ N(0,I)` each round, injecting design spread of order 1 against a
  within-round spread of order 0.02. Real consecutive rounds do not do that. All
  numbers above use an actual FedAvg trajectory.
- A pooled-across-rounds regression with a **common intercept** is biased: each
  round's intercept is `lam*x_t` and the within-round mean displacement is
  `~ -eta K lam x_t`, perfectly anti-correlated, so the between-round cluster
  means trace a slope of the **wrong sign**. Fixed by within-round centring
  (fixed effects). This bug produced an earlier "pinned at 1.000 for every N"
  table that I nearly reported as a finding.
