# Local GP: the trajectory is a free experimental design

## The observation

Local SGD runs `K` local steps on each of `N` workers, then averages the endpoints.

Per round it computes **`N·K` gradients at `N·K` distinct points**, and keeps
**`N` endpoint displacements**, which it then collapses to **one** average.

```
worker i visits   x_t = x_i0, x_i1, ..., x_iK
            computes  g_i0, ..., g_i(K-1)   at those points
server receives   Delta_i = x_iK - x_t = -eta * sum_k g_ik
server uses       mean_i(Delta_i)
```

The intermediate `(point, gradient)` pairs are a **design matrix for the local
gradient field**, and they are discarded. The proposal is to fit the field to
that design and step with the fit, instead of averaging endpoints.

## Why this regime and not the one we just spent 112 experiments on

Three things change at K>1. Legs 1 and 3 hold as stated; **leg 2 was wrong as
originally written and is corrected in place** rather than quietly dropped.

### 1. The sufficiency obstruction does not apply

At K=1 every worker evaluates at the same `x_t`, so `g_i = theta + xi_i` with a
common mean. The sample mean is sufficient; the spread is **independent of the
mean** (Cochran/Basu). Measured directly:

```
corr(R_j, signed error gbar-theta) : -0.0074     <- no direction information
corr(R_j, |error|)                 : +0.5494     <- scale information only
```

The cloud can calibrate, never correct. Every K=1 lane died of this: cloud →
error scale → magnitude → absorbed by the LR tuner. `SCALAR_LR_EXPLAINS`,
`ADAM_V_EXPLAINS`, `SCHEDULE_TRANSFERS_ENOUGH`, `EFFICIENCY_ONLY`,
`PRIOR_ONLY_EXPLAINS` are five names for one theorem.

**At K>1 the gradients have different means** — `E[g_ik] = grad f(x_ik)` and the
`x_ik` differ. `gbar` stops being sufficient. The discarded data carries signal
about the *field*, not noise about the *estimate*. The theorem that closed the
last project does not transfer.

### 2. The design estimates curvature *magnitude* well where it is sharp

**Corrected 2026-10-03.** This leg originally read: "`K` multiplies available
rank by `K`, so N=8,K=8 gives covered rank ~57-64, clearing the ~10-15 Hessian
outliers." wanger refuted it and I have reproduced the refutation
(`ushmoo_design_snr_check.py`): nominal rank is not aligned coverage. Endpoint-gap
variance is **5x larger in flat directions than sharp ones**, and the top-12
gap-variance overlap with the true sharp set is **0.000** in every stable regime.
The design does not point at the outliers. Do not cite `N*K` as coverage.

What survives is a different and narrower statement. Relative curvature error
`|lamhat-lam|/lam` from the `N*K` design:

| directions | relative error | invertible? |
|---|---|---|
| sharp (lam~10) | **0.069** | yes |
| flat (lam~0.5) | **3.080** | 41/200 coords give `lamhat <= 0` |

So the design is **misaligned for subspace identification** but
**well-conditioned for sharp curvature magnitude** — least spread, best relative
estimate, because `lam` enters both the spread and the normalisation. These are
different questions with opposite answers and the literature conflates them.

The consequence is not favourable and is developed in
`ushmoo_audit_bandha_20261003.md` §4: the sharp directions, where estimation
works, are already driven to the Newton step by the `K` local steps themselves
and have **no gain left to give**; the flat directions hold all 31% of the
available gain and are the ones estimated at 308% error. That is the lane's
governing constraint, and the project's real problem is an estimator problem in
the small-`lam` regime, not a coverage problem.

### 3. The data is free

Cloud 3 died on matched-cost accounting: an extra gradient per worker per round.
**These gradients are already computed and already paid for** — doing K local
steps is the entire point of local SGD. Only the aggregation is a design choice.
This is the first lane in the programme where the measurement costs nothing.

## Position against the nearest neighbour

**arXiv:2605.27739, "Worker Disagreement Reveals Sharp Directions in Local SGD"**
is close and it **does** change the aggregation rule:

> *replace the standard Local SGD synchronization update `p̄^c` with the filtered
> update `p_filt^c := alpha * p_dom^c + gamma * p_bulk^c`*

What it leaves open:

1. **Endpoint gaps only** — `Delta_i` is the final displacement. The `K-1`
   intermediate points are unused. Design size `N`, not `N·K`.
2. **A subspace split, not a field** — two buckets, two scalars, no curvature
   magnitudes and no secants. `alpha, gamma` are hyperparameters.
3. **No convergence theory for the modified rule.**

See `ushmoo_localsgd_trajectory_related_work_20261003.md` for the full survey.

## Theory adaptation

Standard local-SGD bounds decompose as optimisation + `variance/sqrt(NKT)` +
**client drift**, with drift `O(eta^2 K^2 G^2)` — growing with K, charged
entirely as error.

**Reframe: drift is the design, not the error.** If workers never diverged there
would be nothing to fit. Replace the smoothness step with a fitted residual:

```
|| grad f(x_ik) - grad f(x_t) ||  <=  L || x_ik - x_t ||
    ->   || (H - H_hat)(x_ik - x_t) ||  +  O(rho || x_ik - x_t ||^2)
```

The drift term gets multiplied by the **unexplained fraction of the field** —
measured at ~0.45 in the prior project (the probe explained ~55%, at the
noise-efficient limit).

**Non-obvious consequence, and the thing worth proving.** Standard theory: larger
K is monotonically worse. Under field fitting, larger K also improves the fit
(more points, better conditioning), so the two partly cancel and the optimal
`K*` shifts **upward**. Both halves are already measured: the competing term is
the third-order Taylor error, which grew from residual 0.587 to 0.762 between
K=2 and K=8.

## First test, pre-registered

**Offline. No training runs. No closed loop.**

Narrowed 2026-10-03 by the §2 correction and
`ushmoo_audit_bandha_20261003.md` §4. The question is no longer "does the
trajectory carry field information" — on a quadratic it demonstrably does, worth
**1.310x** a tuned-scale baseline with true curvature. The question is the one
that binds:

> Can **any** estimator recover the small-`lam` contraction factors from a
> `K`-step local-SGD trajectory well enough to beat a tuned-scale baseline?

The gain is entirely in the flat directions and the slope regression returns
308% relative error there. Everything turns on that one number.

| | |
|---|---|
| **Primary metric** | scale-invariant alignment (cosine) with a large-batch audit gradient. **Not MSE**, not a projection — see guardrail 5 |
| **Arms** | (a) endpoint average + tuned scalar **[baseline]**, (b) slope-regression `lamhat`, (c) **AR(1) contraction estimated from the temporal structure of the path**, (d) **frozen curvature pooled across rounds**, (e) `alpha*p_dom + gamma*p_bulk` with tuned constants (the 2605.27739 rule), (f) true-curvature oracle |
| **Pass** | some estimator arm beats (a) *and* (e) on cosine, paired, CI excluding zero |
| **Fail** | only (f) beats (a) — the idea is right and no estimator reaches it |
| **Void** | (f) does not beat (a): no headroom on this bed, report and move beds |

Prediction on record, from §4: **(d) beats (b)**. The binding constraint is
estimator variance and `lam` is near-static, so pooling across rounds should
dominate refitting per round — the opposite of what "local GP" suggests.

Report in every table: per-direction relative curvature error, and the count of
coordinates with non-invertible `lamhat`. On the synthetic bed those were
0.069 / 3.080 / 41-of-200 and they explained the result completely.
