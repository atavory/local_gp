# Shadow framing: what the reader should believe afterwards

bandha's framing and mine differ on one axis, and it is the axis the complaint
about the LaTeX is actually about. Not prose. Content.

> The writing describes the process, which doesn't interest people, and doesn't
> describe what we want them to take away.

A process sentence says what we did. A takeaway sentence says what is true. The
test for every paragraph: **delete it and ask whether a reader's beliefs
change.** If only their picture of our workflow changes, cut it.

---

## 1. The problem with the current framing

> *Distributed optimization as global inference from worker queries.*
> *Beyond Averaging: Distributed Optimization as Sequential Experimental Design.*

Four problems, in order of severity.

**It contains no falsifiable claim.** Points 1-4 of the description are all
reframings. Nothing in them could come out false. A viewpoint is not a result,
and venues reject viewpoints that arrive without a theorem the viewpoint
produces and the old view cannot. This is the same gap as the acc-fl envy from
the start of the programme: dense math does not come from a frame, it comes from
a mechanism with a closed form. We have one. It is not in the description.

**The frame is wider than the method by two orders of magnitude.** "Sequential
experimental design over a shared information state of all points, gradients,
losses, probes, accepted and rejected steps" promises an acquisition function
and a policy. What the code does is estimate a diagonal-plus-low-rank curvature
and rescale the FedAvg displacement. A reviewer measures the distance between
the abstract's promise and Section 5's delivery, and that distance is the
rejection.

**"Beyond Averaging" picks the wrong enemy, and we have the result that proves
it.** Averaging is not naive. It is *exactly* the Bayes-optimal read-out under a
specific, checkable prior. Making the baseline a special case of your method is
strictly stronger than calling it insufficient, and it is what we can actually
prove.

**The four points are an autobiography of the idea** — single worker, then many,
then global, then async. That is chronology. Invert it: result, mechanism,
scope.

---

## 2. What the reader should take away

Three claims. Each is falsifiable, each has a number, none is in the current
description.

### Claim 1 — Averaging is optimal, for a prior that is wrong in a specific way

Put a GP prior on the loss; worker trajectories are derivative observations.

> **Proposition.** If the prior's gradient field is almost surely constant
> (infinite lengthscale), the posterior mean of `grad f` is the unweighted mean
> of the observed gradients, and the Bayes step is proportional to the averaged
> displacement.

So FedAvg is not a heuristic to beat. It is the `lengthscale = infinity` case,
and the entire literature of filtered updates, weighted averaging and
"averaging is probably not the optimum" is **prior misspecification** rather than
a bag of tricks. This reframes the baseline rather than attacking it, which is
the stronger rhetorical and mathematical position.

### Claim 2 — Local SGD is already a preconditioner, so the obvious correction is wrong

On a quadratic, with `a_j = 1 - eta lam_j`, K local steps give

```
Delta-bar_j  =  -g*_j (1 - a_j^K) / lam_j
```

(verified: observed/predicted 1.0218 sharp, 0.9346 flat). The implied
per-coordinate step interpolates **gradient step -> Newton step with crossover at
`eta lam_j K ~ 1`**:

| direction | implied step | Newton `1/lam` | gradient `eta K` |
|---|---|---|---|
| sharp, lam=10 | **0.0996** | 0.1000 | 0.4000 |
| flat, lam=0.5 | **0.3667** | 2.0000 | 0.4000 |

The sharp directions have **already arrived at the Newton step before
aggregation.** Therefore:

> Preconditioning the averaged displacement by the inverse Hessian
> double-applies a correction the local steps already performed. With the
> **exact** Hessian it scores **0.945x** a tuned scalar baseline — worse than
> doing nothing, with perfect knowledge.

The correct operator is `[I - (I - eta H)^K]^{-1}`, worth **1.310x**.

This is the best single thing we have. It is closed-form, verified,
counterintuitive, and it changes what a practitioner does tomorrow. It is also
a *negative result about an obvious idea*, which is the most readable kind.

### Claim 3 — The gain and the measurability are the same quantity, in opposition

Write the correction owed as `C_j = [1 - (1 - eta lam_j)^K]^{-1}`.

- Benefit of applying it: `(1 - 1/C_j)^2`.
- Relative error of estimating it: `eps_j ∝ C_j`, **in both design branches.**

So their ratio does not depend on curvature at all.

> **The directions with gain to give are exactly the directions you cannot
> measure.**

Measured: relative curvature error **0.069** in sharp directions, **3.080** in
flat ones, 41/200 coordinates non-invertible. The value-times-precision product
`(1-1/C)^2 / C` has an interior maximum at `C = 3`, i.e. `rho = eta lam K ~ 0.4`
(measured peak 0.24). **Benefit is non-monotone in K**, which contradicts the
standard reading of local-SGD theory, where larger K is monotonically worse.

### The empirical claim that makes it not-L-BFGS

Server-side temporal secants — exactly the pairs an L-BFGS running on the
aggregated iterates would use, free, requiring no extra logging — capture
**none** of the available headroom (1.00). Within-round worker geometry captures
most of it. Without this control the work reads as reinvented quasi-Newton; with
it, the intermediate local points are doing something classical methods
provably cannot.

---

## 3. Title and abstract

Name the mechanism, not the viewpoint.

> **Local SGD Is Already a Preconditioner:
> Why Inverse-Hessian Correction of FedAvg Makes Things Worse**

Alternates: *The Correction Local SGD Already Applied*; *Averaging Is Optimal
For the Wrong Prior*.

Draft abstract, as a demonstration of the register:

> Local SGD and FedAvg aggregate worker updates by averaging. We show this is
> exactly the Bayes-optimal read-out under a prior in which the gradient field
> is constant over the region the workers explored, so the question is not
> whether to average but whether that prior holds.
>
> It does not, and the correction is not the one you would guess. On a
> quadratic, K local steps produce a mean displacement that interpolates between
> a gradient step and a Newton step, with crossover at `eta lam K = 1`. Sharp
> directions therefore reach the Newton step before aggregation; preconditioning
> the averaged displacement by the inverse Hessian double-applies that
> correction and makes the step worse — 0.945x a tuned scalar baseline, with
> exact curvature. The correct operator is `[I - (I - eta H)^K]^{-1}`.
>
> What remains to be corrected lies entirely in the flat directions, and these
> are precisely the directions worker geometry estimates worst: the relative
> error of the curvature estimate and the size of the correction owed both scale
> as `C_j`, so their ratio is independent of curvature. Value times precision
> peaks at `C = 3`, making the benefit of worker geometry non-monotone in K.
>
> The remaining gap is real and classically inaccessible: server-side secants,
> the pairs an L-BFGS on the aggregated iterates would use, capture none of the
> available headroom, while within-round worker geometry captures most of it.

Every sentence is a claim. None describes an activity.

---

## 4. Before and after

The pathology is mechanical and so is the fix.

| process (cut) | takeaway (keep) |
|---|---|
| "We then replace the standard synchronization update with the filtered update." | "The standard update is optimal only when the gradient field is constant across the workers' span; otherwise it errs by `(1-1/C_j)` per direction." |
| "In Subsection 3.2 we construct the estimator and in 3.3 we analyse it." | "The estimator's relative error scales as `C_j`, the same quantity as the correction it estimates — which is why more workers do not help." |
| "We ran arms A, B, C and D at three worker counts." | "No tuned scalar reproduces the effect, and no classical secant recovers it." |
| "The appendix reviews these connections." | Delete. Put the connection in the text or drop it. |

Rule of thumb: if a sentence's subject is "we," "this section," or "the
appendix," it is probably process. Subjects should be the objects of study —
the displacement, the operator, the error, the directions.

---

## 5. Scope: what we cannot claim

Stating this crisply is what buys the right to state Claims 1-3 strongly.

- **No real-network result.** Everything is synthetic. E010 separable bed:
  shape beats tuned scalar by 0.302/0.574/0.402, CIs excluding zero. E022 dense
  correlated bed: gated version beats tuned scalar by 0.038/0.103/0.074, CIs
  excluding zero, but captures only ~6-10% of oracle headroom.
- **The displacement law is exact algebra on a quadratic.** Everything inherits
  that. Non-quadraticity is untested and is where the competing literature's
  mechanism must live.
- **The dense trust layer costs unpriced compute.** The probe gradients are
  free — already computed by local SGD. The validation batches are not. That is
  the matched-cost objection returning, and it must appear in the ledger.
- **Homogeneous workers throughout.** The heterogeneity story is a multi-task-GP
  extension we have not run.

---

## 6. The positioning difference, which is the real disagreement

bandha's framing — controller, policy, architecture, information state — is a
**method/systems paper**. We cannot support one: no real workload, a gated
dense gain of 4-10% with unpriced validation cost, and no policy beyond a
per-coordinate shrinkage.

Mine is a **theory paper with a sharp negative and synthetic validation**. We
can support that today: a closed-form law, a duality with an interior optimum, a
counterintuitive negative about an obvious method, and a control showing the
classical alternative gets nothing.

That difference decides what has to be true before submission. Under bandha's
framing the blocker is a real-workload win, which is months away and may not
come. Under mine, the blocker is breaking the quadratic assumption and pricing
the gate — both weeks, both already specified.

A theory paper that is right is worth more than a method paper that is pending.
