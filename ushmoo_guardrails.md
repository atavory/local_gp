# Guardrails, carried over from 112 experiments

These are not style preferences. Each one has a specific corpse behind it. The
previous project ran ~112 experiments and shipped no method; the postmortem is
that most of them were **uninterpretable by construction**, not negative. An arm
that cannot distinguish its hypothesis from a confound produces a number, a
decision label, and no knowledge.

Read this before designing any experiment in this repo.

---

## 1. Match the effective step, always

Any rule that rescales the update — a preconditioner, a fusion weight, a trust
coefficient, a shrinkage factor — changes **gain** and **shape** at once. Gain is
what the learning rate already tunes.

Decompose every proposed step into `c_j * g_j`, split `c` into its mean (gain)
and its spread (shape), and report both. At zero spread the whole intervention is
a learning-rate change in disguise.

Measured: at spread 0.0 the direction change is exactly `0.00000`. Six separate
"wins" in the last project were this and nothing else.

> **In this repo:** a fitted-field step has a different norm than the endpoint
> average. If you compare them at the same `eta` you are measuring a step-size
> change. Normalise, or sweep `eta` for both arms.

## 2. The control for a measured quantity is its tuned constant

If the method measures something per-round and uses it, the baseline is **the
same method with the measurement replaced by a tuned constant** — plus,
ideally, a shuffled and a one-round-delayed version.

A measured controller lost to its own tuned constant in E73→E76, E96→E97, E100,
E102, and E106. Five times. It is the single most replicated finding of the
previous campaign and it is the first thing a reviewer will ask.

> **In this repo:** `alpha * p_dom + gamma * p_bulk` with measured coefficients
> versus the same rule with tuned-constant `(alpha, gamma)`. This control is
> also what the nearest-neighbour paper appears not to have run — it is both our
> critique of them and the bar we have to clear ourselves.

## 3. Include a frozen / prior-only arm

Does the **live** measurement add anything over a frozen one learned offline,
or over the prior alone? This isolates "the structure is real" from "measuring
it each round is worth it". They are different claims and only the second
justifies a method.

## 4. The baseline gets equal tuning budget

A two-knob method compared against a one-knob baseline on the same grid budget
wins on budget, not on mechanism. Count the knobs; equalise the search.

## 5. Score alignment, not MSE

MSE has a known answer: **James-Stein guarantees the shrinkage estimator wins in
dimension ≥ 3**, for every dataset, carrying zero information. Worse, MSE rewards
exactly the uniform rescaling a tuned learning rate undoes — on a quadratic, the
loss decrease at optimal step size

```
decrease = <theta, theta_hat>^2 / (2 * theta_hat' H theta_hat)
```

is invariant to any uniform scaling of `theta_hat`.

Use **scale-invariant alignment** (cosine against a large-batch audit gradient).
Measured consequence: an empirical-Bayes shrinker's directional gain is
**exactly zero** under homogeneous noise, and rises to 0.070 / 0.175 / 0.236 only
as the per-coordinate noise spread grows. MSE ranked it first in all four cases.

## 6. Measure headroom before chasing the effect

Compute the oracle ceiling first. If the ceiling is below the effect size you
are chasing, stop — you have saved a week.

**And validate the oracle**: a "ceiling" that loses to arms beneath it is not a
ceiling, it is a bug. The previous project reported an oracle headroom number
that lost to every transfer arm; the number was withdrawn. Sanity-check that the
oracle actually dominates before quoting it.

## 7. Check the rank before interpreting

Low-rank curvature estimated from `r` directions buys **exactly 1.0000** — no
gain at all, not a small gain — until `r` reaches the Hessian outlier count, then
drops to ~0.12-0.19. Coverage also needs `N >= spikes + 1`.

Report the covered rank and the outlier count in every result table. Below the
cliff, a null is not evidence about the method.

## 8. Vary one thing per arm

E71's arm C had its coefficient slaved to arm A's, which voided its own control.
If two knobs move together the experiment cannot attribute the outcome.

## 9. Watch for catastrophic cancellation in the estimator

`[sum g^2 - N * gbar^2]` at low precision is a difference of near-equal large
numbers: at 4-bit and signal-to-mean ratio 0.02 the error was **78x the signal**.
Compute variances in the centred form, and state the precision any quantity is
measured at.

## 10. Label the failure mode, not just the outcome

"Collapses to a scalar" and "collapses to the prior" are different diagnoses with
different fixes. Four distinct decision labels in the last project turned out to
be one theorem (sample mean ⊥ sample covariance ⟹ scale, never direction) — which
nobody noticed for ~90 experiments because the labels hid the common cause.

Write down the *mechanism* you believe produced the result, then design the arm
that would falsify that mechanism specifically.

## 11. Charge the measurement's cost — except here

Extra gradients, extra bytes, extra memory all go on the ledger, and cloud 3 died
on exactly this accounting.

**This is the one guardrail that relaxes in this repo**, and it is the structural
reason the lane is worth running: the `N*K` intermediate gradients are already
computed and already paid for by local SGD itself. Only the *aggregation* is a
design choice. Communication cost still goes on the ledger if the scheme sends
more than `N` vectors to the server — say so explicitly in every proposal.

## 12. Short runs do not settle anything

A ten-minute run that shows no effect is not a dead hypothesis. Three separate
times I was **right about the asymptotic limit and wrong about the first ten
rounds** — most sharply on DC gain, which is 1 in steady state and badly wrong in
the transient where the damage happens. If a claim is about the limit, do not
test it on a transient; if it is about the transient, do not justify it with a
steady-state derivation.

Related: the `d / r_eff` advantage of a scalar method over a low-rank one is
exact at fixed max-stable step and **reverses completely** under per-horizon
tuning (82x the other way at T=30000). State the tuning protocol as part of the
claim, because the claim changes sign without it.

---

## The one-line version

> Before running an arm, write down what result would make you abandon the
> hypothesis. If no result would, the arm is decoration.
