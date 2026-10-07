# Sufficiency Null for HPO-Aware Aggregation
wanger, 2026-10-06; updated 2026-10-07. Status: theory note; proofs sketched, not refereed.

**Thesis.** With no HPO variation, workers are exchangeable; the averaged
update / momentum / acceleration trajectory is sufficient; everything beyond
it is ancillary noise for estimating the common update.  A probe-weighting rule
can still move realized weights because update and probe batches are finite.
The null claim is subtler: without designed HPO variation, any gain is
stochastic-update filtering, not persistent hyperparameter signal, and must be
validated against shuffled/held-out controls.  Probes become HPO-aware
aggregation only when exchangeability is deliberately broken by designed
spread, tagged heteroscedasticity, or a pooled server-side control.

## 0. Setup

Round $t$, $N$ workers, common init $\theta_t$, COMMON hyperparameters $h$
(no HPO variation). Worker $i$ returns $u_i \in \mathbb{R}^d$. Model:

$$u_i = \mu + \varepsilon_i, \qquad \varepsilon_i \ \text{exchangeable},\
E[\varepsilon_i]=0,\ \mathrm{Cov}(\varepsilon_i)=\Sigma.$$

$\mu$ is the estimand (ideal round update). Server deploys an aggregate
$\hat g = \sum_i w_i u_i$, weights possibly data-dependent, then an outer
step. Loss $L(\mu, \delta)$ is convex in the action and invariant under
worker permutation (any sane deployment loss is: workers are unlabeled).

"HPO variation" means: $h_i \neq h_j$ by design (LR groups, $(\alpha,\beta)$
candidate grid, $K$ groups). Then $\mu_i = f(h_i)$ differ and
exchangeability breaks. That is Section 6.

## 1. Symmetrization: worker identities are inadmissible

**Theorem 1 (Rao-Blackwell symmetrization).** For any rule
$\delta(u_1,\dots,u_N)$ and any permutation-invariant convex loss, the
symmetrized rule $\bar\delta = \frac{1}{N!}\sum_\pi \delta(\pi\cdot u)$
satisfies $R(\mu,\bar\delta) \le R(\mu,\delta)$ for all $\mu$, strict
unless $\delta$ is a.s. permutation-invariant.

*Proof.* Risk of $\bar\delta$ is the average over permuted risks by
exchangeability; Jensen (convex loss) pulls the average inside. $\square$

**Consequence.** Restrict to symmetric rules w.l.o.g. Any rule that treats
worker 3 differently from worker 7 *by identity* is inadmissible. The only
admissible inputs are symmetric statistics (mean, order statistics,
empirical distribution). E035's UCB-over-workers failed for exactly this
reason: it spent statistical budget learning identities that carry no
signal.

## 2. Gaussian location: the mean is sufficient, the cloud is ancillary

**Theorem 2 (Sufficiency + ancillarity).** If $\varepsilon_i \sim
\mathcal{N}(0,\Sigma)$ i.i.d. with $\Sigma$ known: $\bar u$ is complete
sufficient for $\mu$; the centered cloud $\{u_i - \bar u\}$ is ancillary
(distribution free of $\mu$); by Basu's theorem they are independent.

*Proof.* Exponential family / Fisher-Neyman for sufficiency; ancillarity
is immediate since centering kills $\mu$; Basu gives independence of
complete-sufficient and ancillary. $\square$

**Corollary 2a (the cloud is uninformative).** $I_{\text{cloud}}(\mu) = 0$:
the Fisher information about $\mu$ in the centered cloud is exactly zero.
Any within-round reweighting $w(u)$ fit to exchangeable scatter adds
variance and zero signal. This is the Double Fan autopsy in one line:
same-point clouds measure $\Sigma$ (noise), and $\Sigma$ was assumed
known / uninformative about $\mu$.

**Remark (unknown $\Sigma$).** With unknown $\Sigma$, sufficiency needs
$(\bar u, S)$ with scatter $S$. But $S$ informs $\Sigma$, not $\mu$
(independent by Basu). Unless the loss depends on $\Sigma$ (it doesn't
in our deployment), the extra statistic buys nothing. Estimating
heteroscedasticity from one round's scatter to reweight is pure noise
fitting — measured: E034 top-1 hit rates 0.04-0.40 with the oracle gap
persisting.

## 3. Momentum and acceleration: the averaged trajectory is sufficient

Now rounds $t = 1,\dots,T$ with drifting $\mu_t$, independent Gaussian
rounds. Per-round sufficiency stacks: $\{\bar u_t\}$ is sufficient for
$\{\mu_t\}$.

**Corollary 3a (server momentum sufficiency).** Server momentum
$m_t = \beta m_{t-1} + (1-\beta)\bar u_t$ with $m_0$ fixed is a BIJECTION
of the mean trajectory: $\bar u_t = (m_t - \beta m_{t-1})/(1-\beta)$.
Hence $\sigma(m_1,\dots,m_T) = \sigma(\bar u_1,\dots,\bar u_T)$: the
server momentum trajectory carries exactly the sufficient information.
Worker-resolved momentum paths $\{m_{i,t}\}$ add only ancillary noise.

**Corollary 3b (acceleration sufficiency).** For drift
$d_t = \mu_t - \mu_{t-1}$, averaged acceleration
$\bar a_t = \bar u_t - \bar u_{t-1}$ is sufficient; worker accelerations
$a_{i,t} = u_{i,t} - u_{i,t-1}$ add noise. Same proof, one difference up.

**Corollary 3c (Kalman form).** Under random-walk drift
$\mu_t = \mu_{t-1} + d_t$, the Kalman filter state $(m_t, P_t)$ is driven
by $\{\bar u_t\}$ only. Worker-resolved paths never enter the filter.
The Double Fan "$P_t$ problem" (predicted uncertainty unidentifiable from
same-point clouds) is structural: $P_t$ concerns cross-time dynamics and
no within-round statistic, however refined, identifies it. It needs
cross-time data (schedules = offline) or spread-based probes (Section 6).

This is the literal requested claim: **without HPO variation, average of
momentum / acceleration is a sufficient statistic.**

## 4. Zero-spread weighting regret: leaving uniform only costs

**Theorem 3 (variance cost of non-uniformity).** Under zero spread
(common $\mu$, isotropic noise $\sigma^2 I$), fixed weights $w$
($\sum w_i = 1$):

$$E\|\sum_i w_i u_i - \mu\|^2 = \sigma^2 \sum_i w_i^2 \ge \sigma^2/N,$$

equality iff $w$ uniform. For data-dependent $\hat w(u)$ the same
decomposition holds conditionally: every tilt pays variance, and with
zero spread there is no signal term to offset it.

**Corollary 4a (winner's curse, quantified).** Score
$y_i = V + \eta_i$ with COMMON value $V$ (zero spread) and noise
$\eta_i \sim (0,\tau^2)$. Top-1 selection $\hat\imath = \arg\max y_i$
deploys luck $E[\eta_{\hat\imath}] \approx \tau\sqrt{2\log N}$, which
does not transfer; meanwhile variance jumps from $\sigma^2/N$ (uniform)
to $\ge \sigma^2$ (single worker, worse if $\eta \leftrightarrow
\varepsilon$ correlated — selection favors large-noise realizations).
Top-1 under zero spread is strictly worse than uniform by
$\approx \sigma^2(1 - 1/N)$ plus selection bias. Measured: E034.

## 5. Shrinkage with spread: the Wiener gain (null predicted)

Scalar reduction: true values $V_i$ with spread
$\mathrm{Var}(V_i) = s^2$; probes $y_i = V_i + \eta_i$,
$\eta_i \sim (0, \tau^2/m)$. Posterior value:

$$\hat V_i = \bar V + \lambda^*(y_i - \bar y), \qquad
\lambda^* = \frac{s^2}{s^2 + \tau^2/m}.$$

**Theorem 4 (probe-win inequality).** Weighting by $\hat V$ beats
uniform iff spread clears noise plus variance cost:

$$s^2 \cdot g(\lambda^*) > \sigma^2 \cdot c(\lambda^*),$$

for geometry constants $g, c$ (gain shape vs tilt cost); in particular
$\lambda^* \to 0$ as $s^2 \to 0$, $m \to 0$, or $\tau \to \infty$ —
recovering uniform averaging as the population shrinkage target.  A finite
sample controller can still produce non-uniform weights at zero spread; those
weights are driven by probe/update noise.  The measured 0.000 spread-1 rows
show no persistent HPO gain under the frozen grouped rule, not that every
realized weight vector is literally uniform.  The 5-12% capture wall is the
small-$\lambda^*$ regime: $s^2 \sim \tau^2/m$, gain real but thin.

## 6. What breaks sufficiency: the HPO program (everything live = spread)

Sufficiency of $\bar u$ fails iff exchangeability fails. Exhaustive
routes:

**(i) Designed spread (HPO variation).** $\mu_i = f(h_i)$ with KNOWN
labels $h_i$: LR groups (E060/61), $(\alpha,\beta)$ candidate grid
(E078/80 — the outer grid IS $h$-variation at the server level),
$K$-groups. Mean insufficient; need group weighting / response
selection. Labels are free (we assigned $h_i$), spread is designed
large ($s^2$ chosen, not hoped for). **This is the entire live
program.**

**(ii) Tagged heteroscedasticity.** $\mathrm{Cov}(\varepsilon_i) =
\sigma_i^2$ with $\sigma_i$ known via tags: GLS weights
$w_i \propto 1/\sigma_i^2$ (E028). Requires tags (side information);
estimating $\sigma_i$ from one round's scatter is the noise-fitting
trap of Section 2.

**(iii) Heavy tails.** Symmetric robust rules (median, trimmed mean)
beat the mean — but they are functions of order statistics, still
symmetric, still not probe-weighted. Probes don't estimate the median
better than the sample does. Not a probe story.

**(iv) Nonlinear deployment value.** If loss depends on higher moments
of the aggregate (risk-sensitive serving), moments beyond the mean
matter — but they are estimated from the same exchangeable sample;
probes add nothing without spread. Changes the estimand, not the
sufficiency logic.

**Unification.** Every live branch is (i) or (ii), plus the server-side
outer-control case where candidate controls are applied to the same collected
updates: designed $h$-spread with free labels, tagged variance, or pooled
control variation.  Every dead branch tried to extract durable signal from
exchangeable worker identities: dense maps, per-worker ranking among
exchangeables, UCB-over-workers, top-1 selection.  The project slogan in one
line: *name the exchangeability-breaker, then use probes to aggregate the HPO
experiment back into the shared trajectory.*

## 7. Fisher decomposition (one paragraph)

$I_{\text{total}}(\mu) = N\cdot I_1$, all accessible via $\bar u$;
$I_{\text{cloud}}(\mu) = 0$. Probe information splits: common-mode
(pooled over $Nm$, $\sqrt{Nm}$ rate — round quality, $\alpha_t$
selection) vs differential (worker spread — needs $s^2 > 0$ and pays
winner's-curse tax). Stateless per-round probes should target pooled
scalars; spread exploitation needs designed $s^2$ or tags.

## 8. Predictions vs measurements

| prediction | measurement | status |
|---|---|---|
| no persistent HPO gain at zero spread | spread-1 rows show 0.000 gain; realized weight-motion audit pending | CONFIRMED / AUDIT |
| top-1 strictly worse than uniform, zero spread | E034 hit 0.04-0.40, gap persists | CONFIRMED |
| symmetric rules only; identity-learning fails | E035 UCB negative ($-0.250\pm0.154$) | CONFIRMED |
| designed spread + shrinkage wins | E060/61 groups; E078/80 $+0.177\pm0.049$ vs shuffled | CONFIRMED |
| tagged variance wins | E028 (cost caveat open) | PARTIAL |
| small-$\lambda^*$ thin capture | 5-12% everywhere | CONFIRMED |
| $P_t$ unidentifiable from same-point clouds | Double Fan freeze | CONFIRMED |
| outer grid = $h$-variation, same theory | E078/80 design | CONFIRMED |

## 9. Prescription (HPO-aware aggregation)

Design spread (grids, groups — free labels, chosen $s^2$); tag variance
(E028 mechanism); pool server-side controls; pool time (round scalars,
hysteresis); never sell exchangeable identity ranking as HPO.  Frozen grids +
shrinkage first; response-surface modeling only behind demonstrated grid-edge
pinning; $K$-adaptation behind a comm ledger.  The theory above is the referee
for every future branch: name the exchangeability-breaker, measure the IID
noise-selection null, and compare against fixed-compute external HPO.
