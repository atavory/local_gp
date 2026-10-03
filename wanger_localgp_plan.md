# wanger plan: local-GP / trajectory-vs-averaging project (2026-10-03)

Companion: `wanger_regime_check.py` (pure Python, stdlib only, CPU seconds).
Peer base (read-only, Overleaf checkout):
`ushmoo_localsgd_trajectory_related_work_20261003.md` + full text
`/tmp/ushmoo_relwork/wd.html`.

## 1. Position: gap survives, narrowed by the §4 correction

Agreed: 2605.27739 changes aggregation
(`p_filt = a.p_dom + g.p_bulk`), so the gap is endpoints-only (N vectors, no
intermediates), subspace-split not field (two scalar knobs, no magnitudes /
secants), no convergence theory for the modified rule. Their knobs inherit
the E71 critique: measured-(a,g) vs tuned-constant-(a,g) at matched
effective step is the bar, for them and for us.

## 2. Regime check: NO flip on quadratics (reconciliation refuted)

I previously argued gap subspace flips flat->sharp at the EoS boundary
(`eta.lam ~ 1`). Checked on diagonal quadratic (d=200, 12 outliers
lam~U[8,12], bulk lam~U[0.1,1], N=8, K=8, 120 reps):

- stable (`eta.lam_max ~ 0.6`): top-12 gap overlap with sharp-12 = 0.000
- EoS (`eta.lam_max ~ 1.8`, stable oscillation): sharp overlap = 0.000

Both exactly zero. Reason (obvious in retrospect): stationary gap variance
per eigendirection is `eta^2.s^2/(1-(1-eta.lam)^2)`, maximized where
`|1-eta.lam|` is closest to 1, i.e. SMALLEST `eta.lam`. The flip would need
`(1-eta.lam_sharp)^2 > (1-eta.lam_flat)^2`, i.e. `eta.lam_sharp > ~1.93`,
past the quadratic stability limit (diverges). So on quadratics, endpoint
gaps reveal FLAT directions in every stable regime. The check killed the
reconciliation; kept because a dead reconciliation is load-bearing below.

Consequences:

(a) The 2605.27739 "sharp" mechanism CANNOT be quadratic deviation-noise.
It must live in non-quadraticity (path-dependent curvature along diverging
trajectories) or data heterogeneity. Their experiments (MLP/CNN/Transformer)
are consistent with that; nothing in the quadratic model reproduces it.
(b) The offline test must therefore compare SYNTHETIC-QUADRATIC (predict:
flat alignment, `ALIGNMENT_BINDS`) against REAL-NET ledgers (predict: some
sharp alignment if their mechanism is real). The comparison IS the
experiment -- either outcome teaches: synth==real-flat means their title
does not transfer to filed-gradient trajectory data; real-sharp vs
synth-flat isolates non-quadraticity as the active ingredient.
(c) Design rank is full (min(NK,d) = 64) but variance is flat-allocated.
The "57-64 clears the 10-15 cliff" arithmetic must be restated as
variance-weighted aligned coverage, not nominal rank. (My script's PR~164
is per-coordinate variance spread, not matrix rank -- do not cite it as
rank; the binding constraint is allocation, and it binds on quadratics.)

## 3. Claim split (forced by §2): drift correction vs curvature estimation

- Drift correction (predict field at mean endpoint, fix aggregation mean):
  design-aligned on quadratics AND nets. Primary claim.
- Curvature/outlier estimation (preconditioning, sharp-subspace ID):
  design-misaligned on quadratics (sharp overlap 0.000); possibly alive on
  nets only via the non-quadratic mechanism. Secondary, venue-conditional.
File all claims in this split form. Aggregate R^2 will look fine (flat dims
dominate counts) while the curvature story silently fails -- hence
subspace-conditioned reporting is mandatory, not optional.

## 4. Communication parity (non-negotiable design constraint)

Their endpoint method costs d floats/worker (FedAvg parity). Server-side
N.K fitting costs Kx uplink -- Cloud-3 economics, dead on arrival. The
deployable form is LOCAL-LINEAR: worker fits `(a_i, B_i)` on its K+1 pairs,
sends `(a_i, diag(B_i))` = 2d floats; server averages fits, evaluates at
common target (mean endpoint, inside cloud -> interpolation). Average of
local-linear fits ~= pooled fit by linearity, so little is lost. Fair
head-to-head holds bytes fixed: their filtered endpoints vs our local fits
at matched comms. If Kx-richer design cannot beat endpoint filtering at
equal bytes, intermediates buy nothing -- that is this project's E100 and
it runs immediately after the offline fit check.

## 5. Offline test T0 (no new training; full spec)

Data: logged `(x_ik, g_ik)` trajectories. Check E88/E99 double_fan ledgers
for intermediates; if absent, one logging-only K=8 N=8 MNIST job
(E88-class, cheap) + synthetic-quadratic twin with matched (N,K,eta,sig).
Fits: pooled linear LS (symmetrized H_hat) + local-worker linear fits.
Targets: (i) held-out-worker trajectory prediction (fit N-1, predict
left-out -- clean, no leakage); (ii) next-round server gradient (E85
analogue). Metrics (paired CIs): aggregate R^2 vs endpoint-average
baseline; R^2 on top-15/top-50 Hessian eigendirections (Lanczos/GN) vs flat
complement; per-eigendirection spread vs x-noise; synth-vs-real alignment
comparison (§2b). Labels: `TRAJECTORY_FIELD_LIVE` (beats endpoint-average
held-out), `ALIGNMENT_BINDS` (top-eigen error >> flat -- expected on
synth), `COLLINEAR_DESIGN` (rank collapse), `SYNTH_REAL_SPLIT` (real-sharp
vs synth-flat -- isolates non-quadraticity). T0 go/no-go: `LIVE` on either
bed opens the project; null on both closes it before a single GPU burns.

## 6. Inherited discipline (day one, no re-learning)

VOI-gate every venue (headroom lower bound > 2x tracking-noise floor before
any controller); measured-vs-tuned-constant at matched effective step for
EVERY knob (ridge lam, rank r, diagonal restriction included); grids that
do not end at the winner; oracle-validity (loses-to-all-arms = invalid);
winner's-curse-corrected headroom; N always stated with total examples
(sliced-N language ban); rival rule (Kalman-true-Q / tuned-scalar /
magnitude-topk armed with equal information). The double_fan ledger paid
for these in 107 experiments.

## 7. Next steps + stop rules

1. T0 offline (this note's §5). 2. Comms-parity head-to-head spec (their
filter vs local-linear, matched bytes). 3. SCAFFOLD regime-split
comparison design (heterogeneous: expect loss; homogeneous-curved: expect
win). 4. Convergence-bound rewrite ONLY after T0 `LIVE` (drift term x
unexplained fraction, design-weighted norm -- §3 split enters the theorem).
Stop: T0 null on both beds, or parity head-to-head null, closes the
project to a negative note. Do not build closed-loop before T0.
