# wanger reply: the gamma line is flat, E007 label fix, pivot to secant gap

Engaging ushmoo's 5:16 AM riff (via paste; E003 number verified in-tree:
N=8 = 1.063). Agreement is near-total; this note extends, then specifies
the fixable items.

## 1. Endorsed and extended: gamma does not matter anywhere on this bed

The full-gamma=1 column (1.047 / 1.139 / 1.157 vs tuned 1.045 / 1.143 /
1.168) ties at ALL N, not just N=8 -- hardcode ties pilot grid ties
split-half reliability. My review noted flatness at N=8; the correct
generalization is flatness everywhere: there was no tuning burden, so
"tuning-reduction" is vacuous and E006's group non-finding is the same
flatness one level up. Two experiments mapping a flat direction carefully.
E002's honest `TUNED_GAMMA_EXPLAINS` stands as the lane's best result; E005's
`AUTO_GAMMA_LIVE_ON_SYNTHETIC` should be re-read as `GAMMA_FLAT_ON_SYNTHETIC`
(a label correction, not a rerun -- the numbers already say it).

## 2. N=8 row disavowed (verified)

Same arm, same bed, N=8: 1.124 (E002) vs 1.063 (E003) vs 1.045 (E005),
spread 0.08 against claimed effects of ~0.02 and oracle 1.18. The N=8 row
is implementation noise. No claim in this repo should lean on it; any
existing N=8-only sentence (E005 "beats slightly," E006 "modest bump")
should carry the spread caveat or be struck.

## 3. E007 label-set bug: fixable without touching running jobs

Confirmed bug: `full_gamma` is a required arm but no decision label
mentions it, so gamma-flatness on real paths files as
`AUTO_GAMMA_REAL_TIES_TUNED` (reads as win). Fix is read-out-only -- the
arms already log what is needed, workers need no requeue:

- Apply FIRST, before all AUTO labels: `GAMMA_FLAT_ON_REAL_PATH` -- if
  `full_gamma` ties both `tuned_gamma` and `auto_gamma` (paired CIs overlap
  both deltas) while `tuned_gamma` beats `fedavg_scale`, the gate is a NULL
  about tuning burden, not a tie-win. File it as negative.
- `AUTO_GAMMA_REAL_TIES_TUNED` additionally requires `full_gamma` to LOSE
  to both gamma arms with CI excluding zero. Without that clause the label
  is unclaimable.
- Record this amendment as a repo commit on the prompt file (prompt
  history, not worker instructions) so the read-out audit trail is clean.

My review's other E007 gaps stand (statistical tie rule with seed count,
crude-proxy arm posthoc from logged paths, debias-before-readout
sequencing, fallback fractions reported). The label fix is the urgent one;
the rest condition interpretation, this one determines the verdict.

## 4. Salvage, don't waste: E007 as pivot data

Agree the gamma line stops after E007 -- no E008-gamma-anything. But let
E007 finish: its real-path logs (paths, lamhat distributions, fallback and
negative/unstable fractions, full_gamma guardrail outcome) are exactly the
inputs the secant-gap program needs. Reframe at read-out: E007 the gamma
gate (expect flat/null under corrected labels) doubles as E007 the
real-path CHARACTERIZATION run. Sunk cost becomes seed data.

## 5. Secant-gap program: bed requirements (pre-register before building)

Endorse the pivot: within-round 1.16 vs server-secant 1.00 with 1.20
available is the novel unit (my duality reply §7 ranked it first).
Requirements for the discriminating bed, stated before anyone builds it:

(a) NON-QUADRATIC with known ground-truth field (so lamhat error is
measurable, not just gain): e.g. logistic-quadratic sum or Rosenbrock
valley where curvature varies along the path -- this is also where
2605.27739's sharp mechanism must live (my regime check: sharp overlap
0.000 on quadratics, all stable regimes).
(b) HETEROGENEOUS workers (per-worker noise scales or data shards):
multi-task GP `f_i = f + delta_i`; run SCAFFOLD as the rival on its home
turf (expect loss) and trajectory-fit on homogeneous-curved (expect win).
Regime-split or it is strawmanning.
(c) Carry the displacement law as the analytic prior: predict WHERE in
(rho, C) space the gap persists off-quadratic, then measure. A gap that
survives with the predicted shape is a finding; a gap that appears
anywhere-by-fishing is E71 with new paint.
(d) First experiment on the new bed re-runs the E002 control FIRST (tuned
gamma on C-shape vs measured readout), not after a method is built. The
control order from the theory note's item 1 becomes the template for every
bed: controls before claims.

## 6. Housekeeping: my untracked notes

`wanger_duality_reply.md` and `wanger_e002_e007_review.md` are in my working
tree, uncommitted -- the git safety rule bars me from committing without an
explicit user ask in-session. Nothing hidden; flagging for push authority
so the loop can read them. This note joins them, same status.
