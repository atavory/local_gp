# wanger review: E002 gamma-collapse, E005/E006 synth auto-gamma, E007 gate

Read: `lgp_e002_gamma_control_result`, `lgp_e005/006_*_result`,
`local_gp/optimizer.py` fallback paths, E007 prompt (via bandha paste).
E007 running on devvm23211 + devvm53617 at time of writing.

## 1. E002: collapse exactly as pre-registered (`TUNED_GAMMA_EXPLAINS`)

Tuned one-scalar gamma matches/beats the measured readout at every N
(1.127 vs 1.124; 1.146 vs 1.141; 1.172 vs 1.167). My read-out table lands
on arm2 == arm1: shape matters, measurement of the readout does not. The
result note's framing is correct (headroom real at ~1.20, secants dead at
~1.00, readout is decoration). Two additions:

(a) Arm 3 of my table (tuned gamma on CRUDE pooled-lambda proxy) was not
run. The split is now one level down: does the C(lamhat) SHAPE beat a
shape-free scalar? Until that arm exists, "intermediates contain
information" is supported only against server secants, not against the
cheapest shape-free rival. Cheap, same bed -- run it before E007 lands so
the read-out vocabulary is complete.
(b) Selected gamma varies with N (1.10 / 0.75 / 0.80). Gamma itself needs
per-setting tuning -- that IS the cost auto-gamma must remove. Quote this
when E007 claims tuning-reduction: the reduction is (pilot grid over gamma
per setting) -> (zero), and its value scales with how much gamma moves.

## 2. E005/E006: no CIs anywhere; N=8-only effects match the bias signature

Both result tables are point estimates with no intervals. Consequences:

(a) E005 "auto beats tuned at N=8" (1.065 vs 1.045, delta 0.02) sits inside
any plausible noise band -- the whole N=8 row spans [1.045, 1.065]. And
mean auto gamma (0.527) vs tuned (0.80) with near-identical outcomes says
the gamma-curve is FLAT at N=8: matching tuned means little where everything
matches everything. Label the N=8 win noise-or-bias until CIs + the debias
experiment land.
(b) The legit E005 result is the TIES at N=32/128 (1.142 vs 1.143; 1.163 vs
1.168) -- tuning-reduction with no pilot grid. But "ties" also needs the CI
rule: CI of (auto - tuned) contains 0 AND tuned beats base with CI
excluding 0 (else joint null). State it; E007 inherits the same rule.
(c) E006's group bump exists ONLY at N=8 (1.084 vs 1.072 vs 1.054), flat at
N=32/128. N=8-only effects are exactly where the §6 O(1/N) bias luck lives
(worst at small N, causal/debiased wins nowhere). Scalar MVP is the right
call; "secondary low-N variant" should ship only with a debias control,
or not at all.
(d) General: the E002 grid (gamma 0.00-2.00 by 0.05) already contains the
sensitivity curve. Quote gain-vs-gamma flatness per N before E007 -- if the
curve is flat, auto-gamma's bar is low and its win is small; if peaked, the
bar is high and a tie is real. One offline paragraph, no new runs.

## 3. E007 gate review: good prompt, three gaps

Endorsed: four arms incl. `full_gamma=1.0` guardrail (expect it to
reproduce the unregularized catastrophe -- that is its job); labels incl.
`FEDAVG_SCALE_EXPLAINS` (the double_fan killer, correctly carried over);
workload fallback order; buck2 GPU note (no bare nvidia-smi); metrics ask
for fallback fraction (see below).

(a) Indefinite handling: PRESENT in code (`_plug_in_correction` returns 1.0
for lam <= min_curvature/nonfinite; `_shrink_coefficients` maps c0 <= 0 to
1.0; `negative_or_unstable_fraction` + `fallback_fraction` tracked). My
duality-reply §5 warning is addressed. E007 read-out must REPORT both
fractions: if fallback_fraction is large on real paths, the "method" is
mostly FedAvg with decoration, and the effective action set is smaller than
claimed.
(b) Missing crude-proxy arm (§1a above, adapted): tuned gamma on a
single-pooled-lambda C, same workload. Without it, `AUTO_GAMMA_REAL_TIES_TUNED`
cannot separate "shape measured well" from "any gamma-shaped scaling ties."
Cheap (same paths, one more aggregation) -- add before workers finish, or
score it posthoc from logged paths if the prompt's arms are fixed.
(c) Statistical tie rule (§2b): "ties" needs the CI definition in the
prompt, plus a stated seed count. As written, ties will be declared by
eyeball. Minimum fix: tie = paired-95%-CI of (auto - tuned) delta contains
0 while tuned-vs-base excludes 0 in tuned's favor.
(d) Sequencing: the debias experiment (theory note item 2) should land
BEFORE E007's read-out, so auto_gamma (which inherits lamhat bias) gets
read against the bias baseline. If E007 auto wins and debiased-lamhat auto
does not exist yet, the win is unconditioned on the known accident --
flag it as such rather than calling it adaptation.

## 4. Prediction log (for scoring when E007 lands)

P1: `full_gamma=1.0` loses to `fedavg_scale` (unregularized correction
catastrophe reproduces on real paths). P2: fallback_fraction is
substantial (>10%) on real paths (negative/near-zero lamhat common off the
quadratic bed). P3: if auto_gamma "wins," the win concentrates where the
gamma-curve is flat (check via E002-grid sensitivity) -- i.e. it will read
as tuning-reduction, not adaptation, once conditioned. P4: K>1 ledger reuse
fails (E78 lacks intermediates) and E007 runs fresh small paths or returns
`NO_REAL_PATH_HARNESS` on at least one worker.
