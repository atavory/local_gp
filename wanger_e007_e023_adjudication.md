# wanger adjudication: E007/E021/E022/E023 landings (2026-10-03)

Read: E007/E021/E022/E023 result notes + run logs in-tree, E022 prompt,
E007 full-gamma amendment, busy_wait `state.md`, Manifold dropboxes
(E022/E023 CPU results present; no E018/E019 GPU results).

## 1. P1-P4 scored (from `wanger_e002_e007_review.md`)

- P1 CONFIRMED: `full_gamma` loses to `fedavg_scale` (+0.109 loss, CI
  excludes 0). Unregularized catastrophe reproduces on real paths.
- P2 REFUTED: fallback_fraction 0.000000, not >10%. (Bed is a linear
  classifier — near-quadratic loss, curvature positive; the indefinite
  hole did not bind HERE. P2 stands as a prediction for real nets, not
  this bed.)
- P3 MOOT (stronger): auto did not "win flat" — it LOST (+0.060 vs
  fedavg_scale, +0.026 vs tuned, both CI-excluded). And pilot-selected
  source gamma was 0.050 with scale 1.900: the pilot nearly zeroed the
  correction itself. Same signature as E021's combo_tuned -> 0.000.
- P4 CONFIRMED: fresh small PyTorch run (1970-param linear), not E78
  ledger reuse.

E007 files `FEDAVG_SCALE_EXPLAINS`. The gamma line is now dead on BOTH
synthetic (E002) and real-path (E007) beds. My duality-reply §9 binding
constraint is satisfied — the gamma number came back negative. No
gamma work should be queued anywhere.

## 2. E021: the honest catastrophe (`DENSE_GEOMETRY_BREAKS_METHOD`)

Combo -22.671 at N=8 (destructive, huge CI), tuned-gamma upper bound
selects 0.00-0.05 (pilot rejects the correction), oracle-H headroom
2.66-3.30 (bed has room; estimator cannot use it). True-subspace combo
less destructive but still below scalar. Best-worker ~raw, secants weak.
Correct verdict, correct representation call (low-rank-plus-diagonal
needs a stricter trust layer). Two wanger notes:

(a) The pilot-rejects-correction signature (gamma -> 0) now appears in
E007 (0.050) and E021 (0.00-0.05) independently. Promote it to a
STOPRULE: any future correction arm whose pilot-tuned gamma selects the
grid floor is filed as pilot-rejected without further argument. It is
the cheapest negative in the repo.
(b) "Residual probes deceptively small" (shrinkage fails to fire) is the
mechanism. It matters for E022 below: gates built on the SAME residual
probes inherit the deception.

## 3. E023: live method signal on separable beds (`TWO_N_RANDOM_CONFIRMED`)

2N random probes beat tuned scalar at N>=32 with tight CIs (N=32:
+0.331/+0.394; N=128: +0.478/+0.621), near matched full-random; reps=1
sufficient (not `PROBE_REPS_CRITICAL`); N=8 still wants full budget.
Controls all hold: scalar well below, secants DESTRUCTIVE (negatives:
-8.15/-6.33/-4.49 -- stronger negative control than quadratic's 1.00),
best-worker below FedAvg (elitist fallback dead here), risk-vs-tuned
~0 (residual shrinkage ties tuned gamma where the shape wins).
Boundary now mapped: separable (diag/rotated/block) geometry WORKS,
dense correlated geometry BREAKS (E021). Both directions have CIs.
This is the first live-method result with hard controls in either
project. The claim-boundary paste ("controls still hard enough?") is
answered yes for this bed: scalar, secant, best-worker, full/hardcoded
all ran and the win survived all four.

## 4. E022 design review + pre-registered read-out

Prompt is well-built (K-step operator fixed, oracle gates labeled as
upper bounds, INVALID on target leakage, paired-CI requirements). Three
wanger probes, logged before the result:

(a) Gate-quality ordering: E021's failure was deceptive residual probes.
E022 trust options (2) probe-gradient proxy and (3) secant consistency
are built on the SAME residual information -- predict they fail or add
nothing (same deception, one level up). Only option (1) held-out
mini-loss proxy measures the OBJECTIVE independently. Pre-register:
(1) is the real test; (2)/(3) succeeding would contradict E021's
mechanism and demand an explanation, not a celebration.
(b) Budget-accounting hole: E022's winning gate (if any) consumes
held-out probes/loss evals. E023's "2N budget" counts FIT probes only.
If the trust gate needs substantial validation budget, 2N must be
restated as (2N fit + V validation). The read-out must price V or the
budget conclusion leaks. State the total or the claim is incomplete.
(c) Gamma-specter on combo_val_gamma: grid-selected gamma on held-out
proxy is pilot-tuning with a different name. Legitimate IF the grid is
tiny and the proxy is cheap and non-oracle -- the prompt enforces this;
the read-out must verify the proxy cost, else E002's lesson re-applies
(tuned constant eats the method, here BY CONSTRUCTION as the method).

Prediction log for E022 scoring:
- Q1: (2)/(3) fail-or-null (deception inherits); (1) prevents harm;
  lean `TRUST_LAYER_PREVENTS_HARM_ONLY` over `RECOVERS_DENSE` (oracle
  gate managed only +0.026 over scalar on the tiny sanity run; the
  accept-side needs proxy SNR this bed may not give).
- Q2: winning gate's TOTAL probe cost (fit + validation) exceeds 2N;
  the 2N conclusion gets restated with V priced.
- Q3: combo_tuned again selects ~0 (E021 replication). If it selects
  >>0, the bed shifted and E021-comparability breaks -- check first.

## 5. E022 read-out against the pre-registration

E022 files `TRUST_LAYER_RECOVERS_DENSE`, and the result is better than my
lean. Held-out mini-loss `combo_val_gamma` beats scalar with positive CIs:
N=8 +0.038, N=32 +0.103, N=128 +0.074. The E021 residual combo is still
catastrophic (-6.028/-5.464/-1.793), so the trust layer is not decorative; it
is the thing preventing the crash.

Pre-registration accounting:

- Q1: partly wrong in the useful direction. The held-out mini-loss proxy (1)
  does more than prevent harm; it recovers a small real gain. Probe/secant
  trust variants were not the winning story.
- Q2: still open. The winning gate uses `LOSS_PROXY_REPS=8`,
  `coord_batch=24`, and `lr_batch=4`. The 2N probe-budget claim must be stated
  as fit probes plus validation-proxy cost when dense trust is active.
- Q3: confirmed. Combo tuned remains near the floor (0.00/0.05/0.10), matching
  the pilot-rejects-correction signature from E007/E021.

Verdict: dense geometry is not dead, but it is trust-gated and still far from
oracle-H (2.476/2.661/2.421). The next dense task is representation/proxy-cost
engineering, not relaxing the gate.

## 6. Status

E022 and E023 are now imported. E018/E019 GPU: prompts + bundles only, nothing
back. Nothing here reopens gamma (everything closes it further). The open list:
GPU decides real-path transfer; dense follow-up should improve the trusted
low-rank-plus-diagonal representation and price validation cost. Do not invent
results.
