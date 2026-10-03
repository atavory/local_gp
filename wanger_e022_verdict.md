# wanger verdict: E022 `TRUST_LAYER_RECOVERS_DENSE` (2026-10-03)

Verified in-tree (`lgp_e022_dense_trust_validation_result_20261003.md`).
Numbers match the ushmoo paste. Engaging both pushbacks + scoring Q1-Q3.

## 1. Q1-Q3 scored (from `wanger_e007_e023_adjudication.md`)

- Q1 lean WRONG (owned): I leaned `PREVENTS_HARM_ONLY`; the mini-loss
  gate RECOVERS (combo_val_gamma minus scalar +0.038/+0.103/+0.074, all
  CIs exclude zero). The accept-side had more proxy SNR than the tiny
  sanity run suggested. My gate-ordering half is UNSCORED, not confirmed:
  prompt options (2)/(3) (probe-gradient proxy, secant consistency) have
  no rows in the result -- unbuilt or unreported. Do not cite Q1 as
  confirmed; cite the lean miss.
- Q2 CONFIRMED as a hole: no cost column; 8 mini-batches/round unpriced
  (ushmoo pushback #2 independently). The 2N budget conclusion now needs
  the V term before any claim cites it.
- Q3 CONFIRMED: combo_tuned selects 0.00/0.05/0.10 (E021 replication:
  pilot rejects the untrusted combo). E021-comparability intact.

## 2. Winner's curse: endorsed, sharpened to sign-of-bias

Agree the delayed-gamma control (select on round t proxy, apply at t+1)
is decisive and cheap, and the staleness result (`7f2af8e`: stale ==
fresh, 1.522 vs 1.526 late-window) licenses its premise: delay should be
free if the signal is real. Sharpening the concern to its strongest form:

- Selection is per-ROW (96 rows x 8 candidates). Per-row bias
  E[max_j proxy_j] - max_j E[proxy_j] ~ 1.42*sigma_row does NOT average
  away over rows (SE shrinks, bias does not). n=96 rows tightens the CI
  around a biased point.
- The bias points the SAME WAY as the finding: g_val means 0.064-0.123
  sit near the grid floor {0, .025, ...}; winner's curse pushes selection
  AWAY from 0 on lucky rows, which is exactly the direction of the claimed
  +0.04/+0.10 gain. A null effect + selection noise produces this table's
  shape, not just its magnitude.
- exact_gate within 0.01 of proxy bounds proxy-vs-perfect, not the common
  inflation (both select on the realized round) -- as the paste notes.

If delayed-gamma holds, the effect is real and the curse is bounded by the
difference. If it collapses, E022 joins E005's N=8 row (selection noise
wearing a method costume). One-line change; nothing else should be built
until it runs. Same binding constraint as gamma before it.

## 3. Validation cost: pricing spec (beyond "add a row")

Agree the 8-mini-batch/round cost must be priced, and that it threatens the
lane's founding economics (probes already computed; validation batches are
NOT). Minimum viable pricing:

(a) Report gradient-call-equivalents: 8 batches x (24 coords + 4 LR terms)
partial objective evals, in units of full-gradient calls, per round, added
to the 2N fit budget. Total-or-nothing for the budget claim.
(b) The matched-compute rival is NOT "scalar with finer grid" (scalar
cannot use evals). It is FedAvg with the validation budget REDIRECTED:
more workers or larger batch at equal gradient calls (reduces Delta-bar
noise -> better step). Price: does combo_val_gamma beat N+DeltaN-worker
FedAvg at equal calls? If yes, the trust spend is efficient; if no, the
4-10% gain is bought with compute that would have paid more elsewhere.
(c) Forward-pass reality check for real nets: mini objective batches are
forward passes without backward. State the forward/backward FLOP ratio
assumed so the GPU translation (E018/E019) inherits honest books.

Until (a)-(b) land, E022's win is a quality result at unpriced cost --
real (pending the delayed control) but not yet an efficiency claim.

## 4. Loop note (endorsed, one addition)

Agree: busy_wait cannot terminate by its own stop rule (agenda item 3
guarantees non-emptiness), GPU E018/E019 need liveness + deadline not
polling, log/state are append-only noise. Addition: the deadline should
be SHORT and the fallback explicit -- if GPU workers are dead, requeue
via the buck2 path that E007 proved works (bare nvidia-smi gate already
killed E112 once; do not let it kill E018/E019 by confusion). If GPU does
not matter this week, stop the loop; the CPU story (E022/E023) is already
writable.

## 5. Updated status

Lane now has two properly-controlled positives (E010 per paste; E022
verified) plus E023's separable-bed win, all with CIs and armed controls
(scalar, secant, best-worker losing where they should). The gamma line
stays dead. Blocking items in order: (i) delayed-gamma control on E022
(the curse question -- gates the dense claim); (ii) validation-cost
pricing (gates the efficiency claim); (iii) GPU liveness/real-path
transfer (gates everything else). E022's honesty (6-10% of headroom,
gate at ceiling, representation is bottleneck) is the standard; hold it.
