# Local GP Research Control Board

## Prime Directive

Do not repeat the Double Fan failure mode.

The goal is not to invent another controller and then explain away a loss. The
old same-point cloud-momentum line is frozen in
`double_fan_freeze_20261003.md`: reliable measurements, no surviving optimizer
product. The live paper starts from the different observation available in
K>1 local SGD: workers move through a neighborhood and thereby produce local
map data.

The first question is whether local worker paths or explicit probe pairs contain
held-out predictive information that simple schedules, scalar FedAvg-scale
baselines, server/time secants, and best-worker fallback do not contain.

## Project Boundary

This project is separate from:

- same-point cloud momentum;
- dense centered-R preconditioning;
- EB-to-zero gradient shrinkage;
- SNR compression;
- worker fault sentinels.

Those branches produced useful diagnostics but no optimizer product. Local GP
gets a clean board because it uses different observations: K-step worker paths,
not only same-point gradient scatter.

The one sentence bridge is:

```text
alignment estimates noise; misalignment can estimate curvature.
```

That bridge is motivation only. It is not evidence for an optimizer claim.

## Hypothesis

For K>1 local SGD, worker paths reveal local geometry. A GP or kernel predictor
over path/probe descriptors may predict:

- which proposal direction is safer;
- how much of local history to trust;
- which scalar action from a small action grid transfers;
- whether a round-only schedule is about to fail.

The first claim is prediction. The optimizer claim comes later.

Current paper hypothesis, after E014-E021: local path/probe observations should
be treated as a noisy estimate of the K-step update map, not as a fresh Newton
step and not as a GP over all parameters. The corrected quadratic operator is
\[
  [I-(I-\eta H)^K]^{-1}\Delta_K.
\]
The outside controller is a stochastic trust-region method with a ratio test,
not Optuna/TPE over high-dimensional actions.

## Evidence Required

Before closed loop, require an offline held-out prediction win:

- source-only fit;
- held-out target seeds/settings;
- paired confidence intervals;
- shuffled-feature control;
- round-only and path-norm baselines;
- target oracle headroom diagnostic.

## Current Status

E001 has run and returned `SCHEDULE_EXPLAINS`.

Summary:

- existing E78 K>1 ledger only; no fresh training and no controller;
- target K=8 oracle headroom over frozen source-selected constant:
  0.001031 [0.000504, 0.001615];
- selected `rbf_path_krr` regret: 0.001031 [0.000513, 0.001618];
- round-only regret: 0.001031;
- GP/KRR advantage over round-only: 0.000000 [0.000000, 0.000000].

Interpretation: the cheap scalar path descriptors in E78 do not add held-out
prediction over round shape. This blocks a closed-loop optimizer based on E001.
The broader local-GP idea remains open only if we collect richer information:
fuller per-worker path vectors or shared-probe function-space features.

Operational note: `devvm23211` can run GPU code if tasks use Buck2
`gpu_python_binary(..., keep_gpu_sections = True)` or
`-c fbcode.enable_gpu_sections=true`. Bare `nvidia-smi` can fail even when the
Buck2 GPU path works.

E002 has run locally and returned `TUNED_GAMMA_EXPLAINS`.

Summary:

- the diagonal-quadratic local-trajectory curvature headroom is real;
- oracle-C remains around 1.20x tuned-scale FedAvg;
- the measured within-round readout is 1.124/1.141/1.167x at N=8/32/128;
- a one-scalar tuned gamma control applied to the same measured correction
  shape scores 1.127/1.146/1.172x.

Interpretation: intermediate local points expose a useful correction shape, but
the current posterior readout is explained by tuned shrinkage. This blocks a
closed-loop method claim from the current readout. The next gate must show that
a principled Bayesian/GP evidence rule can set shrinkage at least as well as
tuned gamma without target tuning.

E003 built the first optimizer artifact:
`local_gp.optimizer.LocalGPAggregator`.

Synthetic diagonal-quadratic smoke:

- N=8: local GP/base 1.063, oracle-C/base 1.190;
- N=32: local GP/base 1.132, oracle-C/base 1.233;
- N=128: local GP/base 1.148, oracle-C/base 1.207.

Interpretation: the aggregator is now concrete and positive on the synthetic
bed, but it remains gamma-tuned and synthetic. It is an object to test, not a
real-data method claim.

E004 returned `INVALID`.

The worker did not run workload code: the external repository / Mr. Gitties
legitimacy gate failed on the worker. This is a plumbing failure, not evidence
against the optimizer. The replacement real-path gate must be self-contained,
for example via a Manifold source bundle, and must not depend on worker GitHub
access.

E007 has one completed real-path shard and returned `FEDAVG_SCALE_EXPLAINS`.

Summary:

- workload: CUDA MNIST local-SGD paths, N=8, K=8, eta=0.2, linear classifier;
- selected source gamma: 0.050;
- selected source FedAvg scale: 1.900;
- target one-step validation loss:
  - FedAvg scale: 0.396898;
  - tuned gamma: 0.430577;
  - auto gamma: 0.456850;
  - full gamma: 0.506164;
  - raw FedAvg: 0.431406.

Interpretation: synthetic auto-gamma did not transfer. More strongly, tuned
FedAvg scale beat the gamma arms on held-out target loss. This is a one-shard
real-path gate, not a closed-loop verdict; the remaining E007 worker is still
running at time of recording.

E005 has run locally and returned `GAMMA_FLAT_ON_SYNTHETIC`.

Summary:

- `gamma=None` now estimates shrinkage from split-half reliability of the
  correction vector on each round;
- held-out synthetic ratios over FedAvg:
  - N=8: tuned gamma 1.045, auto-gamma 1.065, oracle 1.171;
  - N=32: tuned gamma 1.143, auto-gamma 1.142, oracle 1.247;
  - N=128: tuned gamma 1.168, auto-gamma 1.163, oracle 1.248.

Interpretation: auto-gamma matching tuned gamma is not a contribution on this
bed because hardcoded `gamma=1` also ties. The gamma objective is flat enough
that automating the knob is not meaningful evidence.

E006 has run locally and returned `GROUP_GAMMA_FLAT_ON_SYNTHETIC`.

Summary:

- compared scalar auto-gamma to 4-bin and 8-bin group-wise auto-gamma;
- held-out synthetic ratios over FedAvg:
  - N=8: tuned 1.054, auto1 1.072, auto4 1.081, auto8 1.084;
  - N=32: tuned 1.129, auto1 1.129, auto4 1.127, auto8 1.126;
  - N=128: tuned 1.138, auto1 1.135, auto4 1.135, auto8 1.135.

Interpretation: group-wise shrinkage also mostly inherits the flat gamma axis.
It gives a small low-N bump but does not justify expanding the action layer.
The gamma line should stop unless a real-path result shows that `full_gamma`
fails.

The durable positive signal is elsewhere: intermediate local points carry
curvature information that server-side secants did not capture on the synthetic
bed. That secant gap, not gamma automation, is the next target.

E008 has run locally and returned `FULL_CORRECTION_UNSTABLE`.

Summary on a nonlinear/heterogeneous synthetic bed:

- local full correction/FedAvg:
  - N=8: 0.019 +/- 0.282;
  - N=32: -1.490 +/- 0.642;
  - N=128: -8.054 +/- 3.262;
- auto-gamma/FedAvg:
  - N=8: 0.746 +/- 0.170;
  - N=32: -0.134 +/- 0.337;
  - N=128: -7.268 +/- 3.270;
- server secant/FedAvg:
  - N=8: 0.880 +/- 0.146;
  - N=32: 1.002 +/- 0.114;
  - N=128: 1.106 +/- 0.091.

Interpretation: raw exact-quadratic correction does not survive nonlinearity /
heterogeneity. It overcorrects badly.

E009 has run locally and returned `CONSERVATIVE_GAMMA_NEEDED`.

Summary on the same nonlinear/heterogeneous bed:

- source-tuned conservative gamma is 0.40/0.35/0.30 at N=8/32/128;
- tuned gamma/FedAvg is 1.460/1.473/1.501;
- auto-gamma/FedAvg is 1.124/0.387/-4.449;
- full gamma/FedAvg is 0.598/-0.398/-4.871;
- server secant/FedAvg is 0.936/0.954/1.066.

Interpretation: the secant gap is real on a bed that can tell actions apart.
Intermediate local paths contain useful correction shape that server-side
secants do not recover. But the current auto-gamma rule is wrong: it measures
estimator agreement, not model-bias risk. The next method must shrink based on
nonlinearity / heterogeneity risk, not only split-half reliability.

E010 has run locally and returned `LOCAL_PATH_SHAPE_BEATS_SCALAR`.

Summary on the same nonlinear/heterogeneous bed:

- N=8:
  - tuned scalar/FedAvg: 1.089 +/- 0.052;
  - local-path shape/FedAvg: 1.391 +/- 0.164;
  - shape minus scalar: 0.302 +/- 0.115;
  - server secant/FedAvg: 0.878 +/- 0.182.
- N=32:
  - tuned scalar/FedAvg: 1.100 +/- 0.032;
  - local-path shape/FedAvg: 1.674 +/- 0.243;
  - shape minus scalar: 0.574 +/- 0.211;
  - server secant/FedAvg: 0.897 +/- 0.183.
- N=128:
  - tuned scalar/FedAvg: 1.066 +/- 0.026;
  - local-path shape/FedAvg: 1.468 +/- 0.099;
  - shape minus scalar: 0.402 +/- 0.074;
  - server secant/FedAvg: 1.043 +/- 0.131.

Interpretation: this is the first clean positive in the local-GP lane. The
local-path correction shape beats both tuned scalar scale and server-side
secants on a nonlinear/heterogeneous bed. The live method is not full GP and
not auto-gamma. It is local-path shape plus conservative, bias-aware shrinkage.

E011-E013 opened a separate event-triggered multisecant branch.

Motivation: under a pay-by-wire cost model, workers should run silently until a
drift/quorum event, then send full useful state. The controller forms centered
worker secant pairs
\[
  s_i=x_i-\bar x,\qquad y_i=g_i-\bar g
\]
and computes the consolidation step. No \(d^2\) object is formed.

Results:

- E011 (`PASSIVE_DRIFT_SECANTS_WEAK`): passive stochastic drift did not produce
  useful spatial secants. One-pair server BB beat spatial arms.
- E012 (`SEEDED_PROBES_NOT_ENOUGH_WITH_SCALAR_BB`): controller-seeded probe
  offsets gave a small spatial-BB bump over scalar extrapolation, but still lost
  to server BB.
- E013 (`MULTISECANT_WORKS_WITH_LOW_NOISE_ONLY`): actual spatial L-BFGS
  multisecants on a rotated block quadratic failed with ordinary stochastic
  secants, but worked in the noiseless sanity check:
  - N=32: space L-BFGS 1.170 vs time L-BFGS 1.098;
  - N=128: space L-BFGS 1.259 vs time L-BFGS 1.103.

Interpretation: the architecture is mechanically sound only when secant SNR is
high enough. Ordinary independent stochastic gradients make the spatial secant
bank too noisy. The next event-triggered experiment must use noise-controlled
secants: same-batch probes, repeated event gradients, a shared probe set, or a
large-batch/large-N regime.

E014 corrected the event-branch operator and returned
`DYNAMICS_AWARE_PROBES_WIN_ON_EXACT_QUADRATIC`.

E013's plain L-BFGS step was the wrong consolidation: K local steps already
apply part of the Newton correction. On a quadratic with no momentum, the
FedAvg displacement is
\[
  \Delta_K=-H^{-1}[I-(I-\eta H)^K]g_0,
\]
so the remaining correction is
\[
  [I-(I-\eta H)^K]^{-1}\Delta_K.
\]

Summary on the rotated-block quadratic with noise-controlled `N*K` probes:

- N=8: tuned scalar 1.054, directed probe 1.332, random probe 1.358;
- N=32: tuned scalar 1.047, directed probe 1.301, random probe 1.301;
- N=128: tuned scalar 1.051, directed probe 1.326, random probe 1.326.

The centered local-trajectory estimate is weaker but improves with worker count
(1.061/1.097/1.214). This keeps the rank/noise concern active while validating
the corrected operator. E014 is a Stage-0 mechanism positive, not a product
claim: it relies on an exact quadratic, known 2D block structure, extra probe
points, and repeated gradients.

Current local-GP state:

- gamma automation is dead;
- plain BB/L-BFGS consolidation is dead;
- local-path shape is live from E010;
- dynamics-aware correction is live from E014;
- the next useful gate is the corrected operator on the nonlinear/heterogeneous
  E008-E010 bed, then a real-path gate only if that survives tuned scalar scale
  and server/time secants.

E015 ran that CPU gate and returned
`DYNAMICS_AWARE_PROBES_SURVIVE_NONLINEAR_BED`.

Summary on the nonlinear/heterogeneous bed:

- N=8: tuned scalar 1.069, random probe 1.306, coordinate probe 1.134;
- N=32: tuned scalar 1.053, random probe 1.445, coordinate probe 1.424;
- N=128: tuned scalar 1.101, random probe 2.009, coordinate probe 2.034.

Server/time secants remain weak (0.911/0.974/1.113), and path-only centered
trajectory pairs select gamma zero and tie FedAvg. The CPU-positive mechanism is
therefore probe-fed local-map correction, not plain worker-scatter
multisecants.

E016 ran the rotation gate and returned
`DYNAMICS_AWARE_SURVIVES_ROTATION`.

Summary on rotated 2D nonlinear blocks:

- N=8: tuned scalar 1.032, random probe 1.254, directed probe 1.268;
- N=32: tuned scalar 1.057, random probe 1.497, directed probe 1.497;
- N=128: tuned scalar 1.132, random probe 2.124, directed probe 2.122.

The best-worker oracle fallback does not explain the result (0.729/0.799/0.876),
server/time secants are unstable and strongly negative, and path-only pairs
again select gamma zero. The E015 gain is therefore not a diagonal-coordinate
artifact on this CPU bed.

E017 ran the shrinkage/fallback gate and returned `RISK_SHRINKAGE_WORKS`.

Summary on the E015 nonlinear/heterogeneous bed:

- N=8: tuned scalar 1.075, tuned probe 1.288, risk probe 1.226;
- N=32: tuned scalar 1.070, tuned probe 1.554, risk probe 1.460;
- N=128: tuned scalar 1.127, tuned probe 2.039, risk probe 1.773.

Risk-aware gamma is selected from split-probe residuals, not an arbitrary gamma
grid, and stays within roughly 5%, 6%, and 13% of the tuned-gamma probe arm. Full
gamma is destructive at N=8, so the residual rule is doing real shrinkage. The
best-worker oracle fallback again loses.

E020 ran the probe-budget / active-design gate and returned
`RANDOM_BUDGET_REQUIRED`.

Summary on the rotated nonlinear block bed with residual shrinkage:

- full `N*K` random probes still beat tuned scalar scale:
  - N=8: full random 1.252 vs scalar 1.061;
  - N=32: full random 1.480 vs scalar 1.072;
  - N=128: full random 1.941 vs scalar 1.115.
- the current `value_weighted` active policy usually underperforms matched
  random probes;
- `uncertainty_value` is the better active variant and sometimes ties matched
  random, but it does not provide a robust budget reduction;
- the cheapest plausible policy is still random probes, around `2N` when
  `N >= 32`, with full `N*K` for very small `N`.

Interpretation: random probe budget is a real product cost. The live mechanism
does not require active design v1, and active design v1 should not replace
random probes.

E023 ran the random-probe budget confirmation gate and returned
`TWO_N_RANDOM_CONFIRMED`.

Summary on the same rotated nonlinear block bed, now with eight held-out reps
and `probe_reps in {1, 4}`:

- at N=32, `2N` random probes beat tuned scalar by 0.331 +/- 0.048 with
  `probe_reps=1` and by 0.394 +/- 0.045 with `probe_reps=4`;
- at N=128, `2N` random probes beat tuned scalar by 0.478 +/- 0.089 with
  `probe_reps=1` and by 0.621 +/- 0.115 with `probe_reps=4`;
- matched against full `N*K` random probes, the `2N` gaps are within the
  requested 10-15% band for N=32/128;
- at N=8, `2N` is still not clearly enough, while full `N*K` remains stable;
- `probe_reps=1` does not fail, although `probe_reps=4` improves the absolute
  ratio when extra gradient repeats are cheap.

Interpretation: the default CPU-bed random budget is now `2N` for `N >= 32`,
with full `N*K` reserved for very small worker counts. Probe repeats are a
quality/variance knob, not a requirement for the basic budget result.

E021 ran the dense low-rank / correlated-noise stress and returned
`DENSE_GEOMETRY_BREAKS_METHOD`.

Summary:

- tuned scalar remains near 1.04-1.05 on the dense bed;
- oracle dense-H correction reaches 2.658/3.304/2.778 at N=8/32/128, so the
  bed has large non-scalar headroom;
- diagonal, random low-rank, learned low-rank, and true-subspace low-rank arms
  do not beat tuned scalar with positive confidence;
- learned low-rank-plus-diagonal with E017 shrinkage is unsafe and can be
  destructive;
- pilot-tuned combined gamma nearly zeros the combined learned update, which is
  itself evidence that the representation needs a stricter trust gate.

Interpretation: dense low-rank-plus-diagonal geometry is the next hard problem.
The current residual shrinkage is not enough under correlated low-rank noise.
Do not ship the combined estimator without held-out loss/probe validation or an
equivalent trust layer that can reject the correction.

E022 ran the dense trust-layer gate and returned
`TRUST_LAYER_RECOVERS_DENSE`.

Summary on the same dense low-rank correlated-noise bed:

- the residual-shrunk learned combo remains destructive:
  - N=8: -6.028 +/- 3.875;
  - N=32: -5.464 +/- 8.616;
  - N=128: -1.793 +/- 2.381.
- rowwise `combo_val_gamma`, selected by held-out mini-loss proxy, beats tuned
  scalar with positive CIs:
  - N=8: +0.038 +/- 0.015;
  - N=32: +0.103 +/- 0.039;
  - N=128: +0.074 +/- 0.020.
- the accept/reject gate mostly prevents harm but lands near scalar;
- oracle-H is still much larger at 2.476/2.661/2.421, so most dense headroom is
  not recovered by the current representation;
- best-worker fallback and server/time secants do not explain the trusted gain.

Interpretation: dense geometry is not intrinsically dead, but it is only safe
behind rowwise held-out validation. The next dense work should improve
subspace/model identification and explicitly account for validation-proxy cost.

Updated local-GP state:

- gamma automation from the old aggregator line is dead;
- plain BB/L-BFGS consolidation is dead;
- path-only worker scatter is not enough;
- probe-fed dynamics-aware local-map correction is alive across exact,
  nonlinear, and rotated nonlinear CPU beds;
- risk-aware residual shrinkage has replaced target gamma tuning on the current
  CPU bed;
- random probe budget remains material; active design v1 is not enough to cut
  it reliably, but E023 confirms `2N` random probes as the default for
  `N >= 32` on the rotated CPU bed;
- dense low-rank-plus-diagonal geometry has oracle headroom; E022 shows a
  held-out validation-gamma layer can recover a small safe gain, but the
  representation still misses most oracle-H benefit;
- the controller should use empirical-Bayes local-map fitting plus stochastic
  trust-region adaptation, not Optuna/TPE over high-dimensional actions;
- the next useful work is real-path plumbing (E018/E019), better dense
  subspace/model identification, validation-budget accounting, and
  function-space probes.

## Success Labels

- `PATH_SIGNAL_CONFIRMED`: GP/kernel predictor beats round-only and path-norm
  baselines on held-out proposal-quality or oracle-action prediction.
- `TRANSFER_CONFIRMED`: predictor trained on source setting improves held-out
  target prediction and preserves performance under a K/model/batch shift.
- `CLOSED_LOOP_READY`: offline prediction win is large enough to justify one
  constrained optimizer run.

## Failure Labels

- `SCHEDULE_EXPLAINS`: round-only schedule matches the GP.
- `PATH_NORM_EXPLAINS`: simple path length/norm features match the GP.
- `NO_ORACLE_HEADROOM`: even the target oracle action barely beats the frozen
  baseline.
- `NO_HELDOUT_SIGNAL`: source fit works but held-out prediction fails.
- `INVALID`: target leakage, no held-out split, no target oracle diagnostic, or
  no baseline stronger than random.

## Near-Term Plan

1. Do not run a closed-loop optimizer from E001.
2. Do not run a closed-loop optimizer from E002's measured readout.
3. Do not continue the gamma line unless `full_gamma` fails. Any E007 positive
   must beat or at least improve over `full_gamma`, not merely tuned gamma.
4. Replace E004 with a self-contained real-path gate that does not depend on
   worker GitHub/Mr. Gitties access.
5. In the real-path gate, compare FedAvg/source-tuned scale, pilot-tuned gamma,
   source-free auto-gamma, and full gamma. `FULL_GAMMA_EXPLAINS` is a null for
   gamma estimation.
6. E010/E015-E017 show the method target: noise-controlled probes estimate a
   dynamics-aware local-map correction that beats tuned scalar scale on CPU
   synthetic beds.
7. Richer real-data logging remains needed: probe gradients or shared-probe
   function-space features, not only scalar endpoint summaries.
8. Event-triggered multisecant branch: stop plain BB/L-BFGS consolidation; use
   the K-step local-map operator.
9. Run E018 as a real-path readiness smoke, not a product claim.
10. Use residual-shrunk random probes as the default budget policy for now:
    around `2N` when `N >= 32`, full `N*K` for very small `N`.
11. For dense low-rank-plus-diagonal geometry, add a held-out validation/trust
    gate before applying the combined update.
12. Do not run a closed-loop optimizer before a held-out real-path/control gate
   passes against tuned scalar scale and best-worker fallback.
13. Rewrite the paper around the reset thesis:
    - Double Fan / same-point cloud momentum is archived as a failed approach;
    - FedAvg is the infinite-lengthscale / averaging baseline;
    - local drift/probes estimate local-map curvature unavailable to
      server/time secants;
    - the product path is dynamics-aware local-map correction plus a
      trust-region controller;
    - dense correlated geometry is the current failure boundary.
