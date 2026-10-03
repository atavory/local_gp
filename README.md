# Local GP

Local GP is a fresh research lane for worker-local SGD.

The project is intentionally separate from Double Fan / cloud momentum. The old
project tried to turn same-point worker variance into a first-moment optimizer
controller. This project starts from a different object:

> In local SGD, workers do not only produce noisy gradients. They trace short
> local paths from the same synchronized point. Those paths are observations of
> a local update field.

The old Double Fan direction is frozen in
`double_fan_freeze_20261003.md`. Its useful lesson is negative and precise:
same-point alignment estimates noise, but it does not estimate curvature or
provide an optimizer action by itself. The new paper should not be a rescue of
that line.

The central question is whether those path observations can predict which
proposal, trust action, or synchronization update will work on held-out rounds
and seeds, before paying for a target-setting grid search.

## Current Claim Status

No optimizer claim.

Paper reset, 2026-10-03: the manuscript is now built around local SGD drift as
curvature-probe data. The live mechanism is not "GP over parameters" and not
Optuna over a sync threshold. It is:

```text
local/probe gradient pairs
    -> dynamics-aware estimate of the K-step update map
    -> residual or held-out trust gate
    -> stochastic trust-region consolidation
```

The system is data-rich, not data-poor: one event can produce many gradient
observations. We therefore import the trust-region, batching, pruning, and async
ideas from BO-like systems, but not their high-dimensional black-box samplers.

E001 returned `SCHEDULE_EXPLAINS`: on the existing E78 local-SGD ledger, an RBF
kernel model over scalar path descriptors matched the round-only baseline
exactly on held-out K=8 regret. The target had measurable oracle headroom, but
the logged scalar descriptors did not predict it beyond round shape.

E002 returned `TUNED_GAMMA_EXPLAINS`: on the synthetic diagonal-quadratic bed,
intermediate local points exposed a real curvature correction shape, but a
one-scalar tuned gamma shrinkage matched or slightly beat the measured posterior
readout. This means the current readout is not yet a method.

E003 built the first actual optimizer artifact:
`local_gp.optimizer.LocalGPAggregator`. On the synthetic diagonal-quadratic
smoke test it improves over FedAvg by 1.063x, 1.132x, and 1.148x at
N=8/32/128. This is still synthetic and gamma-tuned, but now there is a concrete
aggregator to test.

E004 was invalid because the worker could not access the repository / Mr.
Gitties path. It did not run workload code.

E007 replaced it with a self-contained bundle and produced one completed
real-path shard: `FEDAVG_SCALE_EXPLAINS`. On CUDA MNIST local-SGD paths
(N=8, K=8), source-tuned FedAvg scale had lower held-out one-step validation
loss than tuned gamma, auto-gamma, full gamma, and raw FedAvg. This is a
negative for the gamma aggregator on a real path.

E005 added source-free shrinkage: `LocalGPConfig(gamma=None)` estimates the
correction shrinkage from split-half reliability on each round. On the same
synthetic bed it matches pilot-tuned gamma:

- N=8: tuned 1.045x, auto 1.065x over FedAvg;
- N=32: tuned 1.143x, auto 1.142x;
- N=128: tuned 1.168x, auto 1.163x.

But this is now labeled `GAMMA_FLAT_ON_SYNTHETIC`, not a win: hardcoded
`gamma=1` also ties the tuned and auto-gamma arms. The gamma knob is too flat on
this bed to make auto-selection a contribution.

E006 tested group-wise gamma. It gave a small N=8 bump but no benefit at
N=32/128, so it is also a flat-gamma result rather than a new action-space
method.

The durable positive signal is the secant gap from the theory note: intermediate
local points carry curvature information that server-side secants did not
capture. That is the next target.

E008/E009 moved onto a nonlinear and heterogeneous synthetic bed. Full
exact-quadratic correction fails badly there, but a conservative tuned gamma
wins strongly:

- tuned gamma/FedAvg: 1.460x, 1.473x, 1.501x at N=8/32/128;
- server secant/FedAvg: 0.936x, 0.954x, 1.066x;
- full gamma is destructive.

So the useful signal is not gamma automation. It is the secant gap plus the need
for bias-aware shrinkage: local paths expose a correction shape that server-side
secants miss, but the correction must shrink when nonlinearity / heterogeneity
violates the exact-quadratic model.

E010 added the missing tuned scalar-scale control on that same bed. The
local-path shape still wins:

- N=8: tuned scalar 1.089x, local-path shape 1.391x;
- N=32: tuned scalar 1.100x, local-path shape 1.674x;
- N=128: tuned scalar 1.066x, local-path shape 1.468x.

This is the first clean positive: not full GP, not auto-gamma, not scalar LR,
and not server-side secants. The live claim is that intermediate local worker
paths contain useful correction shape unavailable to FedAvg-scale and
across-round secants.

The next useful experiment should build a bias-aware shrinkage rule for that
shape, with tuned scalar scale, `full_gamma`, and server-side secants as null
controls. A closed loop is not justified from E001 or E002.

A separate event-triggered multisecant branch is now recorded. Under a
pay-by-wire cost model, workers run silently until a drift/quorum event, then
send full positions and gradients. The controller forms centered secant pairs
and chooses the consolidation step. E011-E013 showed that plain spatial
L-BFGS/BB consolidation is not the product: ordinary independent stochastic
gradients make the secant bank too noisy, and a fresh quasi-Newton step
double-applies curvature already used by the K local steps.

E014 fixed that operator. For K-step local GD on a quadratic, the right
remaining correction is
\[
  [I-(I-\eta H)^K]^{-1}\Delta_K,
\]
applied to the FedAvg displacement, not \(H^{-1}g\). With noise-controlled
`N*K` probes on the rotated-block quadratic, that dynamics-aware shape beat
tuned scalar FedAvg scale by roughly 0.25-0.30 improvement-ratio points.

This is a corrected Stage-0 mechanism positive, not a product claim. It still
needs to survive broader synthetic beds and then a real-path gate.

E015 moved the corrected operator to the E008-E010 nonlinear/heterogeneous CPU
bed. Probe-fed dynamics-aware shape survived the tuned scalar FedAvg-scale null:

- N=8: random probe 1.306x vs scalar 1.069x;
- N=32: random probe 1.445x vs scalar 1.053x;
- N=128: random probe 2.009x vs scalar 1.101x.

Path-only trajectory scatter still failed, selecting gamma zero. The live CPU
mechanism is therefore noise-controlled probing of the local update map, not
plain worker drift.

CPU workers then ran the two follow-up gates:

- E016 returned `DYNAMICS_AWARE_SURVIVES_ROTATION`. On rotated 2D nonlinear
  blocks, random/directed probes still beat tuned scalar scale:
  - N=8: random 1.254x vs scalar 1.032x;
  - N=32: random 1.497x vs scalar 1.057x;
  - N=128: random 2.124x vs scalar 1.132x.
- E017 returned `RISK_SHRINKAGE_WORKS`. Split-probe residual shrinkage replaces
  arbitrary target gamma tuning on the current CPU bed:
  - N=8: risk probe 1.226x vs scalar 1.075x;
  - N=32: risk probe 1.460x vs scalar 1.070x;
  - N=128: risk probe 1.773x vs scalar 1.127x.

In both gates, path-only worker scatter fails, server/time secants do not
explain the gain, and the oracle best-worker fallback loses. The live claim is
now precise: probe-fed dynamics-aware local-map correction is a real CPU
mechanism. It still needs a real-path gate.

E020 returned `RANDOM_BUDGET_REQUIRED`. Full `N*K` random probes still beat
tuned scalar scale under residual shrinkage, but the current active policies do
not robustly reduce the probe budget. The cheapest plausible rule from this gate
is residual-shrunk random probes around `2N` for `N >= 32`, with full `N*K`
still needed for very small worker counts.

E023 returned `TWO_N_RANDOM_CONFIRMED`. With more held-out reps and an explicit
`probe_reps=1` versus `probe_reps=4` check, `2N` random probes beat tuned scalar
and stayed within the 10-15% band of matched full `N*K` random probes for
N=32/128. The default CPU-bed budget is now `2N` random probes for `N >= 32`;
use full `N*K` for very small N and treat repeat probes as a quality knob.

E021 returned `DENSE_GEOMETRY_BREAKS_METHOD`. The true dense Hessian correction
has large headroom, but the current diagonal, sketched low-rank, and combined
low-rank-plus-diagonal estimators do not safely capture it under correlated
low-rank noise. The next dense-geometry direction is not a larger blind update;
it is low-rank-plus-diagonal estimation with a stricter held-out validation or
trust layer that can reject the combined correction.

E022 returned `TRUST_LAYER_RECOVERS_DENSE`. On the same dense correlated bed,
rowwise validation-gamma gating over a held-out mini-loss proxy prevents the
E021 destructive update and beats tuned scalar at N=8/32/128. The gain is
modest and still far below oracle-H, so dense geometry stays behind a trust
layer while the next work improves subspace/model identification and prices the
validation budget.

The algorithmic stack is not Optuna/TPE. The working plan is empirical-Bayes
local-map estimation inside each event, residual shrinkage from held-out probes,
and a stochastic trust-region controller outside the event. Optuna remains
useful only for pruning or experiment bookkeeping, not as the sampler that
chooses high-dimensional actions.

The first milestone is an offline prediction gate:

1. collect local worker path/proposal summaries;
2. project them into a model-free shared probe space when possible;
3. fit a GP or kernel predictor on pilot rounds/seeds;
4. score held-out prediction of proposal quality or oracle action;
5. compare against round-only schedules, path-norm baselines, and scalar
   source-tuned baselines.

Only if the offline gate passes should we run a closed-loop optimizer.

## Key Files

- `double_fan_freeze_20261003.md`: frozen negative record for the old
  cloud-momentum direction.
- `research_control_board.md`: live project map and stop rules.
- `model.md`: first-principles model for local GP over worker paths.
- `event_triggered_multisecant_plan.md`: pay-by-wire event-triggered
  multisecant branch.
- `algorithm_stack.md`: why this is empirical-Bayes / stochastic trust-region,
  not Optuna.
- `experiments/lgp_e001_path_prediction_gate.md`: first experiment spec.
- `experiments/lgp_e001_result_20261003.md`: first experiment result.
- `experiments/lgp_e002_gamma_control.py`: tuned-gamma control script.
- `experiments/lgp_e002_gamma_control_result_20261003.md`: tuned-gamma result.
- `local_gp/optimizer.py`: local-SGD trajectory aggregator prototype.
- `experiments/lgp_e003_optimizer_smoke.py`: public optimizer smoke test.
- `experiments/lgp_e003_optimizer_smoke_result_20261003.md`: smoke result.
- `experiments/lgp_e005_auto_gamma_gate.py`: source-free gamma gate.
- `experiments/lgp_e005_auto_gamma_gate_result_20261003.md`: auto-gamma result.
- `experiments/lgp_e006_group_gamma_gate.py`: small action-space gate.
- `experiments/lgp_e006_group_gamma_gate_result_20261003.md`: group-gamma result.
- `experiments/lgp_e007_real_path_auto_gamma_result_20261003.md`: real-path
  auto-gamma result, one shard.
- `experiments/lgp_e008_secant_gap_nonlinear.py`: nonlinear secant-gap bed.
- `experiments/lgp_e008_secant_gap_nonlinear_result_20261003.md`: E008 result.
- `experiments/lgp_e009_nonlinear_gamma_sweep.py`: conservative gamma sweep.
- `experiments/lgp_e009_nonlinear_gamma_sweep_result_20261003.md`: E009 result.
- `experiments/lgp_e010_shape_vs_scalar_nonlinear.py`: shape-vs-scalar control.
- `experiments/lgp_e010_shape_vs_scalar_nonlinear_result_20261003.md`: E010 result.
- `experiments/lgp_e011_event_multisecant_stage0.py`: passive event secants.
- `experiments/lgp_e011_event_multisecant_stage0_result_20261003.md`: E011 result.
- `experiments/lgp_e012_seeded_probe_multisecant.py`: seeded probe secants.
- `experiments/lgp_e012_seeded_probe_multisecant_result_20261003.md`: E012 result.
- `experiments/lgp_e013_block_lbfgs_multisecant.py`: rotated-block L-BFGS secants.
- `experiments/lgp_e013_block_lbfgs_multisecant_result_20261003.md`: E013 result.
- `experiments/lgp_e014_dynamics_aware_correction.py`: dynamics-aware local-map correction.
- `experiments/lgp_e014_dynamics_aware_correction_result_20261003.md`: E014 result.
- `experiments/lgp_e015_dynamics_aware_nonlinear_probe.py`: corrected probes on nonlinear bed.
- `experiments/lgp_e015_dynamics_aware_nonlinear_probe_result_20261003.md`: E015 result.
- `experiments/lgp_e016_rotated_nonlinear_probe.py`: rotated nonlinear CPU gate.
- `experiments/lgp_e016_rotated_nonlinear_probe_result_20261003.md`: E016 result.
- `experiments/lgp_e016_rotated_nonlinear_probe_run_20261003.txt`: E016 run log.
- `experiments/lgp_e017_risk_shrinkage_fallback.py`: residual shrinkage and fallback gate.
- `experiments/lgp_e017_risk_shrinkage_fallback_result_20261003.md`: E017 result.
- `experiments/lgp_e020_probe_budget_active_design.py`: probe-budget and active-design gate.
- `experiments/lgp_e020_probe_budget_active_design_result_20261003.md`: E020 result.
- `experiments/lgp_e020_probe_budget_active_design_run_20261003.txt`: E020 run log.
- `experiments/lgp_e021_dense_lowrank_correlated_stress.py`: dense low-rank correlated stress.
- `experiments/lgp_e021_dense_lowrank_correlated_stress_result_20261003.md`: E021 result.
- `experiments/lgp_e021_dense_lowrank_correlated_stress_run_20261003.txt`: E021 run log.
- `experiments/lgp_e022_cpu_worker_prompt_20261003.md`: completed dense trust-layer CPU gate.
- `experiments/lgp_e022_dense_trust_validation.py`: E022 dense trust validation.
- `experiments/lgp_e022_dense_trust_validation_result_20261003.md`: E022 result.
- `experiments/lgp_e022_dense_trust_validation_run_20261003.txt`: E022 run log.
- `experiments/lgp_cpu1_hold_after_e022_20261003.md`: CPU-1 hold note after E022.
- `experiments/lgp_e023_cpu_worker_prompt_20261003.md`: completed random-probe budget CPU gate.
- `experiments/lgp_e023_random_probe_budget_confirm.py`: E023 random budget confirmation.
- `experiments/lgp_e023_random_probe_budget_confirm_result_20261003.md`: E023 result.
- `experiments/lgp_e023_random_probe_budget_confirm_run_20261003.txt`: E023 run log.
- `experiments/lgp_cpu2_hold_after_e023_20261003.md`: CPU-2 hold note after E023.
- `experiments/lgp_cpu_worker_manifest_20261003.md`: CPU worker dropbox manifest.
- `experiments/lgp_gpu_worker_manifest_20261003.md`: GPU worker dropbox manifest.
- `ops/devvm23211_gpu_buck2_result_20261003.md`: confirmed GPU/Buck2 invocation.
- `wanger_duality_reply.md`: review of the local-GP duality theory note.
- `wanger_e002_e007_review.md`: gamma-collapse and E007 gate review.
- `wanger_gamma_line_reply.md`: gamma-flat pivot and secant-gap plan.
