# Local GP

Local GP is a fresh research lane for worker-local SGD.

The project is intentionally separate from Double Fan / cloud momentum. The old
project tried to turn same-point worker variance into a first-moment optimizer
controller. This project starts from a different object:

> In local SGD, workers do not only produce noisy gradients. They trace short
> local paths from the same synchronized point. Those paths are observations of
> a local update field.

The central question is whether those path observations can predict which
proposal, trust action, or synchronization update will work on held-out rounds
and seeds, before paying for a target-setting grid search.

## Current Claim Status

No optimizer claim.

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

E005 added source-free shrinkage: `LocalGPConfig(gamma=None)` estimates the
correction shrinkage from split-half reliability on each round. On the same
synthetic bed it matches pilot-tuned gamma:

- N=8: tuned 1.045x, auto 1.065x over FedAvg;
- N=32: tuned 1.143x, auto 1.142x;
- N=128: tuned 1.168x, auto 1.163x.

This shifts the plausible product from "full GP over high-dimensional weights"
to "small Bayesian action layer over a local-SGD aggregation correction." The
first action is scalar gamma; a later version can try group-wise gammas or
probe-space actions.

E006 tested that next step. Group-wise gamma gave a small N=8 bump but no
benefit at N=32/128, so the MVP should stay scalar:

- estimate a high-dimensional correction shape from local worker paths;
- estimate one source-free reliability shrinkage scalar;
- apply the corrected FedAvg update.

The next useful experiment must run the actual aggregator on real local-SGD
paths with a self-contained source bundle, because E004 failed before workload
execution. The key comparison is FedAvg/source-tuned scale vs pilot-tuned gamma
vs source-free auto-gamma. A closed loop is not justified from E001 or E002.

The first milestone is an offline prediction gate:

1. collect local worker path/proposal summaries;
2. project them into a model-free shared probe space when possible;
3. fit a GP or kernel predictor on pilot rounds/seeds;
4. score held-out prediction of proposal quality or oracle action;
5. compare against round-only schedules, path-norm baselines, and scalar
   source-tuned baselines.

Only if the offline gate passes should we run a closed-loop optimizer.

## Key Files

- `research_control_board.md`: live project map and stop rules.
- `model.md`: first-principles model for local GP over worker paths.
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
- `ops/devvm23211_gpu_buck2_result_20261003.md`: confirmed GPU/Buck2 invocation.
