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

The next useful experiment must either collect richer path/probe information or
show that a principled Bayesian/GP evidence rule sets shrinkage better than the
tuned gamma control. A closed loop is not justified from E001 or E002.

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
- `ops/devvm23211_gpu_buck2_result_20261003.md`: confirmed GPU/Buck2 invocation.
