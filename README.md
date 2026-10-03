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

No optimizer claim yet.

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
