# Local GP Research Control Board

## Prime Directive

Do not repeat the Double Fan failure mode.

The goal is not to invent another controller and then explain away a loss. The
goal is first to ask whether local worker paths contain held-out predictive
information that simple schedules and scalar baselines do not contain.

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

## Hypothesis

For K>1 local SGD, worker paths reveal local geometry. A GP or kernel predictor
over path/probe descriptors may predict:

- which proposal direction is safer;
- how much of local history to trust;
- which scalar action from a small action grid transfers;
- whether a round-only schedule is about to fail.

The first claim is prediction. The optimizer claim comes later.

## Evidence Required

Before closed loop, require an offline held-out prediction win:

- source-only fit;
- held-out target seeds/settings;
- paired confidence intervals;
- shuffled-feature control;
- round-only and path-norm baselines;
- target oracle headroom diagnostic.

## Current Status

No experiments have been run in this repo yet.

The first experiment is `experiments/lgp_e001_path_prediction_gate.md`.

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

1. Run E001 as an offline prediction gate.
2. If E001 passes, design a tiny closed-loop action set.
3. If E001 fails, inspect whether the failure is no headroom, wrong features,
   or schedule/path-norm sufficiency.
4. Do not run a closed-loop optimizer before the prediction gate passes.
