# Amendment To LGP-E007: Full-Gamma Null Control

E007 is already running, but its decision labels are incomplete.

The `full_gamma` arm is required in the prompt, but the label set does not say
what to do if `gamma=1.0` ties `auto_gamma` and `tuned_gamma`.

That case is not an auto-gamma win. It means the gamma axis is flat enough that
hardcoding full correction works about as well as estimating or tuning gamma.

## Corrected Scoring

Use these amended labels:

- `AUTO_GAMMA_REAL_WINS`: auto-gamma beats tuned-gamma, full-gamma, and
  FedAvg/source scale.
- `AUTO_GAMMA_REAL_TIES_TUNED`: auto-gamma ties tuned-gamma, beats full-gamma,
  and beats FedAvg/source scale. This is a tuning-reduction result.
- `FULL_GAMMA_EXPLAINS`: full-gamma ties or beats auto-gamma and tuned-gamma.
  This is a null for the gamma-estimation claim.
- `TUNED_GAMMA_REQUIRED`: tuned-gamma beats auto-gamma and full-gamma.
- `FEDAVG_SCALE_EXPLAINS`: source-tuned FedAvg scale beats all gamma arms.
- `NO_REAL_PATH_HARNESS`: bundle works but no usable real local-SGD path source
  exists on the worker.
- `INVALID_BUNDLE`: bundle sanity checks fail.

## Required Interpretation

If `full_gamma` ties, do not write the result as "source-free auto-gamma
matched tuning." That is true but vacuous: a hardcoded constant matched it too.

The real question becomes whether the path-derived correction shape itself beats
the free classical alternatives, especially server-side secants. The gamma line
should not continue unless gamma materially changes the target score.
