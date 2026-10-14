# E097 Tiny Causal-LM One-Hot Probe HPO

E097 reruns the E096 tiny causal-LM tuning audit with one-hot probe selection
instead of soft probe-weighted aggregation.

Decision: `TINY_LM_PROBE_LABEL_CONTENT`.

Pooled readout:

- default fixed LR: `0.880 +/- 0.028`
- internal one-hot probe HPO: `1.177 +/- 0.031`
- shuffled-probe internal: `1.000 +/- 0.034`
- external validation-selected HPO: `1.178 +/- 0.031`
- internal minus default: `+0.298 +/- 0.011`
- internal minus shuffled: `+0.177 +/- 0.015`
- internal minus external validation-selected HPO: `-0.000 +/- 0.000`

Interpretation: the LM probe labels matter and recover the validation-selected
external HPO arm, but this tiny fixed-LR grid is simple enough that external HPO
matches the internal controller.
