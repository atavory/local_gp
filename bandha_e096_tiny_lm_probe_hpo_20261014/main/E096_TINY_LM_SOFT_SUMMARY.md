# E096 Tiny Causal-LM Soft Probe HPO

E096 adds a lightweight language-model tuning audit: a tiny character-level
causal Transformer trained on an embedded text corpus with LR arms
`(0.5, 1, 2, 4)`.

Policy: soft probe-weighted parameter-delta aggregation.

Decision: `TINY_LM_PROBE_LABEL_CONTENT`.

Pooled readout:

- default fixed LR: `0.880 +/- 0.028`
- internal probe-weighted HPO: `1.081 +/- 0.035`
- shuffled-probe internal: `1.048 +/- 0.038`
- external validation-selected HPO: `1.178 +/- 0.031`
- internal minus default: `+0.202 +/- 0.011`
- internal minus shuffled: `+0.033 +/- 0.021`
- internal minus external validation-selected HPO: `-0.096 +/- 0.015`

Interpretation: soft LM aggregation shows probe-label content but does not beat
validation-selected external HPO on this tiny fixed-LR grid.
