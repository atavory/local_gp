# E098 Tiny Causal-LM Domain-Shift HPO

E098 adds the adaptive LM setting missing from E096/E097.  The training stream
shifts from off-domain prose to code-like text, while probe/select and target
text are code-like.

Decision: `TINY_LM_DOMAIN_SHIFT_HPO_WIN`.

Pooled 16-seed readout:

- default fixed LR: `0.925 +/- 0.044`
- internal one-hot probe HPO: `1.374 +/- 0.042`
- shuffled-probe internal: `0.909 +/- 0.051`
- external validation-selected HPO: `1.331 +/- 0.046`
- internal minus default: `+0.450 +/- 0.030`
- internal minus shuffled: `+0.465 +/- 0.037`
- internal minus external validation-selected HPO: `+0.043 +/- 0.034`

Interpretation: same-round probe selection adapts across the domain shift and
beats validation-selected external fixed-LR HPO.
