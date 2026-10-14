# E099 Tiny Causal-LM Domain-Shift 32-Seed HPO

E099 reruns E098 with 32 held-out seeds.

Decision: `TINY_LM_DOMAIN_SHIFT_HPO_WIN`.

Pooled readout:

- default fixed LR: `0.920 +/- 0.025`
- internal one-hot probe HPO: `1.372 +/- 0.026`
- shuffled-probe internal: `0.906 +/- 0.031`
- external validation-selected HPO: `1.330 +/- 0.028`
- internal minus default: `+0.452 +/- 0.022`
- internal minus shuffled: `+0.466 +/- 0.029`
- internal minus external validation-selected HPO: `+0.043 +/- 0.026`
- probe overhead estimate: `16.67%`

This is the usable LM tuning result for the paper: still tiny-scale, but it
shows the adaptive-HPO mechanism in a language-model setting.
