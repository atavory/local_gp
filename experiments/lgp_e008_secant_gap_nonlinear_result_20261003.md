# LGP-E008 Secant Gap Nonlinear Result - 2026-10-03

## Decision

`FULL_CORRECTION_UNSTABLE`

## Question

Does the local-path correction survive a bed that breaks the exact diagonal
quadratic assumptions?

Bed:

- nonlinear diagonal global objective:
  `0.5 * lambda * x^2 + 0.25 * q * x^4`;
- centered worker heterogeneity;
- stochastic gradient noise;
- actual one-step global loss decrease as the metric.

## Result

| N | local full / FedAvg | auto gamma / FedAvg | server secant / FedAvg |
|---:|---:|---:|---:|
| 8 | 0.019 +/- 0.282 | 0.746 +/- 0.170 | 0.880 +/- 0.146 |
| 32 | -1.490 +/- 0.642 | -0.134 +/- 0.337 | 1.002 +/- 0.114 |
| 128 | -8.054 +/- 3.262 | -7.268 +/- 3.270 | 1.106 +/- 0.091 |

## Interpretation

Raw displacement-law correction is not robust to this nonlinear/heterogeneous
bed. The server-side secant control is roughly FedAvg-to-slightly-positive,
while full local correction is destructive, especially as N grows and the
curvature estimate becomes more confident.

This is not a total death of the correction shape. It says the exact-quadratic
readout is too aggressive under model error. The next question is whether a
conservative shrinkage can recover useful signal.

That is E009.
