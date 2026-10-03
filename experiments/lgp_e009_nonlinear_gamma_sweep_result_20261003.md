# LGP-E009 Nonlinear Gamma Sweep Result - 2026-10-03

## Decision

`CONSERVATIVE_GAMMA_NEEDED`

## Question

E008 showed that full displacement-law correction is destructive on a nonlinear
and heterogeneous synthetic bed. E009 asks whether the correction shape still
has value after conservative shrinkage, or whether FedAvg / server-side secants
dominate.

## Bed

- diagonal but nonlinear global objective:
  `0.5 * lambda * x^2 + 0.25 * q * x^4`;
- centered worker heterogeneity;
- stochastic gradient noise;
- actual one-step global loss decrease as the metric;
- source/pilot gamma sweep, held-out scoring.

## Result

| N | tuned gamma | tuned/FedAvg | auto/FedAvg | full/FedAvg | secant/FedAvg | mean auto gamma |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.40 | 1.460 +/- 0.221 | 1.124 +/- 0.273 | 0.598 +/- 0.389 | 0.936 +/- 0.131 | 0.814 |
| 32 | 0.35 | 1.473 +/- 0.124 | 0.387 +/- 0.482 | -0.398 +/- 0.476 | 0.954 +/- 0.124 | 0.666 |
| 128 | 0.30 | 1.501 +/- 0.062 | -4.449 +/- 4.207 | -4.871 +/- 3.937 | 1.066 +/- 0.056 | 0.229 |

## Interpretation

This is the first bed in this lane where gamma is not flat.

The correction shape has value: a conservative tuned gamma gives roughly
1.46-1.50x FedAvg and beats the server-side secant control. But the current
Bayesian auto-gamma is wrong on this bed. It trusts estimator agreement and does
not account for model bias from nonlinearity / heterogeneity, so it overcorrects
and becomes destructive as N grows.

This resolves the E005/E006 confusion:

- on the exact-quadratic bed, gamma was flat, so auto-gamma matching tuned gamma
  was vacuous;
- on a bed that can tell actions apart, conservative shrinkage matters, and
  the current auto-gamma does not set it correctly.

## Consequence

The live target is not "choose gamma from split-half estimator reliability."
The live target is a bias-aware trust rule:

1. estimate the within-round path correction shape;
2. estimate or bound model-error risk from path nonlinearity / worker
   heterogeneity;
3. shrink aggressively when model-error risk is high.

The secant gap remains alive: server-side secants do not recover the
conservative local-path gain on this bed.
