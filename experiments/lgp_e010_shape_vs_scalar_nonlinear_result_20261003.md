# LGP-E010 Shape Vs Scalar Nonlinear Result - 2026-10-03

## Decision

`LOCAL_PATH_SHAPE_BEATS_SCALAR`

## Question

E009 showed that a conservative tuned gamma beats FedAvg and server-side secants
on a nonlinear/heterogeneous bed. E010 adds the missing control:

> Is that win merely a tuned scalar/LR effect, or does the per-coordinate
> local-path correction shape add value?

## Result

| N | tuned scale | tuned gamma | scalar/FedAvg | shape/FedAvg | shape - scalar | secant/FedAvg |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.40 | 0.40 | 1.089 +/- 0.052 | 1.391 +/- 0.164 | 0.302 +/- 0.115 | 0.878 +/- 0.182 |
| 32 | 1.30 | 0.30 | 1.100 +/- 0.032 | 1.674 +/- 0.243 | 0.574 +/- 0.211 | 0.897 +/- 0.183 |
| 128 | 1.30 | 0.30 | 1.066 +/- 0.026 | 1.468 +/- 0.099 | 0.402 +/- 0.074 | 1.043 +/- 0.131 |

## Interpretation

This is the first clean positive in the local-GP lane:

- it is not explained by tuned scalar scale;
- it is not recovered by server-side secants;
- it uses a bed where full correction is destructive, so shrinkage matters;
- the value comes from the local-path correction shape.

The current source-free auto-gamma still fails on this bed because it estimates
agreement of the curvature estimator, not model bias. The product direction is
therefore:

1. keep the local-path correction shape;
2. use conservative, bias-aware shrinkage;
3. benchmark against tuned scalar scale and server-side secants.

## Caveat

This is still synthetic. It is deliberately more hostile than the original
diagonal quadratic bed, but it is not a real-network claim.
