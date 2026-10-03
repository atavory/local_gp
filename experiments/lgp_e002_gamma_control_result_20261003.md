# LGP-E002 Gamma-Control Result - 2026-10-03

## Decision

`TUNED_GAMMA_EXPLAINS`

## Purpose

Ushmoo's theory note found that within-round local-SGD curvature estimates can
beat tuned-scale FedAvg on a diagonal-quadratic bed, while server-side secants
capture almost none of the headroom. The note also identified a likely
confound: the winning estimator is less accurate but more upward-biased in flat
directions, accidentally shrinking the correction toward one where it is least
trustworthy.

The required control is:

```text
c_j(gamma) = 1 + gamma * (c_j - 1)
```

where \(c_j\) is the measured per-coordinate correction shape and \(\gamma\) is
one tuned scalar. If this matches the measured readout, the live posterior
readout is not yet a method.

## Protocol

Script: `experiments/lgp_e002_gamma_control.py`.

- diagonal quadratic;
- d=120;
- 10 sharp directions;
- K=8;
- eta=0.05;
- rounds=30;
- N in {8, 32, 128};
- gamma grid 0.00..2.00 by 0.05;
- gamma selected on pilot reps only;
- held-out reps scored separately;
- metric is one-step gain ratio over tuned-scale FedAvg.

## Result

| N | selected gamma | within/base | tuned-gamma/base | oracle-C/base |
|---:|---:|---:|---:|---:|
| 8 | 1.10 | 1.124 | 1.127 | 1.200 |
| 32 | 0.75 | 1.141 | 1.146 | 1.220 |
| 128 | 0.80 | 1.167 | 1.172 | 1.219 |

The tuned one-scalar gamma control matches or slightly beats the measured
within-round posterior readout on every tested N.

## Interpretation

This is not a null about the information in local trajectories. The headroom is
real:

- oracle-C remains around 1.20x tuned-scale FedAvg;
- server-side secants from the previous check capture about 1.00x;
- intermediate local points expose a correction shape that classical server
  secants miss.

But the current readout is not a method. Its gain is explainable by a tuned
constant shrinkage applied to the measured shape. This is the same control class
that killed several Double Fan arms.

What survives:

- FedAvg as infinite-lengthscale GP posterior is a clean theoretical framing;
- finite-lengthscale/local-trajectory curvature headroom exists on the synthetic
  quadratic bed;
- intermediate local points contain information not captured by free
  server-side secants.

What does not survive:

- claiming the measured posterior readout itself as an optimizer method;
- running closed-loop from this readout;
- calling the bias-shaped win a Bayesian product before replacing the accidental
  regularizer with a principled, source-fit prior or marginal-likelihood rule
  that beats tuned gamma.

## Next Gate

The next live question is no longer "does within-round curvature contain
information?" It does.

The next live question is:

> Can a principled GP/Bayesian evidence rule set the shrinkage shape as well as
> or better than the tuned gamma constant, without target tuning?

Until that is answered positively, this lane has theory/headroom but not a
method.
