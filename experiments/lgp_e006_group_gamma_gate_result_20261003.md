# LGP-E006 Group-Gamma Gate Result - 2026-10-03

## Decision

`GROUP_GAMMA_FLAT_ON_SYNTHETIC`

## Question

Full GP over parameter space is too large. E005 showed that scalar source-free
`gamma` can match pilot-tuned gamma on the synthetic diagonal bed. E006 asks
whether the next useful smaller object is a small group-wise action layer:
split the coefficient vector into correction-magnitude bins and estimate one
split-half reliability shrinkage per bin.

## Result

| N | tuned gamma | tuned/base | auto1/base | auto4/base | auto8/base | oracle/base |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.90 | 1.054 | 1.072 | 1.081 | 1.084 | 1.179 |
| 32 | 0.85 | 1.129 | 1.129 | 1.127 | 1.126 | 1.219 |
| 128 | 0.70 | 1.138 | 1.135 | 1.135 | 1.135 | 1.198 |

Mean estimated gamma:

| N | auto1 | auto4 | auto8 |
|---:|---:|---:|---:|
| 8 | 0.542 | 0.699 | 0.732 |
| 32 | 0.883 | 0.871 | 0.871 |
| 128 | 0.902 | 0.899 | 0.899 |

## Interpretation

The group-wise gamma axis is also mostly flat. It gives a modest N=8 bump, but
at N=32 and N=128 it provides no improvement over scalar auto-gamma, and E005
already showed that hardcoded `gamma=1` is competitive on the same bed.

This should not be cited as evidence for group-wise action Bayes. It is evidence
that this synthetic bed is poor for judging gamma selection.

## Consequence

For the next gate, stop iterating on gamma unless `full_gamma` fails. The
durable signal is the secant gap: intermediate local points expose curvature
information that server-side secants did not capture in the original synthetic
analysis. That is the direction to test on a bed with nonlinearity or worker
heterogeneity.
