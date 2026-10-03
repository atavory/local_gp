# LGP-E006 Group-Gamma Gate Result - 2026-10-03

## Decision

`SCALAR_ACTION_MVP`

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

The action space should stay small.

Group-wise shrinkage gives a modest N=8 bump, but at N=32 and N=128 it provides
no improvement over scalar auto-gamma. The clean MVP is therefore:

1. estimate the local-SGD displacement-law correction from worker paths;
2. choose one source-free shrinkage gamma from split-half reliability;
3. apply the corrected FedAvg update.

This is a smaller and more plausible Bayesian object than a full GP:
high-dimensional worker paths estimate a correction shape, while the Bayesian
layer only decides how much of that shape to trust.

## Consequence

For the next real-path gate, do not spend complexity on a full parameter-space
GP or many group-wise weights. Test scalar auto-gamma first. Group-wise gamma is
a secondary low-N variant, not the product MVP.
