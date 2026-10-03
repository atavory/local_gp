# LGP-E005 Auto-Gamma Gate Result - 2026-10-03

## Decision

`AUTO_GAMMA_LIVE_ON_SYNTHETIC`

## Question

E003 built a concrete optimizer artifact, but it still used a pilot-tuned scalar
`gamma`. E005 asks whether the optimizer can set that shrinkage itself from the
worker cloud.

Candidate: `LocalGPConfig(gamma=None)`, which estimates shrinkage from
split-half reliability of the correction vector on each round.

Controls:

- FedAvg;
- pilot-tuned scalar gamma;
- full unshrunk correction (`gamma=1`);
- oracle curvature correction.

## Result

| N | tuned gamma | mean auto gamma | tuned/base | auto/base | full/base | oracle/base |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.80 | 0.527 | 1.045 | 1.065 | 1.047 | 1.171 |
| 32 | 0.80 | 0.880 | 1.143 | 1.142 | 1.139 | 1.247 |
| 128 | 0.75 | 0.906 | 1.168 | 1.163 | 1.157 | 1.248 |

## Interpretation

This is the first evidence that the optimizer can reduce the tuning burden:
the source-free reliability shrinker matches the pilot-tuned scalar on the
synthetic diagonal bed, and beats it at N=8 in this held-out split.

This is not yet a real-data claim. It says the right small Bayesian object is
probably not a full GP over parameters. It is a low-dimensional decision layer
over aggregation actions, starting with scalar `gamma` and later a small number
of group-wise gammas.

## Consequence

The next real-path gate should not attempt a full parameter-space GP. It should
test three small-action arms:

1. FedAvg/source-tuned scale;
2. pilot-tuned scalar gamma;
3. source-free auto-gamma from split-half reliability.

If auto-gamma ties tuned gamma on real local paths, the contribution is reduced
tuning/search cost. If it beats tuned gamma, the contribution is both tuning
reduction and adaptation.
