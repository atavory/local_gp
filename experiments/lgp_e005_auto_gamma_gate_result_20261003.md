# LGP-E005 Auto-Gamma Gate Result - 2026-10-03

## Decision

`GAMMA_FLAT_ON_SYNTHETIC`

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

The source-free reliability shrinker matches the pilot-tuned scalar on the
synthetic diagonal bed, but this is not evidence of a useful auto-gamma method:
the hardcoded `gamma=1` arm also ties both learned/tuned gamma arms.

At N=8, auto-gamma chooses 0.527 while tuned gamma chooses 0.80, yet the scores
move by only about two percent. That is a flat action axis, not a successful
adaptive controller.

This is not a real-data claim and not a tuning-reduction claim. It says the
synthetic bed is too insensitive to gamma to adjudicate gamma selection.

## Consequence

The next real-path gate must keep `full_gamma` as a null control:

1. FedAvg/source-tuned scale;
2. pilot-tuned scalar gamma;
3. source-free auto-gamma from split-half reliability.
4. full correction with `gamma=1`.

If full gamma ties the other gamma arms, the gamma line is flat there too and
should not be written as an auto-gamma win.
