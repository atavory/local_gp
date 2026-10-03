# LGP-E013 Block L-BFGS Multisecant Result - 2026-10-03

## Decision

`MULTISECANT_WORKS_WITH_LOW_NOISE_ONLY`

## Question

E013 moves from diagonal inverse estimates to actual L-BFGS two-loop recursion
on a rotated block-SPD quadratic. This removes the diagonal cheat and tests the
real multisecant claim:

> Do centered worker secant pairs beat one-pair server/time L-BFGS and tuned
> scalar extrapolation?

## Main Run: Stochastic Secants

Parameters: `sigma=0.1`, `probe_radius=0.2`, `delta=0.45`.

| N | steps | scalar/FedAvg | time L-BFGS/FedAvg | space L-BFGS/FedAvg |
|---:|---:|---:|---:|---:|
| 8 | 7.41 | 1.007 +/- 0.005 | 1.097 +/- 0.006 | 1.000 +/- 0.000 |
| 32 | 6.99 | 1.014 +/- 0.004 | 1.099 +/- 0.005 | 1.000 +/- 0.000 |
| 128 | 7.58 | 1.011 +/- 0.006 | 1.104 +/- 0.008 | 1.000 +/- 0.000 |

The tuned spatial L-BFGS damping selected zero in all three cases.

## Sanity Check: Noiseless Secants

Same script with `sigma=0.0`:

| N | scalar/FedAvg | time L-BFGS/FedAvg | space L-BFGS/FedAvg | space - time |
|---:|---:|---:|---:|---:|
| 8 | 1.005 +/- 0.006 | 1.098 +/- 0.014 | 1.088 +/- 0.013 | -0.009 +/- 0.010 |
| 32 | 1.015 +/- 0.007 | 1.098 +/- 0.008 | 1.170 +/- 0.017 | 0.072 +/- 0.013 |
| 128 | 1.005 +/- 0.008 | 1.103 +/- 0.011 | 1.259 +/- 0.021 | 0.157 +/- 0.014 |

With clean secants, the spatial multisecant mechanism works and improves with
worker count.

## Interpretation

The architecture is mechanically sound, but ordinary independent stochastic
gradients make spatial secant pairs too noisy. This is the exact load-bearing
condition the theory must quantify.

The next viable version needs noise-controlled secants:

- same-batch / common-random-number probes;
- repeated microbatch gradients at event time;
- a shared probe set in function space;
- or very large batches / large N where secant noise is small.

Do not claim the event-triggered multisecant architecture from ordinary noisy
worker gradients. Claim only the conditional result: spatial multisecants are
valuable when the secant equations have sufficient SNR.
