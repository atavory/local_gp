# LGP-E012 Seeded Probe Multisecant Result - 2026-10-03

## Decision

`SEEDED_PROBES_NOT_ENOUGH_WITH_SCALAR_BB`

## Question

E011 showed passive stochastic worker drift is too noisy. E012 asks whether the
controller can seed small centered probe offsets at broadcast time and recover a
useful spatial secant signal.

## Result

Using `probe_radius=0.18`, `delta=0.5`, `sigma=0.2`:

| N | steps | scalar/FedAvg | timeBB/FedAvg | spaceDiag/FedAvg | spaceBB/FedAvg |
|---:|---:|---:|---:|---:|---:|
| 8 | 6.90 | 1.027 +/- 0.008 | 1.109 +/- 0.011 | 1.000 +/- 0.000 | 1.062 +/- 0.009 |
| 32 | 7.07 | 1.029 +/- 0.013 | 1.109 +/- 0.012 | 1.003 +/- 0.012 | 1.063 +/- 0.008 |
| 128 | 7.26 | 1.026 +/- 0.007 | 1.107 +/- 0.008 | 1.021 +/- 0.008 | 1.063 +/- 0.005 |

## Interpretation

Controller-seeded probes improve the scalar spatial BB arm relative to scalar
extrapolation, but they still do not beat the one-pair server/time BB step.
Diagonal spatial inverse remains too fragile.

Seeded probe diversity alone is not enough. The next test must use the actual
multi-direction mechanism, not only scalar BB: limited-memory multisecant
updates on a non-diagonal Hessian.

That is E013.
