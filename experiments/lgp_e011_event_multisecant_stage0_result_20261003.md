# LGP-E011 Event Multisecant Stage-0 Result - 2026-10-03

## Decision

`PASSIVE_DRIFT_SECANTS_WEAK`

## Question

Under the pay-by-wire event-triggered architecture, can passive worker drift
produce centered secant pairs useful enough to beat scalar extrapolation or a
one-pair server BB step?

Protocol:

- workers start aligned and run local heavy-ball silently;
- event fires when a quorum crosses a drift radius;
- event gathers full positions and gradients;
- compare FedAvg, tuned scalar extrapolation, server/time BB, diagonal
  server-time diagnostic, spatial diagonal secants, and spatial BB.

## Result

Using `delta=0.5`, `sigma=0.2`, quorum 0.5:

| N | mean local steps | scalar/FedAvg | timeBB/FedAvg | spaceDiag/FedAvg | spaceBB/FedAvg |
|---:|---:|---:|---:|---:|---:|
| 8 | 6.96 | 1.022 +/- 0.007 | 1.099 +/- 0.006 | 1.000 +/- 0.000 | 1.038 +/- 0.006 |
| 32 | 6.67 | 1.035 +/- 0.017 | 1.113 +/- 0.019 | 1.026 +/- 0.007 | 1.047 +/- 0.008 |
| 128 | 7.11 | 1.028 +/- 0.015 | 1.112 +/- 0.015 | 0.995 +/- 0.023 | 1.047 +/- 0.006 |

Spatial diagonal secants do not beat scalar extrapolation; spatial BB is only a
small scalar-like bump. A one-pair server BB step is better on this passive
drift bed.

## Interpretation

Passive stochastic worker drift is not enough. The positions differ mostly
because of minibatch noise, and the same noise pollutes the centered gradient
differences. The architecture should not rely on accidental drift as the
experimental design.

This changes the next step:

- keep the event-triggered pay-by-wire framing;
- let the controller deliberately seed probe diversity at broadcast;
- test whether designed worker offsets create useful spatial secants.

That is E012.
