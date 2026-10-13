# E092 Longer-Round Stress Run

E092 reran the E090B fixed-compute outer-control HPO audit with `rounds=16`
and fresh held-out seeds `998200..998231`.

Decision: `FIXED_COMPUTE_OUTER_CONTROL_HPO_LIVE`.

Pooled readout:

- internal minus stale/default control: `+2.838 +/- 1.979`
- internal minus fresh/theory control: `+0.619 +/- 0.472`
- internal minus shuffled internal: `+0.771 +/- 0.632`
- internal minus external validation-selected HPO: `+0.330 +/- 2.151`
- internal minus external oracle HPO: `-0.559 +/- 2.182`
- external validation-selected HPO minus stale/default: `+2.508 +/- 0.642`

Interpretation: the probe-controlled outer-control mechanism remains live over
a longer ledger, but the validation-selected external-HPO margin is high
variance and not CI-cleared at this budget.
