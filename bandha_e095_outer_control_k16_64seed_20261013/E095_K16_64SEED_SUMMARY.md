# E095 K=16 64-Seed Support

E095 reran the fixed-compute outer-control HPO audit at `K=16`, `rounds=16`,
with 64 fresh held-out seeds `998500..998563`.

Decision: `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`.

Pooled readout:

- internal minus stale/default control: `+0.094 +/- 0.061`
- internal minus fresh/theory control: `+0.115 +/- 0.056`
- internal minus shuffled internal: `+0.075 +/- 0.048`
- internal minus external validation-selected HPO: `+0.516 +/- 0.262`
- internal minus external oracle HPO: `-0.067 +/- 0.277`
- external validation-selected HPO minus stale/default: `-0.423 +/- 0.252`

Interpretation: the large-K margin is smaller, but the fixed-compute comparison
still favors the internal probe-controlled aggregate over validation-selected
external HPO. Probe cost is `0.89%` at `N=8` and `0.45%` at `N=16`.
