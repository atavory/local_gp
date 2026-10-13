# E091 Fresh-Seed Replication

E091 reran the E090B fixed-compute outer-control HPO audit with fresh held-out
seeds `998100..998131`.

Decision: `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`.

Pooled readout:

- internal minus stale/default control: `+3.878 +/- 0.488`
- internal minus fresh/theory control: `+0.044 +/- 0.062`
- internal minus shuffled internal: `+0.093 +/- 0.073`
- internal minus external validation-selected HPO: `+1.148 +/- 0.511`
- internal minus external oracle HPO: `+0.413 +/- 0.345`
- external validation-selected HPO minus stale/default: `+2.730 +/- 0.602`

This supports the E090B conclusion that deliberately diverse outer-control
aggregation clears the fixed-compute external-HPO comparator in this local-SGD
update-map bed.
