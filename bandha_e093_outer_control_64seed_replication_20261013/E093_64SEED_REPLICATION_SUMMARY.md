# E093 64-Seed Replication

E093 reran the E090B fixed-compute outer-control HPO audit at the main
`rounds=8` setting with 64 fresh held-out seeds `998300..998363`.

Decision: `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`.

Pooled readout:

- internal minus stale/default control: `+3.829 +/- 0.378`
- internal minus fresh/theory control: `+0.095 +/- 0.057`
- internal minus shuffled internal: `+0.140 +/- 0.058`
- internal minus external validation-selected HPO: `+1.224 +/- 0.313`
- internal minus external oracle HPO: `+0.343 +/- 0.255`
- external validation-selected HPO minus stale/default: `+2.605 +/- 0.443`

This is the strongest replication packet for the TMLR fixed-compute HPO claim:
the internal probe-controlled aggregate beats both shuffled probes and
validation-selected external HPO under the same worker budget.
