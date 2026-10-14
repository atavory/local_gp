# E094 Longer-Round 64-Seed Win

E094 reran the E090B fixed-compute outer-control HPO audit with `rounds=16`
and 64 fresh held-out seeds `998400..998463`.

Decision: `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`.

Pooled readout:

- internal minus stale/default control: `+4.403 +/- 0.417`
- internal minus fresh/theory control: `+0.194 +/- 0.072`
- internal minus shuffled internal: `+0.233 +/- 0.076`
- internal minus external validation-selected HPO: `+1.275 +/- 0.239`
- internal minus external oracle HPO: `+0.465 +/- 0.222`
- external validation-selected HPO minus stale/default: `+3.128 +/- 0.396`

This upgrades the longer-round story from E092's high-variance live result to a
CI-cleared fixed-compute HPO win.
