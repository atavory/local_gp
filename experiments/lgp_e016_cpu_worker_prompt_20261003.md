# LGP-E016 CPU Worker Ping - Rotated / Dense Nonlinear Gate

You are running the next CPU gate for `local_gp`. Use the GitHub repo
`https://github.com/atavory/local_gp`.

## Mission

E015 showed that probe-fed dynamics-aware correction beats tuned scalar FedAvg
scale on a diagonal nonlinear/heterogeneous CPU bed. E016 asks whether that was
a coordinate artifact.

Build and run a CPU-only experiment:

```text
experiments/lgp_e016_rotated_nonlinear_probe.py
```

## Starting Point

Start from:

```text
experiments/lgp_e015_dynamics_aware_nonlinear_probe.py
```

Keep the same main comparison:

- tuned scalar FedAvg scale;
- oracle dynamics-aware correction;
- path-only local trajectory correction;
- random-probe dynamics-aware correction;
- coordinate / directed-probe dynamics-aware correction;
- server/time secant;
- best-worker fallback if cheap to include.

## Change The Bed

Replace the diagonal objective with one where coordinates are not privileged.
Two acceptable CPU variants:

1. Rotated 2D nonlinear blocks:

```text
f_b(z) = 0.5 z^T H_b z + 0.25 q_1 (v_1^T z)^4 + 0.25 q_2 (v_2^T z)^4
```

where `(v_1, v_2)` are the eigenvectors of `H_b`.

2. Dense low-rank plus diagonal nonlinear bed:

```text
H = diag(flat/scalar spectrum) + U diag(spikes) U^T
```

with quartic terms in the same low-rank/eigendirections.

Variant 1 is preferred because it is easy to debug and matches E014's block
helpers.

## Correct Operator

Do not use a plain quasi-Newton update. The correction must be the K-step
local-map operator:

```text
[I - (I - eta Hhat)^K]^{-1} Delta_K
```

applied to the FedAvg displacement.

For rotated 2D blocks, estimate a 2x2 `Hhat` per block from secant/probe pairs,
then apply the 2x2 map correction blockwise.

## Nulls

The main null is tuned scalar FedAvg scale selected on pilot rounds. Also report
server/time secants. If you include best-worker fallback, report it as a strong
baseline, not as part of the method.

## Decision Labels

Use exactly one:

- `DYNAMICS_AWARE_SURVIVES_ROTATION`: random or non-coordinate probes beat tuned
  scalar scale with a positive CI on held-out rounds.
- `COORDINATE_ARTIFACT`: only coordinate/oracle probes win; random/non-coordinate
  probes fail against tuned scalar.
- `SCALAR_OR_FALLBACK_EXPLAINS`: tuned scalar or best-worker fallback explains
  the gain.
- `INVALID`: implementation or scoring bug.

## Required Output

Write:

```text
experiments/lgp_e016_rotated_nonlinear_probe_result_20261003.md
```

Include:

- parameters;
- selected pilot scale/gamma values;
- held-out table with ratios over FedAvg;
- CIs for probe minus scalar and best-worker minus probe if included;
- short interpretation scoped to CPU synthetic only.

Do not claim a real optimizer product from this gate.
