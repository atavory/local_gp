# LGP-E021 CPU Worker Ping - Dense / Correlated Geometry Stress

You are running the next CPU stress gate for `local_gp`. Use the GitHub repo
`https://github.com/atavory/local_gp`, or the Manifold source bundle listed in
the CPU manifest if GitHub access is inconvenient.

## Mission

E016 says the method survives rotated 2D nonlinear blocks. E021 asks whether the
mechanism survives when the true curvature/noise is not separable into tiny
independent coordinate/block pieces.

Build and run:

```text
experiments/lgp_e021_dense_lowrank_correlated_stress.py
```

## Bed

Construct a CPU synthetic bed with:

- dimension `D` around 80-200;
- Hessian:

```text
H = diag(base spectrum) + U diag(spikes) U^T
```

where `U` is a random orthonormal low-rank basis (`rank=8` or `16`);

- nonlinear quartic terms in the same low-rank directions plus weak diagonal
  quartic terms;
- centered worker heterogeneity;
- correlated gradient noise:

```text
noise = diagonal_noise + U lowrank_noise
```

Keep the objective globally tractable and the scoring exact.

## Arms

Required:

- raw FedAvg;
- tuned scalar FedAvg scale;
- diagonal random-probe dynamics-aware correction;
- low-rank/sketched probe dynamics-aware correction;
- combined low-rank-plus-diagonal correction if feasible;
- oracle-H dynamics-aware correction;
- server/time secant;
- best-worker true fallback.

The correction must be the K-step local-map operator:

```text
[I - (I - eta Hhat)^K]^{-1} Delta_K
```

not a plain Newton or L-BFGS step.

## Low-Rank / Sketch Variant

The point is not to form a huge dense matrix. This is CPU synthetic, so forming
small `D x D` matrices is acceptable for validation, but the reported method
should mimic a scalable representation:

- choose a sketch/subspace dimension `k` (`8`, `16`, maybe `32`);
- estimate curvature action inside that subspace from probe pairs;
- correct the FedAvg displacement inside the subspace;
- leave orthogonal residual as FedAvg or diagonal-corrected.

Compare random subspace, true low-rank oracle subspace, and if cheap a learned
subspace from recent probe residuals.

## Shrinkage

Use split-probe residual shrinkage from E017. Include tuned-gamma only as an
upper-bound control if useful.

## Decision Labels

Use exactly one:

- `LOWRANK_GEOMETRY_SURVIVES`: low-rank/sketched or combined correction beats
  tuned scalar and diagonal-only correction with positive CI.
- `DIAGONAL_SUFFICES`: diagonal correction matches low-rank/sketched correction;
  dense geometry adds no value on this bed.
- `DENSE_GEOMETRY_BREAKS_METHOD`: oracle has headroom but estimated low-rank /
  diagonal arms fail against tuned scalar.
- `NO_ORACLE_HEADROOM`: even oracle-H correction cannot beat tuned scalar.
- `SCALAR_OR_FALLBACK_EXPLAINS`: tuned scalar or best-worker fallback explains
  the gain.
- `INVALID`: implementation/scoring bug.

## Required Output

Write:

```text
experiments/lgp_e021_dense_lowrank_correlated_stress_result_20261003.md
```

Upload script, result note, and run log to:

```text
aai_research_tlv/tree/local_gp/cpu_pings/20261003/
```

The result must say whether the next representation should be diagonal,
block-diagonal, low-rank/sketched, or low-rank-plus-diagonal.
