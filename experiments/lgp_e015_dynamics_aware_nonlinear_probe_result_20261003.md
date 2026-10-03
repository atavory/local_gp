# LGP-E015 Dynamics-Aware Nonlinear Probe Result - 2026-10-03

## Decision

`DYNAMICS_AWARE_PROBES_SURVIVE_NONLINEAR_BED`

## Question

E014 showed that the corrected K-step local-map operator works on an exact
rotated-block quadratic:
\[
  [I-(I-\eta \hat H)^K]^{-1}\Delta_K.
\]

E015 moves that operator to the E008-E010 nonlinear/heterogeneous CPU bed:

- diagonal nonlinear global objective \(0.5\lambda x^2 + 0.25 qx^4\);
- centered worker heterogeneity;
- stochastic gradients;
- K=8 local steps.

The control is source-selected scalar FedAvg scale. The tested shape sources
are:

- centered local trajectory pairs (`path`);
- random probe points around the round start (`random_probe`);
- coordinate-directed probe points around the round start (`coord_probe`);
- true local diagonal curvature (`oracle`).

All non-scalar shape arms source-select one gamma on pilot rounds.

## Result

Parameters: `d=80`, `K=8`, `eta=0.025`, `sigma=0.35`, `het=0.35`,
`probe_reps=4`, `probe_radius=0.25`.

Ratios are held-out loss improvement divided by raw FedAvg improvement.

| N | scale* | oracle* | path* | rand* | coord* | scalar | oracle | path | random probe | coord probe | server secant | coord - scalar | random - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.30 | 0.60 | 0.00 | 0.35 | 0.35 | 1.069 +/- 0.027 | 1.507 +/- 0.142 | 1.000 +/- 0.000 | 1.306 +/- 0.094 | 1.134 +/- 0.049 | 0.911 +/- 0.177 | 0.064 +/- 0.029 | 0.237 +/- 0.069 |
| 32 | 1.30 | 0.95 | 0.00 | 0.60 | 0.60 | 1.053 +/- 0.045 | 1.650 +/- 0.295 | 1.000 +/- 0.000 | 1.445 +/- 0.193 | 1.424 +/- 0.185 | 0.974 +/- 0.120 | 0.371 +/- 0.141 | 0.391 +/- 0.149 |
| 128 | 1.30 | 1.00 | 0.00 | 0.95 | 1.00 | 1.101 +/- 0.038 | 2.134 +/- 0.265 | 1.000 +/- 0.000 | 2.009 +/- 0.248 | 2.034 +/- 0.256 | 1.113 +/- 0.153 | 0.934 +/- 0.219 | 0.909 +/- 0.211 |

## Interpretation

This is the CPU work Muse was pointing at. The useful object is not a plain
quasi-Newton step and not worker trajectory scatter by itself. It is a
dynamics-aware correction to the local-update map, fed by noise-controlled
curvature probes.

What survives:

- probe-fed dynamics-aware shape beats tuned scalar FedAvg scale;
- it also beats server/time secants;
- the gain grows with probe budget / worker count;
- random probes are already enough on this diagonal nonlinear bed;
- coordinate probes catch up and slightly beat random probes at N=128.

What does not survive:

- path-only centered trajectory pairs select gamma zero and tie FedAvg;
- server secants remain weak;
- this is still synthetic CPU, not a real optimizer product.

## Consequence

The local-GP branch now has a concrete CPU-positive mechanism:

1. use local/probe geometry to estimate the local update map;
2. apply the K-step dynamics correction to the FedAvg displacement;
3. shrink by pilot-selected or risk-aware gamma;
4. beat tuned scalar FedAvg scale on a nonlinear/heterogeneous bed.

The next CPU gates should be:

1. remove the oracle-like synthetic coordinate structure: rotated nonlinear
   blocks or dense low-rank curvature;
2. replace pilot-selected gamma with a risk-aware rule tied to probe residual /
   nonlinearity;
3. add an elitist best-worker fallback baseline;
4. then move to a real-path gate only after those survive.
