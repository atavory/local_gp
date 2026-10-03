# LGP-E014 Dynamics-Aware Correction Result - 2026-10-03

## Decision

`DYNAMICS_AWARE_PROBES_WIN_ON_EXACT_QUADRATIC`

## Question

E013 used worker secants inside a plain quasi-Newton consolidation step. That
double-applies curvature: K local steps have already moved partway from a
gradient step toward a Newton step.

On a quadratic with no momentum,
\[
  \Delta_K
    = x_K - x_0
    = -H^{-1}\left[I-(I-\eta H)^K\right]g_0.
\]

The remaining correction is therefore
\[
  \left[I-(I-\eta H)^K\right]^{-1}\Delta_K,
\]
not another \(H^{-1}g\) update.

E014 tests that corrected operator on the rotated-block quadratic used by the
event-triggered branch. It estimates block Hessians from:

- centered local trajectory pairs (`path`);
- random probe offsets with budget `N*K`;
- oracle-directed probe offsets with budget `N*K`;
- true Hessian (`oracle`).

All non-scalar arms use source-selected gamma on pilot events. The null is a
tuned scalar FedAvg scale selected on the same pilot events.

## Main Run

Parameters: `d=40`, `K=8`, `eta=0.018`, `sigma=0.04`,
`probe_reps=4`, `probe_radius=0.45`, `C_max=30`.

Ratios are actual held-out one-event loss improvement divided by raw FedAvg
improvement.

| N | scale* | oracle* | path* | rand* | dir* | scalar | oracle | path | random probe | directed probe | directed - scalar | path - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.30 | 1.00 | 0.05 | 1.00 | 0.85 | 1.054 +/- 0.004 | 1.359 +/- 0.024 | 1.061 +/- 0.007 | 1.358 +/- 0.024 | 1.332 +/- 0.024 | 0.278 +/- 0.021 | 0.007 +/- 0.005 |
| 32 | 1.30 | 1.00 | 0.25 | 1.00 | 1.00 | 1.047 +/- 0.003 | 1.301 +/- 0.021 | 1.097 +/- 0.024 | 1.301 +/- 0.021 | 1.301 +/- 0.021 | 0.255 +/- 0.019 | 0.050 +/- 0.023 |
| 128 | 1.30 | 1.00 | 0.55 | 1.00 | 1.00 | 1.051 +/- 0.004 | 1.326 +/- 0.023 | 1.214 +/- 0.022 | 1.326 +/- 0.023 | 1.326 +/- 0.023 | 0.275 +/- 0.020 | 0.163 +/- 0.021 |

## Stress Sanity

Same code with `sigma=0.10`, `pilot_events=40`, `heldout_events=80`
without editing the experiment file:

| N | scalar | oracle | path | random probe | directed probe | directed - scalar | path - scalar |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.053 +/- 0.006 | 1.356 +/- 0.052 | 1.050 +/- 0.010 | 1.350 +/- 0.051 | 1.327 +/- 0.048 | 0.273 +/- 0.043 | -0.004 +/- 0.007 |
| 32 | 1.047 +/- 0.008 | 1.308 +/- 0.031 | 1.073 +/- 0.048 | 1.307 +/- 0.031 | 1.307 +/- 0.031 | 0.260 +/- 0.025 | 0.027 +/- 0.045 |
| 128 | 1.051 +/- 0.006 | 1.347 +/- 0.047 | 1.231 +/- 0.030 | 1.346 +/- 0.047 | 1.346 +/- 0.047 | 0.295 +/- 0.042 | 0.180 +/- 0.025 |

## Interpretation

The architecture bug is real and actionable. Once the surrogate corrects the
local-update dynamics rather than taking a fresh quasi-Newton step, the shape
beats a tuned scalar FedAvg scale by about 0.25-0.30 improvement-ratio points.

The positive is conditional:

- this is an exact quadratic with known 2D block structure;
- the successful estimated arms use `N*K` extra probe points and repeated
  noisy gradients;
- random and directed probes tie because the bed is exactly block quadratic and
  the probe budget is large enough;
- trajectory-only centered path secants are much weaker at N=8 and only become
  useful at N=128.

So E014 is not a product claim. It is a corrected Stage-0 mechanism result:
if we can get noise-controlled curvature-action probes, the dynamics-aware
correction has shape value that scalar FedAvg scale cannot absorb.

## Consequence

Stop testing plain L-BFGS / BB consolidation as the event-triggered product.
The live branch is now:

1. estimate local curvature or the local-update map from path/probe geometry;
2. apply \([I-(I-\eta \hat H)^K]^{-1}\) to the FedAvg displacement;
3. shrink by model-risk, not by the old gamma-flat reliability heuristic;
4. compare against tuned scalar FedAvg scale and server/time secants.

The next gate should move this corrected operator off the exact quadratic:
first to the E008-E010 nonlinear/heterogeneous bed, then to a real path if the
shape survives.
