# LGP-E016 Rotated Nonlinear Probe Result - 2026-10-03

## Decision

`DYNAMICS_AWARE_SURVIVES_ROTATION`

## Question

E015 showed that probe-fed dynamics-aware correction beat a tuned scalar
FedAvg scale on a diagonal nonlinear/heterogeneous CPU bed. E016 asks whether
that was just a coordinate artifact.

This run replaces the diagonal objective with rotated 2D nonlinear blocks:

\[
  f_b(z) = 0.5 z^T H_b z
    + 0.25 q_1 (v_1^T z)^4
    + 0.25 q_2 (v_2^T z)^4,
\]

where \(v_1, v_2\) are the eigenvectors of \(H_b\). All shape arms apply the
K-step local-map correction

\[
  [I - (I - \eta \hat H)^K]^{-1}\Delta_K
\]

blockwise to the FedAvg displacement, then use one pilot-selected shrinkage
gamma. The scalar null is a tuned FedAvg scale selected on the same pilot
rounds.

## Parameters

`d=40`, `blocks=20`, `sharp_blocks=4`, `K=8`, `eta=0.025`, `sigma=0.35`,
`het=0.35`, `rounds=40`, `pilot_reps=4`, `heldout_reps=4`, `probe_reps=4`,
`probe_radius=0.25`, `C_max=25`, `secant_window=8`.

Probe budget is `N*K` offsets per round for each probe arm.

## Held-Out Ratios

Ratios are held-out loss improvement divided by raw FedAvg improvement.

| N | scale* | oracle* | path* | rand* | coord* | dir* | scalar | oracle | path | random probe | coord probe | directed probe | server secant | best worker | random - scalar | directed - scalar | best worker - random |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.30 | 0.60 | 0.00 | 0.35 | 0.25 | 0.25 | 1.032 +/- 0.028 | 1.323 +/- 0.119 | 1.000 +/- 0.000 | 1.254 +/- 0.103 | 1.271 +/- 0.106 | 1.268 +/- 0.103 | -5.545 +/- 3.016 | 0.729 +/- 0.100 | 0.221 +/- 0.078 | 0.236 +/- 0.079 | -0.524 +/- 0.174 |
| 32 | 1.50 | 1.00 | 0.00 | 0.70 | 0.70 | 0.70 | 1.057 +/- 0.058 | 1.646 +/- 0.220 | 1.000 +/- 0.000 | 1.497 +/- 0.183 | 1.491 +/- 0.172 | 1.497 +/- 0.161 | -4.619 +/- 2.661 | 0.799 +/- 0.125 | 0.441 +/- 0.141 | 0.441 +/- 0.114 | -0.698 +/- 0.272 |
| 128 | 1.40 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.132 +/- 0.055 | 2.227 +/- 0.273 | 1.000 +/- 0.000 | 2.124 +/- 0.268 | 2.125 +/- 0.268 | 2.122 +/- 0.259 | -5.349 +/- 2.313 | 0.876 +/- 0.036 | 0.992 +/- 0.217 | 0.990 +/- 0.206 | -1.248 +/- 0.248 |

## Interpretation

The E015 gain does not look like a diagonal-coordinate artifact on this CPU
synthetic gate. Random probes and eigen-directed non-coordinate probes both
beat the tuned scalar scale with positive held-out CIs at N=8, 32, and 128.
Coordinate probes are comparable, but they are not uniquely responsible for
the gain.

The best-worker true fallback does not explain the result here: it is below
raw FedAvg and well below random-probe correction. This is plausible because
averaging cancels worker noise and centered heterogeneity on this bed, while a
single endpoint remains noisy.

The server/time secant baseline is unstable and strongly negative on this
rotated nonlinear bed. Path-only local trajectory pairs again select gamma
zero and tie raw FedAvg.

This is still only a synthetic CPU gate. It supports keeping the
dynamics-aware local-map correction alive for the next shrinkage/fallback
checks, but it is not a real optimizer product claim.

## Command

```bash
python3 experiments/lgp_e016_rotated_nonlinear_probe.py
```
