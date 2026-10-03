# LGP-E021 Dense Low-Rank / Correlated-Noise Stress Result - 2026-10-03

## Decision

`DENSE_GEOMETRY_BREAKS_METHOD`

## Question

E016 showed that probe-fed dynamics-aware correction survives rotated 2D
nonlinear blocks. E021 asks whether the mechanism survives a denser bed where
the curvature and noise are not separable into independent coordinate or tiny
block pieces.

## Bed

- `D=96`, low-rank dimension `rank=8`, sketched dimension `8`;
- Hessian base: `diag(base spectrum) + U diag(spikes) U^T`;
- nonlinear terms: weak diagonal quartics plus stronger quartics in the same
  low-rank directions;
- centered worker heterogeneity with both diagonal and low-rank components;
- correlated gradient noise:
  `noise = diagonal_noise + U lowrank_noise`;
- `K=8`, `eta=0.025`, `rounds=24`, `pilot_reps=3`,
  `heldout_reps=4`;
- `probe_reps=3`, full random probe budget capped at `128`, subspace probe
  budget `64`;
- estimated correction arms use the E017 residual shrinkage rule
  `gamma = sqrt(clip(1 - rho, 0, 1))`.

All correction arms apply the K-step local-map operator
\[
  [I - (I - \eta \hat H)^K]^{-1}\Delta_K
\]
to the FedAvg displacement. Held-out ratios are loss improvement divided by raw
FedAvg improvement.

## Arms

- `scalar`: tuned FedAvg scale selected on pilot rounds;
- `diag`: diagonal random-probe curvature plus residual shrinkage;
- `lr_rand`: random sketched subspace correction plus residual shrinkage;
- `lr_learn`: residual-learned sketched subspace correction plus residual
  shrinkage;
- `lr_true`: true low-rank subspace, probe-estimated curvature, residual
  shrinkage;
- `combo`: learned low-rank-plus-diagonal correction plus residual shrinkage;
- `combo_true`: true low-rank subspace plus diagonal correction, residual
  shrinkage;
- `combo_tuned`: learned low-rank-plus-diagonal correction with pilot-tuned
  gamma as an upper-bound control;
- `oracle`: true dense Hessian correction with pilot-tuned gamma;
- `secant`: server/time diagonal secant;
- `best`: oracle best worker endpoint after the local round.

## Result

| N | scale* | oracle* | combo tuned gamma* | mean diag gamma | mean learned-lr gamma | mean combo gamma | raw | scalar | diag | lr random | lr learned | lr true-subspace | combo learned | combo true-subspace | combo tuned | oracle-H | secant | best worker | combo - scalar | combo - diag | best - combo |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.25 | 0.85 | 0.00 | 0.167 | 0.938 | 0.256 | 1.000 +/- 0.000 | 1.052 +/- 0.061 | 0.942 +/- 0.259 | 0.057 +/- 0.566 | 0.895 +/- 0.149 | 0.999 +/- 0.001 | -22.671 +/- 17.767 | 0.322 +/- 0.592 | 1.000 +/- 0.000 | 2.658 +/- 1.190 | 0.643 +/- 0.278 | 0.900 +/- 0.071 | -23.723 +/- 17.713 | -23.613 +/- 17.571 | 23.571 +/- 17.824 |
| 32 | 1.15 | 1.00 | 0.05 | 0.200 | 0.942 | 0.336 | 1.000 +/- 0.000 | 1.052 +/- 0.041 | 0.877 +/- 0.333 | 0.742 +/- 0.160 | 0.965 +/- 0.025 | 1.001 +/- 0.001 | -1.457 +/- 0.585 | 0.986 +/- 0.026 | 1.101 +/- 0.084 | 3.304 +/- 1.790 | 0.483 +/- 0.418 | 0.972 +/- 0.013 | -2.509 +/- 0.549 | -2.335 +/- 0.322 | 2.429 +/- 0.594 |
| 128 | 1.20 | 1.00 | 0.05 | 0.198 | 0.941 | 0.331 | 1.000 +/- 0.000 | 1.042 +/- 0.023 | 0.808 +/- 0.221 | 0.818 +/- 0.071 | 0.606 +/- 0.725 | 1.002 +/- 0.001 | -3.225 +/- 4.033 | 0.959 +/- 0.181 | 1.061 +/- 0.044 | 2.778 +/- 0.781 | 0.743 +/- 0.144 | 1.002 +/- 0.016 | -4.267 +/- 4.026 | -4.033 +/- 3.845 | 4.227 +/- 4.019 |

## Interpretation

This gate does not clear. The true dense Hessian has large headroom: oracle-H
beats tuned scalar by roughly 1.6-2.3 ratio points on held-out rounds. So the
bed is not saying that scalar FedAvg is inherently enough.

The estimated representations do not capture that headroom:

- diagonal random probes do not beat tuned scalar;
- random and learned sketched low-rank probes do not beat tuned scalar;
- even the true low-rank subspace with probe-estimated curvature is basically
  raw FedAvg, which means low-rank-only leaves too much orthogonal dynamics
  uncorrected;
- learned low-rank-plus-diagonal with E017 residual shrinkage is unsafe here;
- the tuned-gamma combined upper bound falls back to nearly zero gamma, so the
  pilot itself rejects the learned combined correction;
- true-subspace low-rank-plus-diagonal is much less destructive but still does
  not beat tuned scalar with positive CI.

The best-worker fallback does not explain the gain. It is near raw FedAvg and
below the tuned scalar / oracle-H gap. Server/time secants are also weak.

## Representation Call

Do not move forward with a diagonal or tiny block-diagonal representation for
this denser regime. The oracle result says the missing object is dense
low-rank-plus-diagonal geometry, but the current probe estimator and E017
residual shrinkage are not reliable enough under correlated low-rank noise.

The next representation should therefore be **low-rank-plus-diagonal**, but
only with a stricter trust layer: held-out loss proxy validation, stronger
subspace identification, or a shrinkage rule that can zero the combined update
when residual probes remain deceptively small.

This is a CPU synthetic stress result only, not an optimizer product claim.

## Command

```bash
python3 experiments/lgp_e021_dense_lowrank_correlated_stress.py
```
