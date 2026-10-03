# LGP-E017 Risk Shrinkage / Fallback Result - 2026-10-03

## Decision

`RISK_SHRINKAGE_WORKS`

## Question

E015 showed that probe-fed dynamics-aware correction beats a tuned scalar
FedAvg scale on the nonlinear/heterogeneous CPU bed, but it still used a
pilot-selected gamma. E017 asks whether a target-free residual shrinkage rule
can replace that gamma tuning, and whether an oracle best-worker endpoint
fallback explains the gain.

## Bed

- E015 diagonal nonlinear/heterogeneous global objective;
- `d=80`, `K=8`, `eta=0.025`;
- `sigma=0.35`, centered worker heterogeneity `het=0.35`;
- random probes with `probe_reps=4`, `probe_radius=0.25`;
- held-out ratios are loss improvement divided by raw FedAvg improvement.

The tuned probe arm fits curvature with the full random-probe budget and
selects one gamma on pilot rounds. The risk-aware arm splits the probes, fits
on half, scores residual risk on held-out probe pairs, then applies one of:

- `risk_reliability`: `gamma = clip(1 - rho, 0, 1)`;
- `risk_sqrt`: `gamma = clip(sqrt(max(0, 1 - rho)), 0, 1)`.

The rule is selected only between those two options on pilot rounds and frozen
for held-out scoring.

## Result

| N | scale* | gamma* | risk rule | pilot risk gamma | heldout risk gamma | raw | scalar | tuned probe | full probe | risk probe | server secant | best worker true | probe - scalar | risk - tuned | best worker - probe |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.35 | 0.40 | risk_reliability | 0.201 | 0.202 | 1.000 +/- 0.000 | 1.075 +/- 0.030 | 1.288 +/- 0.087 | -0.008 +/- 0.117 | 1.226 +/- 0.066 | 0.959 +/- 0.103 | 0.620 +/- 0.068 | 0.213 +/- 0.061 | -0.061 +/- 0.046 | -0.667 +/- 0.146 |
| 32 | 1.45 | 0.65 | risk_sqrt | 0.460 | 0.462 | 1.000 +/- 0.000 | 1.070 +/- 0.033 | 1.554 +/- 0.127 | 1.321 +/- 0.051 | 1.460 +/- 0.107 | 1.052 +/- 0.061 | 0.647 +/- 0.202 | 0.484 +/- 0.095 | -0.094 +/- 0.030 | -0.907 +/- 0.324 |
| 128 | 1.35 | 0.95 | risk_sqrt | 0.466 | 0.465 | 1.000 +/- 0.000 | 1.127 +/- 0.024 | 2.039 +/- 0.115 | 2.038 +/- 0.116 | 1.773 +/- 0.075 | 1.093 +/- 0.108 | 0.695 +/- 0.045 | 0.912 +/- 0.092 | -0.266 +/- 0.041 | -1.345 +/- 0.125 |

## Interpretation

Risk-aware shrinkage clears this CPU gate. It beats tuned scalar FedAvg scale
at every worker count and stays within roughly 5%, 6%, and 13% of tuned-gamma
random-probe correction for N=8, 32, and 128 respectively. That is inside the
requested 10-15% band, while removing arbitrary target gamma tuning.

Full gamma is not safe at low worker count: it is destructive at N=8. The
residual rule is doing real shrinkage, not just rubber-stamping the full
correction.

The best-worker endpoint fallback does not explain the probe gain on this bed.
Even the oracle `best_worker_true` endpoint is materially below tuned probe and
below risk-aware probe at all three worker counts. Averaging cancels the
centered worker bias better than selecting one local endpoint.

Server/time secants also do not explain the gain. They are close to raw FedAvg
and remain far below the probe-fed dynamics-aware arms.

This is still a CPU synthetic result only. The product-relevant positive is:
for this bed, random probe residuals are enough to set a useful non-targeted
shrinkage rule, and the gain is not matched by a cheap elitist fallback.
