# LGP-E020 Probe Budget / Active Design Result - 2026-10-03

## Decision

`RANDOM_BUDGET_REQUIRED`

## Question

E016 showed that probe-fed dynamics-aware local-map correction survives a
rotated nonlinear synthetic CPU bed. E020 asks whether simple causal active
probe designs can reduce the probe budget versus plain random probes.

All probe arms use the K-step dynamics-aware correction

\[
  [I - (I - \eta \hat H)^K]^{-1}\Delta_K
\]

with E017-style residual shrinkage. The only target-tuned scalar is the FedAvg
scale null. For each `N x budget x policy`, the shrinkage rule is selected
between `risk_reliability` and `risk_sqrt` on pilot rounds and then frozen for
held-out scoring.

## Parameters

Primary bed: E016 rotated 2D nonlinear blocks with quartic terms in each block
eigenbasis.

`d=40`, `blocks=20`, `K=8`, `eta=0.025`, `sigma=0.35`, `het=0.35`,
`rounds=32`, `pilot_reps=3`, `heldout_reps=4`, `probe_reps=4`,
`probe_radius=0.25`, `prior_ridge=0.001`, `prior_blend=0.35`.

Budgets are `max(4, N//2)`, `N`, `2N`, and `N*K`.

Policies:

- `random`: Rademacher offsets.
- `value_weighted`: causal-prior eigen-directions sampled by
  `(1 - 1/C)^2 / max(C, eps)`.
- `uncertainty_value`: two-stage policy; half random exploration, then highest
  prior-value times split-estimate disagreement directions.

## Control Summary

Ratios are held-out loss improvement divided by raw FedAvg improvement. CIs are
paired across held-out replicate batches.

| N | scale* | scalar | full random | full rule | full gamma pilot | full gamma heldout | oracle-H | server secant | best worker true | full random - scalar | best worker - full random |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.25 | 1.061 +/- 0.020 | 1.252 +/- 0.067 | risk_reliability | 0.362 | 0.366 | 1.051 +/- 0.259 | -9.095 +/- 1.887 | 0.593 +/- 0.269 | 0.191 +/- 0.050 | -0.658 +/- 0.300 |
| 32 | 1.30 | 1.072 +/- 0.014 | 1.480 +/- 0.113 | risk_sqrt | 0.614 | 0.615 | 1.715 +/- 0.243 | -5.994 +/- 5.468 | 0.774 +/- 0.069 | 0.408 +/- 0.101 | -0.706 +/- 0.162 |
| 128 | 1.35 | 1.115 +/- 0.034 | 1.941 +/- 0.126 | risk_sqrt | 0.619 | 0.617 | 2.201 +/- 0.160 | -4.892 +/- 4.081 | 0.826 +/- 0.070 | 0.826 +/- 0.105 | -1.115 +/- 0.192 |

## Budget / Policy Table

`policy-randB` is the paired difference against the random policy at the same
budget. `policy-full` is the paired difference against full `N*K` random.

| N | budget | policy | rule | gamma pilot | gamma heldout | ratio | policy - scalar | policy - randB | policy - full random | best worker - policy |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 4 | random | risk_reliability | 0.249 | 0.283 | 0.857 +/- 0.177 | -0.204 +/- 0.193 | 0.000 +/- 0.000 | -0.395 +/- 0.209 | -0.263 +/- 0.095 |
| 8 | 4 | value_weighted | risk_sqrt | 0.192 | 0.177 | 1.016 +/- 0.006 | -0.045 +/- 0.017 | 0.159 +/- 0.181 | -0.236 +/- 0.065 | -0.423 +/- 0.273 |
| 8 | 4 | uncertainty_value | risk_reliability | 0.160 | 0.166 | 0.942 +/- 0.056 | -0.119 +/- 0.070 | 0.085 +/- 0.124 | -0.310 +/- 0.092 | -0.349 +/- 0.218 |
| 8 | 8 | random | risk_reliability | 0.280 | 0.287 | 1.074 +/- 0.054 | 0.012 +/- 0.050 | 0.000 +/- 0.000 | -0.178 +/- 0.043 | -0.480 +/- 0.260 |
| 8 | 8 | value_weighted | risk_sqrt | 0.138 | 0.168 | 1.055 +/- 0.036 | -0.006 +/- 0.023 | -0.018 +/- 0.073 | -0.197 +/- 0.061 | -0.462 +/- 0.303 |
| 8 | 8 | uncertainty_value | risk_reliability | 0.141 | 0.143 | 1.060 +/- 0.032 | -0.001 +/- 0.029 | -0.013 +/- 0.023 | -0.191 +/- 0.044 | -0.467 +/- 0.267 |
| 8 | 16 | random | risk_reliability | 0.304 | 0.298 | 1.141 +/- 0.089 | 0.080 +/- 0.078 | 0.000 +/- 0.000 | -0.111 +/- 0.037 | -0.547 +/- 0.287 |
| 8 | 16 | value_weighted | risk_sqrt | 0.235 | 0.212 | 1.089 +/- 0.015 | 0.028 +/- 0.026 | -0.052 +/- 0.080 | -0.163 +/- 0.064 | -0.496 +/- 0.263 |
| 8 | 16 | uncertainty_value | risk_reliability | 0.176 | 0.174 | 1.131 +/- 0.026 | 0.069 +/- 0.012 | -0.010 +/- 0.073 | -0.121 +/- 0.045 | -0.537 +/- 0.283 |
| 8 | 64 | random | risk_reliability | 0.362 | 0.366 | 1.252 +/- 0.067 | 0.191 +/- 0.050 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | -0.658 +/- 0.300 |
| 8 | 64 | value_weighted | risk_sqrt | 0.306 | 0.310 | 1.212 +/- 0.041 | 0.151 +/- 0.045 | -0.039 +/- 0.057 | -0.039 +/- 0.057 | -0.619 +/- 0.248 |
| 8 | 64 | uncertainty_value | risk_reliability | 0.233 | 0.234 | 1.261 +/- 0.080 | 0.200 +/- 0.062 | 0.009 +/- 0.032 | 0.009 +/- 0.032 | -0.668 +/- 0.325 |
| 32 | 16 | random | risk_reliability | 0.310 | 0.307 | 1.279 +/- 0.083 | 0.207 +/- 0.073 | 0.000 +/- 0.000 | -0.201 +/- 0.032 | -0.505 +/- 0.137 |
| 32 | 16 | value_weighted | risk_sqrt | 0.232 | 0.199 | 1.131 +/- 0.073 | 0.059 +/- 0.062 | -0.148 +/- 0.025 | -0.349 +/- 0.054 | -0.357 +/- 0.124 |
| 32 | 16 | uncertainty_value | risk_reliability | 0.171 | 0.177 | 1.205 +/- 0.076 | 0.133 +/- 0.065 | -0.074 +/- 0.008 | -0.275 +/- 0.039 | -0.431 +/- 0.130 |
| 32 | 32 | random | risk_reliability | 0.350 | 0.342 | 1.317 +/- 0.179 | 0.246 +/- 0.166 | 0.000 +/- 0.000 | -0.163 +/- 0.093 | -0.543 +/- 0.212 |
| 32 | 32 | value_weighted | risk_sqrt | 0.269 | 0.271 | 1.262 +/- 0.100 | 0.190 +/- 0.089 | -0.055 +/- 0.108 | -0.218 +/- 0.026 | -0.488 +/- 0.144 |
| 32 | 32 | uncertainty_value | risk_sqrt | 0.456 | 0.449 | 1.326 +/- 0.195 | 0.254 +/- 0.182 | 0.009 +/- 0.025 | -0.154 +/- 0.117 | -0.552 +/- 0.223 |
| 32 | 64 | random | risk_reliability | 0.363 | 0.365 | 1.422 +/- 0.151 | 0.351 +/- 0.138 | 0.000 +/- 0.000 | -0.058 +/- 0.043 | -0.648 +/- 0.193 |
| 32 | 64 | value_weighted | risk_sqrt | 0.296 | 0.295 | 1.355 +/- 0.127 | 0.284 +/- 0.116 | -0.067 +/- 0.036 | -0.125 +/- 0.024 | -0.582 +/- 0.170 |
| 32 | 64 | uncertainty_value | risk_sqrt | 0.480 | 0.484 | 1.410 +/- 0.149 | 0.338 +/- 0.136 | -0.013 +/- 0.024 | -0.070 +/- 0.051 | -0.636 +/- 0.184 |
| 32 | 256 | random | risk_sqrt | 0.614 | 0.615 | 1.480 +/- 0.113 | 0.408 +/- 0.101 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | -0.706 +/- 0.162 |
| 32 | 256 | value_weighted | risk_sqrt | 0.341 | 0.327 | 1.441 +/- 0.136 | 0.369 +/- 0.125 | -0.039 +/- 0.027 | -0.039 +/- 0.027 | -0.667 +/- 0.186 |
| 32 | 256 | uncertainty_value | risk_sqrt | 0.586 | 0.585 | 1.517 +/- 0.173 | 0.445 +/- 0.160 | 0.037 +/- 0.065 | 0.037 +/- 0.065 | -0.743 +/- 0.211 |
| 128 | 64 | random | risk_sqrt | 0.602 | 0.605 | 1.655 +/- 0.074 | 0.540 +/- 0.062 | 0.000 +/- 0.000 | -0.286 +/- 0.060 | -0.828 +/- 0.141 |
| 128 | 64 | value_weighted | risk_sqrt | 0.285 | 0.306 | 1.575 +/- 0.112 | 0.460 +/- 0.098 | -0.080 +/- 0.047 | -0.366 +/- 0.033 | -0.749 +/- 0.181 |
| 128 | 64 | uncertainty_value | risk_sqrt | 0.481 | 0.481 | 1.702 +/- 0.103 | 0.587 +/- 0.077 | 0.047 +/- 0.048 | -0.239 +/- 0.044 | -0.876 +/- 0.161 |
| 128 | 128 | random | risk_sqrt | 0.610 | 0.609 | 1.762 +/- 0.151 | 0.647 +/- 0.124 | 0.000 +/- 0.000 | -0.179 +/- 0.034 | -0.935 +/- 0.212 |
| 128 | 128 | value_weighted | risk_sqrt | 0.315 | 0.319 | 1.618 +/- 0.068 | 0.503 +/- 0.054 | -0.144 +/- 0.088 | -0.323 +/- 0.060 | -0.792 +/- 0.137 |
| 128 | 128 | uncertainty_value | risk_sqrt | 0.496 | 0.496 | 1.766 +/- 0.096 | 0.651 +/- 0.073 | 0.004 +/- 0.059 | -0.175 +/- 0.033 | -0.940 +/- 0.160 |
| 128 | 256 | random | risk_sqrt | 0.615 | 0.615 | 1.834 +/- 0.116 | 0.719 +/- 0.094 | 0.000 +/- 0.000 | -0.107 +/- 0.017 | -1.008 +/- 0.180 |
| 128 | 256 | value_weighted | risk_sqrt | 0.329 | 0.341 | 1.712 +/- 0.145 | 0.597 +/- 0.120 | -0.122 +/- 0.046 | -0.229 +/- 0.032 | -0.886 +/- 0.208 |
| 128 | 256 | uncertainty_value | risk_sqrt | 0.587 | 0.587 | 1.848 +/- 0.138 | 0.734 +/- 0.118 | 0.014 +/- 0.026 | -0.093 +/- 0.018 | -1.022 +/- 0.204 |
| 128 | 1024 | random | risk_sqrt | 0.619 | 0.617 | 1.941 +/- 0.126 | 0.826 +/- 0.105 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | -1.115 +/- 0.192 |
| 128 | 1024 | value_weighted | risk_sqrt | 0.353 | 0.358 | 1.727 +/- 0.120 | 0.613 +/- 0.100 | -0.213 +/- 0.010 | -0.213 +/- 0.010 | -0.901 +/- 0.186 |
| 128 | 1024 | uncertainty_value | risk_sqrt | 0.621 | 0.620 | 1.946 +/- 0.130 | 0.831 +/- 0.111 | 0.005 +/- 0.016 | 0.005 +/- 0.016 | -1.119 +/- 0.197 |

## Interpretation

Random probe budget still matters. Full `N*K` random beats tuned scalar with a
positive CI at all worker counts, but smaller budgets degrade monotonically on
the hardest small-N case. At N=8, only the full 64-probe budget clears scalar
comfortably.

The active policies do not provide a robust budget reduction. `value_weighted`
usually underperforms matched-budget random probes. `uncertainty_value` is the
better active policy: it roughly ties matched random at N=32 and N=128, and has
a small positive matched-budget edge at N=128 / budget 64, but the CI includes
zero and the same small budget random arm is already close to the 10-15%
full-random band.

The scalar and fallback controls do not explain the probe gain. Tuned scalar is
well below full random, and oracle `best_worker_true` is below every serious
probe arm. Server/time secants are unstable and not competitive on this bed.

## Recommendation

The cheapest plausible policy from this gate is still random, not active:
use residual-shrunk random probes with budget around `2N` when `N >= 32`, and
fall back to full `N*K` when `N` is very small. Keep `uncertainty_value` as a
candidate for a better active design, but do not replace random probes with the
current value-weighted policy.

## Command

```bash
python3 -u experiments/lgp_e020_probe_budget_active_design.py
```
