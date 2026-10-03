# LGP-E023 Random Probe Budget Confirmation Result - 2026-10-03

## Decision

`TWO_N_RANDOM_CONFIRMED`

## Question

E020 showed that the current active probe policies do not robustly reduce the
probe budget, but random probes remain strong. E023 asks for a cleaner random
budget confirmation with more held-out reps and a direct check of
`probe_reps=1` versus `probe_reps=4`.

All probe arms use the K-step dynamics-aware local-map correction

\[
  [I - (I - \eta \hat H)^K]^{-1}\Delta_K
\]

with the residual shrinkage rule selected on pilot rounds between
`risk_reliability` and `risk_sqrt`. The only target-tuned controls are the
FedAvg scalar scale and the labeled tuned-gamma upper bound.

## Parameters

Primary bed: E016 rotated 2D nonlinear blocks with quartic terms in each block
eigenbasis.

`d=40`, `blocks=20`, `K=8`, `eta=0.025`, `sigma=0.35`, `het=0.35`,
`rounds=32`, `pilot_reps=3`, `heldout_reps=8`,
`probe_rep_options=(1, 4)`, `probe_radius=0.25`,
`prior_ridge=0.001`, `prior_blend=0.35`.

Budgets are `N`, `2N`, `4N`, and `N*K`, plus `max(4, N//2)` for `N=8`.

## Control Summary

Ratios are held-out loss improvement divided by raw FedAvg improvement. CIs are
paired across held-out replicate batches.

| N | scale* | scalar | full r1 | full r4 | oracle-H | secant | best worker | full r4 - scalar | best worker - full r4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.35 | 1.099 +/- 0.055 | 1.385 +/- 0.271 | 1.461 +/- 0.279 | 1.478 +/- 0.364 | -8.152 +/- 3.593 | 0.700 +/- 0.182 | 0.362 +/- 0.230 | -0.761 +/- 0.456 |
| 32 | 1.75 | 1.032 +/- 0.117 | 1.445 +/- 0.112 | 1.535 +/- 0.138 | 1.743 +/- 0.190 | -6.334 +/- 3.178 | 0.864 +/- 0.051 | 0.503 +/- 0.054 | -0.671 +/- 0.183 |
| 128 | 1.45 | 1.116 +/- 0.062 | 1.608 +/- 0.130 | 1.827 +/- 0.181 | 2.054 +/- 0.231 | -4.487 +/- 2.672 | 0.890 +/- 0.098 | 0.711 +/- 0.133 | -0.937 +/- 0.227 |

## Budget Table

`budget - scalar` is the paired difference against tuned scalar FedAvg.
`budget - full` is the paired difference against full `N*K` random probes with
the same `probe_reps`.

| N | budget | reps | rule | gamma pilot | gamma heldout | ratio | budget - scalar | budget - full | tuned gamma | tuned ratio | risk - tuned |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 4 | 1 | risk_reliability | 0.092 | 0.096 | 1.016 +/- 0.021 | -0.083 +/- 0.040 | -0.369 +/- 0.254 | 0.20 | 1.158 +/- 0.101 | -0.143 +/- 0.085 |
| 8 | 4 | 4 | risk_reliability | 0.278 | 0.270 | 0.999 +/- 0.030 | -0.100 +/- 0.037 | -0.462 +/- 0.251 | 0.25 | 1.169 +/- 0.118 | -0.170 +/- 0.091 |
| 8 | 8 | 1 | risk_reliability | 0.041 | 0.047 | 1.079 +/- 0.056 | -0.020 +/- 0.038 | -0.306 +/- 0.230 | 0.15 | 1.226 +/- 0.150 | -0.146 +/- 0.102 |
| 8 | 8 | 4 | risk_reliability | 0.271 | 0.277 | 1.206 +/- 0.099 | 0.107 +/- 0.050 | -0.255 +/- 0.183 | 0.25 | 1.286 +/- 0.145 | -0.080 +/- 0.049 |
| 8 | 16 | 1 | risk_reliability | 0.044 | 0.053 | 1.094 +/- 0.042 | -0.005 +/- 0.036 | -0.292 +/- 0.236 | 0.20 | 1.290 +/- 0.216 | -0.197 +/- 0.179 |
| 8 | 16 | 4 | risk_reliability | 0.309 | 0.306 | 1.312 +/- 0.240 | 0.213 +/- 0.197 | -0.149 +/- 0.072 | 0.30 | 1.354 +/- 0.261 | -0.042 +/- 0.028 |
| 8 | 32 | 1 | risk_sqrt | 0.286 | 0.279 | 1.125 +/- 0.353 | 0.026 +/- 0.330 | -0.260 +/- 0.245 | 0.20 | 1.313 +/- 0.225 | -0.188 +/- 0.210 |
| 8 | 32 | 4 | risk_reliability | 0.344 | 0.339 | 1.399 +/- 0.272 | 0.300 +/- 0.224 | -0.062 +/- 0.032 | 0.30 | 1.412 +/- 0.275 | -0.012 +/- 0.012 |
| 8 | 64 | 1 | risk_sqrt | 0.333 | 0.329 | 1.385 +/- 0.271 | 0.286 +/- 0.224 | 0.000 +/- 0.000 | 0.25 | 1.411 +/- 0.266 | -0.026 +/- 0.018 |
| 8 | 64 | 4 | risk_reliability | 0.366 | 0.369 | 1.461 +/- 0.279 | 0.362 +/- 0.230 | 0.000 +/- 0.000 | 0.30 | 1.455 +/- 0.278 | 0.006 +/- 0.007 |
| 32 | 32 | 1 | risk_sqrt | 0.283 | 0.285 | 1.338 +/- 0.101 | 0.307 +/- 0.044 | -0.106 +/- 0.050 | 0.30 | 1.350 +/- 0.105 | -0.012 +/- 0.009 |
| 32 | 32 | 4 | risk_reliability | 0.341 | 0.345 | 1.356 +/- 0.085 | 0.324 +/- 0.060 | -0.179 +/- 0.062 | 0.40 | 1.371 +/- 0.096 | -0.014 +/- 0.012 |
| 32 | 64 | 1 | risk_sqrt | 0.329 | 0.330 | 1.363 +/- 0.106 | 0.331 +/- 0.048 | -0.082 +/- 0.027 | 0.30 | 1.366 +/- 0.106 | -0.003 +/- 0.014 |
| 32 | 64 | 4 | risk_reliability | 0.365 | 0.368 | 1.426 +/- 0.113 | 0.394 +/- 0.045 | -0.109 +/- 0.031 | 0.40 | 1.426 +/- 0.116 | 0.000 +/- 0.004 |
| 32 | 128 | 1 | risk_sqrt | 0.349 | 0.351 | 1.416 +/- 0.107 | 0.385 +/- 0.045 | -0.028 +/- 0.020 | 0.40 | 1.415 +/- 0.099 | 0.002 +/- 0.015 |
| 32 | 128 | 4 | risk_reliability | 0.375 | 0.376 | 1.452 +/- 0.116 | 0.420 +/- 0.035 | -0.083 +/- 0.031 | 0.45 | 1.460 +/- 0.122 | -0.009 +/- 0.010 |
| 32 | 256 | 1 | risk_sqrt | 0.355 | 0.360 | 1.445 +/- 0.112 | 0.413 +/- 0.042 | 0.000 +/- 0.000 | 0.40 | 1.449 +/- 0.111 | -0.004 +/- 0.006 |
| 32 | 256 | 4 | risk_sqrt | 0.612 | 0.616 | 1.535 +/- 0.138 | 0.503 +/- 0.054 | 0.000 +/- 0.000 | 0.60 | 1.528 +/- 0.139 | 0.007 +/- 0.003 |
| 128 | 128 | 1 | risk_sqrt | 0.352 | 0.348 | 1.580 +/- 0.129 | 0.464 +/- 0.084 | -0.028 +/- 0.026 | 0.40 | 1.593 +/- 0.132 | -0.013 +/- 0.009 |
| 128 | 128 | 4 | risk_sqrt | 0.613 | 0.610 | 1.650 +/- 0.136 | 0.534 +/- 0.093 | -0.177 +/- 0.059 | 0.50 | 1.651 +/- 0.136 | -0.001 +/- 0.016 |
| 128 | 256 | 1 | risk_sqrt | 0.360 | 0.360 | 1.594 +/- 0.133 | 0.478 +/- 0.089 | -0.014 +/- 0.037 | 0.40 | 1.612 +/- 0.138 | -0.018 +/- 0.006 |
| 128 | 256 | 4 | risk_sqrt | 0.615 | 0.615 | 1.737 +/- 0.161 | 0.621 +/- 0.115 | -0.090 +/- 0.027 | 0.60 | 1.731 +/- 0.161 | 0.006 +/- 0.003 |
| 128 | 512 | 1 | risk_sqrt | 0.364 | 0.364 | 1.607 +/- 0.134 | 0.491 +/- 0.090 | -0.001 +/- 0.020 | 0.60 | 1.654 +/- 0.159 | -0.047 +/- 0.046 |
| 128 | 512 | 4 | risk_sqrt | 0.618 | 0.617 | 1.792 +/- 0.171 | 0.676 +/- 0.124 | -0.035 +/- 0.014 | 0.80 | 1.788 +/- 0.169 | 0.004 +/- 0.019 |
| 128 | 1024 | 1 | risk_sqrt | 0.365 | 0.365 | 1.608 +/- 0.130 | 0.492 +/- 0.091 | 0.000 +/- 0.000 | 0.65 | 1.733 +/- 0.164 | -0.125 +/- 0.041 |
| 128 | 1024 | 4 | risk_sqrt | 0.620 | 0.619 | 1.827 +/- 0.181 | 0.711 +/- 0.133 | 0.000 +/- 0.000 | 0.95 | 1.891 +/- 0.201 | -0.064 +/- 0.036 |

## Probe Repeat Check

The table reports `rep1 - rep4` at each spatial budget.

| N | budget | rep1 - rep4 |
|---:|---:|---:|
| 8 | 4 | 0.017 +/- 0.020 |
| 8 | 8 | -0.127 +/- 0.056 |
| 8 | 16 | -0.219 +/- 0.203 |
| 8 | 32 | -0.274 +/- 0.229 |
| 8 | 64 | -0.076 +/- 0.037 |
| 32 | 32 | -0.018 +/- 0.062 |
| 32 | 64 | -0.063 +/- 0.038 |
| 32 | 128 | -0.035 +/- 0.029 |
| 32 | 256 | -0.090 +/- 0.035 |
| 128 | 128 | -0.070 +/- 0.023 |
| 128 | 256 | -0.143 +/- 0.045 |
| 128 | 512 | -0.185 +/- 0.041 |
| 128 | 1024 | -0.219 +/- 0.057 |

## Interpretation

This gate confirms the E020 budget call for the rotated nonlinear CPU bed. At
`N=32`, `2N` random probes beat tuned scalar by `0.331 +/- 0.048` with
`probe_reps=1` and by `0.394 +/- 0.045` with `probe_reps=4`; the same arms are
only `0.082 +/- 0.027` and `0.109 +/- 0.031` below their full `N*K` matched-rep
baselines. At `N=128`, `2N` beats tuned scalar by `0.478 +/- 0.089` and
`0.621 +/- 0.115`, and is only `0.014 +/- 0.037` and `0.090 +/- 0.027` below
matched full random.

The small-worker case still wants more probes. At `N=8`, `2N` is not clearly
better than tuned scalar, while `N*K` remains the stable arm. This matches the
E020 warning that very small worker counts need the full random budget.

`probe_reps=4` improves absolute ratios, especially at `N=128`, but
`probe_reps=1` does not fail: it clears tuned scalar for `N >= 32` and stays
close to the matched `probe_reps=1` full-random baseline. Therefore the result
is not `PROBE_REPS_CRITICAL`. Repeats are a variance/quality knob, not a
requirement for the basic `2N` budget conclusion.

The tuned scalar, server/time secant, and best-worker controls do not explain
the probe gain. Tuned scalar is well below the `2N` and full random arms,
server/time secants are destructive, and the oracle best-worker fallback stays
below raw FedAvg-scale correction.

## Recommendation

Use residual-shrunk random probes with spatial budget `2N` as the default for
`N >= 32` on this CPU bed. Use `probe_reps=1` when gradient-call budget is the
binding cost, and reserve `probe_reps=4` for variance reduction when extra
global-gradient repeats are cheap. Keep full `N*K` random probes for very small
worker counts such as `N=8`.

This is still a synthetic CPU result. It should set the default random-probe
budget for the local-map estimator, but not replace the pending real-path GPU
gates or the dense trust-layer follow-up.

## Command

```bash
python3 experiments/lgp_e023_random_probe_budget_confirm.py
```
