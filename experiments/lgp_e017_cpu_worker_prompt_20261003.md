# LGP-E017 CPU Worker Ping - Shrinkage / Fallback Gate

You are running the second CPU gate for `local_gp`. Use the GitHub repo
`https://github.com/atavory/local_gp`.

## Mission

E015 showed that probe-fed dynamics-aware correction beats tuned scalar FedAvg
scale, but it still used pilot-selected gamma. E017 asks whether we can replace
that tuning with a risk-aware shrinkage rule, and whether a cheap elitist
fallback explains the gain.

Build and run a CPU-only experiment:

```text
experiments/lgp_e017_risk_shrinkage_fallback.py
```

## Starting Point

Start from:

```text
experiments/lgp_e015_dynamics_aware_nonlinear_probe.py
```

Use the E015 nonlinear/heterogeneous bed first. If E016 has already landed,
optionally add the rotated bed as a second section.

## Arms

Required arms:

- raw FedAvg;
- tuned scalar FedAvg scale;
- tuned-gamma dynamics-aware random-probe correction;
- fixed full-gamma dynamics-aware random-probe correction;
- risk-aware-gamma dynamics-aware random-probe correction;
- server/time secant;
- best-worker fallback.

Optional arms:

- coordinate-probe correction;
- oracle-H correction;
- path-only correction.

## Risk-Aware Gamma

Use split probes:

1. Fit `Hhat` on half of the probe pairs.
2. Score residuals on the held-out probe pairs:

```text
residual = ||y - Hhat s||^2
target = ||y||^2
rho = residual / max(target, eps)
```

Try two no-target-tuning rules:

```text
gamma_reliability = clip(1 - rho, 0, 1)
gamma_sqrt = clip(sqrt(max(0, 1 - rho)), 0, 1)
```

If you need one selected rule, select between these two on source/pilot and
freeze it for held-out. Do not grid over arbitrary gamma values except for the
tuned-gamma oracle control.

## Best-Worker Fallback

Report at least one elitist fallback:

- `best_worker_true`: choose the worker endpoint with best true synthetic global
  loss after the local round. This is an oracle/strong baseline.

If cheap, also report:

- `best_worker_probe`: choose by noisy/probe loss or by local objective proxy.

If best-worker beats everything, say so. Do not bury it.

## Decision Labels

Use exactly one:

- `RISK_SHRINKAGE_WORKS`: risk-aware gamma beats tuned scalar and is within
  roughly 10-15% of tuned gamma on held-out rounds.
- `TUNED_GAMMA_STILL_REQUIRED`: tuned gamma wins materially and risk-aware gamma
  does not.
- `BEST_WORKER_EXPLAINS`: best-worker fallback matches or beats the probe
  correction.
- `SCALAR_EXPLAINS`: tuned scalar explains the gain.
- `INVALID`: implementation or scoring bug.

## Required Output

Write:

```text
experiments/lgp_e017_risk_shrinkage_fallback_result_20261003.md
```

Include:

- selected scalar/gamma/rule on pilot;
- held-out ratios over FedAvg;
- probe-minus-scalar, risk-minus-tuned-gamma, and best-worker-minus-probe CIs;
- a blunt interpretation.

This gate is about product viability: can we remove target gamma tuning, and can
we beat the stupid strong fallback?
