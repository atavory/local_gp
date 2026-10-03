# LGP-E022 CPU Worker Ping - Dense Trust-Layer Gate

You are running the next CPU gate for `local_gp`. Use the GitHub repo
`https://github.com/atavory/local_gp`, or the Manifold source bundle listed in
the CPU manifest if GitHub access is inconvenient.

## Mission

E021 returned `DENSE_GEOMETRY_BREAKS_METHOD`: oracle dense-H correction has
large headroom, but the learned low-rank-plus-diagonal update was destructive
under E017-style residual shrinkage. E022 asks whether a stricter trust layer
can prevent that failure and recover any dense-geometry headroom.

Build and run:

```text
experiments/lgp_e022_dense_trust_validation.py
```

## Starting Point

Start from:

```text
experiments/lgp_e021_dense_lowrank_correlated_stress.py
```

Use the same dense low-rank correlated-noise bed first. Keep the run CPU-only.
If time is short, run `N=32` and `N=128` first, then add `N=8`.

Pre-launch sanity from the coordinator, not a result: on a tiny `N=32`,
`rounds=6`, `probe_reps=2`, single-heldout run, the E021 residual combo scored
`-6.396`, pilot-tuned combo gamma selected `0.000` and scored `1.000`, and an
exact-loss rowwise gate over `gamma in {0, .025, .05, .1, .2, .4, .7, 1}`
scored `1.042` versus tuned scalar `1.016` and oracle-H `1.681`. Treat this
only as evidence that rejection can mechanically prevent the crash; the primary
E022 gate must use a non-oracle held-out probe/loss proxy.

## Required Operator

Every correction arm must use the K-step local-map operator:

```text
[I - (I - eta Hhat)^K]^-1 Delta_K
```

Do not use plain Newton, BB, or L-BFGS consolidation.

## Trust-Layer Designs

Keep the E021 residual-shrunk arms as controls, then add at least two stricter
trust gates for the low-rank-plus-diagonal combined update.

Required controls:

- raw FedAvg;
- tuned scalar FedAvg scale;
- diagonal residual-shrunk correction;
- learned low-rank residual-shrunk correction;
- learned low-rank-plus-diagonal residual-shrunk correction;
- true-subspace low-rank-plus-diagonal residual-shrunk correction if cheap;
- pilot-tuned combined gamma as an upper-bound control;
- oracle-H dynamics-aware correction;
- server/time secant;
- best-worker true fallback.

Required trust gates:

1. `combo_val_gamma`: fit the combined model on one probe split, then choose
   gamma from a tiny grid such as `0, .025, .05, .1, .2, .4, .7, 1` using only a
   held-out validation proxy from the same round. The grid is a trust gate, not
   a target-scoring oracle: do not use held-out result rows to choose it.
2. `combo_val_reject`: accept the combined correction only if held-out probe or
   loss-proxy validation says it improves over raw FedAvg by a margin. Otherwise
   return FedAvg or tuned scalar.

Preferred validation proxies, in order:

- held-out mini/objective-loss proxy at `x0 + update` versus `x0 + fedavg`;
- held-out probe-gradient quadratic proxy for candidate decrease;
- held-out secant consistency plus update-norm/radius penalty.

Because this is a synthetic bed, exact global loss may be used for diagnostics
and final scoring, but label any gate that uses exact target loss as an oracle
upper bound. The primary trust gate should be implementable with held-out
probes or validation loss only.

## Decision Labels

Use exactly one:

- `TRUST_LAYER_RECOVERS_DENSE`: a non-oracle trust-gated combined correction
  beats tuned scalar and diagonal-only correction with positive CI, while
  staying below oracle-H as expected.
- `TRUST_LAYER_PREVENTS_HARM_ONLY`: the gate reliably avoids the destructive
  combined update but falls back to scalar/FedAvg and does not beat tuned scalar.
- `TRUST_LAYER_FAILS`: trust-gated combined updates are still destructive or
  worse than scalar with clear CI.
- `SCALAR_OR_FALLBACK_EXPLAINS`: tuned scalar or best-worker fallback explains
  the apparent gain.
- `NO_ORACLE_HEADROOM`: oracle-H no longer beats tuned scalar.
- `INVALID`: implementation/scoring bug or target leakage in the primary gate.

## Required Output

Write:

```text
experiments/lgp_e022_dense_trust_validation_result_20261003.md
```

Upload script, result note, and run log to:

```text
aai_research_tlv/tree/local_gp/cpu_pings/20261003/
```

The result must include paired CIs for:

- trust-gated combo minus tuned scalar;
- trust-gated combo minus E021 residual-shrunk combo;
- trust-gated combo minus diagonal-only;
- best-worker fallback minus trust-gated combo.

Be blunt about whether the trust layer recovers dense geometry or merely keeps
the method from harming itself.
