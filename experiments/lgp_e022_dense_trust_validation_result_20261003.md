# LGP-E022 Dense Trust Validation Result - 2026-10-03

## Decision

`TRUST_LAYER_RECOVERS_DENSE`

## Question

E021 returned `DENSE_GEOMETRY_BREAKS_METHOD`: oracle dense-H correction had
large headroom, but learned low-rank-plus-diagonal correction was destructive
under split-probe residual shrinkage. E022 asks whether a stricter same-round
trust layer can prevent that failure and recover useful dense-geometry signal.

## Bed And Operator

This uses the E021 dense low-rank correlated-noise bed:

- `D=96`, `rank=8`, `sketch_dim=8`;
- Hessian background `diag(base) + U diag(spikes) U^T`;
- weak diagonal quartics plus stronger quartics in the low-rank subspace;
- centered worker heterogeneity with diagonal and low-rank components;
- correlated gradient noise `diag_noise + U lowrank_noise`.

All correction arms use the K-step local-map operator

\[
  [I - (I - \eta \hat H)^K]^{-1}\Delta_K
\]

applied to the FedAvg displacement. No plain Newton/BB/L-BFGS consolidation is
used.

Run parameters: `K=8`, `eta=0.025`, `rounds=24`, `pilot_reps=3`,
`heldout_reps=4`, `probe_reps=3`, `probe_budget_cap=128`,
`subspace_probe_budget=64`.

## Trust Gates

Primary non-oracle gate:

- `combo_val_gamma`: fit learned low-rank-plus-diagonal on probe fit split, then
  choose rowwise gamma from `{0, .025, .05, .1, .2, .4, .7, 1}` using a
  held-out mini-loss proxy from the same round.

Secondary gate:

- `combo_val_reject`: accept the E021 residual-shrunk learned combo only if a
  separate held-out mini-loss proxy beats FedAvg by a small margin; otherwise
  fall back to tuned scalar.

Validation proxy: average of 8 held-out mini objective component batches,
sampling 24 coordinates and 4 low-rank terms per batch, with a mild update
radius penalty. Exact global loss is used only for final scoring and the
diagnostic `exact_gate` upper-bound row.

## Main Table

Ratios are held-out loss improvement divided by raw FedAvg improvement. CIs are
paired across held-out replicate batches.

| N | scale* | oracle* | combo* | g_diag | g_lrL | g_combo | g_val | g_exact | reject rate | raw | scalar | diag | lr learned | E021 combo | combo_val_gamma | combo_val_reject | combo true | combo tuned | exact gate | oracle-H | secant | best worker |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.25 | 0.85 | 0.00 | 0.166 | 0.939 | 0.239 | 0.064 | 0.070 | 0.042 | 1.000 +/- 0.000 | 1.046 +/- 0.029 | 0.784 +/- 0.408 | 0.832 +/- 0.213 | -6.028 +/- 3.875 | 1.085 +/- 0.032 | 1.048 +/- 0.029 | 0.300 +/- 0.417 | 1.000 +/- 0.000 | 1.089 +/- 0.033 | 2.476 +/- 0.654 | 0.316 +/- 0.603 | 0.911 +/- 0.031 |
| 32 | 1.20 | 1.00 | 0.05 | 0.197 | 0.943 | 0.338 | 0.123 | 0.166 | 0.125 | 1.000 +/- 0.000 | 1.040 +/- 0.023 | 0.864 +/- 0.178 | 0.995 +/- 0.005 | -5.464 +/- 8.616 | 1.143 +/- 0.061 | 1.063 +/- 0.034 | 0.720 +/- 0.411 | 1.062 +/- 0.059 | 1.153 +/- 0.065 | 2.661 +/- 0.719 | 0.917 +/- 0.059 | 0.966 +/- 0.018 |
| 128 | 1.30 | 1.00 | 0.10 | 0.198 | 0.941 | 0.330 | 0.121 | 0.171 | 0.062 | 1.000 +/- 0.000 | 1.032 +/- 0.015 | 0.596 +/- 0.247 | 0.983 +/- 0.020 | -1.793 +/- 2.381 | 1.107 +/- 0.032 | 1.041 +/- 0.019 | 1.046 +/- 0.066 | 1.009 +/- 0.053 | 1.114 +/- 0.033 | 2.421 +/- 0.376 | 0.860 +/- 0.189 | 0.998 +/- 0.010 |

## Required Paired Differences

Primary trust gate: `combo_val_gamma`.

| N | val gamma - scalar | val gamma - E021 combo | val gamma - diag | best worker - val gamma |
|---:|---:|---:|---:|---:|
| 8 | 0.038 +/- 0.015 | 7.113 +/- 3.851 | 0.300 +/- 0.395 | -0.174 +/- 0.060 |
| 32 | 0.103 +/- 0.039 | 6.607 +/- 8.636 | 0.279 +/- 0.118 | -0.177 +/- 0.075 |
| 128 | 0.074 +/- 0.020 | 2.900 +/- 2.412 | 0.510 +/- 0.270 | -0.109 +/- 0.035 |

Secondary trust gate: `combo_val_reject`.

| N | val reject - scalar | val reject - E021 combo | val reject - diag | best worker - val reject |
|---:|---:|---:|---:|---:|
| 8 | 0.002 +/- 0.002 | 7.076 +/- 3.859 | 0.264 +/- 0.392 | -0.138 +/- 0.053 |
| 32 | 0.023 +/- 0.022 | 6.527 +/- 8.642 | 0.199 +/- 0.151 | -0.097 +/- 0.050 |
| 128 | 0.009 +/- 0.006 | 2.834 +/- 2.398 | 0.445 +/- 0.259 | -0.043 +/- 0.022 |

## Interpretation

The trust layer fixes the E021 safety failure. The residual-shrunk learned
combo is still destructive, but `combo_val_gamma` selects small rowwise gammas
and beats tuned scalar with positive CIs at all worker counts. It also beats
the E021 residual-shrunk combo by a large margin.

This is a modest recovery, not the dense optimum. `combo_val_gamma` remains far
below oracle-H, so most dense geometry headroom is still uncaptured. Against
diagonal-only it is clearly positive at N=32 and N=128; at N=8 the point
estimate is positive but the CI is wide because diagonal-only is unstable.

`combo_val_reject` is mostly a safety fallback. It prevents harm and lands
near tuned scalar, but it does not recover much additional value.

Best-worker fallback does not explain the trust-gated gain: `best_worker` is
below `combo_val_gamma` at every worker count. Server/time secants remain weak
and unstable on this bed.

## Recommendation

Keep dense low-rank-plus-diagonal only behind a rowwise validation-gamma trust
layer. The current learned dense representation is not product-ready, but the
failure is no longer intrinsic: held-out validation can reject the bad
correction and recover a small, real gain over scalar. The next gate should
improve subspace/model identification rather than relax the trust layer.

## Command

```bash
python3 -u experiments/lgp_e022_dense_trust_validation.py
```
