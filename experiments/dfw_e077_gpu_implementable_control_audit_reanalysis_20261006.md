# DFW-E077 Reanalysis - Probe-Active Split

Status: post-hoc reanalysis of existing E077 result summaries.  No new GPU run.

## Question

E077's best-looking rule, `small_k_probe_fixed_large_k`, uses probes only for
`K<=8`.  For `K>8` it returns `cell.fixed`, so pooled comparisons dilute the
probe-content test with mechanical fixed-fallback cells.

The decisive question is therefore:

```text
Does the rule beat its shuffled-probe control on K<=8, where probes actually
act?
```

## Method

Parsed the E077 reported cell summaries and reconstructed pooled CIs from the
per-cell means and 95% CIs, using `n=12` held-out seeds per cell.

Rows included for probe-active split:

```text
N=8,  K=4
N=8,  K=8
N=16, K=4
N=16, K=8
```

Rows included for fixed-fallback split:

```text
N=8,  K=16
N=8,  K=32
N=16, K=16
N=16, K=32
```

## Result

| component | minus fixed | minus schedule / pilot-best | minus held-out-best ceiling | minus shuffled |
|---|---:|---:|---:|---:|
| K<=8 probe-active cells | 0.124+/-0.176 | 0.400+/-0.216 | 0.031+/-0.155 | 0.089+/-0.154 |
| K>8 fixed-fallback cells | 0.000+/-0.000 | 0.306+/-0.172 | -0.065+/-0.062 | 0.000+/-0.000 |
| all pooled cells | 0.062+/-0.089 | 0.353+/-0.138 | -0.017+/-0.084 | 0.044+/-0.077 |

## Readout

Decision: `PROBE_ACTIVE_SHUFFLED_NULL`.

The K<=8 probe-active contrast versus shuffled is `0.089+/-0.154`, so the CI
includes zero.  The probe effect does not concentrate into a clean content win
when isolated to the rows where probes act.

The clean positive in E077 is not probe extraction.  It is fixed fallback
beating an overfit schedule/pilot baseline at large K:

```text
K>8 fixed fallback minus schedule/pilot = 0.306+/-0.172
```

So E077 should not be cited as evidence that probes carry reliable value
information.  It is evidence for two separate facts:

1. the current probe selector has no proven content beyond shuffled probes;
2. transferred/pilot schedules are fragile, and fixed fallback avoids that
   fragility in this bed.

Before running a GP/bandit on this branch, the next experiment must replicate
the K<=8 probe-active contrast directly and clear the shuffled control.
