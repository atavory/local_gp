# E100 Multidomain Shift HPO 64-Seed Summary

Decision: `MULTIDOMAIN_SHIFT_HPO_WIN`.

Run:

- script: `experiments/dfw_e100_multidomain_shift_hpo.py`
- output dir: `bandha_e100_multidomain_shift_hpo_20261014/main64_5domain_sgd_lr0.02_probe64/`
- seeds: `999300..999363`
- domains: tabular classification, tabular regression, PFN-style tabular in-context classification, time-series forecasting, synthetic 8x8 vision
- optimizer arms: SGD with LR multipliers `(0, 0.25, 0.5, 1, 2, 4)`
- policy: one-hot probe-weighted aggregation (`argmin`)
- rounds: `24`, target-domain switch at round `18`
- batch: `64`, probe batch: `64`, eval batches: `2`
- estimated probe overhead: `33.33%`
- compute: Buck/PyTorch `TORCH_CPU`, `torch=2.16.0a0+fb`

Pooled readout:

- internal minus default: `+0.194 +/- 0.026`
- internal minus shuffled probes: `+0.206 +/- 0.029`
- internal minus validation-selected external fixed-LR HPO: `+0.112 +/- 0.020`
- internal minus external oracle fixed-LR HPO: `+0.111 +/- 0.020`

Per-domain readout:

| domain | decision | internal-default | internal-shuffled | internal-external |
|---|---|---:|---:|---:|
| tabular_class | `MULTIDOMAIN_SHIFT_HPO_WIN` | `+0.019 +/- 0.001` | `+0.021 +/- 0.001` | `+0.013 +/- 0.001` |
| tabular_reg | `MULTIDOMAIN_SHIFT_HPO_WIN` | `+0.642 +/- 0.032` | `+0.706 +/- 0.035` | `+0.450 +/- 0.025` |
| tabular_context | `MULTIDOMAIN_SHIFT_HPO_WIN` | `+0.132 +/- 0.007` | `+0.139 +/- 0.007` | `+0.090 +/- 0.005` |
| timeseries | `MULTIDOMAIN_SHIFT_HPO_WIN` | `+0.170 +/- 0.007` | `+0.154 +/- 0.006` | `+0.017 +/- 0.008` |
| vision8x8 | `MULTIDOMAIN_SHIFT_LABEL_CONTENT` | `+0.009 +/- 0.002` | `+0.009 +/- 0.002` | `-0.007 +/- 0.012` |

Interpretation:

E100 now has a 64-seed replication.  The pooled fixed-compute comparison
clears validation-selected external HPO with a tighter CI than the 32-seed run,
and four of five domains clear individually.  Vision remains a label-content
row rather than an external-HPO win, but it is not needed for the pooled result.
