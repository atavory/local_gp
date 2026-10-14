# E100 Multidomain Shift HPO Summary

Decision: `MULTIDOMAIN_SHIFT_HPO_WIN`.

Run:

- script: `experiments/dfw_e100_multidomain_shift_hpo.py`
- output dir: `bandha_e100_multidomain_shift_hpo_20261014/main32_5domain_sgd_lr0.02_probe64/`
- seeds: `999300..999331`
- domains: tabular classification, tabular regression, PFN-style tabular in-context classification, time-series forecasting, synthetic 8x8 vision
- optimizer arms: SGD with LR multipliers `(0, 0.25, 0.5, 1, 2, 4)`
- policy: one-hot probe-weighted aggregation (`argmin`)
- rounds: `24`, target-domain switch at round `18`
- batch: `64`, probe batch: `64`, eval batches: `2`
- estimated probe overhead: `33.33%`
- compute: Buck/PyTorch `TORCH_CPU`, `torch=2.16.0a0+fb`

Pooled readout:

- internal minus default: `+0.189 +/- 0.036`
- internal minus shuffled probes: `+0.199 +/- 0.040`
- internal minus validation-selected external fixed-LR HPO: `+0.108 +/- 0.026`
- internal minus external oracle fixed-LR HPO: `+0.107 +/- 0.026`

Per-domain readout:

| domain | decision | internal-default | internal-shuffled | internal-external |
|---|---|---:|---:|---:|
| tabular_class | `MULTIDOMAIN_SHIFT_HPO_WIN` | `+0.019 +/- 0.002` | `+0.021 +/- 0.002` | `+0.013 +/- 0.001` |
| tabular_reg | `MULTIDOMAIN_SHIFT_HPO_WIN` | `+0.614 +/- 0.048` | `+0.676 +/- 0.054` | `+0.426 +/- 0.035` |
| tabular_context | `MULTIDOMAIN_SHIFT_HPO_WIN` | `+0.135 +/- 0.009` | `+0.141 +/- 0.009` | `+0.093 +/- 0.007` |
| timeseries | `MULTIDOMAIN_SHIFT_HPO_LIVE` | `+0.167 +/- 0.010` | `+0.151 +/- 0.009` | `+0.008 +/- 0.009` |
| vision8x8 | `MULTIDOMAIN_SHIFT_HPO_LIVE` | `+0.009 +/- 0.002` | `+0.009 +/- 0.002` | `+0.000 +/- 0.005` |

Interpretation:

E100 broadens the positive HPO-aware aggregation story beyond the local-SGD
outer-control bed and tiny-LM setting.  In a deliberately shifted stream,
internal same-round probe weighting reinvests LR-arm exploration into one
shared trajectory, while external HPO must choose one fixed LR trajectory for
the whole run.  The pooled fixed-compute comparison clears, and the three
tabular-style domains clear individually, including the PFN-style in-context
tabular task.
