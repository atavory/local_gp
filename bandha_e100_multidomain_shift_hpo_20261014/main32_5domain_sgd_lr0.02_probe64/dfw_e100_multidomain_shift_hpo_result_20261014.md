# DFW-E100 Result - Multidomain Shift HPO

Decision label: `MULTIDOMAIN_SHIFT_HPO_WIN`

## Compute Status

`TORCH_CPU`: `torch=2.16.0a0+fb cuda_available=False`.

## Pooled

- internal minus default: `0.189+/-0.036`
- internal minus shuffled: `0.199+/-0.040`
- internal minus external validation-selected HPO: `0.108+/-0.026`
- internal minus external oracle HPO: `0.107+/-0.026`

## Domains

| domain | decision | internal-default | internal-shuffled | internal-external | cost |
|---|---|---:|---:|---:|---:|
| tabular_class | MULTIDOMAIN_SHIFT_HPO_WIN | 0.019+/-0.002 | 0.021+/-0.002 | 0.013+/-0.001 | 33.33% |
| tabular_reg | MULTIDOMAIN_SHIFT_HPO_WIN | 0.614+/-0.048 | 0.676+/-0.054 | 0.426+/-0.035 | 33.33% |
| tabular_context | MULTIDOMAIN_SHIFT_HPO_WIN | 0.135+/-0.009 | 0.141+/-0.009 | 0.093+/-0.007 | 33.33% |
| timeseries | MULTIDOMAIN_SHIFT_HPO_LIVE | 0.167+/-0.010 | 0.151+/-0.009 | 0.008+/-0.009 | 33.33% |
| vision8x8 | MULTIDOMAIN_SHIFT_HPO_LIVE | 0.009+/-0.002 | 0.009+/-0.002 | 0.000+/-0.005 | 33.33% |

## Interpretation

The multidomain shift audit clears the pooled fixed-compute comparator: same-round probe-weighted aggregation beats default controls, shuffled probes, and validation-selected fixed-LR external HPO across the domain suite.
