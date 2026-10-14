# DFW-E100 Result - Multidomain Shift HPO

Decision label: `MULTIDOMAIN_SHIFT_HPO_WIN`

## Compute Status

`TORCH_CPU`: `torch=2.16.0a0+fb cuda_available=False`.

## Pooled

- internal minus default: `0.194+/-0.026`
- internal minus shuffled: `0.206+/-0.029`
- internal minus external validation-selected HPO: `0.112+/-0.020`
- internal minus external oracle HPO: `0.111+/-0.020`

## Domains

| domain | decision | internal-default | internal-shuffled | internal-external | cost |
|---|---|---:|---:|---:|---:|
| tabular_class | MULTIDOMAIN_SHIFT_HPO_WIN | 0.019+/-0.001 | 0.021+/-0.001 | 0.013+/-0.001 | 33.33% |
| tabular_reg | MULTIDOMAIN_SHIFT_HPO_WIN | 0.642+/-0.032 | 0.706+/-0.035 | 0.450+/-0.025 | 33.33% |
| tabular_context | MULTIDOMAIN_SHIFT_HPO_WIN | 0.132+/-0.007 | 0.139+/-0.007 | 0.090+/-0.005 | 33.33% |
| timeseries | MULTIDOMAIN_SHIFT_HPO_WIN | 0.170+/-0.007 | 0.154+/-0.006 | 0.017+/-0.008 | 33.33% |
| vision8x8 | MULTIDOMAIN_SHIFT_LABEL_CONTENT | 0.009+/-0.002 | 0.009+/-0.002 | -0.007+/-0.012 | 33.33% |

## Interpretation

The multidomain shift audit clears the pooled fixed-compute comparator: same-round probe-weighted aggregation beats default controls, shuffled probes, and validation-selected fixed-LR external HPO across the domain suite.
