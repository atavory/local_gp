# DFW-E101 Result - More-Domain Shift HPO

Decision label: `MORE_DOMAINS_SHIFT_HPO_WIN`

## Compute Status

`TORCH_CPU`: `torch=2.16.0a0+fb cuda_available=False`.

## Pooled

- internal minus default: `0.118+/-0.024`
- internal minus shuffled: `0.133+/-0.028`
- internal minus external validation-selected HPO: `0.077+/-0.016`
- internal minus external oracle HPO: `0.074+/-0.016`

## Domains

| domain | decision | internal-default | internal-shuffled | internal-external | cost |
|---|---|---:|---:|---:|---:|
| multiclass_tabular | MORE_DOMAIN_SHIFT_HPO_WIN | 0.023+/-0.002 | 0.023+/-0.002 | 0.022+/-0.003 | 8.33% |
| sparse_text | MORE_DOMAIN_SHIFT_HPO_WIN | 0.004+/-0.001 | 0.004+/-0.001 | 0.004+/-0.001 | 8.33% |
| ranking_pair | MORE_DOMAIN_SHIFT_HPO_WIN | 0.029+/-0.002 | 0.033+/-0.003 | 0.017+/-0.002 | 8.33% |
| graph_node | MORE_DOMAIN_SHIFT_HPO_WIN | 0.006+/-0.001 | 0.007+/-0.001 | 0.004+/-0.001 | 8.33% |
| sequence_class | MORE_DOMAIN_SHIFT_HPO_WIN | 0.042+/-0.003 | 0.044+/-0.003 | 0.036+/-0.003 | 8.33% |
| matrix_interaction | MORE_DOMAIN_SHIFT_HPO_WIN | 0.602+/-0.065 | 0.685+/-0.073 | 0.378+/-0.050 | 8.33% |

## Interpretation

The added-domain audit clears the pooled fixed-compute comparator: same-round probe-selected aggregation beats default fixed LR, shuffled probes, and validation-selected external fixed-LR HPO across six additional domain shapes.
