# DFW-E101 Result - More-Domain Shift HPO

Decision label: `MORE_DOMAINS_SHIFT_HPO_WIN`

## Compute Status

`TORCH_CPU`: `torch=2.16.0a0+fb cuda_available=False`.

## Pooled

- internal minus default: `0.125+/-0.024`
- internal minus shuffled: `0.137+/-0.026`
- internal minus external validation-selected HPO: `0.085+/-0.016`
- internal minus external oracle HPO: `0.085+/-0.016`

## Domains

| domain | decision | internal-default | internal-shuffled | internal-external | cost |
|---|---|---:|---:|---:|---:|
| multiclass_tabular | MORE_DOMAIN_SHIFT_HPO_WIN | 0.025+/-0.001 | 0.025+/-0.001 | 0.022+/-0.002 | 33.33% |
| sparse_text | MORE_DOMAIN_SHIFT_HPO_WIN | 0.005+/-0.001 | 0.005+/-0.001 | 0.005+/-0.001 | 33.33% |
| ranking_pair | MORE_DOMAIN_SHIFT_HPO_WIN | 0.031+/-0.002 | 0.035+/-0.002 | 0.019+/-0.001 | 33.33% |
| graph_node | MORE_DOMAIN_SHIFT_HPO_WIN | 0.008+/-0.001 | 0.009+/-0.001 | 0.005+/-0.001 | 33.33% |
| sequence_class | MORE_DOMAIN_SHIFT_HPO_WIN | 0.043+/-0.002 | 0.045+/-0.002 | 0.036+/-0.002 | 33.33% |
| matrix_interaction | MORE_DOMAIN_SHIFT_HPO_WIN | 0.638+/-0.041 | 0.701+/-0.043 | 0.425+/-0.029 | 33.33% |

## Interpretation

The added-domain audit clears the pooled fixed-compute comparator: same-round probe-selected aggregation beats default fixed LR, shuffled probes, and validation-selected external fixed-LR HPO across six additional domain shapes.
