# E101 More-Domain Shift HPO Summary

Decision: `MORE_DOMAINS_SHIFT_HPO_WIN`.

Run:

- script: `experiments/dfw_e101_more_domains_shift_hpo.py`
- output dir: `/data/users/atavory/local_gp/bandha_e101_more_domains_shift_hpo_20261014/main64_antitarget`
- seeds: `999400..999463` (64 seeds)
- domains: multiclass_tabular, sparse_text, ranking_pair, graph_node, sequence_class, matrix_interaction
- optimizer arms: SGD with LR multipliers `(0, 0.25, 0.5, 1, 2, 4)`
- policy: one-hot probe selection (`argmin`)
- rounds: `24`, target-domain switch at round `18`
- batch: `64`, probe batch: `64`, eval batches: `2`
- estimated probe overhead: `33.33%`
- compute: Buck/PyTorch `TORCH_CPU`, `torch=2.16.0a0+fb cuda_available=False`

Pooled readout:

- internal minus default: `0.125+/-0.024`
- internal minus shuffled probes: `0.137+/-0.026`
- internal minus validation-selected external fixed-LR HPO: `0.085+/-0.016`
- internal minus external oracle fixed-LR HPO: `0.085+/-0.016`

Per-domain readout:

| domain | decision | internal-default | internal-shuffled | internal-external |
|---|---|---:|---:|---:|
| multiclass_tabular | `MORE_DOMAIN_SHIFT_HPO_WIN` | `0.025+/-0.001` | `0.025+/-0.001` | `0.022+/-0.002` |
| sparse_text | `MORE_DOMAIN_SHIFT_HPO_WIN` | `0.005+/-0.001` | `0.005+/-0.001` | `0.005+/-0.001` |
| ranking_pair | `MORE_DOMAIN_SHIFT_HPO_WIN` | `0.031+/-0.002` | `0.035+/-0.002` | `0.019+/-0.001` |
| graph_node | `MORE_DOMAIN_SHIFT_HPO_WIN` | `0.008+/-0.001` | `0.009+/-0.001` | `0.005+/-0.001` |
| sequence_class | `MORE_DOMAIN_SHIFT_HPO_WIN` | `0.043+/-0.002` | `0.045+/-0.002` | `0.036+/-0.002` |
| matrix_interaction | `MORE_DOMAIN_SHIFT_HPO_WIN` | `0.638+/-0.041` | `0.701+/-0.043` | `0.425+/-0.029` |

Interpretation:

E101 is a breadth extension of E100.  It keeps the same adaptive-HPO controller and fixed-compute external-HPO comparator, but moves to six new synthetic domain shapes.  A pooled win here strengthens the paper's claim that the mechanism is not tied to one tabular toy, one tiny-LM stream, or one local-SGD outer-control ledger.
