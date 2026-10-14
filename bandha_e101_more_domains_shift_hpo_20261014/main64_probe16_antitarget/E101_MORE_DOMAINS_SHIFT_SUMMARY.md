# E101 More-Domain Shift HPO Summary

Decision: `MORE_DOMAINS_SHIFT_HPO_WIN`.

Run:

- script: `experiments/dfw_e101_more_domains_shift_hpo.py`
- output dir: `/data/users/atavory/local_gp/bandha_e101_more_domains_shift_hpo_20261014/main64_probe16_antitarget`
- seeds: `999400..999463` (64 seeds)
- domains: multiclass_tabular, sparse_text, ranking_pair, graph_node, sequence_class, matrix_interaction
- optimizer arms: SGD with LR multipliers `(0, 0.25, 0.5, 1, 2, 4)`
- policy: one-hot probe selection (`argmin`)
- rounds: `24`, target-domain switch at round `18`
- batch: `64`, probe batch: `16`, eval batches: `2`
- estimated probe overhead: `8.33%`
- compute: Buck/PyTorch `TORCH_CPU`, `torch=2.16.0a0+fb cuda_available=False`

Pooled readout:

- internal minus default: `0.118+/-0.024`
- internal minus shuffled probes: `0.133+/-0.028`
- internal minus validation-selected external fixed-LR HPO: `0.077+/-0.016`
- internal minus external oracle fixed-LR HPO: `0.074+/-0.016`

Per-domain readout:

| domain | decision | internal-default | internal-shuffled | internal-external |
|---|---|---:|---:|---:|
| multiclass_tabular | `MORE_DOMAIN_SHIFT_HPO_WIN` | `0.023+/-0.002` | `0.023+/-0.002` | `0.022+/-0.003` |
| sparse_text | `MORE_DOMAIN_SHIFT_HPO_WIN` | `0.004+/-0.001` | `0.004+/-0.001` | `0.004+/-0.001` |
| ranking_pair | `MORE_DOMAIN_SHIFT_HPO_WIN` | `0.029+/-0.002` | `0.033+/-0.003` | `0.017+/-0.002` |
| graph_node | `MORE_DOMAIN_SHIFT_HPO_WIN` | `0.006+/-0.001` | `0.007+/-0.001` | `0.004+/-0.001` |
| sequence_class | `MORE_DOMAIN_SHIFT_HPO_WIN` | `0.042+/-0.003` | `0.044+/-0.003` | `0.036+/-0.003` |
| matrix_interaction | `MORE_DOMAIN_SHIFT_HPO_WIN` | `0.602+/-0.065` | `0.685+/-0.073` | `0.378+/-0.050` |

Interpretation:

E101 is a breadth extension of E100.  It keeps the same adaptive-HPO controller and fixed-compute external-HPO comparator, but moves to six new synthetic domain shapes.  A pooled win here strengthens the paper's claim that the mechanism is not tied to one tabular toy, one tiny-LM stream, or one local-SGD outer-control ledger.
