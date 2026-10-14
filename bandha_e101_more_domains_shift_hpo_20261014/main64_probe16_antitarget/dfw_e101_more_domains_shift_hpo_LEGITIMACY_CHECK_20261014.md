# DFW-E101 Legitimacy Check

- Compute: `TORCH_CPU` / `torch=2.16.0a0+fb cuda_available=False`.
- Domains: multiclass tabular, sparse text-like bags, pairwise ranking, graph/node summaries, sequence classification, matrix-interaction regression.
- Optimizer family: `sgd` with LR multipliers `0, 0.25, 0.5, 1, 2, 4`.
- Training stream: off-target or label-corrupted before the switch, target-domain after the switch.
- Probe/select and target batches are target-domain.
- Same candidate-update budget: internal and external HPO both train one candidate update per LR arm per round.
- Internal HPO uses `argmin` same-round probe selection from one shared base state each round.
- External HPO keeps separate fixed-LR trajectories and selects by held-out validation loss.
- Shuffled control preserves candidate updates but permutes probe-to-arm labels.
- This is an additional small synthetic domain-suite audit, not a foundation-scale benchmark.
