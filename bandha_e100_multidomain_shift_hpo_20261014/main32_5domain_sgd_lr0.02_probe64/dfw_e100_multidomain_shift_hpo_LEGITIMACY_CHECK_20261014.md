# DFW-E100 Legitimacy Check

- Compute: `TORCH_CPU` / `torch=2.16.0a0+fb cuda_available=False`.
- Domains: tabular classification, tabular regression, PFN-style tabular in-context classification, time-series forecasting, synthetic vision.
- Optimizer family: `sgd` with LR multipliers `0, 0.25, 0.5, 1, 2, 4`.
- Training stream: off-target or label-flipped before the switch, target-domain after the switch.
- Probe/select and target batches are target-domain.
- Same candidate-update budget: internal and external HPO both train one candidate update per LR arm per round.
- Internal HPO uses `argmin` probe-weighted aggregation from one shared base state each round.
- External HPO keeps separate fixed-LR trajectories and selects by held-out validation loss.
- Shuffled control preserves candidate updates but permutes probe-to-arm labels.
- This is a small synthetic domain-suite audit, not a replacement for large task-family experiments.
