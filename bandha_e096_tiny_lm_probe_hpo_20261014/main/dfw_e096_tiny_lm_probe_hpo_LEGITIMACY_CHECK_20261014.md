# DFW-E096 Legitimacy Check

- Compute: `TORCH_CPU` / `torch=2.16.0a0+fb cuda_available=False`.
- Model: tiny character-level causal Transformer language model.
- Dataset: embedded text corpus split into train, probe/select, and target text.
- Same candidate-update budget: internal and external HPO both train one candidate trajectory per LR arm per round.
- Internal HPO aggregates candidate parameter deltas from one shared base state each round.
- External HPO keeps separate fixed-LR trajectories and selects by held-out validation loss.
- Shuffled control preserves candidate updates but permutes probe-to-arm labels.
- Target text is used for scoring and oracle diagnostics, not for the probe controller.
- This is a lightweight LM tuning result, not a full-scale LLM pretraining run.

