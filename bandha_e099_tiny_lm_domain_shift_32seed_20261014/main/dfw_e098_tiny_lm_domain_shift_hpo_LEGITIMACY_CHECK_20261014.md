# DFW-E098 Legitimacy Check

- Compute: `TORCH_CPU` / `torch=2.16.0a0+fb cuda_available=False`.
- Model: tiny character-level causal Transformer language model.
- Training stream: off-domain prose before the switch, code-like text after the switch.
- Probe/select and target streams: held-out code-like text.
- Same candidate-update budget: internal and external HPO both train one candidate update per LR arm per round.
- Internal HPO aggregates by one-hot probe selection from one shared base state each round.
- External HPO keeps separate fixed-LR trajectories and selects by held-out validation loss.
- Shuffled control preserves candidate updates but permutes probe-to-arm labels.
- Target text is used for scoring and oracle diagnostics, not for the probe controller.
- This is a tiny-LM domain-shift tuning result, not a full-scale LLM pretraining run.

