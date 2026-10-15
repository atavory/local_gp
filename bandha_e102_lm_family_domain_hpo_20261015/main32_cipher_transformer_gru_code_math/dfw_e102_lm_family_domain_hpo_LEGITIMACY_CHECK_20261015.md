# DFW-E102 Legitimacy Check

- Compute: `TORCH_CPU` / `torch=2.16.0a0+fb cuda_available=False`.
- Models: character-level causal LMs from transformer, recurrent, and causal-convolution families.
- Training stream: off-target text before the switch, target-domain text after the switch.
- Probe/select and target streams: held-out target-domain text split disjointly.
- Same candidate-update budget: internal and external HPO both train one candidate update per LR arm per round.
- Internal HPO selects one candidate next state from a shared base model each round.
- External HPO keeps separate fixed-LR trajectories and selects by held-out validation loss.
- Shuffled control preserves candidate updates but permutes probe-to-arm labels.
- Target-eval split is used for scoring and oracle diagnostics, not for the probe controller.
- This is a lightweight LM-family audit, not a full-scale LLM pretraining run.

