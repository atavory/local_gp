# DFW-E103 Legitimacy Check

- Compute: `TORCH_CPU` / `torch=2.16.0a0+fb cuda_available=False`.
- Models: token-level Transformer, recurrent, and causal-convolution language models.
- Tokenization: deterministic word/punctuation tokenizer implemented in the script.
- Training stream: source-domain tokens before the switch, target-domain tokens after the switch.
- Probe/select and target streams: held-out target-domain token splits.
- Same candidate-update budget: internal and external HPO both train one candidate update per LR arm per round.
- Internal HPO selects one candidate next state from a shared base model each round.
- External HPO keeps separate fixed-LR trajectories and selects by held-out validation loss.
- Shuffled control preserves candidate updates but permutes probe-to-arm labels.
- Target-eval split is used for scoring and oracle diagnostics, not for the probe controller.
- This is a token-level scale-up audit, not a foundation-scale LLM pretraining run.

