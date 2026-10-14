# DFW-E096 Result - Tiny Causal-LM Probe HPO

Decision label: `TINY_LM_PROBE_LABEL_CONTENT`

## Compute Status

`TORCH_CPU`: `torch=2.16.0a0+fb cuda_available=False`.

## Design

A tiny character-level causal Transformer language model is trained on an embedded held-out text corpus. Internal HPO evaluates LR-arm candidate next states from the same base model each round, scores them on held-out probe text, and probe-weights the parameter deltas into one shared trajectory. External HPO spends the same candidate-update budget on separate fixed-LR trajectories and selects by held-out validation loss.

## Pooled Target-Loss Improvement

- initial target loss: `3.690+/-0.022`
- default fixed LR: `0.880+/-0.028`
- internal probe-weighted HPO: `1.177+/-0.031`
- shuffled-probe internal: `1.000+/-0.034`
- external validation-selected HPO: `1.178+/-0.031`
- external target oracle HPO: `1.178+/-0.031`
- internal minus default: `0.298+/-0.011`
- internal minus shuffled: `0.177+/-0.015`
- internal minus external validation-selected HPO: `-0.000+/-0.000`
- internal minus external oracle HPO: `-0.000+/-0.000`
- external validation-selected HPO minus default: `0.298+/-0.011`
- probe overhead estimate: `16.67%`

## Interpretation

The lightweight LM audit is a boundary result for language modeling; use the pooled margins rather than promoting it as a main result.

## Run Log

```text
DFW-E096 tiny causal-LM probe HPO
host=devvm44191.lla0.facebook.com python=/data/users/atavory/fbsource/buck-out/v2/art/fbcode/c95bb19b4474c2e4/aai/double_fan_gpu_runner/__run_local_gp_script_static__/run_local_gp_script_static#link-tree/runtime/bin/run_local_gp_script_static#native-main#platform-runtime#python#py_version_3_12 compute_status=TORCH_CPU torch=2.16.0a0+fb cuda_available=False
rounds=12 block=32 batch=32 probe_batch=16 eval_batches=2
base_lr=0.003 weight_decay=0.01 policy=argmin tau_scale=0.5 lr_mults=(0.5, 1, 2, 4) seeds=16
POOLED internal-default=  0.298+/-0.011 internal-shuffled=  0.177+/-0.015 internal-external_valid= -0.000+/-0.000 internal-external_oracle= -0.000+/-0.000 external_valid-default=  0.298+/-0.011 internal-internal_oracle=  0.000+/-0.001
DECISION TINY_LM_PROBE_LABEL_CONTENT
compute_status=TORCH_CPU
```
