# DFW-E098 Result - Tiny Causal-LM Domain-Shift HPO

Decision label: `TINY_LM_DOMAIN_SHIFT_HPO_WIN`

## Compute Status

`TORCH_CPU`: `torch=2.16.0a0+fb cuda_available=False`.

## Design

The tiny causal-LM training stream shifts from off-domain prose to code-like text, while probe/select and target text are code-like. Internal HPO uses one-hot probe selection over LR-arm candidate next states from the shared model each round. External HPO spends the same candidate-update budget on separate fixed-LR trajectories and selects one by held-out validation loss.

## Pooled Target-Loss Improvement

- initial target loss: `4.158+/-0.020`
- default fixed LR: `0.920+/-0.025`
- internal probe HPO: `1.372+/-0.026`
- shuffled-probe internal: `0.906+/-0.031`
- external validation-selected HPO: `1.330+/-0.028`
- external target oracle HPO: `1.330+/-0.028`
- internal minus default: `0.452+/-0.022`
- internal minus shuffled: `0.466+/-0.029`
- internal minus external validation-selected HPO: `0.043+/-0.026`
- internal minus external oracle HPO: `0.043+/-0.026`
- external validation-selected HPO minus default: `0.409+/-0.021`
- probe overhead estimate: `16.67%`

## Interpretation

The domain-shift LM audit clears the fixed-compute comparator: same-round probe selection beats the default fixed LR, shuffled probes, and validation-selected fixed-LR external HPO.

## Run Log

```text
DFW-E098 tiny causal-LM domain-shift HPO
host=devvm44191.lla0.facebook.com python=/data/users/atavory/fbsource/buck-out/v2/art/fbcode/c95bb19b4474c2e4/aai/double_fan_gpu_runner/__run_local_gp_script_static__/run_local_gp_script_static#link-tree/runtime/bin/run_local_gp_script_static#native-main#platform-runtime#python#py_version_3_12 compute_status=TORCH_CPU torch=2.16.0a0+fb cuda_available=False
rounds=16 switch_round=8 block=32 batch=32 probe_batch=16 eval_batches=2
base_lr=0.004 weight_decay=0.01 lr_mults=(0, 0.25, 0.5, 1, 2, 4) seeds=32
POOLED internal-default=  0.452+/-0.022 internal-shuffled=  0.466+/-0.029 internal-external_valid=  0.043+/-0.026 internal-external_oracle=  0.043+/-0.026 external_valid-default=  0.409+/-0.021 internal-internal_oracle=  0.000+/-0.000
DECISION TINY_LM_DOMAIN_SHIFT_HPO_WIN
compute_status=TORCH_CPU
```
