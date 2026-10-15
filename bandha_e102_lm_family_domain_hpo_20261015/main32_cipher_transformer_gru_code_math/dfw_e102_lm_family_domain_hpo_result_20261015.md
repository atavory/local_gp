# DFW-E102 Result - LM Family/Domain Adaptive HPO

Decision label: `LM_FAMILY_DOMAIN_HPO_WIN`

## Compute Status

`TORCH_CPU`: `torch=2.16.0a0+fb cuda_available=False`.

## Design

E102 broadens E099 from one tiny Transformer/code-shift run to a family/domain sweep. Each cell trains a character-level causal LM whose early stream is off-target and whose later stream, probe/select split, and target split are target-domain text. Internal HPO uses one-hot same-round probe selection over LR-arm candidate next states. External HPO trains separate fixed-LR trajectories and selects by target-domain validation loss under the same candidate-update budget.
The `cipher_*` domains are adversarial textual shifts: the early stream has the target alphabet but a substitution-cipher transition structure. They are stress tests for whether same-round probes can suppress harmful early LM updates and exploit later target-domain rounds.

- model families: `transformer_small, gru`
- domain shifts: `cipher_to_code, cipher_to_math`
- held-out seeds per cell: `32`
- LR arms: `(0, 0.25, 0.5, 1, 2, 4)`
- probe overhead estimate: `16.67%`

## Pooled and Slice Margins

| slice | internal-default | internal-shuffled | internal-external valid | internal-external oracle | cleared cells |
|---|---:|---:|---:|---:|---:|
| pooled | `0.661+/-0.025` | `0.605+/-0.030` | `0.098+/-0.011` | `0.098+/-0.011` | `4/4` |
| family=transformer_small | `0.538+/-0.020` | `0.486+/-0.031` | `0.079+/-0.015` | `0.079+/-0.015` | `2/2` |
| family=gru | `0.783+/-0.020` | `0.725+/-0.031` | `0.118+/-0.015` | `0.118+/-0.015` | `2/2` |
| domain=cipher_to_code | `0.680+/-0.036` | `0.616+/-0.041` | `0.088+/-0.015` | `0.088+/-0.015` | `2/2` |
| domain=cipher_to_math | `0.641+/-0.036` | `0.594+/-0.044` | `0.108+/-0.016` | `0.108+/-0.016` | `2/2` |

## Cell Margins

| family | domain | internal-default | internal-shuffled | internal-external valid | clear? |
|---|---|---:|---:|---:|---:|
| transformer_small | cipher_to_code | `0.551+/-0.022` | `0.497+/-0.038` | `0.055+/-0.015` | `yes` |
| transformer_small | cipher_to_math | `0.526+/-0.033` | `0.476+/-0.049` | `0.102+/-0.023` | `yes` |
| gru | cipher_to_code | `0.809+/-0.023` | `0.736+/-0.042` | `0.121+/-0.020` | `yes` |
| gru | cipher_to_math | `0.757+/-0.030` | `0.713+/-0.046` | `0.115+/-0.023` | `yes` |

## Interpretation

The LM-family/domain audit clears the main fixed-compute adaptive-HPO comparators: same-round internal probe selection beats default, shuffled probes, and validation-selected external fixed-LR HPO in the pooled result, with most model/domain cells clearing individually.

## Run Log

```text
DFW-E102 LM family/domain adaptive HPO
host=devvm44191.lla0.facebook.com python=/data/users/atavory/fbsource/buck-out/v2/art/fbcode/c95bb19b4474c2e4/aai/double_fan_gpu_runner/__run_local_gp_script_static__/run_local_gp_script_static#link-tree/runtime/bin/run_local_gp_script_static#native-main#platform-runtime#python#py_version_3_12 compute_status=TORCH_CPU torch=2.16.0a0+fb cuda_available=False
rounds=12 switch_round=6 block=40 batch=24 probe_batch=12 eval_batches=2
base_lr=0.004 weight_decay=0.01 lr_mults=(0, 0.25, 0.5, 1, 2, 4) families=(transformer_small, gru) domains=(cipher_to_code, cipher_to_math) seeds=32
POOLED internal-default=  0.661+/-0.025 internal-shuffled=  0.605+/-0.030 internal-external_valid=  0.098+/-0.011 internal-external_oracle=  0.098+/-0.011
clear_cells=4/4
DECISION LM_FAMILY_DOMAIN_HPO_WIN
compute_status=TORCH_CPU
```
