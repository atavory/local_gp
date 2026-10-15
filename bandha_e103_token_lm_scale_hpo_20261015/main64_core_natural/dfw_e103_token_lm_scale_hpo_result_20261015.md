# DFW-E103 Result - Token-Level LM Scale Adaptive HPO

Decision label: `TOKEN_LM_SCALE_HPO_WIN`

## Compute Status

`TORCH_CPU`: `torch=2.16.0a0+fb cuda_available=False`.

## Design

E103 moves from character-level prediction to word/punctuation token prediction, with larger Transformer, recurrent, and convolutional causal LMs. Each cell uses the same fixed-compute protocol as E102: early rounds train on a source domain, later rounds train on target-domain tokens, and probe/select plus target-eval splits are held-out target-domain tokens.

- model families: `token_transformer, token_gru`
- domain shifts: `prose_to_code, dialogue_to_json, manual_to_sql, recipe_to_math`
- held-out seeds per cell: `64`
- LR arms: `(0, 0.25, 0.5, 1, 2, 4)`
- probe overhead estimate: `8.33%`

## Pooled and Slice Margins

| slice | internal-default | internal-shuffled | internal-external valid | internal-external oracle | cleared cells |
|---|---:|---:|---:|---:|---:|
| pooled | `1.332+/-0.048` | `1.639+/-0.055` | `0.474+/-0.030` | `0.474+/-0.030` | `8/8` |
| family=token_transformer | `1.127+/-0.065` | `1.460+/-0.076` | `0.590+/-0.048` | `0.589+/-0.048` | `4/4` |
| family=token_gru | `1.537+/-0.062` | `1.818+/-0.074` | `0.358+/-0.029` | `0.358+/-0.029` | `4/4` |
| domain=prose_to_code | `1.495+/-0.067` | `1.742+/-0.079` | `0.645+/-0.033` | `0.644+/-0.033` | `2/2` |
| domain=dialogue_to_json | `0.518+/-0.035` | `0.974+/-0.110` | `0.167+/-0.030` | `0.166+/-0.030` | `2/2` |
| domain=manual_to_sql | `1.790+/-0.015` | `2.046+/-0.067` | `0.703+/-0.071` | `0.703+/-0.071` | `2/2` |
| domain=recipe_to_math | `1.526+/-0.058` | `1.796+/-0.084` | `0.381+/-0.043` | `0.381+/-0.043` | `2/2` |

## Cell Margins

| family | domain | internal-default | internal-shuffled | internal-external valid | clear? |
|---|---|---:|---:|---:|---:|
| token_transformer | prose_to_code | `1.126+/-0.030` | `1.411+/-0.060` | `0.639+/-0.046` | `yes` |
| token_transformer | dialogue_to_json | `0.355+/-0.024` | `0.700+/-0.071` | `0.138+/-0.030` | `yes` |
| token_transformer | manual_to_sql | `1.798+/-0.021` | `2.134+/-0.085` | `1.085+/-0.042` | `yes` |
| token_transformer | recipe_to_math | `1.230+/-0.051` | `1.595+/-0.120` | `0.499+/-0.072` | `yes` |
| token_gru | prose_to_code | `1.863+/-0.022` | `2.072+/-0.090` | `0.651+/-0.047` | `yes` |
| token_gru | dialogue_to_json | `0.682+/-0.033` | `1.247+/-0.185` | `0.196+/-0.051` | `yes` |
| token_gru | manual_to_sql | `1.781+/-0.021` | `1.958+/-0.098` | `0.322+/-0.025` | `yes` |
| token_gru | recipe_to_math | `1.822+/-0.016` | `1.997+/-0.096` | `0.264+/-0.023` | `yes` |

## Interpretation

The token-level LM scale audit clears the fixed-compute adaptive-HPO comparators. This is stronger than the character-level stress test, though still far below foundation-scale pretraining.

## Run Log

```text
DFW-E103 token-level LM scale adaptive HPO
host=devvm44191.lla0.facebook.com python=/data/users/atavory/fbsource/buck-out/v2/art/fbcode/c95bb19b4474c2e4/aai/double_fan_gpu_runner/__run_local_gp_script_static__/run_local_gp_script_static#link-tree/runtime/bin/run_local_gp_script_static#native-main#platform-runtime#python#py_version_3_12 compute_status=TORCH_CPU torch=2.16.0a0+fb cuda_available=False
rounds=16 switch_round=8 block=32 batch=32 probe_batch=8 eval_batches=2
base_lr=0.003 weight_decay=0.01 lr_mults=(0, 0.25, 0.5, 1, 2, 4) families=(token_transformer, token_gru) domains=(prose_to_code, dialogue_to_json, manual_to_sql, recipe_to_math) seeds=64
POOLED internal-default=  1.332+/-0.048 internal-shuffled=  1.639+/-0.055 internal-external_valid=  0.474+/-0.030 internal-external_oracle=  0.474+/-0.030
clear_cells=8/8
DECISION TOKEN_LM_SCALE_HPO_WIN
compute_status=TORCH_CPU
```
