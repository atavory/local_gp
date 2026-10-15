# DFW-E103 Result - Token-Level LM Scale Adaptive HPO

Decision label: `TOKEN_LM_SCALE_HPO_WIN`

## Compute Status

`TORCH_CPU`: `torch=2.16.0a0+fb cuda_available=False`.

## Design

E103 moves from character-level prediction to word/punctuation token prediction, with larger Transformer, recurrent, and convolutional causal LMs. Each cell uses the same fixed-compute protocol as E102: early rounds train on a source domain, later rounds train on target-domain tokens, and probe/select plus target-eval splits are held-out target-domain tokens.

- model families: `token_transformer, token_gru`
- domain shifts: `prose_to_code, dialogue_to_json, manual_to_sql, recipe_to_math`
- held-out seeds per cell: `32`
- LR arms: `(0, 0.25, 0.5, 1, 2, 4)`
- probe overhead estimate: `8.33%`

## Pooled and Slice Margins

| slice | internal-default | internal-shuffled | internal-external valid | internal-external oracle | cleared cells |
|---|---:|---:|---:|---:|---:|
| pooled | `1.331+/-0.068` | `1.632+/-0.076` | `0.482+/-0.042` | `0.482+/-0.042` | `8/8` |
| family=token_transformer | `1.125+/-0.092` | `1.466+/-0.108` | `0.593+/-0.068` | `0.593+/-0.068` | `4/4` |
| family=token_gru | `1.537+/-0.086` | `1.798+/-0.099` | `0.371+/-0.042` | `0.371+/-0.042` | `4/4` |
| domain=prose_to_code | `1.502+/-0.095` | `1.733+/-0.110` | `0.656+/-0.044` | `0.656+/-0.044` | `2/2` |
| domain=dialogue_to_json | `0.525+/-0.049` | `0.975+/-0.140` | `0.175+/-0.044` | `0.175+/-0.044` | `2/2` |
| domain=manual_to_sql | `1.777+/-0.022` | `2.019+/-0.090` | `0.711+/-0.101` | `0.710+/-0.101` | `2/2` |
| domain=recipe_to_math | `1.520+/-0.084` | `1.801+/-0.125` | `0.386+/-0.058` | `0.385+/-0.058` | `2/2` |

## Cell Margins

| family | domain | internal-default | internal-shuffled | internal-external valid | clear? |
|---|---|---:|---:|---:|---:|
| token_transformer | prose_to_code | `1.132+/-0.042` | `1.416+/-0.081` | `0.630+/-0.060` | `yes` |
| token_transformer | dialogue_to_json | `0.355+/-0.032` | `0.722+/-0.113` | `0.148+/-0.047` | `yes` |
| token_transformer | manual_to_sql | `1.793+/-0.030` | `2.112+/-0.113` | `1.101+/-0.052` | `yes` |
| token_transformer | recipe_to_math | `1.221+/-0.077` | `1.615+/-0.189` | `0.495+/-0.099` | `yes` |
| token_gru | prose_to_code | `1.871+/-0.033` | `2.050+/-0.134` | `0.682+/-0.065` | `yes` |
| token_gru | dialogue_to_json | `0.695+/-0.040` | `1.229+/-0.226` | `0.202+/-0.074` | `yes` |
| token_gru | manual_to_sql | `1.762+/-0.032` | `1.926+/-0.133` | `0.321+/-0.036` | `yes` |
| token_gru | recipe_to_math | `1.820+/-0.021` | `1.987+/-0.138` | `0.278+/-0.031` | `yes` |

## Interpretation

The token-level LM scale audit clears the fixed-compute adaptive-HPO comparators. This is stronger than the character-level stress test, though still far below foundation-scale pretraining.

## Run Log

```text
DFW-E103 token-level LM scale adaptive HPO
host=devvm44191.lla0.facebook.com python=/data/users/atavory/fbsource/buck-out/v2/art/fbcode/c95bb19b4474c2e4/aai/double_fan_gpu_runner/__run_local_gp_script_static__/run_local_gp_script_static#link-tree/runtime/bin/run_local_gp_script_static#native-main#platform-runtime#python#py_version_3_12 compute_status=TORCH_CPU torch=2.16.0a0+fb cuda_available=False
rounds=16 switch_round=8 block=32 batch=32 probe_batch=8 eval_batches=2
base_lr=0.003 weight_decay=0.01 lr_mults=(0, 0.25, 0.5, 1, 2, 4) families=(token_transformer, token_gru) domains=(prose_to_code, dialogue_to_json, manual_to_sql, recipe_to_math) seeds=32
POOLED internal-default=  1.331+/-0.068 internal-shuffled=  1.632+/-0.076 internal-external_valid=  0.482+/-0.042 internal-external_oracle=  0.482+/-0.042
clear_cells=8/8
DECISION TOKEN_LM_SCALE_HPO_WIN
compute_status=TORCH_CPU
```
