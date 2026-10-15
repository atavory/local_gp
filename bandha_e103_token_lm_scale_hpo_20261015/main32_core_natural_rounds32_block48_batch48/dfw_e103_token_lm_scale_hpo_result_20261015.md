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
| pooled | `0.761+/-0.042` | `1.495+/-0.092` | `0.221+/-0.033` | `0.221+/-0.033` | `8/8` |
| family=token_transformer | `0.869+/-0.073` | `1.781+/-0.150` | `0.392+/-0.050` | `0.392+/-0.050` | `4/4` |
| family=token_gru | `0.652+/-0.035` | `1.209+/-0.080` | `0.051+/-0.013` | `0.051+/-0.013` | `4/4` |
| domain=prose_to_code | `1.015+/-0.027` | `1.783+/-0.142` | `0.362+/-0.092` | `0.362+/-0.092` | `2/2` |
| domain=dialogue_to_json | `0.339+/-0.047` | `1.237+/-0.112` | `0.128+/-0.022` | `0.128+/-0.022` | `2/2` |
| domain=manual_to_sql | `0.792+/-0.074` | `1.482+/-0.241` | `0.208+/-0.059` | `0.208+/-0.059` | `2/2` |
| domain=recipe_to_math | `0.897+/-0.068` | `1.479+/-0.190` | `0.188+/-0.060` | `0.188+/-0.060` | `2/2` |

## Cell Margins

| family | domain | internal-default | internal-shuffled | internal-external valid | clear? |
|---|---|---:|---:|---:|---:|
| token_transformer | prose_to_code | `1.065+/-0.039` | `2.062+/-0.209` | `0.680+/-0.094` | `yes` |
| token_transformer | dialogue_to_json | `0.167+/-0.025` | `0.931+/-0.115` | `0.115+/-0.028` | `yes` |
| token_transformer | manual_to_sql | `1.084+/-0.032` | `2.174+/-0.328` | `0.405+/-0.066` | `yes` |
| token_transformer | recipe_to_math | `1.161+/-0.035` | `1.959+/-0.282` | `0.366+/-0.083` | `yes` |
| token_gru | prose_to_code | `0.965+/-0.027` | `1.503+/-0.136` | `0.043+/-0.008` | `yes` |
| token_gru | dialogue_to_json | `0.511+/-0.030` | `1.543+/-0.120` | `0.142+/-0.034` | `yes` |
| token_gru | manual_to_sql | `0.499+/-0.017` | `0.789+/-0.101` | `0.010+/-0.002` | `yes` |
| token_gru | recipe_to_math | `0.632+/-0.015` | `0.999+/-0.100` | `0.011+/-0.003` | `yes` |

## Interpretation

The token-level LM scale audit clears the fixed-compute adaptive-HPO comparators. This is stronger than the character-level stress test, though still far below foundation-scale pretraining.

## Run Log

```text
DFW-E103 token-level LM scale adaptive HPO
host=devvm44191.lla0.facebook.com python=/data/users/atavory/fbsource/buck-out/v2/art/fbcode/c95bb19b4474c2e4/aai/double_fan_gpu_runner/__run_local_gp_script_static__/run_local_gp_script_static#link-tree/runtime/bin/run_local_gp_script_static#native-main#platform-runtime#python#py_version_3_12 compute_status=TORCH_CPU torch=2.16.0a0+fb cuda_available=False
rounds=32 switch_round=16 block=48 batch=48 probe_batch=12 eval_batches=3
base_lr=0.003 weight_decay=0.01 lr_mults=(0, 0.25, 0.5, 1, 2, 4) families=(token_transformer, token_gru) domains=(prose_to_code, dialogue_to_json, manual_to_sql, recipe_to_math) seeds=32
POOLED internal-default=  0.761+/-0.042 internal-shuffled=  1.495+/-0.092 internal-external_valid=  0.221+/-0.033 internal-external_oracle=  0.221+/-0.033
clear_cells=8/8
DECISION TOKEN_LM_SCALE_HPO_WIN
compute_status=TORCH_CPU
```
