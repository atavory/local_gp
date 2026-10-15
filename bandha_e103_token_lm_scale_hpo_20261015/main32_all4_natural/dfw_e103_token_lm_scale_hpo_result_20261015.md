# DFW-E103 Result - Token-Level LM Scale Adaptive HPO

Decision label: `TOKEN_LM_SCALE_HPO_WIN`

## Compute Status

`TORCH_CPU`: `torch=2.16.0a0+fb cuda_available=False`.

## Design

E103 moves from character-level prediction to word/punctuation token prediction, with larger Transformer, recurrent, and convolutional causal LMs. Each cell uses the same fixed-compute protocol as E102: early rounds train on a source domain, later rounds train on target-domain tokens, and probe/select plus target-eval splits are held-out target-domain tokens.

- model families: `token_transformer, token_gru, token_lstm, token_causal_conv`
- domain shifts: `prose_to_code, dialogue_to_json, manual_to_sql, recipe_to_math`
- held-out seeds per cell: `32`
- LR arms: `(0, 0.25, 0.5, 1, 2, 4)`
- probe overhead estimate: `8.33%`

## Pooled and Slice Margins

| slice | internal-default | internal-shuffled | internal-external valid | internal-external oracle | cleared cells |
|---|---:|---:|---:|---:|---:|
| pooled | `1.183+/-0.054` | `1.569+/-0.057` | `0.196+/-0.036` | `0.195+/-0.036` | `10/16` |
| family=token_transformer | `1.125+/-0.092` | `1.466+/-0.108` | `0.593+/-0.068` | `0.593+/-0.068` | `4/4` |
| family=token_gru | `1.537+/-0.086` | `1.798+/-0.099` | `0.371+/-0.042` | `0.371+/-0.042` | `4/4` |
| family=token_lstm | `1.473+/-0.103` | `1.761+/-0.107` | `0.038+/-0.032` | `0.038+/-0.032` | `2/4` |
| family=token_causal_conv | `0.598+/-0.055` | `1.250+/-0.117` | `-0.220+/-0.048` | `-0.221+/-0.048` | `0/4` |
| domain=prose_to_code | `1.321+/-0.099` | `1.654+/-0.089` | `0.300+/-0.077` | `0.300+/-0.077` | `3/4` |
| domain=dialogue_to_json | `0.452+/-0.036` | `0.923+/-0.103` | `0.044+/-0.041` | `0.043+/-0.041` | `2/4` |
| domain=manual_to_sql | `1.572+/-0.069` | `1.968+/-0.075` | `0.356+/-0.081` | `0.356+/-0.081` | `3/4` |
| domain=recipe_to_math | `1.387+/-0.088` | `1.731+/-0.100` | `0.082+/-0.070` | `0.082+/-0.070` | `2/4` |

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
| token_lstm | prose_to_code | `1.776+/-0.040` | `1.930+/-0.122` | `0.207+/-0.059` | `yes` |
| token_lstm | dialogue_to_json | `0.469+/-0.048` | `1.110+/-0.259` | `-0.098+/-0.075` | `no` |
| token_lstm | manual_to_sql | `1.826+/-0.034` | `2.018+/-0.117` | `0.062+/-0.024` | `yes` |
| token_lstm | recipe_to_math | `1.820+/-0.024` | `1.985+/-0.142` | `-0.019+/-0.021` | `no` |
| token_causal_conv | prose_to_code | `0.503+/-0.084` | `1.221+/-0.176` | `-0.318+/-0.082` | `no` |
| token_causal_conv | dialogue_to_json | `0.291+/-0.066` | `0.630+/-0.101` | `-0.078+/-0.071` | `no` |
| token_causal_conv | manual_to_sql | `0.908+/-0.054` | `1.815+/-0.206` | `-0.060+/-0.051` | `no` |
| token_causal_conv | recipe_to_math | `0.689+/-0.097` | `1.335+/-0.228` | `-0.424+/-0.106` | `no` |

## Interpretation

The token-level LM scale audit clears the fixed-compute adaptive-HPO comparators. This is stronger than the character-level stress test, though still far below foundation-scale pretraining.

## Run Log

```text
DFW-E103 token-level LM scale adaptive HPO
host=devvm44191.lla0.facebook.com python=/data/users/atavory/fbsource/buck-out/v2/art/fbcode/c95bb19b4474c2e4/aai/double_fan_gpu_runner/__run_local_gp_script_static__/run_local_gp_script_static#link-tree/runtime/bin/run_local_gp_script_static#native-main#platform-runtime#python#py_version_3_12 compute_status=TORCH_CPU torch=2.16.0a0+fb cuda_available=False
rounds=16 switch_round=8 block=32 batch=32 probe_batch=8 eval_batches=2
base_lr=0.003 weight_decay=0.01 lr_mults=(0, 0.25, 0.5, 1, 2, 4) families=(token_transformer, token_gru, token_lstm, token_causal_conv) domains=(prose_to_code, dialogue_to_json, manual_to_sql, recipe_to_math) seeds=32
POOLED internal-default=  1.183+/-0.054 internal-shuffled=  1.569+/-0.057 internal-external_valid=  0.196+/-0.036 internal-external_oracle=  0.195+/-0.036
clear_cells=10/16
DECISION TOKEN_LM_SCALE_HPO_WIN
compute_status=TORCH_CPU
```
