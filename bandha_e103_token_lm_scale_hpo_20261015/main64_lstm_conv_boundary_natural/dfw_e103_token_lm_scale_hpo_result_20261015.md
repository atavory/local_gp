# DFW-E103 Result - Token-Level LM Scale Adaptive HPO

Decision label: `TOKEN_LM_SCALE_LABEL_CONTENT`

## Compute Status

`TORCH_CPU`: `torch=2.16.0a0+fb cuda_available=False`.

## Design

E103 moves from character-level prediction to word/punctuation token prediction, with larger Transformer, recurrent, and convolutional causal LMs. Each cell uses the same fixed-compute protocol as E102: early rounds train on a source domain, later rounds train on target-domain tokens, and probe/select plus target-eval splits are held-out target-domain tokens.

- model families: `token_lstm, token_causal_conv`
- domain shifts: `prose_to_code, dialogue_to_json, manual_to_sql, recipe_to_math`
- held-out seeds per cell: `64`
- LR arms: `(0, 0.25, 0.5, 1, 2, 4)`
- probe overhead estimate: `8.33%`

## Pooled and Slice Margins

| slice | internal-default | internal-shuffled | internal-external valid | internal-external oracle | cleared cells |
|---|---:|---:|---:|---:|---:|
| pooled | `1.030+/-0.056` | `1.531+/-0.061` | `-0.104+/-0.023` | `-0.105+/-0.023` | `2/8` |
| family=token_lstm | `1.464+/-0.074` | `1.806+/-0.078` | `0.028+/-0.023` | `0.028+/-0.023` | `2/4` |
| family=token_causal_conv | `0.596+/-0.038` | `1.256+/-0.081` | `-0.236+/-0.033` | `-0.237+/-0.033` | `0/4` |
| domain=prose_to_code | `1.116+/-0.116` | `1.558+/-0.102` | `-0.088+/-0.063` | `-0.089+/-0.063` | `1/2` |
| domain=dialogue_to_json | `0.377+/-0.029` | `0.948+/-0.123` | `-0.099+/-0.034` | `-0.100+/-0.034` | `0/2` |
| domain=manual_to_sql | `1.378+/-0.085` | `1.943+/-0.083` | `-0.008+/-0.022` | `-0.008+/-0.022` | `1/2` |
| domain=recipe_to_math | `1.250+/-0.103` | `1.675+/-0.108` | `-0.221+/-0.049` | `-0.221+/-0.049` | `0/2` |

## Cell Margins

| family | domain | internal-default | internal-shuffled | internal-external valid | clear? |
|---|---|---:|---:|---:|---:|
| token_lstm | prose_to_code | `1.759+/-0.028` | `1.952+/-0.084` | `0.207+/-0.041` | `yes` |
| token_lstm | dialogue_to_json | `0.444+/-0.032` | `1.231+/-0.214` | `-0.125+/-0.051` | `no` |
| token_lstm | manual_to_sql | `1.850+/-0.025` | `2.051+/-0.085` | `0.060+/-0.017` | `yes` |
| token_lstm | recipe_to_math | `1.805+/-0.020` | `1.990+/-0.105` | `-0.031+/-0.020` | `no` |
| token_causal_conv | prose_to_code | `0.473+/-0.059` | `1.165+/-0.126` | `-0.384+/-0.061` | `no` |
| token_causal_conv | dialogue_to_json | `0.310+/-0.043` | `0.664+/-0.071` | `-0.073+/-0.044` | `no` |
| token_causal_conv | manual_to_sql | `0.905+/-0.033` | `1.834+/-0.137` | `-0.076+/-0.033` | `no` |
| token_causal_conv | recipe_to_math | `0.695+/-0.068` | `1.359+/-0.153` | `-0.412+/-0.069` | `no` |

## Interpretation

The token-level LM run should be treated as a boundary result under this budget; use the slice margins to choose the next rerun.

## Run Log

```text
DFW-E103 token-level LM scale adaptive HPO
host=devvm44191.lla0.facebook.com python=/data/users/atavory/fbsource/buck-out/v2/art/fbcode/c95bb19b4474c2e4/aai/double_fan_gpu_runner/__run_local_gp_script_static__/run_local_gp_script_static#link-tree/runtime/bin/run_local_gp_script_static#native-main#platform-runtime#python#py_version_3_12 compute_status=TORCH_CPU torch=2.16.0a0+fb cuda_available=False
rounds=16 switch_round=8 block=32 batch=32 probe_batch=8 eval_batches=2
base_lr=0.003 weight_decay=0.01 lr_mults=(0, 0.25, 0.5, 1, 2, 4) families=(token_lstm, token_causal_conv) domains=(prose_to_code, dialogue_to_json, manual_to_sql, recipe_to_math) seeds=64
POOLED internal-default=  1.030+/-0.056 internal-shuffled=  1.531+/-0.061 internal-external_valid= -0.104+/-0.023 internal-external_oracle= -0.105+/-0.023
clear_cells=2/8
DECISION TOKEN_LM_SCALE_LABEL_CONTENT
compute_status=TORCH_CPU
```
