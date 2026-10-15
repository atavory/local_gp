# DFW-E103 Result - Token-Level LM Scale Adaptive HPO

Decision label: `TOKEN_LM_SCALE_HPO_WIN`

## Compute Status

`TORCH_CPU`: `torch=2.16.0a0+fb cuda_available=False`.

## Design

E103 moves from character-level prediction to word/punctuation token prediction, with larger Transformer, recurrent, and convolutional causal LMs. Each cell uses the same fixed-compute protocol as E102: early rounds train on a source domain, later rounds train on target-domain tokens, and probe/select plus target-eval splits are held-out target-domain tokens.

- model families: `token_transformer, token_gru, token_lstm, token_causal_conv`
- domain shifts: `prose_to_code, dialogue_to_json, manual_to_sql, recipe_to_math`
- held-out seeds per cell: `8`
- LR arms: `(0, 0.25, 0.5, 1, 2, 4)`
- probe overhead estimate: `8.33%`

## Pooled and Slice Margins

| slice | internal-default | internal-shuffled | internal-external valid | internal-external oracle | cleared cells |
|---|---:|---:|---:|---:|---:|
| pooled | `1.190+/-0.107` | `1.519+/-0.115` | `0.205+/-0.071` | `0.204+/-0.071` | `10/16` |
| family=token_transformer | `1.148+/-0.180` | `1.455+/-0.218` | `0.603+/-0.131` | `0.602+/-0.131` | `4/4` |
| family=token_gru | `1.553+/-0.169` | `1.786+/-0.183` | `0.384+/-0.061` | `0.384+/-0.061` | `4/4` |
| family=token_lstm | `1.474+/-0.203` | `1.693+/-0.223` | `0.043+/-0.060` | `0.043+/-0.060` | `2/4` |
| family=token_causal_conv | `0.584+/-0.119` | `1.143+/-0.235` | `-0.212+/-0.101` | `-0.214+/-0.101` | `0/4` |
| domain=prose_to_code | `1.280+/-0.221` | `1.568+/-0.184` | `0.245+/-0.164` | `0.244+/-0.165` | `3/4` |
| domain=dialogue_to_json | `0.473+/-0.079` | `0.869+/-0.227` | `0.048+/-0.086` | `0.047+/-0.086` | `2/4` |
| domain=manual_to_sql | `1.579+/-0.126` | `1.966+/-0.143` | `0.375+/-0.145` | `0.375+/-0.145` | `3/4` |
| domain=recipe_to_math | `1.427+/-0.166` | `1.673+/-0.174` | `0.151+/-0.144` | `0.150+/-0.143` | `2/4` |

## Cell Margins

| family | domain | internal-default | internal-shuffled | internal-external valid | clear? |
|---|---|---:|---:|---:|---:|
| token_transformer | prose_to_code | `1.122+/-0.074` | `1.379+/-0.105` | `0.591+/-0.127` | `yes` |
| token_transformer | dialogue_to_json | `0.365+/-0.091` | `0.568+/-0.066` | `0.118+/-0.113` | `yes` |
| token_transformer | manual_to_sql | `1.739+/-0.050` | `2.186+/-0.136` | `1.038+/-0.121` | `yes` |
| token_transformer | recipe_to_math | `1.364+/-0.081` | `1.685+/-0.228` | `0.667+/-0.165` | `yes` |
| token_gru | prose_to_code | `1.882+/-0.074` | `2.020+/-0.276` | `0.618+/-0.086` | `yes` |
| token_gru | dialogue_to_json | `0.736+/-0.086` | `1.392+/-0.535` | `0.291+/-0.113` | `yes` |
| token_gru | manual_to_sql | `1.772+/-0.046` | `1.875+/-0.261` | `0.338+/-0.030` | `yes` |
| token_gru | recipe_to_math | `1.824+/-0.047` | `1.857+/-0.193` | `0.289+/-0.059` | `yes` |
| token_lstm | prose_to_code | `1.764+/-0.084` | `1.860+/-0.273` | `0.222+/-0.113` | `yes` |
| token_lstm | dialogue_to_json | `0.492+/-0.082` | `0.995+/-0.553` | `-0.126+/-0.103` | `no` |
| token_lstm | manual_to_sql | `1.822+/-0.087` | `1.971+/-0.203` | `0.090+/-0.053` | `yes` |
| token_lstm | recipe_to_math | `1.819+/-0.040` | `1.945+/-0.317` | `-0.015+/-0.039` | `no` |
| token_causal_conv | prose_to_code | `0.351+/-0.172` | `1.014+/-0.299` | `-0.451+/-0.184` | `no` |
| token_causal_conv | dialogue_to_json | `0.296+/-0.160` | `0.521+/-0.210` | `-0.091+/-0.177` | `no` |
| token_causal_conv | manual_to_sql | `0.985+/-0.080` | `1.832+/-0.438` | `0.033+/-0.073` | `no` |
| token_causal_conv | recipe_to_math | `0.703+/-0.130` | `1.206+/-0.411` | `-0.338+/-0.172` | `no` |

## Interpretation

The token-level LM scale audit clears the fixed-compute adaptive-HPO comparators. This is stronger than the character-level stress test, though still far below foundation-scale pretraining.

## Run Log

```text
DFW-E103 token-level LM scale adaptive HPO
host=devvm44191.lla0.facebook.com python=/data/users/atavory/fbsource/buck-out/v2/art/fbcode/c95bb19b4474c2e4/aai/double_fan_gpu_runner/__run_local_gp_script_static__/run_local_gp_script_static#link-tree/runtime/bin/run_local_gp_script_static#native-main#platform-runtime#python#py_version_3_12 compute_status=TORCH_CPU torch=2.16.0a0+fb cuda_available=False
rounds=16 switch_round=8 block=32 batch=32 probe_batch=8 eval_batches=2
base_lr=0.003 weight_decay=0.01 lr_mults=(0, 0.25, 0.5, 1, 2, 4) families=(token_transformer, token_gru, token_lstm, token_causal_conv) domains=(prose_to_code, dialogue_to_json, manual_to_sql, recipe_to_math) seeds=8
POOLED internal-default=  1.190+/-0.107 internal-shuffled=  1.519+/-0.115 internal-external_valid=  0.205+/-0.071 internal-external_oracle=  0.204+/-0.071
clear_cells=10/16
DECISION TOKEN_LM_SCALE_HPO_WIN
compute_status=TORCH_CPU
```
