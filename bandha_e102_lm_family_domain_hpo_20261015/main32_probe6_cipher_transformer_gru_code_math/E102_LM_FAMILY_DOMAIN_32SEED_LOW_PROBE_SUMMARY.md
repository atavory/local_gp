# E102 LM Family/Domain 32-Seed Low-Probe HPO

This reruns the positive E102 Transformer-small/GRU cipher-shift audit with
probe batch size 6, reducing estimated probe overhead from `16.67%` to
`8.33%`.

- script: `experiments/dfw_e102_lm_family_domain_hpo.py`
- output dir: `bandha_e102_lm_family_domain_hpo_20261015/main32_probe6_cipher_transformer_gru_code_math/`
- model families: Transformer-small and GRU character-level causal LMs
- domains: substitution-cipher code to real code, and substitution-cipher theorem/math text to real theorem/math text
- held-out seeds per family/domain cell: 32
- decision: `LM_FAMILY_DOMAIN_HPO_WIN`
- probe overhead: `8.33%`

Pooled margins:

- internal minus default fixed LR: `+0.662 +/- 0.027`
- internal minus shuffled probes: `+0.603 +/- 0.032`
- internal minus validation-selected external fixed-LR HPO: `+0.098 +/- 0.012`
- internal minus external oracle fixed-LR HPO: `+0.098 +/- 0.012`
- cells cleared individually: `4/4`

Cell external-HPO margins:

- Transformer-small / cipher-to-code: `+0.053 +/- 0.015`
- Transformer-small / cipher-to-math: `+0.102 +/- 0.024`
- GRU / cipher-to-code: `+0.113 +/- 0.022`
- GRU / cipher-to-math: `+0.123 +/- 0.027`

The low-probe rerun preserves the external-HPO win with the same qualitative
cell pattern as the full-probe E102 run.
