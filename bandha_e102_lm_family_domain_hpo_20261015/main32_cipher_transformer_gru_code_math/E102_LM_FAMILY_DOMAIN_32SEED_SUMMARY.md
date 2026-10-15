# E102 LM Family/Domain 32-Seed HPO

E102 expands the LM evidence beyond one tiny Transformer/code-shift row.

- script: `experiments/dfw_e102_lm_family_domain_hpo.py`
- output dir: `bandha_e102_lm_family_domain_hpo_20261015/main32_cipher_transformer_gru_code_math/`
- model families: Transformer-small and GRU character-level causal LMs
- domains: substitution-cipher code to real code, and substitution-cipher theorem/math text to real theorem/math text
- held-out seeds per family/domain cell: 32
- decision: `LM_FAMILY_DOMAIN_HPO_WIN`
- probe overhead: `16.67%`

Pooled margins:

- internal minus default fixed LR: `+0.661 +/- 0.025`
- internal minus shuffled probes: `+0.605 +/- 0.030`
- internal minus validation-selected external fixed-LR HPO: `+0.098 +/- 0.011`
- internal minus external oracle fixed-LR HPO: `+0.098 +/- 0.011`
- cells cleared individually: `4/4`

Cell external-HPO margins:

- Transformer-small / cipher-to-code: `+0.055 +/- 0.015`
- Transformer-small / cipher-to-math: `+0.102 +/- 0.023`
- GRU / cipher-to-code: `+0.121 +/- 0.020`
- GRU / cipher-to-math: `+0.115 +/- 0.023`

This is the strongest current LM-family support row.  It is still a
character-level adversarial-shift stress test, not a foundation-scale LLM run.
