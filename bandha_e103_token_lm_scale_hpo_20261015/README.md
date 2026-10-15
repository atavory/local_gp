# E103 Token-Level LM Scale HPO Summary

E103 answers the character-level/adversarial-shift concern from E102.
It uses word/punctuation token prediction, natural text-domain shifts, and the
same fixed-compute internal-vs-external HPO protocol.

## Headline Run

Directory: `main32_core_natural/`

- Decision: `TOKEN_LM_SCALE_HPO_WIN`.
- Families: token-level Transformer and GRU causal LMs.
- Shifts: prose-to-code, dialogue-to-JSON, manual-to-SQL, recipe-to-math.
- Seeds: 32 held-out seeds per family/domain cell.
- Probe overhead: `8.33%`.
- Pooled internal minus default: `+1.331 +/- 0.068`.
- Pooled internal minus shuffled probes: `+1.632 +/- 0.076`.
- Pooled internal minus validation-selected external fixed-LR HPO:
  `+0.482 +/- 0.042`.
- Cell coverage: all 8 Transformer/GRU family-domain cells clear.

This is the current LM headline: natural token-level shifts, not
character-level cipher shifts.

## Broad-Family Boundary Run

Directory: `main8_all4_natural/`

- Decision: `TOKEN_LM_SCALE_HPO_WIN`.
- Families: token Transformer, GRU, LSTM, and causal-conv.
- Seeds: 8 held-out seeds per family/domain cell.
- Probe overhead: `8.33%`.
- Pooled internal minus default: `+1.190 +/- 0.107`.
- Pooled internal minus shuffled probes: `+1.519 +/- 0.115`.
- Pooled internal minus validation-selected external fixed-LR HPO:
  `+0.205 +/- 0.071`.
- Cell coverage: 10/16 cells clear.

Family interpretation: Transformer and GRU clear all cells; LSTM is partial
(2/4); causal-conv is a boundary family (0/4 against external HPO).
