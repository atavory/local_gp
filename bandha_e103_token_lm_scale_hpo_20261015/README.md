# E103 Token-Level LM Scale HPO Summary

E103 answers the character-level/adversarial-shift concern from E102.
It uses word/punctuation token prediction, natural text-domain shifts, and the
same fixed-compute internal-vs-external HPO protocol.

## Headline Run

Directory: `main64_core_natural/`

- Decision: `TOKEN_LM_SCALE_HPO_WIN`.
- Families: token-level Transformer and GRU causal LMs.
- Shifts: prose-to-code, dialogue-to-JSON, manual-to-SQL, recipe-to-math.
- Seeds: 64 held-out seeds per family/domain cell.
- Probe overhead: `8.33%`.
- Pooled internal minus default: `+1.332 +/- 0.048`.
- Pooled internal minus shuffled probes: `+1.639 +/- 0.055`.
- Pooled internal minus validation-selected external fixed-LR HPO:
  `+0.474 +/- 0.030`.
- Cell coverage: all 8 Transformer/GRU family-domain cells clear.

This is the current LM headline: natural token-level shifts, not
character-level cipher shifts.

## Heavier Trajectory Run

Directory: `main32_core_natural_rounds32_block48_batch48/`

- Decision: `TOKEN_LM_SCALE_HPO_WIN`.
- Families: token-level Transformer and GRU causal LMs.
- Configuration: 32 rounds, switch at round 16, token block 48, batch 48,
  probe batch 12, and 3 target-eval batches.
- Seeds: 32 held-out seeds per family/domain cell.
- Probe overhead: `8.33%`.
- Pooled internal minus default: `+0.761 +/- 0.042`.
- Pooled internal minus shuffled probes: `+1.495 +/- 0.092`.
- Pooled internal minus validation-selected external fixed-LR HPO:
  `+0.221 +/- 0.033`.
- Cell coverage: all 8 Transformer/GRU family-domain cells clear.

This is the heavier LM support row: longer training horizon and larger token
context/batches still clear the fixed-compute adaptive-HPO comparator.

## Broad-Family Boundary Run

Directory: `main32_all4_natural/`

- Decision: `TOKEN_LM_SCALE_HPO_WIN`.
- Families: token Transformer, GRU, LSTM, and causal-conv.
- Seeds: 32 held-out seeds per family/domain cell.
- Probe overhead: `8.33%`.
- Pooled internal minus default: `+1.183 +/- 0.054`.
- Pooled internal minus shuffled probes: `+1.569 +/- 0.057`.
- Pooled internal minus validation-selected external fixed-LR HPO:
  `+0.196 +/- 0.036`.
- Cell coverage: 10/16 cells clear.

Family interpretation: Transformer and GRU clear all cells; LSTM is partial
(2/4); causal-conv is a boundary family (0/4 against external HPO).

## Confirmed Boundary Slice

Directory: `main64_lstm_conv_boundary_natural/`

- Decision: `TOKEN_LM_SCALE_LABEL_CONTENT`.
- Families: LSTM and causal-conv token-level causal LMs.
- Seeds: 64 held-out seeds per family/domain cell.
- Probe overhead: `8.33%`.
- Pooled internal minus default: `+1.030 +/- 0.056`.
- Pooled internal minus shuffled probes: `+1.531 +/- 0.061`.
- Pooled internal minus validation-selected external fixed-LR HPO:
  `-0.104 +/- 0.023`.
- Cell coverage: 2/8 cells clear.

Interpretation: probe labels carry real content for these families, but this
budget does not beat validation-selected external fixed-LR HPO. LSTM remains
partial and causal-conv is negative against external HPO.
