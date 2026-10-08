# TMLR Submission Packet - HPO-Aware Aggregation

Status: submission packet, 2026-10-08.

## Core Claim

HPO-aware aggregation is useful when distributed workers create designed
nonexchangeable optimizer arms and the server can score the arms from the same
round.  The main TMLR claim is not generic better averaging.  It is practical
HPO-aware aggregation under designed nonexchangeability, especially low-K
outer-control HPO.

## Main Evidence

1. E090B fixed-compute outer-control HPO is the main positive:
   `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`.
2. Pooled E090B margins:
   internal minus stale/default `+3.789 +/- 0.515`,
   internal minus shuffled `+0.161 +/- 0.061`,
   internal minus external validation-selected HPO `+1.354 +/- 0.463`,
   internal minus external oracle HPO `+0.605 +/- 0.340`.
3. E080/E081 remain the core replicated outer-control evidence:
   E080 selector minus shuffled `+0.177 +/- 0.049`; E081 packages the rule with
   frozen large-K fallback.
4. E089 is the fixed-compute grouped-LR boundary:
   internal minus external validation-selected HPO `-0.006 +/- 0.003`, while
   internal minus shuffled is `+0.004 +/- 0.002`.
5. E090 is the stricter stale-LR final-trajectory boundary:
   guarded minus stale full-worker `-0.008 +/- 0.002`, but guarded minus
   shuffled `+6.117 +/- 1.841`.
6. E087 is stochastic filtering under exchangeable fake labels, not HPO.

## TMLR Tables

Table 1: IID null/filtering, from E087.

Table 2: designed worker LR arms, from E063/E064.

Table 3: low-K outer-control HPO, from E080/E081.

Table 4: stale-default guarded recovery, from E085, with E090 as the limitation.

Table 5: fixed-compute external HPO:
E089 grouped-LR boundary, E090 stale-LR boundary, and E090B outer-control win.

## Limitations

- E090B is an E080/E081-style one-step local-SGD update-map ledger, not a full
  final-training-trajectory theorem.
- E089 shows grouped-LR internal content but not a win over validation-selected
  external HPO.
- E090 shows the stale-default guard does not clear the stricter full-worker
  stale/default final-trajectory comparator.
- E087 prevents claiming that exchangeable IID worker labels create persistent
  HPO information.
- The current package is one local-SGD classifier/update-map bed; broader task
  families are future work.

## Submission Checklist

- Positive fixed-compute HPO result: yes, E090B.
- Boundary fixed-compute results included: yes, E089 and E090.
- Null/filtering result included: yes, E087.
- Narrowed claim: yes, practical HPO-aware aggregation under designed
  nonexchangeability.
- Durable docs updated: research control board, option map, parallel probe
  program, and paper draft.
- Relevant artifacts ready for commit: E090, E090B, docs, and this packet.
