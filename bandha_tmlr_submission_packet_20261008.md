# TMLR Submission Packet - Deliberately Diverse Aggregation

Status: submission packet, updated 2026-10-13.

Anonymous artifact URL:
https://anonymous.4open.science/r/local_gp-9F83/

## Core Claim

The paper is **Deliberately Diverse Aggregation for Distributed
Optimization**.

Distributed optimization should not treat all worker updates as accidental
exchangeable samples.  We can deliberately create useful update diversity by
assigning optimizer/hyperparameter arms to worker groups or server-side
candidate updates, score those candidate next states with cheap probes, and
aggregate the scored candidates into one shared trajectory.

HPO-aware aggregation is the mechanism.  Deliberate diversity creates the arms;
probes estimate their same-round value; aggregation reinvests exploration into
the model being trained instead of discarding losing HPO trials.

## Main Evidence

1. E090B fixed-compute outer-control HPO is the clearest fixed-compute
   demonstration of deliberately diverse aggregation:
   `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`.
2. Pooled E090B margins:
   internal minus stale/default `+3.789 +/- 0.515`,
   internal minus shuffled `+0.161 +/- 0.061`,
   internal minus external validation-selected HPO `+1.354 +/- 0.463`,
   internal minus external oracle HPO `+0.605 +/- 0.340`.
3. E091 fresh-seed replication of the E090B audit again returns
   `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`: internal minus stale/default
   `+3.878 +/- 0.488`, internal minus shuffled `+0.093 +/- 0.073`, internal
   minus external validation-selected HPO `+1.148 +/- 0.511`, and internal
   minus external oracle HPO `+0.413 +/- 0.345`.
4. E080/E081 remain the replicated outer-control evidence:
   E080 selector minus shuffled `+0.177 +/- 0.049`; E081 packages the rule with
   frozen large-K fallback.
5. E089 is the fixed-compute grouped-LR boundary:
   internal minus external validation-selected HPO `-0.006 +/- 0.003`, while
   internal minus shuffled is `+0.004 +/- 0.002`.
6. E090 is the stricter stale-LR final-trajectory boundary:
   guarded minus stale full-worker `-0.008 +/- 0.002`, but guarded minus
   shuffled `+6.117 +/- 1.841`.
7. E087 is stochastic filtering under exchangeable fake labels, not HPO.

## TMLR Tables

Table 1: IID null/filtering, from E087.

Table 2: designed worker LR arms, from E063/E064.

Table 3: server-side deliberately diverse outer-control candidates, from
E080/E081.

Table 4: stale-default guarded recovery, from E085, with E090 as the limitation.

Table 5: fixed-compute external HPO:
E089 grouped-LR boundary, E090 stale-LR boundary, E090B outer-control win, and
E091 fresh-seed replication.

The table order should make the paper read as one claim: first the
exchangeability boundary, then deliberately diverse worker arms, then
deliberately diverse server-side candidates, then fixed-compute comparisons.

## Limitations

- E090B is an E080/E081-style local-SGD update-map ledger; the paper should
  present it as the clean fixed-compute demonstration of the mechanism and be
  explicit about what trajectory length it covers.
- E089 shows grouped-LR internal content but not a win over validation-selected
  external HPO.
- E090 shows the stale-default guard does not clear the stricter full-worker
  stale/default final-trajectory comparator.
- E087 prevents claiming that exchangeable IID worker labels create persistent
  HPO information.
- The current package is one local-SGD classifier/update-map bed; broader task
  families are future work.

## Submission Checklist

- Positive fixed-compute HPO result: yes, E090B, now replicated by E091.
- Boundary fixed-compute results included: yes, E089 and E090.
- Null/filtering result included: yes, E087.
- Unified claim: yes, deliberately diverse aggregation via HPO-aware probes.
- Anonymous artifact link included: yes,
  `https://anonymous.4open.science/r/local_gp-9F83/`.
- Durable docs updated: research control board, option map, parallel probe
  program, and paper draft.
- Relevant artifacts ready for commit: E090, E090B, E091, docs, and this packet.
