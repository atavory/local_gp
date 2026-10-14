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
4. E093 64-seed replication again returns
   `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`: internal minus stale/default
   `+3.829 +/- 0.378`, internal minus shuffled `+0.140 +/- 0.058`, internal
   minus external validation-selected HPO `+1.224 +/- 0.313`, and internal
   minus external oracle HPO `+0.343 +/- 0.255`.
5. E094 64-seed `rounds=16` audit upgrades the longer-round story to
   `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`: internal minus stale/default
   `+4.403 +/- 0.417`, internal minus fresh/theory `+0.194 +/- 0.072`,
   internal minus shuffled `+0.233 +/- 0.076`, internal minus external
   validation-selected HPO `+1.275 +/- 0.239`, and internal minus external
   oracle HPO `+0.465 +/- 0.222`.
6. E095 large-K audit at `K=16`, `rounds=16`, and 64 seeds also returns
   `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`: internal minus stale/default
   `+0.094 +/- 0.061`, internal minus shuffled `+0.075 +/- 0.048`, internal
   minus external validation-selected HPO `+0.516 +/- 0.262`, and internal
   minus external oracle HPO `-0.067 +/- 0.277`.
7. E097 tiny causal-LM tuning sanity: one-hot probe selection beats default
   fixed LR by `+0.298 +/- 0.011`, beats shuffled probes by
   `+0.177 +/- 0.015`, and ties validation-selected external HPO
   (`-0.000 +/- 0.000`).  E096 soft-weighted LM aggregation is the boundary:
   it beats default and shuffled but trails external HPO by `-0.096 +/- 0.015`.
8. E092 longer-round stress run remains live but does not clear the
   external-HPO CI: internal minus stale/default `+2.838 +/- 1.979`, internal
   minus shuffled `+0.771 +/- 0.632`, internal minus external
   validation-selected HPO `+0.330 +/- 2.151`.
9. E080/E081 remain the replicated outer-control evidence:
   E080 selector minus shuffled `+0.177 +/- 0.049`; E081 packages the rule with
   frozen large-K fallback.
10. E089 is the fixed-compute grouped-LR boundary:
   internal minus external validation-selected HPO `-0.006 +/- 0.003`, while
   internal minus shuffled is `+0.004 +/- 0.002`.
11. E090 is the stricter stale-LR final-trajectory boundary:
   guarded minus stale full-worker `-0.008 +/- 0.002`, but guarded minus
   shuffled `+6.117 +/- 1.841`.
12. E087 is stochastic filtering under exchangeable fake labels, not HPO.

## TMLR Tables

Table 1: IID null/filtering, from E087.

Table 2: designed worker LR arms, from E063/E064.

Table 3: server-side deliberately diverse outer-control candidates, from
E080/E081.

Table 4: stale-default guarded recovery, from E085, with E090 as the limitation.

Table 5: fixed-compute external HPO:
E089 grouped-LR boundary, E090 stale-LR boundary, E090B outer-control win, E091
fresh-seed replication, E093 64-seed replication, E092 longer-round stress
run, E094 longer-round 64-seed win, and E095 large-K support row.

Table 6: lightweight language-model tuning sanity:
E096 soft probe-weighted tiny-LM boundary and E097 one-hot probe-selection
tiny-LM tie against validation-selected external HPO.

The table order should make the paper read as one claim: first the
exchangeability boundary, then deliberately diverse worker arms, then
deliberately diverse server-side candidates, then fixed-compute comparisons.

## Limitations

- E090B is an E080/E081-style local-SGD update-map ledger; the paper should
  present it as the clean fixed-compute demonstration of the mechanism and be
  explicit about what trajectory length it covers.
- E092 shows the same mechanism remains live for a longer `rounds=16` ledger,
  but the validation-selected external-HPO margin is not CI-cleared at this
  budget.
- E094 resolves that longer-round variance at 64 seeds and should supersede
  E092 in the main table if space is tight.
- E095 extends the positive fixed-compute result to `K=16`; margins are smaller
  and oracle gap remains, but this directly addresses low-K-only concerns.
- E096/E097 are tiny causal-LM tuning audits, not full-scale LLM pretraining
  runs.  They can support task-family plausibility and probe-label content, but
  should not be sold as the main empirical claim.
- E089 shows grouped-LR internal content but not a win over validation-selected
  external HPO.
- E090 shows the stale-default guard does not clear the stricter full-worker
  stale/default final-trajectory comparator.
- E087 prevents claiming that exchangeable IID worker labels create persistent
  HPO information.
- The current package is one local-SGD classifier/update-map bed; broader task
  families are future work.

## Submission Checklist

- Positive fixed-compute HPO result: yes, E090B, now replicated by E091 and
  E093.
- Longer-round fixed-compute HPO result included: yes, E094, with E092 as its
  noisy 32-seed precursor.
- Large-K fixed-compute HPO result included: yes, E095.
- Lightweight LM tuning result included: yes, E096/E097.
- Boundary fixed-compute results included: yes, E089 and E090.
- Null/filtering result included: yes, E087.
- Unified claim: yes, deliberately diverse aggregation via HPO-aware probes.
- Anonymous artifact link included: yes,
  `https://anonymous.4open.science/r/local_gp-9F83/`.
- Durable docs updated: research control board, option map, parallel probe
  program, and paper draft.
- Relevant artifacts ready for commit: E090, E090B, E091, E092, E093, E094,
  E095, E096/E097, docs, and this packet.
