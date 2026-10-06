# State

Initialized 2026-10-06.

Turn 1 progress, 2026-10-06:

- E082 LR skew summary diagnostic completed:
  `LR_SKEW_SUMMARY_NO_SAFE_COLLAPSE_TRACE_NEEDED`.
  It reanalyzed durable E060/E061 `REAL_GPU` summaries and found no
  summary-level safe/default collapse, but exact LR arm mass is still missing.
- E083 batch-size/value-per-FLOP arm completed on direct `devvm52518` GPU:
  `BATCH_VALUE_PER_FLOP_NEGATIVE`.  Batch arms have oracle headroom, but the
  simple default-anchored value/FLOP selector does not beat shuffled labels
  cleanly.
- Updated `bandha_internal_hpo_option_map_20261006.md`,
  `bandha_parallel_hparam_probe_program_20261005.md`, and
  `research_control_board.md` with E082/E083 outcomes.
- Local host `devvm44191` has no raw torch import; this was not treated as a
  GPU verdict.  Direct `devvm52518.cln0.facebook.com` was reachable through
  Mr. Gitties and produced the E083 `REAL_GPU` artifact.

Turn 2 progress, 2026-10-06:

- E084 selected-arm trajectory diagnostic completed on direct `devvm52518`
  GPU: `OUTER_TRAJECTORY_STABLE_SOFT_INTERIOR`.  The E080/E081
  `scaled/soft_std2` low-K controller emits a stable, high-entropy interior
  soft pair rather than pinning to grid edges or moving across phases.
- E085 guarded stale-default HP-arm test completed on direct `devvm52518` GPU:
  `STALE_DEFAULT_GUARDED_PROBE_LIVE`.  A stale low-LR default
  (`lr_multiplier=0.25`) remains the fallback, but a 4-slice guarded
  abandonment rule leaves it in about 20% of rounds and improves over
  stale-only in the pooled decision.
- Updated `bandha_internal_hpo_option_map_20261006.md`,
  `bandha_parallel_hparam_probe_program_20261005.md`, and
  `research_control_board.md` with E084/E085 outcomes.

Required artifacts before completion:

- E082 LR skew diagnostic script/result/run note. DONE.
- E083 batch-size/value-per-FLOP arm script/result/run note. DONE.
- E084 selected-arm trajectory diagnostic script/result/run note. DONE.
- E085 bad/stale-default HP-arm script/result/run note. DONE.
- Updated `bandha_internal_hpo_option_map_20261006.md`. DONE for E082-E085.
- Updated `bandha_parallel_hparam_probe_program_20261005.md`. DONE for E082-E085.
- Updated `research_control_board.md`. DONE for E082-E085.
- Git commit and push marker in `INTERNAL_HPO_BUSY_DONE.md`.

Current hard positive to preserve:

```text
E080:
  selector - shuffled = +0.177 +/- 0.049
  selector - fixed    = +0.116 +/- 0.052
```

Current deployment shape:

```text
K<=8: probe-selected scaled/soft_std2 outer (alpha,beta)
K>8: frozen no-probe fallback, preferably theory-derived unless global-fixed
     safety is desired
```

Current additions:

```text
E084: no moving-grid trigger for outer controls; frozen grid sits interior.
E085: guarded stale-default recovery is live; soft default anchoring remains
      negative, but abandon-or-fallback passes.
```
