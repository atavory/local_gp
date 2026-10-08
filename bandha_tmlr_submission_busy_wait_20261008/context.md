# Context

Initialized 2026-10-08 by bandha.

Mission: move the HPO-aware aggregation project from "draft-ready" to a
defensible TMLR submission package.

The paper claim must stay honest:

- We are not selling generic better averaging.
- We are not claiming that IID exchangeable workers create persistent HPO
  information.  E087 shows stochastic-update filtering under finite
  probe/update noise; keep that as a boundary/null nuance.
- We are not claiming E089 beats validation-selected external HPO.  E089
  shows internal label content versus shuffled labels, but external HPO remains
  a binding comparator.
- The strongest current submission path is practical HPO-aware aggregation:
  small nonexchangeable optimizer arms, cheap probes, conservative weighting or
  guards, and reinvestment into one shared trajectory.

Current evidence:

- E063 grouped LR arms: spread 16 gives repeatable probe label-content wins at
  low cost.
- E080/E081 outer-control HPO: strongest low-dimensional server-side HPO win.
  E080 pooled K<=8 selector minus shuffled `+0.177 +/- 0.049`, selector minus
  fixed `+0.116 +/- 0.052`, selector minus schedule `+0.164 +/- 0.057`.
- E085 stale-default recovery: most legible practical HPO win.  Guarded minus
  stale is positive for K=8/16/32 and fallback keeps the stale default in about
  80% of rounds.
- E087 IID fake-arm audit: `IID_STOCHASTIC_FILTERING_POSITIVE`, useful for the
  null/boundary section.
- E089 fixed-compute grouped-LR baseline:
  `INTERNAL_HPO_CONTENT_LIVE_EXTERNAL_HPO_TIE`.  Pooled internal minus
  external validation-selected HPO `-0.006 +/- 0.003`; internal minus shuffled
  `+0.004 +/- 0.002`.  This is content, not a fixed-compute external-HPO win.

Primary missing item:

- E090: fixed-compute stale-default recovery.  Compare a real user stale
  default against internal guarded recovery, naive external HPO with the same
  total worker budget, shuffled controls, fresh/default controls, and oracle
  diagnostics.  This should give a stronger, more practical HPO table than E089.

Completion standard:

- At least one new fixed-compute HPO result beyond E089, preferably E090, must
  land with a positive/live/win decision and valid script/result/run/legitimacy
  artifacts.
- The paper draft must be rewritten into a TMLR submission packet with narrowed
  claims, tables, limitations, and a residual-risk checklist.
- Durable docs must be updated.
- Relevant files must be committed and pushed to `origin/main` through
  Mr. Gitties.
- Write `TMLR_SUBMISSION_BUSY_DONE.md` only after validation passes.
