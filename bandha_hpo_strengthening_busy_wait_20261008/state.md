# State

Initialized 2026-10-08.

Starting local_gp HEAD: `9ccc4be`.

Observed loop start HEAD: `4b8d632` (`origin/main`), which already contained
the busy-wait scaffold.

Current evidence state:

- E087 and E088 have landed and were pushed in commit `9ccc4be`.
- E087 supports the nuanced exchangeability null: identical fake arms can still
  produce useful stochastic-update filtering from finite-sample noise.
- E088 ultra did not clear the fixed-compute HPO baseline and should not be the
  last word.
- E089 now supplies a serious fixed-compute table beyond the E088 smoke:
  `experiments/dfw_e089_fixed_compute_parallel_hpo_headroom.py` ran through
  Buck/PyTorch remote-only as `TORCH_CPU` on `devvm44191`, using the E063
  grouped-LR spread-16 regime (`K={8,16,32}`, 36 rounds, 8 held-out seeds,
  arms `(0.25,1.0,4.0)`, counts `(2,4,2)`).  Decision:
  `INTERNAL_HPO_CONTENT_LIVE_EXTERNAL_HPO_TIE`.
- Main E089 pooled result: internal minus external validation-selected HPO
  `-0.006 +/- 0.003`; internal minus external oracle `-0.007 +/- 0.003`;
  internal minus shuffled internal `+0.004 +/- 0.002`; internal minus
  parallel-arm average `+0.001 +/- 0.001`.
- A supplemental pilot-pair branch ran under
  `bandha_hpo_strengthening_busy_wait_20261008/runs/e089_pilot_pair/` and
  tightened but did not clear the external gap:
  internal minus external validation-selected HPO `-0.004 +/- 0.003`.

Current hard requirement:

- Commit and push the E089 script/artifacts and durable doc updates.
- Write the done sentinel only after the commit is pushed and validation
  passes.

Known workspace caveat:

- The repo has many older dirty/untracked files.  Stage only files created or
  intentionally edited for this loop.
