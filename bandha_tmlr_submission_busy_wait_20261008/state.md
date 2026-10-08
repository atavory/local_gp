# State

Initialized 2026-10-08.

Starting local_gp HEAD: `985e0e0`.

Current readiness:

- TMLR is the right target venue.
- E090 fixed-compute stale-LR recovery was run and did not clear the stricter
  final-trajectory comparator: guarded minus stale full-worker was
  `-0.008 +/- 0.002`, guarded minus external validation-HPO was
  `-0.007 +/- 0.003`, while guarded minus shuffled was
  `+6.117 +/- 1.841`.
- E090B fixed-compute outer-control HPO was run as the next strongest
  comparator and returned `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`.
  Pooled internal minus external validation-selected HPO was
  `+1.354 +/- 0.463`; internal minus shuffled was `+0.161 +/- 0.061`;
  internal minus stale/default was `+3.789 +/- 0.515`.
- The paper draft, durable docs, and TMLR submission packet now reflect the
  narrowed claim: E090B is the positive fixed-compute HPO result, E089/E090 are
  boundaries, and E087 is stochastic filtering under exchangeable fake labels.

Hard completion requirements:

- E090 or equivalent fixed-compute HPO result with positive/live/win decision.
- TMLR submission packet and updated paper draft.
- Updated durable docs.
- Commit and push to `origin/main`.

Remaining work:

- Commit the intentionally changed files.
- Push to `origin/main` through Mr. Gitties.
- Write `TMLR_SUBMISSION_BUSY_DONE.md` only after push and wrapper validation.

Workspace caveat:

- The repo contains many older dirty/untracked files.  Stage only files created
  or intentionally edited for this loop.
