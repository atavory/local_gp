# State

Initialized 2026-10-08.

Starting local_gp HEAD: `9ccc4be`.

Current evidence state:

- E087 and E088 have landed and were pushed in commit `9ccc4be`.
- E087 supports the nuanced exchangeability null: identical fake arms can still
  produce useful stochastic-update filtering from finite-sample noise.
- E088 ultra did not clear the fixed-compute HPO baseline and should not be the
  last word.

Current hard requirement:

- Run E089, a meaningful fixed-compute HPO comparison beyond E088 ultra.
- Update docs and paper draft with the result.
- Commit and push.

Known workspace caveat:

- The repo has many older dirty/untracked files.  Stage only files created or
  intentionally edited for this loop.
