# Context

Initialized 2026-10-08 by bandha.

The current paper claim is HPO-aware aggregation in distributed optimization:
workers or server-side candidates act as hyperparameter arms, probes score the
candidate updates, and the server folds the useful arms back into one shared
trajectory.  This is not ordinary parallel HPO, because ordinary parallel HPO
splits the worker budget into separate trajectories and discards losing arms.

The user wants stronger HPO wins.  Do not spend turns only explaining the next
step.  Run experiments, update docs, and push.

Important framing:

- IID identical arms are exchangeable as labels, but finite probe/update noise
  can still move weights and can produce stochastic-update filtering.  E087 is
  evidence for that nuance; it is not the HPO win.
- The paper needs fixed-compute HPO evidence.  E088 ultra was a tiny smoke in a
  weak regime and did not clear the baseline.
- Main next target: E089, a serious fixed-compute HPO baseline in the E063
  grouped-LR headroom regime.
- Prefer real GPU/Buck execution when available, but do not call the project
  blocked because one route is slow or because raw Python on a worker lacks
  torch.  TORCH_CPU is acceptable for algorithmic local runs if the artifact
  labels it honestly and the experiment is still informative.

Key existing evidence to preserve:

- E063 grouped LR arms:
  - spread 16 gives positive HPO content at K=8/16/32.
  - K=8 raw-fixed `+0.251 +/-0.153`, raw-shuffled `+0.447 +/-0.207`.
  - K=16 raw-fixed `+0.162 +/-0.439`, raw-shuffled `+0.439`.
  - K=32 raw-fixed `+0.271`, raw-shuffled `+0.824`.
  - overhead is low, roughly sub-percent to low-single-digit.
- E080/E081:
  - low-K outer controls are the clean server-side HPO story.
  - E080 K<=8 selector-shuffled `+0.177 +/-0.049`;
    selector-fixed `+0.116 +/-0.052`;
    selector-schedule `+0.164 +/-0.057`.
- E085:
  - stale-default guarded LR recovery is practical HPO.
  - guarded minus stale: K=8 `+0.997 +/-0.739`, K=16 `+0.395 +/-0.654`,
    K=32 `+0.727 +/-1.114`; fallback remains around 80%.
- E087:
  - `IID_STOCHASTIC_FILTERING_POSITIVE` on Buck-declared PyTorch CPU.
  - selector-uniform `+0.621 +/-0.320`;
    selector-shuffled `+0.476 +/-0.222`;
    weight L1 `0.137`; entropy `0.991`; probe/target Spearman `0.406`.
- E088 ultra:
  - `FIXED_COMPUTE_BASELINE_NOT_CLEARED`.
  - internal-external validation HPO `-0.006 +/-0.002`;
    internal-shuffled `+0.003 +/-0.003`;
    internal-parallel-average `0.000 +/-0.001`.

Completion gate:

- Produce at least one valid E089 result beyond the E088 ultra smoke:
  script, run log, result markdown, and legitimacy check.
- E089 must be a fixed-compute comparison against naive parallel HPO in a
  regime with real HPO headroom, preferably E063 grouped-LR spread 16.
- Update the durable docs and paper draft with the new result.
- Commit relevant files and push to `origin/main` through Mr. Gitties.
- Write `HPO_STRENGTHENING_BUSY_DONE.md` only after the gate validates.
