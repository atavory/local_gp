# Agenda

Run this loop until the fixed-compute HPO story is materially stronger than it
was after E088.

1. E089: build and run a stronger fixed-compute parallel-HPO comparison.
   - Start from `experiments/dfw_e088_fixed_compute_parallel_hpo_baseline.py`
     but move the setting toward the E063 grouped-LR headroom regime.
   - Keep same total worker budget for internal HPO and external HPO.
   - External HPO: split workers into separate LR-arm trajectories, select by
     validation/probe loss, report oracle only as diagnostic.
   - Internal HPO: use the same workers as in-round LR arms, probe candidate
     next states, and weighted-average/recombine into one trajectory.
   - Include shuffled-label, parallel-arm-average, default/full-worker, and
     oracle diagnostics.
   - Run enough seeds/rounds that the result is not just the E088 ultra smoke.
2. If E089 is still weak or ambiguous, run the next most useful HPO win:
   stale-default fixed-compute recovery in the E085 regime.
3. Update:
   - `research_control_board.md`
   - `bandha_internal_hpo_option_map_20261006.md`
   - `bandha_parallel_hparam_probe_program_20261005.md`
   - `bandha_hpo_aware_aggregation_paper_draft_20261007.md`
4. Commit and push only relevant files.  Leave unrelated dirty/untracked files
   alone.
5. Write `HPO_STRENGTHENING_BUSY_DONE.md` with `pushed=yes` and the current
   short HEAD only after all completion gates validate.

Do not end a turn with only a plan.  Run the next experiment, write the next
artifact, or update the docs.
