# Agenda

Execute, in this order unless a result logically blocks the next item:

1. DONE E082: LR skew summary diagnostic over E060/E061.
   Result: no summary-level safe/default collapse; exact LR arm mass still
   needs a trace before free LR search becomes live.
2. DONE E083: batch-size/value-per-FLOP internal-HPO experiment.
   Result: `BATCH_VALUE_PER_FLOP_NEGATIVE`; oracle headroom exists, but the
   simple soft value/FLOP selector does not beat shuffled labels.
3. DONE E084: run the selected-arm trajectory diagnostic for the E080/E081 outer
   `(alpha,beta)` controller.  Determine whether the frozen grid pins to edges,
   moves across training phases, or simply sits in the middle.
   Result: `OUTER_TRAJECTORY_STABLE_SOFT_INTERIOR`; the frozen scaled/soft_std2
   controller sits in a high-entropy interior mixture with negligible phase
   movement, so moving-grid/GP machinery is not triggered.
4. DONE E085: run a bad/stale-default HP-arm test using the internal-HPO framing.
   This must test the actual customer case: a default that is stale or wrong,
   with probes allowed to abandon it under a conservative rule.
   Result: `STALE_DEFAULT_GUARDED_PROBE_LIVE`; a guarded abandon-or-fallback
   rule recovers value from a stale low-LR default while keeping the stale arm
   in about 80% of rounds.
5. DONE update `bandha_internal_hpo_option_map_20261006.md`,
   `bandha_parallel_hparam_probe_program_20261005.md`, and
   `research_control_board.md` after E084/E085.
6. NEXT commit and push the relevant files.
7. NEXT write `INTERNAL_HPO_BUSY_DONE.md` only after all required artifacts and docs
   exist and the push is complete.

Do not spend a turn only explaining.  Run or write the next artifact.
