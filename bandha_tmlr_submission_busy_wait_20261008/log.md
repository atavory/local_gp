# Log

## initialization - 2026-10-08

- Created a guarded busy-wait loop for TMLR submission readiness.
- The main experiment target is E090 fixed-compute stale-default recovery.

## turn 1 progress - 2026-10-08

- Read agent context, project context/state/agenda/log, existing E085/E088/E089
  artifacts, and the E080/E081 outer-control evidence.
- Added and ran `experiments/dfw_e090_fixed_compute_stale_default_recovery.py`
  through `buck2 run --remote-only fbcode//aai/double_fan_gpu_runner:run_local_gp_script_static`.
  Main result: `FIXED_COMPUTE_STALE_DEFAULT_NOT_CLEARED` on `TORCH_CPU`;
  guarded minus stale full-worker `-0.008 +/- 0.002`; guarded minus shuffled
  `+6.117 +/- 1.841`; guarded minus external validation-HPO
  `-0.007 +/- 0.003`.
- Added and ran `experiments/dfw_e090b_fixed_compute_outer_control_hpo.py` as
  the next strongest fixed-compute comparator. Main result:
  `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN` on `TORCH_CPU`; internal minus
  external validation-HPO `+1.354 +/- 0.463`; internal minus shuffled
  `+0.161 +/- 0.061`; internal minus stale/default `+3.789 +/- 0.515`.
- Updated durable docs and paper materials:
  `research_control_board.md`, `bandha_internal_hpo_option_map_20261006.md`,
  `bandha_parallel_hparam_probe_program_20261005.md`,
  `bandha_hpo_aware_aggregation_paper_draft_20261007.md`, and created
  `bandha_tmlr_submission_packet_20261008.md`.
- Next recommended action: stage only the E090/E090B artifacts, docs, paper
  packet, and busy-wait state files; commit; push through Mr. Gitties; validate
  the wrapper; then write the done sentinel.
