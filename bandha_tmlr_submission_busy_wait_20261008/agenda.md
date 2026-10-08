# Agenda

1. Build and run E090 fixed-compute stale-default recovery.
   - Use E085 as the practical-HPO core and E088/E089 as the fixed-compute
     comparator template.
   - Internal: one shared trajectory, worker groups are stale/fresh/high LR
     arms, guarded probes decide when to abandon stale default.
   - External: same total workers split into independent LR-arm trajectories,
     validation-selected HPO chooses one, oracle is diagnostic only.
   - Include stale-only, fresh-only, default/full-worker, shuffled/label-control,
     external validation HPO, external oracle, internal guarded, probe cost, and
     legitimacy check.
   - Run enough seeds/rounds that this is not a smoke.  A reduced smoke can be
     used only to debug; do not complete on it.
2. If E090 does not give a usable positive table, run the next strongest option:
   fixed-compute outer-control HPO based on E080/E081.
3. Rewrite the paper materials for TMLR:
   - create `bandha_tmlr_submission_packet_20261008.md`;
   - update `bandha_hpo_aware_aggregation_paper_draft_20261007.md`;
   - include contribution claims, table plan, limitations, and what E089 does
     and does not say.
4. Update durable docs:
   - `research_control_board.md`
   - `bandha_internal_hpo_option_map_20261006.md`
   - `bandha_parallel_hparam_probe_program_20261005.md`
5. Commit and push only relevant files.
6. Write `TMLR_SUBMISSION_BUSY_DONE.md` with `pushed=yes` and current short HEAD
   only after the wrapper validates the result, docs, paper packet, and push.

Do not end a turn with only a plan.  Run an experiment, write an artifact, or
update the paper.
