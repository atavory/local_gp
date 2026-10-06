# Log

Initialized by bandha at 2026-10-06.

## turn 1 - 2026-10-06

- Read `~/.agent_context.md`, `~/.agent_context.json`, and the busy-wait
  `context.md`/`state.md`/`agenda.md`/`log.md`.
- Ran worker scan:
  `/tmp/project_worker/project_workerctl.py scan`.
  Preferred lanes `assafcohen__devvm23211` and `talsarig__devvm40229` were
  alive; direct `atavory__devvm52518` was stale in the table.
- Checked local compute diagnostic on `devvm44191`; raw
  `/usr/local/fbcode/platform010/bin/python3.12` had no `torch`.  This was
  recorded only as local routing information, not as a GPU verdict.
- Added and ran
  `experiments/dfw_e082_lr_skew_summary_diagnostic.py`.
  Result: `LR_SKEW_SUMMARY_NO_SAFE_COLLAPSE_TRACE_NEEDED`.
  Key summary: 10 nontrivial E060/E061 cells, pilot-best minus fixed
  `+0.245 +/- 0.137`, pilot-best minus shuffled `+0.447 +/- 0.231`.
- Added `experiments/dfw_e083_gpu_batch_value_per_flop_hpo.py` and
  compile-checked it with `python3.12 -m py_compile`.
- Used Mr. Gitties SSH/SCP to run E083 on the known direct GPU lane
  `devvm52518.cln0.facebook.com`; copied the source tarball to `/tmp`, ran
  the experiment with CUDA device 0, and copied back the result/run/legitimacy
  artifacts.
  Result: `BATCH_VALUE_PER_FLOP_NEGATIVE` with `REAL_GPU` PyTorch CUDA
  (`torch=2.14.1+cu130`, `NVIDIA PG509-210`).
  Key shuffled contrasts: K=8 `+0.054 +/- 0.112`, K=16
  `+0.287 +/- 0.405`.
- Updated durable docs:
  `bandha_internal_hpo_option_map_20261006.md`,
  `bandha_parallel_hparam_probe_program_20261005.md`, and
  `research_control_board.md`.
- Files changed/created include E082/E083 scripts, result markdowns, run logs,
  legitimacy checks, the three durable docs, and this busy-wait state.
- Next recommended action: implement and run E084 selected-arm trajectory for
  the E080/E081 `scaled/soft_std2` low-K `(alpha,beta)` controller, preferably
  on `devvm52518.cln0.facebook.com` through Mr. Gitties if still reachable;
  then run E085 stale/default HP-arm follow-up.

## guarded internal-HPO worker - 2026-10-06T10:18:25.927142+00:00

- started: `2026-10-06T10:05:10.562380+00:00`
- inner_returncode: `0`
- inner_timed_out: `False`
- sentinel_status: `continue`
- sentinel_reason: `done sentinel missing or first line is wrong`
- inner_stdout: `/data/users/atavory/local_gp/bandha_internal_hpo_busy_wait_20261006/last_inner_stdout.txt`
- inner_stderr: `/data/users/atavory/local_gp/bandha_internal_hpo_busy_wait_20261006/last_inner_stderr.txt`


## busy_wait turn 1 - 20261006_031825

- status: continue
- returncode: 0
- timed_out: False
- duration_sec: 795.443
- run_dir: /data/users/atavory/local_gp/bandha_internal_hpo_busy_wait_20261006/runs/20261006_030510_turn_0001

## turn 2 - 2026-10-06

- Read `~/.agent_context.md`, `~/.agent_context.json`, and the busy-wait
  `context.md`/`state.md`/`agenda.md`/`log.md`.
- Added and compile-checked
  `experiments/dfw_e084_gpu_outer_control_trajectory_diagnostic.py` with
  `/usr/local/fbcode/platform010/bin/python3.12 -m py_compile`.
- Used Mr. Gitties SSH/SCP to run E084 on
  `devvm52518.cln0.facebook.com`; copied back result/run/legitimacy artifacts.
  Result: `OUTER_TRAJECTORY_STABLE_SOFT_INTERIOR` with `REAL_GPU`
  (`torch=2.14.1+cu130`, `NVIDIA PG509-210`).  Effective alpha was stable near
  `19.7-19.9` for K=4 and `9.95-10.03` for K=8, beta was stable near `0.116`,
  normalized phase spans were `<=0.008`, and entropy was `~0.976`.
- Added and compile-checked
  `experiments/dfw_e085_gpu_stale_default_guarded_hp_arm.py`.
- Used Mr. Gitties SSH/SCP to run E085 on
  `devvm52518.cln0.facebook.com`; copied back result/run/legitimacy artifacts.
  Result: `STALE_DEFAULT_GUARDED_PROBE_LIVE` with `REAL_GPU`.  The guarded
  stale-default rule improved over stale-only by `+0.997+/-0.739` at K=8,
  `+0.395+/-0.654` at K=16, and `+0.727+/-1.114` at K=32, with an abandon rate
  near `0.20`.
- Updated durable docs:
  `bandha_internal_hpo_option_map_20261006.md`,
  `bandha_parallel_hparam_probe_program_20261005.md`, and
  `research_control_board.md`.
- Updated busy-wait `state.md` and `agenda.md`.
- Files changed/created include E084/E085 scripts, result markdowns, run logs,
  legitimacy checks, the three durable docs, and this busy-wait state.
- Committed and pushed `f685489` (`Complete internal HPO busy-wait gates`) to
  `origin/main` through Mr. Gitties.
- Added validation alias symlinks required by
  `bandha_internal_hpo_busy_wait_agent_20261006.py`, then committed and pushed
  `0b23156` (`Add internal HPO validation aliases`) through Mr. Gitties.
- Wrote `INTERNAL_HPO_BUSY_DONE.md` and added `pushed=yes` plus current HEAD.
- Ran wrapper validation:
  `/usr/local/fbcode/platform010/bin/python3.12 bandha_internal_hpo_busy_wait_agent_20261006.py bandha_internal_hpo_busy_wait_20261006/runs/20261006_031825_turn_0002/prompt.md`.
  Result: `BUSY_WAIT_STATUS: complete` at HEAD `0b23156`.
- Next recommended action: none for this busy-wait objective; the done sentinel
  validates.
