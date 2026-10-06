# Internal-HPO Busy Wait Context

Status: initialized 2026-10-06.

The active project is no longer "rank exchangeable workers."  The active
project is internal HPO inside aligned distributed rounds:

```text
all workers start from theta_t
worker groups run different optimizer/resource controls h_i
each group returns a candidate update
a small probe/validation set scores candidate value
the server uses a conservative/shrunk decision rule with explicit fallback
```

Current live evidence:

- E080 is the strongest label-content positive in the low-K outer-control
  regime:
  - selector - shuffled = +0.177 +/- 0.049
  - selector - fixed = +0.116 +/- 0.052
  - selector - schedule = +0.164 +/- 0.057
- E081 is the deployment-shaped audit:
  - K<=8 probe controller plus frozen large-K fallback beats schedule and ties
    implementable fixed/best-control ceilings.
  - theory fallback is the clean no-pilot candidate.
  - K=8 transferred fallback is bad and should be dropped.

The research question for this busy loop:

> Which hyperparameters/resource choices belong inside the distributed round,
> and can probes produce an actionable deployment rule rather than just a
> ceiling or post-hoc screen?

The current option map says the next gates are:

1. LR-arm skew check on E060/E061.
2. Batch-size/value-per-FLOP internal-HPO arm.
3. Selected-arm trajectory diagnostic for the outer `(alpha,beta)` controller.
4. Bad/stale-default HP-arm test.
5. Documentation update and push after each material result.

GPU protocol:

- Do not declare CUDA/PyTorch unusable from raw `nvidia-smi` or from bare raw
  `/usr/local/fbcode/platform010/bin/python3.12 import torch` on a worker.
- For generic worker machines, use a Buck2 GPU path:

```bash
buck2 run -c fbcode.enable_gpu_sections=true <target> -- <args>
```

  or a `gpu_python_binary(..., keep_gpu_sections = True)` target.
- If no Buck2/PyTorch GPU path exists, write an exact diagnostic and label it
  `NO_GPU_HARNESS`.  CPU-only results may be supplemental only and must be
  labeled `CPU_FALLBACK`.
- The previous raw-Python failure on `assafcohen__devvm23211` was not a valid
  GPU verdict.  It only said that raw Python there did not have `torch`.

Known capacity from the last worker table:

| lane | subject/host | GPUs | state | instruction |
|---|---|---:|---|---|
| known direct | `atavory__devvm52518` / `devvm52518` | 1 PG509 | direct PyTorch CUDA known-good from E078-E081 | use for direct real-GPU scripts if reachable |
| worker | `assafcohen__devvm23211` | 1 PG509 | fresh worker heartbeat; raw Python lacks torch | use Buck2 GPU harness check or worker assignment |
| worker | `talsarig__devvm40229` | 2 A100 | fresh worker heartbeat | use Buck2 GPU harness check or worker assignment |
| avoid unless asked | `yoelgri__*` | GPUs | user objected to using these | do not use by default |

Worker protocol:

- controller: `/tmp/project_worker/project_workerctl.py scan|assign|reap`
- project root for messages: `aai_research_tlv/tree/double_fan`
- use one outstanding assignment per worker.
- if Manifold scan hangs, keep local progress moving and record the exact
  diagnostic in `log.md`.

Network protocol:

- Use Mr. Gitties for external network operations, including git push and SSH.
- Do not put secrets in command strings.

Main files to keep updated:

- `bandha_internal_hpo_option_map_20261006.md`
- `bandha_parallel_hparam_probe_program_20261005.md`
- `research_control_board.md`

Completion is deliberately strict.  This busy loop is not done when one turn
finishes.  It is done only when the required gates have artifacts, docs are
updated, and a commit has been pushed.
