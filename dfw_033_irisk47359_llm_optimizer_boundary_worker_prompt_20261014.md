# DFW-033 Worker Prompt - LLM Optimizer Boundary

Run a calibrated optimizer follow-up for the DFW-031 token-boundary result.
This is a CUDA/Buck run.  Do not use `pip`.  Do not fall back to CPU.  Do not
require `~/.agent_context.*` or Mr. Gitties.

Subject: `irisk__devvm47359`

Run root:

`manifold://aai_research_tlv/tree/kalman_ddps/runs/irisk__devvm47359/llm_optimizer_boundary_20261014/`

Reply root:

`manifold://aai_research_tlv/tree/double_fan/messages/from_workers/irisk__devvm47359/`

Source tarball:

`manifold://aai_research_tlv/tree/kalman_ddps/shared/source/it_geom_data_layerprobe_buck_accel_20261014.tgz`

Expected source SHA256:

`30c5c5baa13a3136772b500c4847fb7bd64b79ecc785f2633d418f8e67f3ed03`

Install the source under `/data/repos/fbsource` if present, otherwise
`/data/users/$USER/fbsource`, so that
`$FBROOT/it_geom_data/code/layerprobe_buck` exists.  Create empty package
markers in `it_geom_data/`, `it_geom_data/code/`, and
`it_geom_data/code/layerprobe_buck/`.

Use:

`buck2 --isolation-dir kalman_llm_20261008 run @fbcode//mode/dev-nosan TARGET -- ARGS`

Resolve `fbcode//it_geom_data/code/layerprobe_buck:{gpu_preflight,llm_disagreement}`
first, then `fbsource//...` if needed.

## Experiment Grid

Run two separate calibrated optimizer sweeps, because the optimizer learning
rates differ.

Common settings:

- conditions: `domain,iid,style`
- seeds: `2026101431,2026101432,2026101433`
- nodes: `8`
- seq len: `32`
- vocab/content: `256/240`
- width/layers/heads: `96/2/4`
- samples per worker: `512`
- probe/value/test samples: `256/256/512`
- batch size: `64`
- local steps: `120`
- null permutations: `16`
- value epsilon: `0.01`

Optimizer cells:

1. `sgd_lr002`: `--optimizers sgd --lr 0.02`
2. `adamw_lr003`: `--optimizers adamw --lr 0.003`

Expected cells: `2 optimizers * 3 conditions * 3 seeds = 18`.

## Required Artifacts

Upload the complete local output dir to the run root.  Include:

- `COMMAND_LOG.md`
- `GPU_PREFLIGHT.log`
- per-optimizer CSV/JSON/logs
- `LLM_OPTIMIZER_BOUNDARY_SUMMARY.md`
- `RESULT_VALIDATION.json`
- `RUN_COMMANDS.sh`
- `LEGITIMACY_CHECK.md`
- `WORKER_REPLY.md`

The summary must include per optimizer and condition:

- row count;
- mean `test_nll_gain_vs_average`;
- mean `split_nll_gain_vs_average`;
- mean `content_test_nll_gain_vs_average`;
- mean `content_split_nll_gain_vs_average`;
- full/content split-selection rates;
- mean `null_margin`;
- mean `Ghat`.

Decision labels:

- `TOKEN_OPTIMIZER_BOUNDARY_POSITIVE`: calibrated AdamW preserves the same
  qualitative boundary as SGD: positive domain gains, IID abstention, and style
  full-token gains larger than content-only gains.
- `TOKEN_OPTIMIZER_BOUNDARY_MIXED`: only part of the pattern holds.
- `TOKEN_OPTIMIZER_BOUNDARY_NOT_EVALUATED`: preflight/build/cell failure.

Worker reply must start with `VALID` or `INVALID`, include the decision label,
completed cell count, and run path.
