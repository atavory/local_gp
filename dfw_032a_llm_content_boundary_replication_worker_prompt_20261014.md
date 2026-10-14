# DFW-032A Worker Prompt - LLM Content Boundary Replication

## Mission

Replicate DFW-031 with more seeds using the accelerated source bundle.  This is
a CUDA/Buck run.  Do not use `pip`.  Do not fall back to CPU.  Do not require
`~/.agent_context.*` or Mr. Gitties.

Reply `VALID` only if all requested cells complete on CUDA.  If build,
preflight, or a cell fails, upload logs and reply `INVALID` with the exact
failed command.

## Fixed Context

- Subject: `atavory__devvm26869`
- Run root:
  `manifold://aai_research_tlv/tree/kalman_ddps/runs/atavory__devvm26869/llm_content_boundary_replication_20261014/`
- Reply root:
  `manifold://aai_research_tlv/tree/double_fan/messages/from_workers/atavory__devvm26869/`
- Source tarball:
  `manifold://aai_research_tlv/tree/kalman_ddps/shared/source/it_geom_data_layerprobe_buck_accel_20261014.tgz`
- Expected source SHA256:
  `30c5c5baa13a3136772b500c4847fb7bd64b79ecc785f2633d418f8e67f3ed03`

Install the source so this exists:

`$FBROOT/it_geom_data/code/layerprobe_buck`

Use `/data/repos/fbsource` if present, else `/data/users/$USER/fbsource`.
Create empty package markers in `it_geom_data/`, `it_geom_data/code/`, and
`it_geom_data/code/layerprobe_buck/`.

Use the successful DFW-031 Buck style:

`buck2 --isolation-dir kalman_llm_20261008 run @fbcode//mode/dev-nosan TARGET -- ARGS`

Resolve `fbcode//it_geom_data/code/layerprobe_buck:{gpu_preflight,llm_disagreement}`
first, then `fbsource//...` if needed.

## Experiment Grid

Common settings:

- conditions: `domain,iid,style`
- optimizer: `sgd`
- seeds: `2026101401,2026101402,2026101403,2026101404,2026101405`
- nodes: `8`
- seq len: `32`
- width/layers/heads: `96/2/4`
- samples per worker: `512`
- probe/value/test samples: `256/256/512`
- batch size: `64`
- local steps: `120`
- null permutations: `16`
- lr: `0.02`
- value epsilon: `0.01`

Configs:

1. `v128_c120`: `vocab_size=128`, `content_vocab=120`
2. `v256_c240`: `vocab_size=256`, `content_vocab=240`
3. `v256_c128`: `vocab_size=256`, `content_vocab=128`

Expected cells: `3 configs * 3 conditions * 5 seeds = 45`.

## Required Artifacts

Upload the complete local output dir to the run root above, including:

- `COMMAND_LOG.md`
- `GPU_PREFLIGHT.log`
- per-config CSV/JSON/logs
- `LLM_CONTENT_BOUNDARY_REPLICATION_SUMMARY.md`
- `RESULT_VALIDATION.json`
- `RUN_COMMANDS.sh`
- `LEGITIMACY_CHECK.md`
- `WORKER_REPLY.md`

The summary must include per config and condition:

- row count;
- mean `test_nll_gain_vs_average`;
- mean `split_nll_gain_vs_average`;
- mean `content_test_nll_gain_vs_average`;
- mean `content_split_nll_gain_vs_average`;
- split-selection rates for full and content gates;
- mean `null_margin`;
- mean `Ghat`.

Decision labels:

- `CONTENT_BOUNDARY_REPLICATION_POSITIVE`: domain gains positive in all
  configs, IID selected gains zero or near zero, and style full-token gains
  exceed content-only gains.
- `CONTENT_BOUNDARY_REPLICATION_MIXED`: partial pattern.
- `CONTENT_BOUNDARY_REPLICATION_NOT_EVALUATED`: preflight/build/cell failure.

Worker reply must start with `VALID` or `INVALID`, include the decision label,
completed cell count, and run path.
