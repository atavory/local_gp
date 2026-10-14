# DFW-032B Worker Prompt - LLM Content Boundary Vocab-512 Stress

## Mission

Run the vocab-scale follow-up to DFW-031 using the accelerated source bundle.
This tests whether the content/readout boundary survives at a larger output
vocabulary.  This is a CUDA/Buck run.  Do not use `pip`.  Do not fall back to
CPU.  Do not require `~/.agent_context.*` or Mr. Gitties.

Reply `VALID` only if all requested cells complete on CUDA.  If build,
preflight, or a cell fails, upload logs and reply `INVALID` with the exact
failed command.

## Fixed Context

- Subject: `atavory__devvm68196`
- Run root:
  `manifold://aai_research_tlv/tree/kalman_ddps/runs/atavory__devvm68196/llm_content_boundary_vocab512_20261014/`
- Reply root:
  `manifold://aai_research_tlv/tree/double_fan/messages/from_workers/atavory__devvm68196/`
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
- seeds: `2026101411,2026101412,2026101413`
- nodes: `8`
- seq len: `32`
- width/layers/heads: `96/2/4`
- samples per worker: `512`
- probe/value/test samples: `128/256/512`
- batch size: `64`
- local steps: `120`
- null permutations: `16`
- lr: `0.02`
- value epsilon: `0.01`

Configs:

1. `v512_c500`: `vocab_size=512`, `content_vocab=500`
2. `v512_c256`: `vocab_size=512`, `content_vocab=256`

Expected cells: `2 configs * 3 conditions * 3 seeds = 18`.

## Required Artifacts

Upload the complete local output dir to the run root above, including:

- `COMMAND_LOG.md`
- `GPU_PREFLIGHT.log`
- per-config CSV/JSON/logs
- `LLM_CONTENT_BOUNDARY_VOCAB512_SUMMARY.md`
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

- `CONTENT_BOUNDARY_VOCAB512_POSITIVE`: domain gains positive, IID selected
  gains zero or near zero, and style full-token gains exceed content-only
  gains.
- `CONTENT_BOUNDARY_VOCAB512_MIXED`: partial pattern.
- `CONTENT_BOUNDARY_VOCAB512_NOT_EVALUATED`: preflight/build/cell failure.

Worker reply must start with `VALID` or `INVALID`, include the decision label,
completed cell count, and run path.
