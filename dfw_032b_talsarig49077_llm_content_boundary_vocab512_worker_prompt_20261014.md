# DFW-032B Worker Prompt - LLM Content Boundary Vocab-512 Stress

Run the DFW-031 vocab-scale stress on CUDA/Buck.  Do not use `pip`.  Do not
fall back to CPU.  Do not require `~/.agent_context.*` or Mr. Gitties.

Subject: `talsarig__devvm49077`

Run root:

`manifold://aai_research_tlv/tree/kalman_ddps/runs/talsarig__devvm49077/llm_content_boundary_vocab512_20261014/`

Reply root:

`manifold://aai_research_tlv/tree/double_fan/messages/from_workers/talsarig__devvm49077/`

Source tarball:

`manifold://aai_research_tlv/tree/kalman_ddps/shared/source/it_geom_data_layerprobe_buck_accel_20261014.tgz`

Expected source SHA256:

`30c5c5baa13a3136772b500c4847fb7bd64b79ecc785f2633d418f8e67f3ed03`

Install the source under `/data/repos/fbsource` if present, otherwise
`/data/users/$USER/fbsource`, so that
`$FBROOT/it_geom_data/code/layerprobe_buck` exists.  Create empty package
markers in `it_geom_data/`, `it_geom_data/code/`, and
`it_geom_data/code/layerprobe_buck/`.

Use the DFW-031 Buck style:

`buck2 --isolation-dir kalman_llm_20261008 run @fbcode//mode/dev-nosan TARGET -- ARGS`

Resolve `fbcode//it_geom_data/code/layerprobe_buck:{gpu_preflight,llm_disagreement}`
first, then `fbsource//...` if needed.

Grid:

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

Expected cells: `18`.

Upload the complete local output dir to the run root.  Include:
`COMMAND_LOG.md`, `GPU_PREFLIGHT.log`, per-config CSV/JSON/logs,
`LLM_CONTENT_BOUNDARY_VOCAB512_SUMMARY.md`, `RESULT_VALIDATION.json`,
`RUN_COMMANDS.sh`, `LEGITIMACY_CHECK.md`, and `WORKER_REPLY.md`.

The summary must include per config and condition: row count, mean selected
gain, mean raw split gain, mean content-selected gain, mean content raw split
gain, full/content split-selection rates, mean `null_margin`, and mean `Ghat`.

Decision labels:

- `CONTENT_BOUNDARY_VOCAB512_POSITIVE`: domain gains positive, IID selected
  gains zero or near zero, and style full-token gains exceed content-only gains.
- `CONTENT_BOUNDARY_VOCAB512_MIXED`: partial pattern.
- `CONTENT_BOUNDARY_VOCAB512_NOT_EVALUATED`: preflight/build/cell failure.

Worker reply must start with `VALID` or `INVALID`, include the decision label,
completed cell count, and run path.
