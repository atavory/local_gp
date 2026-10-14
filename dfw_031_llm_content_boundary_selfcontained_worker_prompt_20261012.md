# DFW-031 Worker Prompt - Self-Contained LLM Content Boundary, Correct Dev-Nosan Mode

## Mission

Run the Task 027 LLM disagreement content-boundary follow-up on CUDA through
Buck.  This is self-contained: install the supplied source overlay first, then
run the GPU preflight and all experiment cells.

Do not use `pip`.  Do not fall back to CPU.  Do not require
`~/.agent_context.*` or Mr. Gitties.  If CUDA/Buck fails, upload the logs and
reply `INVALID` with the exact failed command.

## Fixed Context

- Subject: `atavory__devvm26869`
- Run root:
  `manifold://aai_research_tlv/tree/kalman_ddps/runs/atavory__devvm26869/llm_content_boundary_selfcontained_20261012/`
- Reply root:
  `manifold://aai_research_tlv/tree/double_fan/messages/from_workers/atavory__devvm26869/`
- Source tarball:
  `manifold://aai_research_tlv/tree/kalman_ddps/shared/source/it_geom_data_layerprobe_buck_20261012.tgz`
- Expected SHA256:
  `b73a16c24179a3cedf2769d1355301bf4a1efc00b428e3e1ee7c07b27c3a9f6e`

Important: successful Task 027 used this Buck style:

```bash
buck2 --isolation-dir kalman_llm_20261008 run @fbcode//mode/dev-nosan TARGET -- ARGS
```

Use that style here.  The earlier DFW-030 attempt used `--config
fbcode.mode=dev-nosan` and built, but CUDA initialized unavailable on
`devvm68196`; do not repeat that weaker mode invocation.

## Source Overlay

Use `/data/repos/fbsource` if present; otherwise use `/data/users/$USER/fbsource`.
Install the archive so this directory exists:

`$FBROOT/it_geom_data/code/layerprobe_buck`

Exclude archive `__pycache__` members during extraction.  Create package markers:

```bash
touch "$FBROOT/it_geom_data/__init__.py"
touch "$FBROOT/it_geom_data/code/__init__.py"
touch "$FBROOT/it_geom_data/code/layerprobe_buck/__init__.py"
```

Resolve both `llm_disagreement` and `gpu_preflight` under `fbcode//...` first,
then `fbsource//...` if needed.

## Experiment Grid

Common settings:

- conditions: `domain,iid,style`
- optimizer: `sgd`
- seeds: `2026101221,2026101222,2026101223`
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

Content-boundary configs:

1. `vocab_size=128`, `content_vocab=120` (`v128_c120`)
2. `vocab_size=256`, `content_vocab=240` (`v256_c240`)
3. `vocab_size=256`, `content_vocab=128` (`v256_c128`)

This is 27 CUDA cells total.  Do not shrink the grid silently.

## Required Artifacts

Write these files at the local output root before upload:

- `COMMAND_LOG.md`: host, `nvidia-smi`, target root, exact timestamps.
- `GPU_PREFLIGHT.log`.
- per-config stdout/stderr and produced CSV/JSON.
- `LLM_CONTENT_BOUNDARY_SELFCONTAINED_SUMMARY.md`.
- `RESULT_VALIDATION.json`.
- `RUN_COMMANDS.sh`.
- `LEGITIMACY_CHECK.md`.
- `WORKER_REPLY.md`.

The summary must report, for each config and condition:

- row count;
- mean `test_nll_gain_vs_average`;
- mean `split_nll_gain_vs_average`;
- mean `content_test_nll_gain_vs_average`;
- mean `content_split_nll_gain_vs_average`;
- selection rate for `selected_action == split`;
- selection rate for `content_selected_action == split`;
- mean `null_margin`;
- mean `Ghat`.

Suggested decision labels:

- `CONTENT_BOUNDARY_EVIDENCE_POSITIVE`: domain split remains positive, IID stays
  near average/no persistent selection, and style full-token gains exceed
  content-only gains in the high-style-token configs.
- `CONTENT_BOUNDARY_MIXED`: only part of that pattern holds.
- `CONTENT_BOUNDARY_NOT_EVALUATED`: preflight/build failed or zero cells ran.

## Upload and Reply

Upload the complete output directory to:

`aai_research_tlv/tree/kalman_ddps/runs/atavory__devvm26869/llm_content_boundary_selfcontained_20261012/`

Send `WORKER_REPLY.md` to:

`aai_research_tlv/tree/double_fan/messages/from_workers/atavory__devvm26869/`

The reply must start with `VALID` or `INVALID`, include the decision label, and
include the Manifold run path.
