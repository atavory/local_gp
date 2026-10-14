# DFW-030 Worker Prompt - Self-Contained LLM Content Boundary

## Mission

Run a self-contained follow-up to the Task 027 LLM disagreement result.  The
previous atavory lane failed because the benchmark source was not visible in the
worker checkout.  This task includes a source tarball; do not require
`~/.agent_context.*`, Mr. Gitties, or any pre-existing `it_geom_data` checkout.

Use CUDA through Buck.  Do not use `pip`.  Do not fall back to CPU.  If CUDA or
the Buck target is unavailable after installing the source overlay, report
`INVALID` with the exact failing command and logs.

## Source Bundle

Source tarball:

`manifold://aai_research_tlv/tree/kalman_ddps/shared/source/it_geom_data_layerprobe_buck_20261012.tgz`

Expected SHA256:

`b73a16c24179a3cedf2769d1355301bf4a1efc00b428e3e1ee7c07b27c3a9f6e`

Install it under an available fbsource checkout, preferring
`/data/repos/fbsource` and falling back to `/data/users/$USER/fbsource`.  After
extraction the target directory must be:

`$FBROOT/it_geom_data/code/layerprobe_buck`

Also create package markers:

```bash
touch "$FBROOT/it_geom_data/__init__.py"
touch "$FBROOT/it_geom_data/code/__init__.py"
touch "$FBROOT/it_geom_data/code/layerprobe_buck/__init__.py"
```

## Experiment

Run `llm_disagreement` with GPU-only Buck on three content-boundary settings.

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

Content-boundary cells:

1. `vocab_size=128`, `content_vocab=120`
2. `vocab_size=256`, `content_vocab=240`
3. `vocab_size=256`, `content_vocab=128`

This is 27 GPU cells total.  If a single cell fails after source installation
and GPU preflight succeeded, keep the logs and report the exact failed cell
rather than silently shrinking the experiment.

## Suggested Runner

You may use this skeleton, adjusting only for local Buck cell prefix if needed:

```bash
set -euo pipefail

SUBJECT="atavory__devvm68196"
TASK="dfw_030_llm_content_boundary_selfcontained_20261012"
WORK="/tmp/${TASK}"
OUT="${WORK}/out"
SRC="${WORK}/it_geom_data_layerprobe_buck_20261012.tgz"
RUN_ROOT="aai_research_tlv/tree/kalman_ddps/runs/${SUBJECT}/llm_content_boundary_selfcontained_20261012"
REPLY_ROOT="aai_research_tlv/tree/double_fan/messages/from_workers/${SUBJECT}"

rm -rf "$WORK"
mkdir -p "$OUT"

ROSETTA_DISABLE_AOT=1 manifold get --overwrite_local_path \
  aai_research_tlv/tree/kalman_ddps/shared/source/it_geom_data_layerprobe_buck_20261012.tgz \
  "$SRC"
sha256sum "$SRC" > "$OUT/source_sha256.txt"
grep -q b73a16c24179a3cedf2769d1355301bf4a1efc00b428e3e1ee7c07b27c3a9f6e \
  "$OUT/source_sha256.txt"

FBROOT=""
for candidate in /data/repos/fbsource "/data/users/${USER}/fbsource"; do
  if [ -d "$candidate" ]; then
    FBROOT="$candidate"
    break
  fi
done
if [ -z "$FBROOT" ]; then
  echo "INVALID: no fbsource checkout found" | tee "$OUT/INVALID.txt"
  exit 2
fi

mkdir -p "$FBROOT/it_geom_data"
tar -xzf "$SRC" -C "$FBROOT/it_geom_data"
touch "$FBROOT/it_geom_data/__init__.py"
touch "$FBROOT/it_geom_data/code/__init__.py"
touch "$FBROOT/it_geom_data/code/layerprobe_buck/__init__.py"

cd "$FBROOT"
if buck2 query fbsource//it_geom_data/code/layerprobe_buck:llm_disagreement \
    > "$OUT/buck_query_llm.log" 2>&1; then
  CELL="fbsource"
elif buck2 query fbcode//it_geom_data/code/layerprobe_buck:llm_disagreement \
    > "$OUT/buck_query_llm.log" 2>&1; then
  CELL="fbcode"
else
  echo "INVALID: llm_disagreement Buck target missing after source overlay" \
    | tee "$OUT/INVALID.txt"
  cat "$OUT/buck_query_llm.log" >> "$OUT/INVALID.txt"
  exit 3
fi

PREFLIGHT="${CELL}//it_geom_data/code/layerprobe_buck:gpu_preflight"
TARGET="${CELL}//it_geom_data/code/layerprobe_buck:llm_disagreement"

buck2 run --config fbcode.mode=dev-nosan "$PREFLIGHT" \
  > "$OUT/GPU_PREFLIGHT.log" 2>&1

COMMON=(
  --conditions domain,iid,style
  --optimizers sgd
  --seeds 2026101221,2026101222,2026101223
  --nodes 8
  --seq-len 32
  --width 96
  --layers 2
  --heads 4
  --samples-per-worker 512
  --probe-samples 256
  --value-samples 256
  --test-samples 512
  --batch-size 64
  --local-steps 120
  --null-permutations 16
  --lr 0.02
  --value-epsilon 0.01
)

for cfg in v128_c120 v256_c240 v256_c128; do
  case "$cfg" in
    v128_c120) VOCAB=128; CONTENT=120 ;;
    v256_c240) VOCAB=256; CONTENT=240 ;;
    v256_c128) VOCAB=256; CONTENT=128 ;;
  esac
  CFG_OUT="$OUT/${cfg}"
  mkdir -p "$CFG_OUT"
  buck2 run --config fbcode.mode=dev-nosan "$TARGET" -- \
    --out-dir "$CFG_OUT" \
    "${COMMON[@]}" \
    --vocab-size "$VOCAB" \
    --content-vocab "$CONTENT" \
    > "$OUT/${cfg}.stdout.log" \
    2> "$OUT/${cfg}.stderr.log"
done
```

## Summary

Write a compact summary markdown named
`LLM_CONTENT_BOUNDARY_SELFCONTAINED_SUMMARY.md` containing, for each
configuration and condition:

- number of rows;
- mean `test_nll_gain_vs_average`;
- mean `split_nll_gain_vs_average`;
- mean `content_test_nll_gain_vs_average`;
- mean `content_split_nll_gain_vs_average`;
- selection rate for `selected_action == split`;
- selection rate for `content_selected_action == split`;
- mean `null_margin`;
- mean `Ghat`.

Also write:

- `RESULT_VALIDATION.json` with row counts, CUDA preflight status, and SHA256.
- `RUN_COMMANDS.sh` with the exact commands run.
- `LEGITIMACY_CHECK.md` saying whether this was a full CUDA/Buck run, whether
  CPU fallback was used, and whether any cells were omitted.

## Uploads

Upload the entire output directory to:

`manifold://aai_research_tlv/tree/kalman_ddps/runs/atavory__devvm68196/llm_content_boundary_selfcontained_20261012/`

Send a short worker reply to:

`manifold://aai_research_tlv/tree/double_fan/messages/from_workers/atavory__devvm68196/`

The reply must start with one of:

- `VALID`
- `INVALID`

For `VALID`, include the decision label, the main per-condition findings, and
the Manifold run path.
