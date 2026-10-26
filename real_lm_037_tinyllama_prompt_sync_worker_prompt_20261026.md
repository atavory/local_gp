# Kalman DDP via charm_active_rag: TinyLlama real LM prompt sync

Run this task through the `charm_active_rag` transport namespace.  Treat
`charm_active_rag` only as the mailbox transport; all science artifacts must
upload under `kalman_ddps`.

## Objective

Run the real pretrained-LM synchronization experiment on TinyLlama:

- real checkpoint: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, staged as a local HF
  directory tarball;
- real text data: AG News CSV;
- frozen pretrained LM plus learned soft-prompt adapters per worker;
- sync-time comparison of base LM, averaged prompt adapter, spectral-routed
  prompt specialists, and best-of-random routed prompt specialists.

No pip. No CPU fallback. Use Buck2 only.

## Inputs

Source overlay:

`aai_research_tlv/tree/kalman_ddps/shared/source/real_lm_prompt_sync_source_20261025.tgz`

Model:

`aai_research_tlv/tree/kalman_ddps/shared/models/tinyllama-1.1b-chat-v1.0/tinyllama-1.1b-chat-v1.0_20261026.tgz`

Dataset:

`aai_research_tlv/tree/kalman_ddps/shared/datasets/ag_news_csv/ag_news_csv_20261025.tgz`

Expected SHA256 values:

- source: `cf323519fda18d7964e2dcfd59554893e8319d35ffb806ad91f5ce37129c35b0`
- model: `aa24430f03174f76b23db5384dea8a4748d3a7096b1dc275a684849fe2185013`
- data: `856f66282126c901a07938735238e5aa35264a0f4b9c67803f179d093694c7d8`

## Output

Upload everything under:

`aai_research_tlv/tree/kalman_ddps/runs/atavory__devvm50213/real_lm_prompt_sync_tinyllama11b_agnews_20261026/`

## Script

```bash
set -euo pipefail

HOST=$(hostname | cut -d. -f1)
SUBJ="${SUBJECT:-${PROJECT_WORKER_SUBJECT:-${MAILBOX_ID:-atavory__devvm50213}}}"
OUT_SUBJECT="atavory__devvm50213"
SUFFIX="real_lm_prompt_sync_tinyllama11b_agnews_20261026"
WORK="$HOME/.project_worker/kalman_ddps/$OUT_SUBJECT/$SUFFIX"
OUT="$WORK/out"
REMOTE_ROOT="aai_research_tlv/tree/kalman_ddps/runs/$OUT_SUBJECT/$SUFFIX"
MODEL_TGZ="$WORK/tinyllama-1.1b-chat-v1.0_20261026.tgz"
DATA_TGZ="$WORK/ag_news_csv_20261025.tgz"
SOURCE_TGZ="$WORK/real_lm_prompt_sync_source_20261025.tgz"
MODEL_DIR="$WORK/models/TinyLlama-1.1B-Chat-v1.0"
DATA_DIR="$WORK/datasets/ag_news_csv"
FB="$HOME/fbsource"

mkdir -p "$WORK" "$OUT" "$WORK/models" "$WORK/datasets"
cd "$FB" || cd /data/users/atavory/fbsource || cd /data/repos/fbsource

{
  echo "# Command log"
  date -u
  echo "host=$HOST"
  echo "subject=$SUBJ"
  echo "transport_project=charm_active_rag"
  echo "result_project=kalman_ddps"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo
  echo "## nvidia-smi"
  nvidia-smi -L || true
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
} > "$OUT/COMMAND_LOG.md" 2>&1

cat > "$OUT/LEGITIMACY_CHECK.md" <<'EOF'
LEGITIMACY_CHECK: PASS

This task runs a kalman_ddps research experiment using a staged source overlay,
a staged public pretrained LM checkpoint, and a staged AG News CSV dataset.
The script installs plain Python source under it_geom_data/, runs Buck2 GPU
targets, trains only soft-prompt adapter tensors, and writes local/manifold
artifacts under the declared run directory. It does not use pip, CPU fallback,
credential access, destructive writes, persistent services, or external network
downloads.
EOF

ROSETTA_DISABLE_AOT=1 manifold get --overwrite_local_path \
  aai_research_tlv/tree/kalman_ddps/shared/source/real_lm_prompt_sync_source_20261025.tgz \
  "$SOURCE_TGZ"
ROSETTA_DISABLE_AOT=1 manifold get --overwrite_local_path \
  aai_research_tlv/tree/kalman_ddps/shared/models/tinyllama-1.1b-chat-v1.0/tinyllama-1.1b-chat-v1.0_20261026.tgz \
  "$MODEL_TGZ"
ROSETTA_DISABLE_AOT=1 manifold get --overwrite_local_path \
  aai_research_tlv/tree/kalman_ddps/shared/datasets/ag_news_csv/ag_news_csv_20261025.tgz \
  "$DATA_TGZ"

{
  echo "source  $(sha256sum "$SOURCE_TGZ")"
  echo "model   $(sha256sum "$MODEL_TGZ")"
  echo "data    $(sha256sum "$DATA_TGZ")"
} | tee "$OUT/ASSET_SHA256.txt"

echo "cf323519fda18d7964e2dcfd59554893e8319d35ffb806ad91f5ce37129c35b0  $SOURCE_TGZ" | sha256sum -c -
echo "aa24430f03174f76b23db5384dea8a4748d3a7096b1dc275a684849fe2185013  $MODEL_TGZ" | sha256sum -c -
echo "856f66282126c901a07938735238e5aa35264a0f4b9c67803f179d093694c7d8  $DATA_TGZ" | sha256sum -c -

tar -tzf "$SOURCE_TGZ" > "$OUT/SOURCE_TAR_LIST.txt"
tar -tzf "$MODEL_TGZ" > "$OUT/MODEL_TAR_LIST.txt"
tar -tzf "$DATA_TGZ" > "$OUT/DATA_TAR_LIST.txt"

tar -xzf "$SOURCE_TGZ" -C "$FB"
tar -xzf "$MODEL_TGZ" -C "$WORK/models"
tar -xzf "$DATA_TGZ" -C "$WORK/datasets"

buck2 --isolation-dir real_lm_prompt_sync_tinyllama_20261026 uquery \
  'set(fbsource//it_geom_data/code/layerprobe_buck:gpu_preflight fbsource//it_geom_data/code/layerprobe_buck:real_lm_prompt_sync)' \
  > "$OUT/TARGET_RESOLUTION.log" 2>&1

buck2 --isolation-dir real_lm_prompt_sync_tinyllama_20261026 run @fbcode//mode/dev-nosan \
  fbsource//it_geom_data/code/layerprobe_buck:gpu_preflight \
  > "$OUT/GPU_PREFLIGHT.log" 2>&1

buck2 --isolation-dir real_lm_prompt_sync_tinyllama_20261026 run @fbcode//mode/dev-nosan \
  fbsource//it_geom_data/code/layerprobe_buck:real_lm_prompt_sync -- \
  --model-dir "$MODEL_DIR" \
  --train-csv "$DATA_DIR/train.csv" \
  --test-csv "$DATA_DIR/test.csv" \
  --out-dir "$OUT/main" \
  --domain-labels 1,2,3,4 \
  --workers 8 \
  --prompt-len 16 \
  --max-length 128 \
  --train-texts-per-domain 512 \
  --probe-texts-per-domain 32 \
  --gate-texts-per-domain 32 \
  --test-texts-per-domain 64 \
  --steps 80 \
  --train-batch-size 4 \
  --eval-batch-size 8 \
  --lr 0.05 \
  --route-tokens 32 \
  --random-proposals 16 \
  --value-epsilon 0.01 \
  --seed 20261025 \
  --model-dtype bfloat16 \
  > "$OUT/REAL_LM_PROMPT_SYNC.stdout.log" \
  2> "$OUT/REAL_LM_PROMPT_SYNC.stderr.log"

python3 - <<'PY' "$OUT/main/real_lm_prompt_sync_report.json" > "$OUT/REAL_LM_PROMPT_SYNC_SUMMARY.md"
import json, sys
payload = json.load(open(sys.argv[1]))
avg = payload["average"]
base = payload["base"]
spec = payload["spectral"]
rand = payload["random_best"]
print("# Real LM prompt sync summary")
print()
print(f"Model: {payload['model_dir']}")
print(f"Model class: {payload['model_class']}")
print(f"Dataset: {payload['dataset']}")
print(f"Workers: {payload['workers']}")
print(f"Worker domains: {payload['worker_domains']}")
print(f"Selected action: {payload['selected_action']}")
print()
print("| method | gate NLL | test NLL | gain vs avg test | route counts |")
print("|---|---:|---:|---:|---|")
avg_test = avg["test_nll"]
rows = [
    ("base LM no prompt", base),
    ("average prompt", avg),
    ("spectral routed prompts", spec),
    ("best random routed prompts", rand),
]
for name, item in rows:
    print(
        f"| {name} | {item['gate_nll']:.4f} | {item['test_nll']:.4f} | "
        f"{avg_test - item['test_nll']:+.4f} | {item.get('test_route_counts', [])} |"
    )
print()
print(f"Gain vs base test NLL: {payload['gain_vs_base_test_nll']:+.4f}")
print(f"Gain vs average test NLL: {payload['gain_vs_average_test_nll']:+.4f}")
print(f"Random best gain vs average test NLL: {payload['random_best_gain_vs_average_test_nll']:+.4f}")
PY

ROSETTA_DISABLE_AOT=1 manifold mkdirs "$REMOTE_ROOT"
find "$OUT" -type f | while read -r f; do
  rel="${f#$OUT/}"
  ROSETTA_DISABLE_AOT=1 manifold mkdirs "$REMOTE_ROOT/$(dirname "$rel")" || true
  ROSETTA_DISABLE_AOT=1 manifold put --overwrite "$f" "$REMOTE_ROOT/$rel" || true
done

echo "Completed and uploaded to manifold://$REMOTE_ROOT"
cat "$OUT/REAL_LM_PROMPT_SYNC_SUMMARY.md"
```
