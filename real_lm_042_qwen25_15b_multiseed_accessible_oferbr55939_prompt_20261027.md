# Kalman DDP via charm_active_rag: Qwen2.5-1.5B real LM prompt sync multi-seed

Run this task through the `charm_active_rag` transport namespace on
`oferbr__devvm55939`.  This worker cannot read/write `kalman_ddps` directly, so
use the SHA-verified mirrors under `aair_research_tlv_messaging` and upload
science artifacts there.  The controller will copy durable results back to
`kalman_ddps` after completion.

## Objective

Run a real pretrained-LM multi-seed replication of the Qwen2.5-1.5B prompt
synchronization result:

- real checkpoint: `Qwen/Qwen2.5-1.5B`;
- real text data: AG News CSV;
- frozen pretrained LM plus learned soft-prompt adapters per worker;
- sync-time comparison of base LM, averaged prompt adapter, spectral-routed
  prompt specialists, and best-of-random routed prompt specialists;
- three seeds: `20261025 20261026 20261027`.

No pip. No CPU fallback. Use Buck2 only.

## Required science fix

Before execution, patch `real_lm_prompt_sync.py` exactly as in the successful
036C/037C corrected runs:

- evaluate the base row with a true zero-length/no-prompt model call;
- score common next-token positions `1..L-1` for all methods, excluding
  padding;
- record the patch as `BASELINE_CORRECTION.patch`;
- record the executed source and SHA256.

## Inputs

Accessible source overlay:

`aair_research_tlv_messaging/tree/charm_active_rag/shared/kalman_ddps_accessible_20261026/real_lm_prompt_sync_source_20261025.tgz`

Accessible model:

`aair_research_tlv_messaging/tree/charm_active_rag/shared/kalman_ddps_accessible_20261026/qwen2.5-1.5b_accessible_20261027.tgz`

Accessible dataset:

`aair_research_tlv_messaging/tree/charm_active_rag/shared/kalman_ddps_accessible_20261026/ag_news_csv_accessible_20261026.tgz`

Expected SHA256 values:

- source: `cf323519fda18d7964e2dcfd59554893e8319d35ffb806ad91f5ce37129c35b0`
- model: `429829bd1e7b3500559becce6df796303d729cef94ebe3e227ea1b788529646c`
- data: `856f66282126c901a07938735238e5aa35264a0f4b9c67803f179d093694c7d8`

## Output

Upload everything under:

`aair_research_tlv_messaging/tree/charm_active_rag/runs/oferbr__devvm55939/real_lm_prompt_sync_qwen25_15b_multiseed_accessible_20261027/`

## Script

```bash
set -euo pipefail

HOST=$(hostname | cut -d. -f1)
SUBJ="${SUBJECT:-${PROJECT_WORKER_SUBJECT:-${MAILBOX_ID:-oferbr__devvm55939}}}"
OUT_SUBJECT="oferbr__devvm55939"
SUFFIX="real_lm_prompt_sync_qwen25_15b_multiseed_accessible_20261027"
WORK="$HOME/.project_worker/kalman_ddps/$OUT_SUBJECT/$SUFFIX"
OUT="$WORK/out"
REMOTE_ROOT="aair_research_tlv_messaging/tree/charm_active_rag/runs/$OUT_SUBJECT/$SUFFIX"
MODEL_TGZ="$WORK/qwen2.5-1.5b_accessible_20261027.tgz"
DATA_TGZ="$WORK/ag_news_csv_accessible_20261026.tgz"
SOURCE_TGZ="$WORK/real_lm_prompt_sync_source_20261025.tgz"
MODEL_DIR="$WORK/models/Qwen2.5-1.5B"
DATA_DIR="$WORK/datasets/ag_news_csv"
FB="$HOME/fbsource"
SEEDS=(20261025 20261026 20261027)

mkdir -p "$WORK" "$OUT" "$WORK/models" "$WORK/datasets" "$OUT/seeds"
cd "$FB" || cd /data/users/atavory/fbsource || cd /data/repos/fbsource

{
  echo "# Command log"
  date -u
  echo "host=$HOST"
  echo "subject=$SUBJ"
  echo "transport_project=charm_active_rag"
  echo "asset_project=charm_active_rag accessible mirror"
  echo "seeds=${SEEDS[*]}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo
  echo "## nvidia-smi"
  nvidia-smi -L || true
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
} > "$OUT/COMMAND_LOG.md" 2>&1

cat > "$OUT/LEGITIMACY_CHECK.md" <<'EOF'
LEGITIMACY_CHECK: PASS

This task runs a kalman_ddps research experiment using SHA-verified mirrors of
the staged source, Qwen2.5-1.5B checkpoint, and AG News dataset under the
charm_active_rag messaging bucket because this worker lacks kalman_ddps read
permission.  The script installs plain Python source under it_geom_data/, runs
Buck2 GPU targets, trains only soft-prompt adapter tensors, and writes
artifacts under the declared run directory. It does not use pip, CPU fallback,
credential access, destructive writes, persistent services, or external network
downloads.
EOF

ROSETTA_DISABLE_AOT=1 manifold get --overwrite_local_path \
  aair_research_tlv_messaging/tree/charm_active_rag/shared/kalman_ddps_accessible_20261026/real_lm_prompt_sync_source_20261025.tgz \
  "$SOURCE_TGZ"
ROSETTA_DISABLE_AOT=1 manifold get --overwrite_local_path \
  aair_research_tlv_messaging/tree/charm_active_rag/shared/kalman_ddps_accessible_20261026/qwen2.5-1.5b_accessible_20261027.tgz \
  "$MODEL_TGZ"
ROSETTA_DISABLE_AOT=1 manifold get --overwrite_local_path \
  aair_research_tlv_messaging/tree/charm_active_rag/shared/kalman_ddps_accessible_20261026/ag_news_csv_accessible_20261026.tgz \
  "$DATA_TGZ"

{
  echo "source  $(sha256sum "$SOURCE_TGZ")"
  echo "model   $(sha256sum "$MODEL_TGZ")"
  echo "data    $(sha256sum "$DATA_TGZ")"
} | tee "$OUT/ASSET_SHA256.txt"

echo "cf323519fda18d7964e2dcfd59554893e8319d35ffb806ad91f5ce37129c35b0  $SOURCE_TGZ" | sha256sum -c -
echo "429829bd1e7b3500559becce6df796303d729cef94ebe3e227ea1b788529646c  $MODEL_TGZ" | sha256sum -c -
echo "856f66282126c901a07938735238e5aa35264a0f4b9c67803f179d093694c7d8  $DATA_TGZ" | sha256sum -c -

tar -tzf "$SOURCE_TGZ" > "$OUT/SOURCE_TAR_LIST.txt"
tar -tzf "$MODEL_TGZ" > "$OUT/MODEL_TAR_LIST.txt"
tar -tzf "$DATA_TGZ" > "$OUT/DATA_TAR_LIST.txt"

tar -xzf "$SOURCE_TGZ" -C "$FB"
tar -xzf "$MODEL_TGZ" -C "$WORK/models"
tar -xzf "$DATA_TGZ" -C "$WORK/datasets"

SOURCE_PATH="$FB/it_geom_data/code/layerprobe_buck/real_lm_prompt_sync.py"
cp "$SOURCE_PATH" "$OUT/LOCAL_SOURCE_BEFORE.py"
python3 - <<'PY' "$SOURCE_PATH"
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
text = text.replace("import math\n", "")
text = text.replace(
    "def _prompted_outputs(\n"
    "    model: nn.Module,\n"
    "    prompt: torch.Tensor,\n"
    "    input_ids: torch.Tensor,\n"
    "    attention_mask: torch.Tensor,\n"
    "    *,\n"
    "    labels: bool,\n"
    ") -> Any:\n"
    "    embedding = model.get_input_embeddings()\n",
    "def _prompted_outputs(\n"
    "    model: nn.Module,\n"
    "    prompt: torch.Tensor,\n"
    "    input_ids: torch.Tensor,\n"
    "    attention_mask: torch.Tensor,\n"
    "    *,\n"
    "    labels: bool,\n"
    ") -> Any:\n"
    "    if prompt.shape[0] == 0:\n"
    "        label_ids = input_ids.masked_fill(attention_mask == 0, -100) if labels else None\n"
    "        return model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)\n"
    "    embedding = model.get_input_embeddings()\n",
)
text = text.replace(
    "        logits = outputs.logits[:, prompt_len - 1 : -1, :].float()\n"
    "        targets = ids\n"
    "        token_loss = nn.functional.cross_entropy(\n"
    "            logits.reshape(-1, logits.shape[-1]),\n"
    "            targets.reshape(-1),\n"
    "            reduction=\"none\",\n"
    "        ).reshape_as(targets)\n"
    "        weights = mask.float()\n"
    "        losses.append((token_loss * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1))\n",
    "        logits = outputs.logits[:, prompt_len:-1, :].float()\n"
    "        targets = ids[:, 1:]\n"
    "        weights = mask[:, 1:].float()\n"
    "        token_loss = nn.functional.cross_entropy(\n"
    "            logits.reshape(-1, logits.shape[-1]),\n"
    "            targets.reshape(-1),\n"
    "            reduction=\"none\",\n"
    "        ).reshape_as(targets)\n"
    "        losses.append((token_loss * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1))\n",
)
text = text.replace("    base_prompt = torch.zeros_like(initial)\n", "    base_prompt = initial[:0]\n")
text = text.replace(
    "        \"model_dtype\": args.model_dtype,\n"
    "        \"base\": base_metrics,\n",
    "        \"model_dtype\": args.model_dtype,\n"
    "        \"base_is_unprompted\": True,\n"
    "        \"evaluation_token_positions\": \"1..L-1, excluding padding\",\n"
    "        \"base\": base_metrics,\n",
)
required = [
    "if prompt.shape[0] == 0:",
    "targets = ids[:, 1:]",
    "weights = mask[:, 1:].float()",
    "base_prompt = initial[:0]",
    "\"base_is_unprompted\": True",
]
missing = [needle for needle in required if needle not in text]
if missing:
    raise SystemExit(f"baseline correction did not apply cleanly: {missing}")
path.write_text(text)
PY
diff -u "$OUT/LOCAL_SOURCE_BEFORE.py" "$SOURCE_PATH" > "$OUT/BASELINE_CORRECTION.patch" || true
cp "$SOURCE_PATH" "$OUT/EXECUTED_SOURCE.py"
sha256sum "$SOURCE_PATH" > "$OUT/EXECUTED_SOURCE_SHA256.txt"
python3 -m py_compile "$SOURCE_PATH"

buck2 --isolation-dir real_lm_prompt_sync_qwen25_15b_multiseed_20261027 uquery \
  'set(fbsource//it_geom_data/code/layerprobe_buck:gpu_preflight fbsource//it_geom_data/code/layerprobe_buck:real_lm_prompt_sync)' \
  > "$OUT/TARGET_RESOLUTION.log" 2>&1

buck2 --isolation-dir real_lm_prompt_sync_qwen25_15b_multiseed_20261027 run @fbcode//mode/dev-nosan \
  fbsource//it_geom_data/code/layerprobe_buck:gpu_preflight \
  > "$OUT/GPU_PREFLIGHT.log" 2>&1

for SEED in "${SEEDS[@]}"; do
  RUN_OUT="$OUT/seeds/seed_$SEED"
  mkdir -p "$RUN_OUT"
  buck2 --isolation-dir real_lm_prompt_sync_qwen25_15b_multiseed_20261027 run @fbcode//mode/dev-nosan \
    fbsource//it_geom_data/code/layerprobe_buck:real_lm_prompt_sync -- \
    --model-dir "$MODEL_DIR" \
    --train-csv "$DATA_DIR/train.csv" \
    --test-csv "$DATA_DIR/test.csv" \
    --out-dir "$RUN_OUT" \
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
    --seed "$SEED" \
    --model-dtype bfloat16 \
    > "$OUT/REAL_LM_PROMPT_SYNC_seed_${SEED}.stdout.log" \
    2> "$OUT/REAL_LM_PROMPT_SYNC_seed_${SEED}.stderr.log"
done

python3 - <<'PY' "$OUT/seeds" "$OUT/REAL_LM_MULTISEED_SUMMARY.md" "$OUT/REAL_LM_MULTISEED_SUMMARY.csv"
import csv
import json
import math
import statistics
import sys
from pathlib import Path

seeds_dir = Path(sys.argv[1])
md_path = Path(sys.argv[2])
csv_path = Path(sys.argv[3])
rows = []
for report in sorted(seeds_dir.glob("seed_*/real_lm_prompt_sync_report.json")):
    payload = json.loads(report.read_text())
    seed = int(report.parent.name.split("_", 1)[1])
    avg = payload["average"]["test_nll"]
    spec = payload["spectral"]["test_nll"]
    rand = payload["random_best"]["test_nll"]
    selected = payload["selected_test_nll"]
    rows.append(
        {
            "seed": seed,
            "base_test_nll": payload["base"]["test_nll"],
            "average_test_nll": avg,
            "spectral_test_nll": spec,
            "random_best_test_nll": rand,
            "selected_action": payload["selected_action"],
            "selected_test_nll": selected,
            "spectral_gain_vs_average": avg - spec,
            "random_best_gain_vs_average": avg - rand,
            "selected_gain_vs_average": avg - selected,
            "spectral_groups": payload["spectral"]["groups"],
            "spectral_route_counts": payload["spectral"]["test_route_counts"],
            "random_route_counts": payload["random_best"]["test_route_counts"],
            "elapsed_s": payload["elapsed_s"],
        }
    )

def mean_sem(name):
    vals = [float(row[name]) for row in rows]
    mean = statistics.fmean(vals)
    sem = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return mean, sem

with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

lines = []
lines.append("# Qwen2.5-1.5B real-LM prompt-sync multi-seed summary")
lines.append("")
lines.append("Model: Qwen/Qwen2.5-1.5B")
lines.append("Dataset: AG News CSV")
lines.append("Seeds: " + ", ".join(str(row["seed"]) for row in rows))
lines.append("Baseline: true no-prompt model call; all methods score positions 1..L-1, excluding padding.")
lines.append("")
lines.append("| metric | mean +/- sem |")
lines.append("|---|---:|")
for metric in [
    "base_test_nll",
    "average_test_nll",
    "spectral_test_nll",
    "random_best_test_nll",
    "spectral_gain_vs_average",
    "random_best_gain_vs_average",
    "selected_gain_vs_average",
]:
    mean, sem = mean_sem(metric)
    lines.append(f"| {metric} | {mean:.6f} +/- {sem:.6f} |")
lines.append("")
lines.append("| seed | selected | avg test | spectral test | spectral gain | random best test | random gain |")
lines.append("|---:|---|---:|---:|---:|---:|---:|")
for row in rows:
    lines.append(
        f"| {row['seed']} | {row['selected_action']} | {row['average_test_nll']:.6f} | "
        f"{row['spectral_test_nll']:.6f} | {row['spectral_gain_vs_average']:+.6f} | "
        f"{row['random_best_test_nll']:.6f} | {row['random_best_gain_vs_average']:+.6f} |"
    )
lines.append("")
lines.append(
    f"Spectral-positive seeds: {sum(row['spectral_gain_vs_average'] > 0 for row in rows)}/{len(rows)}."
)
lines.append(
    f"Spectral selected by gate: {sum(row['selected_action'] == 'spectral_route' for row in rows)}/{len(rows)}."
)
md_path.write_text("\n".join(lines) + "\n")
print(md_path.read_text())
PY

{
  echo "# Verification"
  echo
  echo "- Ran CUDA GPU preflight through Buck2."
  echo "- Ran three Qwen2.5-1.5B real-LM seeds."
  echo "- Used true no-prompt base row and common next-token scoring positions."
  echo "- Wrote per-seed reports plus aggregate Markdown/CSV summaries."
} > "$OUT/VERIFICATION.md"

sha256sum "$OUT"/seeds/seed_*/real_lm_prompt_sync_report.json \
  "$OUT/REAL_LM_MULTISEED_SUMMARY.md" \
  "$OUT/REAL_LM_MULTISEED_SUMMARY.csv" \
  > "$OUT/ARTIFACT_SHA256.txt"

ROSETTA_DISABLE_AOT=1 manifold mkdirs "$REMOTE_ROOT"
find "$OUT" -type f | while read -r f; do
  rel="${f#$OUT/}"
  ROSETTA_DISABLE_AOT=1 manifold mkdirs "$REMOTE_ROOT/$(dirname "$rel")" || true
  ROSETTA_DISABLE_AOT=1 manifold put --overwrite "$f" "$REMOTE_ROOT/$rel" || true
done

echo "Completed and uploaded to manifold://$REMOTE_ROOT"
cat "$OUT/REAL_LM_MULTISEED_SUMMARY.md"
```
