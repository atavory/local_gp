# DFW-E088 Worker Prompt - Ultra-Reduced Fixed-Compute HPO

## Mission

Run E088 only with an ultra-reduced profile.  This is a fast fixed-compute
parallel-HPO sanity result, not the final high-power table.

`TORCH_CPU` through Buck-declared `//caffe2:torch` is acceptable.  Do not call
this GPU blocked.

## Source Bundle

```bash
RUN_DIR=${RUN_DIR:-/tmp/dfw_e088_ultra_20261007}
mkdir -p "$RUN_DIR"
ROSETTA_DISABLE_AOT=1 manifold get --overwrite_local_path \
  aai_research_tlv/tree/double_fan/source/double_fan_e087_e088_static_overlay_20261007.tgz \
  "$RUN_DIR/double_fan_e087_e088_static_overlay_20261007.tgz"

if [ -d "/data/users/$USER/fbsource" ]; then
  FB="/data/users/$USER/fbsource"
else
  FB="$HOME/fbsource"
fi

mkdir -p "$FB/fbcode/aai/double_fan_gpu_runner"
tar -xzf "$RUN_DIR/double_fan_e087_e088_static_overlay_20261007.tgz" \
  -C "$FB/fbcode/aai/double_fan_gpu_runner"
cd "$FB"
```

## Run

```bash
DFW_E088_K_VALUES=8 \
DFW_E088_ROUNDS=8 \
DFW_E088_PILOT_SEEDS=989000 \
DFW_E088_HELDOUT_SEEDS=990000,990001 \
DFW_E088_ALPHAS=1,2,4 \
DFW_E088_BETAS=0,0.35 \
buck2 run fbcode//aai/double_fan_gpu_runner:run_e087_e088_static -- \
  --profile main \
  --experiment e088 \
  --out-dir "$RUN_DIR/main"
```

## Upload

Upload result markdowns, run logs, legitimacy checks, and console output to:

```text
aai_research_tlv/tree/double_fan/runs/noacohen__devvm14473/dfw_e088_ultra_20261007/
```

If this is not `noacohen__devvm14473`, replace the subject with your mailbox id.

## Report

Report the E088 decision label, compute status, internal-minus-external-HPO,
internal-minus-shuffled, internal-minus-parallel-average, and artifact path.
