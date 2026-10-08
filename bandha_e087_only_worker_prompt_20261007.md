# DFW-E087 Worker Prompt - IID Fake-Arm Null

## Mission

Run E087 only.  This is the IID fake-arm/noise-selection null.  Finish it fast
and upload artifacts.

`TORCH_CPU` through Buck-declared `//caffe2:torch` is acceptable.  Do not call
this GPU blocked.

## Source Bundle

```bash
RUN_DIR=${RUN_DIR:-/tmp/dfw_e087_only_20261007}
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
DFW_E087_K_VALUES=8,16 \
DFW_E087_ROUNDS=12 \
DFW_E087_PILOT_SEEDS=987000 \
DFW_E087_HELDOUT_SEEDS=988000,988001,988002,988003 \
buck2 run fbcode//aai/double_fan_gpu_runner:run_e087_e088_static -- \
  --profile main \
  --experiment e087 \
  --out-dir "$RUN_DIR/main"
```

## Upload

Upload result markdowns, run logs, legitimacy checks, and console output to:

```text
aai_research_tlv/tree/double_fan/runs/talsarig__devvm24439/dfw_e087_only_20261007/
```

If this is not `talsarig__devvm24439`, replace the subject with your mailbox id.

## Report

Report the E087 decision label, compute status, selector-minus-uniform,
selector-minus-shuffled, mean weight L1, entropy, and artifact path.
