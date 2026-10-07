# DFW-E087/E088 Reduced Worker Prompt

## Mission

Run a reduced-but-useful backup for the HPO-aware aggregation strengthening
experiments:

1. E087 IID fake-arm noise-selection audit.
2. E088 fixed-compute external HPO baseline with reduced rounds/seeds/grid.

Use the static Buck overlay.  `TORCH_CPU` from Buck-declared `//caffe2:torch`
is acceptable for these audits; do not call this GPU blocked.

## Source Bundle

```bash
RUN_DIR=${RUN_DIR:-/tmp/dfw_e087_e088_reduced_20261007}
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

## Commands

Smoke:

```bash
buck2 run fbcode//aai/double_fan_gpu_runner:run_e087_e088_static -- \
  --profile smoke \
  --experiment both \
  --out-dir "$RUN_DIR/smoke"
```

Main reduced run:

```bash
DFW_E087_K_VALUES=8,16 \
DFW_E087_ROUNDS=24 \
DFW_E087_HELDOUT_SEEDS=988000,988001,988002,988003,988004,988005,988006,988007 \
DFW_E088_K_VALUES=8 \
DFW_E088_ROUNDS=16 \
DFW_E088_PILOT_SEEDS=989000 \
DFW_E088_HELDOUT_SEEDS=990000,990001,990002,990003 \
DFW_E088_ALPHAS=0.5,1,2,4,8 \
DFW_E088_BETAS=0,0.35,0.7 \
buck2 run fbcode//aai/double_fan_gpu_runner:run_e087_e088_static -- \
  --profile main \
  --experiment both \
  --out-dir "$RUN_DIR/main_reduced"
```

## Upload

Upload result markdowns, run logs, legitimacy checks, and console output to:

```text
aai_research_tlv/tree/double_fan/runs/assafcohen__devvm52244/dfw_e087_e088_reduced_20261007/
```

If this worker is not `assafcohen__devvm52244`, replace the subject with your
mailbox id.

## Report

Report the E087/E088 decision labels, the exact compute status, and the
artifact path.  For E087, include selector-minus-uniform and mean weight L1.
For E088, include internal-minus-external-HPO and internal-minus-shuffled.
