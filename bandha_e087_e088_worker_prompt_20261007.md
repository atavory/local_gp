# DFW-E087/E088 Worker Prompt - HPO-Aware Aggregation Strengthening

## Mission

Run the two HPO-strengthening experiments for the double_fan/internal-HPO paper:

1. `DFW-E087`: IID exchangeability / noise-selection audit.
2. `DFW-E088`: fixed-compute naive parallel HPO baseline.

These are not a CUDA-harness test.  A real CUDA path is welcome, but `TORCH_CPU`
from a Buck target with declared `//caffe2:torch` is acceptable for these two
audits because they use the same PyTorch math and are not GPU-kernel claims.
Do not report "GPU blocked"; report the exact compute status printed by the
scripts.

## Source Bundle

Fetch and unpack the static Buck overlay:

```bash
RUN_DIR=${RUN_DIR:-/tmp/dfw_e087_e088_20261007}
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

## Required Commands

First run a smoke:

```bash
buck2 run fbcode//aai/double_fan_gpu_runner:run_e087_e088_static -- \
  --profile smoke \
  --experiment both \
  --out-dir "$RUN_DIR/smoke"
```

Then run the main audits:

```bash
buck2 run fbcode//aai/double_fan_gpu_runner:run_e087_e088_static -- \
  --profile main \
  --experiment both \
  --out-dir "$RUN_DIR/main"
```

If the main run is too slow, keep E087 complete and run E088 with this reduced
but still useful profile:

```bash
DFW_E088_ROUNDS=20 \
DFW_E088_PILOT_SEEDS=989000 \
DFW_E088_HELDOUT_SEEDS=990000,990001,990002,990003,990004,990005 \
DFW_E088_ALPHAS=0.5,1,2,4,8 \
DFW_E088_BETAS=0,0.35,0.7 \
buck2 run fbcode//aai/double_fan_gpu_runner:run_e087_e088_static -- \
  --profile main \
  --experiment e088 \
  --out-dir "$RUN_DIR/main_e088_reduced"
```

## Upload Artifacts

Upload the result markdowns, run logs, legitimacy checks, and console log to:

```text
aai_research_tlv/tree/double_fan/runs/talsarig__devvm40229/dfw_e087_e088_20261007/
```

If this worker is not `talsarig__devvm40229`, replace that subject with your
mailbox id.

## Decision Readout

Report:

- E087 decision label and pooled selector-minus-uniform / selector-minus-shuffled.
- E087 mean weight L1 and entropy: this is the "weights move under IID noise"
  diagnostic.
- E088 decision label and pooled internal-minus-external-HPO / internal-minus-
  shuffled / internal-minus-parallel-average.
- Exact compute status (`REAL_GPU`, `TORCH_CPU`, or failure route).
- Artifact Manifold path.

The scientific interpretation should distinguish:

- IID fake-arm gains, if any, are stochastic-update filtering, not HPO.
- E088 is the fixed-compute answer to "why not just run HPO in parallel?"
