# GPU/Buck2 Diagnostic For `devvm23211`

You are an Assaf worker under subject `assafcohen`.

## Problem

Earlier CUDA-required jobs on `devvm23211` returned `INVALID` after checking
`nvidia-smi` directly:

- E110;
- E112.

Do **not** assume this means the machine cannot run GPU work. The project owner
believes the correct path may be through Buck2 with GPU sections enabled
(`keep_gpu_sections` or the local equivalent), not through direct ad hoc CUDA
probing.

## Task

Figure out the correct way to run CUDA/GPU workloads on `devvm23211` in this
worker environment.

## Required Investigation

1. Inspect the local worker harness and prior successful GPU task transcripts.
2. Find the project-standard Buck2 invocation for GPU sections.
3. Search internal docs/posts, not just local files. Use Meta CLI command
   discovery before running any `meta` command. Search Workplace, wiki,
   Collab/Google docs, and knowledge search for terms such as:
   - `devvm gpu buck2 keep_gpu_sections`;
   - `gpu sections buck2`;
   - `keep_gpu_sections`;
   - `Buck2 CUDA devvm`;
   - `nvidia-smi exit 9 devvm`;
   - `Assaf worker GPU`.
4. Search local flags/configs with names like:
   - `keep_gpu_sections`;
   - `gpu_sections`;
   - `cuda`;
   - `gpu`;
   - any worker wrapper that allocates a GPU section before running Python.
5. Run the smallest safe GPU smoke test through the correct Buck2/wrapper path.
6. Do not burn a long training job.

## What Not To Do

- Do not return `INVALID` merely because bare `nvidia-smi` fails.
- Do not run a full experiment.
- Do not claim the machine is CPU-only until the Buck2/GPU-section path has
  been checked.
- Do not claim the path is unknown until Workplace/wiki/Collab/internal-search
  results have been checked or explicitly ruled unavailable.

## Required Outputs

Upload a final message and artifacts with:

- whether GPU execution is possible on `devvm23211`;
- the exact command that should be used for future CUDA-required tasks;
- whether the command requires Buck2 sections, a wrapper, or environment vars;
- links or titles for the Workplace/wiki/Collab/internal docs that established
  the command, or an explicit note that internal search found nothing;
- the smallest smoke-test result;
- if still impossible, the exact failure after using the proper path.

Decision labels:

- `GPU_PATH_CONFIRMED`
- `GPU_PATH_REQUIRES_WRAPPER`
- `GPU_PATH_NOT_FOUND`
- `GPU_UNAVAILABLE_AFTER_PROPER_PATH`
- `INVALID_DIAGNOSTIC`
