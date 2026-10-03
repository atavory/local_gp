# devvm23211 GPU/Buck2 Diagnostic Result - 2026-10-03

## Decision

`GPU_PATH_CONFIRMED`

## Correct Command Pattern

Bare `nvidia-smi` is not a valid availability gate on this worker.

The confirmed GPU path is a Buck2 target using `gpu_python_binary` with
`keep_gpu_sections = True`.

Confirmed smoke command:

```bash
buck2 run fbcode//aai/double_fan_gpu_smoke_assafcohen:gpu_smoke
```

Target pattern:

```python
load("@fbcode_macros//build_defs:gpu_python_binary.bzl", "gpu_python_binary")

gpu_python_binary(
    name = "run",
    srcs = ["run.py"],
    keep_gpu_sections = True,
    main_function = ".run.main",
    deps = ["fbsource//third-party/pypi/torch:torch"],
)
```

Run pattern:

```bash
buck2 run fbcode//aai/<package>:run -- <task arguments>
```

If the target cannot be changed, the documented CLI equivalent is:

```bash
buck2 run -c fbcode.enable_gpu_sections=true fbcode//aai/<package>:<target> -- <task arguments>
```

## Smoke Result

The Buck2 smoke target succeeded on `devvm23211`.

- CUDA available: true;
- device: NVIDIA PG509-210, 80 GB class;
- compute capability: 8.0;
- CUDA runtime: 13.3.0;
- PyTorch: 2.15.0a0+fb;
- test `[1, 2] + [3, 4]` returned `[4, 6]`.

Immediately afterward, bare `nvidia-smi -L` still exited 9. Therefore E110/E112
were invalidated by an incomplete availability check, not by proof that the host
could not run GPU code.

## Internal References

- GPU Programs target config wiki:
  `https://www.internalfb.com/wiki/Lowering_Service/Reference/fundamentals/gpu_targets_config/`
- Workplace devenvusers thread:
  `https://fb.workplace.com/groups/devenvusers/permalink/2650036081996745/`
- Workplace python.builds RFC:
  `https://fb.workplace.com/groups/python.builds/permalink/3938671039787642/`

Collab/Google searches found no relevant project-specific document.

## Consequence

Future CUDA-required worker prompts must instruct the worker to use Buck2 GPU
targets with `keep_gpu_sections = True` or
`-c fbcode.enable_gpu_sections=true`. Do not let future tasks return `INVALID`
only from bare `nvidia-smi`.
