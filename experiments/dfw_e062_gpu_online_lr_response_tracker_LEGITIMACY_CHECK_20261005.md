# DFW-E062 Legitimacy Check

- Compute: `REAL_GPU` / `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.
- Target batches are used only for scoring and oracle diagnostics.
- Aligned stationary workers differ only by inner LR group.
- Trackers fit probe observations only; shuffled trackers rotate probe values across groups.

