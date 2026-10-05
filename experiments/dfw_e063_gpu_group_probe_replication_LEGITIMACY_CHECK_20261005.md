# DFW-E063 Legitimacy Check

- Compute: `REAL_GPU` / `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.
- Target batches are used only for scoring and oracle diagnostics.
- Uses frozen `group_mean2_l40_t2`; no pilot policy selection.
- Shuffled control rotates probe values across LR groups.

