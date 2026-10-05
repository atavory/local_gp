# DFW-E066 Legitimacy Check

- Compute: `REAL_GPU` / `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.
- Target batches are used only for scoring and oracle diagnostics.
- Default-only control uses only the lr_multiplier=1 group with its own tuned outer pair.
- Raw group-probe rule uses the E064 cheap budget and frozen damping.
- Shuffled control rotates probe losses across LR groups.

