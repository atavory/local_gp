# DFW-E067 Legitimacy Check

- Compute: `REAL_GPU` / `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.
- Target batches are used only for scoring and oracle diagnostics.
- Default-only control uses only the lr_multiplier=1 group with its own tuned outer pair.
- All-prior raw rule repeats the E064/E066 grouped soft average.
- Default-anchored rule uses a one-hot default prior plus frozen E064 damping.
- Shuffled control rotates probe losses across LR groups for the anchored rule.

