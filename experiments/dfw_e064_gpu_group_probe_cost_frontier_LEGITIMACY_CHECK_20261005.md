# DFW-E064 Legitimacy Check

- Compute: `REAL_GPU` / `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.
- Target batches are used only for scoring and oracle diagnostics.
- Method frozen to `lambda=0.4`, `tau=2.0`; only probe cost varies.
- Four-slice budgets use disjoint prefixes from the two independent select draws.
- Shuffled control rotates probe losses across LR groups.

