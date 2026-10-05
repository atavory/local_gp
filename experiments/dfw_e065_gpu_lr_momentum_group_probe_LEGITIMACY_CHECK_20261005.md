# DFW-E065 Legitimacy Check

- Compute: `REAL_GPU` / `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.
- Target batches are used only for scoring and oracle diagnostics.
- Workers are aligned at the same base point each round.
- All arms have the same local step count; only LR and inner momentum vary.
- Default-only control uses the `(lr=1.0, momentum=0.0)` arm.
- Shuffled control rotates probe losses across LR/momentum groups.

