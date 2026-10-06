# DFW-E085 Legitimacy Check

- Compute: `REAL_GPU` / `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.
- Models a stale/default-wrong LR arm as the fallback: lr_multiplier=0.25.
- Same starting theta per round; worker groups differ only by inner LR multiplier.
- Probe losses use select batches only; target batches score policies and oracle diagnostics.
- Guard abandons the stale default only when the best probed arm beats it by a slice-disagreement margin and at least 3/4 slices agree.
- Shuffled labels rotate probe losses across HP arms before the same guard is applied.
- Reports fresh lr=1 default, oracle single/subset, abandonment rate, and selected-arm mass.

