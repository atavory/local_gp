# DFW-E083 Legitimacy Check

- Compute: `REAL_GPU` / `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.
- Same base state per round; worker groups differ only by local batch-size arm.
- Default batch arm is present and used as the conservative prior.
- Probe selector uses select batches only; target batches score policies and oracle diagnostics.
- Shuffled value/FLOP labels are included.
- FLOP ledger reports mixed training cost and probe overhead relative to batch-size work.
