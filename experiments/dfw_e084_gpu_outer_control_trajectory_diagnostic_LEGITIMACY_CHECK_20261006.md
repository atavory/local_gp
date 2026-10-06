# DFW-E084 Legitimacy Check

- Compute: `REAL_GPU` / `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.
- Uses the E080/E081 frozen controller family: scaled design and soft_std2 policy.
- Restricts live probe selection to K<=8 cells.
- Pilot seeds tune only the fixed and schedule controls used for context.
- Held-out target batches score selector and oracle context; trajectory summaries use select-batch probe losses.
- Shuffled probe labels are retained as a content-control score.
- Reports effective soft pairs, top-1 probe grid points, target-design grid points, edge mass, entropy, and phase movement.

