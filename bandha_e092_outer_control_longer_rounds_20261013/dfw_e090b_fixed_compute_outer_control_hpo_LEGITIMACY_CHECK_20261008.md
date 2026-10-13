# DFW-E090B Legitimacy Check

- Compute: `TORCH_CPU` / `torch=2.16.0a0+fb cuda_available=False`.
- Same total worker budget: external arm counts sum to the internal worker count in every cell.
- Internal HPO uses one shared worker-update pool and does not spend extra worker trajectories for each outer-control arm.
- External HPO splits workers across independent stale/default, fresh/theory, and high outer-control arms, then selects by validation/select improvement.
- Shuffled labels rotate probe losses inside the same internal selector.
- Target losses are used for scoring and oracle diagnostics only.
- This is an E080/E081-style one-step local-SGD update-map ledger, not a final full-training-trajectory claim.
- TORCH_CPU is an algorithmic PyTorch run, not a GPU-kernel performance claim.

