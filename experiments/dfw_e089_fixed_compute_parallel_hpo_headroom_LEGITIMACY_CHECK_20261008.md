# DFW-E089 Legitimacy Check

- Compute: `TORCH_CPU` / `torch=2.15.0a0+fb cuda_available=False`.
- E063 headroom regime: spread `16`, LR arms `(0.25, 1.0, 4.0)`, worker counts `(2, 4, 2)`.
- Seriousness beyond E088 ultra: `K=(8, 16, 32)`, `36` rounds, `8` held-out seeds.
- Same total worker count per round: `sum(ARM_COUNTS) == N_WORKERS`.
- External HPO splits workers across independent arm trajectories and discards non-selected arms.
- Internal HPO starts all arms from one shared theta each round and recombines by probe-derived weights.
- Select loss chooses the deployable external-HPO winner; target loss is used only for oracle diagnostics.
- Default full-worker and best full-worker arms are diagnostics, not part of the fixed-compute external-HPO budget.
- TORCH_CPU is an algorithmic PyTorch run, not a GPU-kernel performance claim.

