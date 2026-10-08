# DFW-E088 Legitimacy Check

- Compute: `TORCH_CPU` / `torch=2.15.0a0+fb cuda_available=False`.
- Same total worker count per round: `sum(ARM_COUNTS) == N_WORKERS`.
- External HPO splits workers across independent arm trajectories and discards non-selected arms.
- Internal HPO starts all arms from one shared theta each round and recombines by probe-derived weights.
- Select loss chooses the deployable external-HPO winner; target loss is used only for oracle diagnostics.

