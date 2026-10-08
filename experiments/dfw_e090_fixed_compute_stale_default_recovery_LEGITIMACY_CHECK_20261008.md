# DFW-E090 Legitimacy Check

- Compute: `TORCH_CPU` / `torch=2.15.0a0+fb cuda_available=False`.
- Fixed-compute accounting: `sum(ARM_COUNTS) == N_WORKERS`; internal and external comparisons both spend that worker budget per round.
- Stale-default recovery setting: the fallback arm is the stale LR multiplier, and the guard can abandon it only on same-round probe evidence.
- External HPO uses independent arm trajectories and validation loss for the deployable choice; target loss is oracle-only.
- Shuffled guard rotates probe-loss labels across LR arms before applying the same guarded decision rule.
- Reports stale, fresh/default, high, best full-worker, validation-selected external HPO, external oracle, abandonment, arm mass, and probe cost.
- TORCH_CPU is an algorithmic PyTorch run, not a GPU-kernel performance claim.

