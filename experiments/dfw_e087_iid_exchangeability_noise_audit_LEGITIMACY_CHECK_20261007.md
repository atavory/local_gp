# DFW-E087 Legitimacy Check

- Compute: `TORCH_CPU` / `torch=2.15.0a0+fb cuda_available=False`.
- All workers use the same inner learning rate; fake arm labels are arbitrary worker partitions.
- Probe batches select weights; target batches are used only for scoring and oracle diagnostics.
- Shuffled labels rotate probe losses across fake groups.
- This is an IID/null audit, not an HPO positive gate.

