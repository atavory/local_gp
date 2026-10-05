# DFW-E070 Legitimacy Check

- Compute: `REAL_GPU` / `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.
- Target batches are used only for scoring and oracle diagnostics.
- Validation-gradient scores use select shards only and compute gradients at theta, not at target labels.
- Default-only control uses only the lr_multiplier=1 group with its own tuned outer pair.
- Forward anchor repeats E067's forward finite-difference default-anchored rule.
- Gradient anchor uses one validation backward per slice and scores all LR groups by `<g_V, action>`.

