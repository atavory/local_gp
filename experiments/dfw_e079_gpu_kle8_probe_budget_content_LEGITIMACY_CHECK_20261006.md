# DFW-E079 Legitimacy Check

- Compute: `REAL_GPU` / `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.
- Isolates K<=8 probe-active cells.
- Sweeps probe batch size with selector/policy frozen.
- Pilot and held-out seeds are fresh relative to E071-E078.
- Selector uses select batches only; scoring uses target batches.
- Shuffled probe labels are the primary content control.

