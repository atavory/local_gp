# DFW-E080 Legitimacy Check

- Compute: `REAL_GPU` / `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.
- Isolates K<=8 probe-active cells.
- Uses the cheapest E078/E079 probe batch: select=384.
- Selector design and policy frozen: scaled/soft_std2.
- Pilot and held-out seeds are fresh relative to E078/E079.
- Selector uses select batches only; scoring uses target batches.
- Shuffled probe labels are the primary content control.

