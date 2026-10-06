# DFW-E078 Legitimacy Check

- Compute: `REAL_GPU` / `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.
- Isolates K<=8 cells where the E077 rule actually used probes.
- Selector design and policy frozen: scaled/soft_std2.
- Pilot and held-out seeds are fresh relative to E071-E077.
- Selector uses select batches only; scoring uses target batches.
- Shuffled probe labels are the primary content control.

