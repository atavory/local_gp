# DFW-E077 Legitimacy Check

- Compute: `REAL_GPU` / `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.
- Comparisons separate implementable controls from held-out oracle ceiling.
- Selector design frozen from E073/E074: scaled design only.
- Pilot and held-out seeds are fresh relative to E071-E076.
- Selector uses select batches only; scoring uses target batches.

