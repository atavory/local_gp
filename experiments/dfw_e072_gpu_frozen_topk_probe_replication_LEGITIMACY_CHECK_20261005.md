# DFW-E072 Legitimacy Check

- Compute: `REAL_GPU` / `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.
- Selector family frozen from E071: scaled design, soft_std2 policy.
- Pilot and held-out seeds are fresh relative to E071.
- Selector uses select batches only; scoring uses target batches.
- Controls include fixed, DiLoCo beta=.9, and target-retuned schedule.

