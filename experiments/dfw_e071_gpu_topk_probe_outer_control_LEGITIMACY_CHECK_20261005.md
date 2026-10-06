# DFW-E071 Legitimacy Check

- Compute: `REAL_GPU` / `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.
- Selector uses select batches only; scoring uses target batches.
- Controls include fixed, DiLoCo beta=.9, and target-retuned schedule.
- Shuffled selector rotates probe labels as a noise control.
- Top-k policies reuse the same probe evaluations as top-1.

