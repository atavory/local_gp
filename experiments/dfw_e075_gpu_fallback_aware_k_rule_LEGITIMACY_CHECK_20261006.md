# DFW-E075 Legitimacy Check

- Compute: `REAL_GPU` / `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.
- Selector design frozen from E073/E074: scaled design only.
- Fallback rules fixed before this run from the E074 readout.
- Pilot and held-out seeds are fresh relative to E071-E074.
- Selector uses select batches only; scoring uses target batches.
- Controls include fixed, DiLoCo beta=.9, and target-retuned schedule.

