# DFW-E081 Legitimacy Check

- Compute: `REAL_GPU` / `torch=2.14.1+cu130 cuda_available=True cuda_version=13.0 device=NVIDIA PG509-210`.
- K<=8 probe controller frozen from E080: scaled/soft_std2, select=384.
- K>8 fallbacks do not use per-K held-out tuning.
- Global and K=8 transfer fallbacks use pilot seeds only.
- Theory fallback uses only known K.
- Shuffled probe labels remain the low-K content control.
- Held-out scoring uses fresh seeds relative to E078-E080.

