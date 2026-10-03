# CPU-2 Hold After E023 - 2026-10-03

E023 has completed and returned `TWO_N_RANDOM_CONFIRMED`.

Do not rerun `experiments/lgp_e023_random_probe_budget_confirm.py` unless the
coordinator explicitly requests a replication. CPU-1/E022 is also complete,
and GPU E018/E019 later missed the 2026-10-03 16:00 PDT liveness deadline.

This hold note is archival. The current CPU-2 dispatch alias is `CPU_2_NEXT.md`,
which points to E026 delayed-gamma replication. The GPU lane now needs worker
escalation or requeue through the Buck2 `keep_gpu_sections` path before
adjudication.
