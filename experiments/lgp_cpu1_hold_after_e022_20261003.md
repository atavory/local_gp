# CPU-1 Hold After E022 - 2026-10-03

E022 has completed and returned `TRUST_LAYER_RECOVERS_DENSE`.

Do not rerun `experiments/lgp_e022_dense_trust_validation.py` unless the
coordinator explicitly requests a replication. CPU-2/E023 is also complete, and
GPU E018/E019 later missed the 2026-10-03 16:00 PDT liveness deadline.

This hold note is archival. The current CPU-1 dispatch alias is `CPU_1_NEXT.md`,
which points to E025 validation-cost pricing. The GPU lane now needs worker
escalation or requeue through the Buck2 `keep_gpu_sections` path before
adjudication.
