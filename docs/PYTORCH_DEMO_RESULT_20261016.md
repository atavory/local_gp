# PyTorch Probe-HPO Demo Result - 2026-10-16

Command, through the checked PyTorch runner:

```bash
PROBE_HPO_DEMO_JSON_OUT=/data/users/atavory/local_gp/examples/pytorch_probe_hpo_demo_result_20261016.json \
  buck2 run fbcode//aai/double_fan_gpu_runner:run_local_gp_script_static -- \
  --script /data/users/atavory/local_gp/examples/pytorch_probe_hpo_demo.py
```

The stream used 90 rounds:

- clean target-like batches: 54;
- shifted batches: 18;
- anti-target batches: 18.

| method | probe loss | eval loss | eval acc | notes |
|---|---:|---:|---:|---|
| fixed LR default | 0.4241 | 0.4073 | 0.903 | `lr=0.08` |
| external fixed-LR HPO | 0.4048 | 0.3848 | 0.905 | selected `lr=0.02` |
| shuffled probe-HPO | 0.6331 | 0.6279 | 0.860 | candidate states real; probe labels shuffled |
| internal probe-HPO | 0.1134 | 0.1007 | 0.979 | selected arms per round |

Internal probe-HPO selected:

```json
{
  "sgd_lr=0": 29,
  "sgd_lr=0.005": 1,
  "sgd_lr=0.02": 11,
  "sgd_lr=0.08": 16,
  "sgd_lr=0.25": 33
}
```

Margins are internal minus comparator eval loss, so negative is better:

- internal minus default: `-0.3066`;
- internal minus external fixed-LR HPO: `-0.2840`;
- internal minus shuffled probe-HPO: `-0.5271`.

Interpretation: this is the practitioner-facing recipe, not another paper-only
ledger. A clean probe lets the training loop skip harmful batches and move
aggressively on useful ones, while external fixed-LR HPO can only choose one
learning rate for the whole stream.
