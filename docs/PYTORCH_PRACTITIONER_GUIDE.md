# PyTorch Practitioner Guide

This repository now contains a small reusable PyTorch reference
implementation:

```python
from hpo_aware_aggregation import ArmSpec, ProbeHPOAggregator
```

Use it when you already have parallel training capacity and want to spend that
capacity on in-training hyperparameter exploration instead of launching
separate HPO jobs whose losing trajectories are discarded.

## What It Does

At one training round:

1. copy the current `nn.Module`;
2. train candidate copies with different optimizer arms;
3. score candidate next states on held-out probe batches;
4. select or weight the candidate states;
5. install the selected/weighted state back into the live model.

The reference implementation is single-process, but the abstraction maps
directly to distributed training: workers evaluate arms from the same base
model, and the server applies the probe-scored aggregate.

## Minimal Usage

```python
import torch.nn.functional as F
from hpo_aware_aggregation import ArmSpec, ProbeHPOAggregator


def loss_fn(model, batch):
    x, y = batch
    return F.cross_entropy(model(x), y)


arms = (
    ArmSpec.sgd("skip", lr=0.0, momentum=0.9),
    ArmSpec.sgd("lr_1e-3", lr=1e-3, momentum=0.9),
    ArmSpec.sgd("lr_3e-3", lr=3e-3, momentum=0.9),
    ArmSpec.sgd("lr_1e-2", lr=1e-2, momentum=0.9),
)

aggregator = ProbeHPOAggregator(
    arms=arms,
    loss_fn=loss_fn,
    selector="argmin",
)

for train_batch, probe_batch in stream:
    result = aggregator.step(
        model,
        train_batches=[train_batch],
        probe_batches=[probe_batch],
    )
```

`result.selected_arm` and `result.candidates` are the per-round diagnostics.

## Controls Practitioners Should Run

Do not trust the method unless these controls are in the same script:

1. default optimizer setting;
2. external fixed-HPO, where each arm gets its own full trajectory and the best
   is selected by held-out validation loss;
3. shuffled-probe control, where candidate states are real but probe labels are
   assigned to the wrong arms;
4. internal probe-HPO.

The useful pattern is:

```text
internal probe-HPO beats default;
internal probe-HPO beats shuffled probes;
internal probe-HPO beats or matches external fixed-HPO at comparable worker cost.
```

If shuffled probes match the real probe, the run is stochastic filtering or
noise. If external fixed-HPO wins, the setting may not need same-round
adaptation.

## Runnable Demo

The demo is in `examples/pytorch_probe_hpo_demo.py`.

It simulates a common practitioner pain point: a training stream mixes useful
target-like batches with off-target or anti-target batches. External fixed-HPO
can choose one learning rate for the whole stream; internal probe-HPO can move
on useful batches and skip harmful batches using an `lr=0` arm.

```bash
python examples/pytorch_probe_hpo_demo.py --rounds 90
```

The checked run in `examples/pytorch_probe_hpo_demo_result_20261016.json`
produced:

| method | probe loss | eval loss | eval acc | note |
|---|---:|---:|---:|---|
| default fixed LR | 0.4241 | 0.4073 | 0.903 | `lr=0.08` |
| external fixed-LR HPO | 0.4048 | 0.3848 | 0.905 | selected `lr=0.02` |
| shuffled probe-HPO | 0.6331 | 0.6279 | 0.860 | probe labels broken |
| internal probe-HPO | 0.1134 | 0.1007 | 0.979 | adapts per round |

Internal probe-HPO lowered eval loss by `0.2840` versus external fixed-LR HPO
and by `0.5271` versus the shuffled-probe control.

Expected table shape:

```text
default fixed LR       worse under harmful batches
external fixed-LR HPO  chooses the best one-size-fits-all LR
shuffled probe-HPO     verifies probe labels matter
internal probe-HPO     uses the probe to adapt per round
```

## When This Is Worth Trying

- data quality changes over time;
- domains/tasks alternate or drift;
- a default optimizer setting is stale;
- a training job is already distributed enough that several arms can be run in
  parallel from the same base model;
- held-out probe batches are cheap compared with the local worker computation.

## When Not To Expect Much

- all workers and arms are truly identical and IID;
- one fixed learning rate or schedule is already near-oracle;
- probe batches are not representative of the target metric;
- probe cost is comparable to the whole training round;
- the model or optimizer state is too large to clone without real distributed
  execution.

## Relation To The Paper

The paper claim is not that weighted averaging is magic. The claim is that
distributed training can make HPO internal to aggregation: create deliberate
optimizer diversity, score the candidate next states, and reinvest the evidence
into one shared trajectory instead of discarding losing external HPO runs.
