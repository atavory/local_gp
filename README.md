# Deliberately Diverse Aggregation for Distributed Optimization

Start here:

- `PROJECT.md` is the stable project definition.
- `AGENT_PROMPT.md` is the operational handoff for a fresh agent.
- `double_fan_freeze_20261003.md` is the frozen negative record of the old
  same-point cloud-momentum line.
- `experiments/` is the experiment ledger. It is evidence, not the narrative.

## What This Project Is

This project treats distributed training as a coupled optimization and HPO
system.  Instead of spending parallel workers only to estimate one update, we
deliberately create optimizer/hyperparameter-arm diversity inside a distributed
round, score the candidate next states with cheap probes, and aggregate or
select the scored candidates back into one shared trajectory.

The current paper claim is:

```text
distributed workers create optimizer/hyperparameter arms;
probes estimate arm value from the same base point;
the server reinvests exploration into the shared trajectory by selecting or
weighting candidate updates.
```

A single worker naturally forms a time series, so momentum and acceleration are
reasonable summaries of that worker's past and predicted future. With many
workers, the default move is to reduce the system back to one synthetic worker
by averaging. We are studying the alternative: workers are query processes in a
shared global information state.

After a minibatch or local path segment, a worker's next action should be chosen
from all known information, not only from its own private history. The
information state may include points, gradients, losses, probes, accepted and
rejected steps, worker tags, noise estimates, latency, cost, and versioning.

The concrete local-SGD form is:

```text
local worker paths/probes
  -> estimate a structured local K-step update map
  -> shrink or reject unsafe corrections with held-out evidence
  -> take a trust-region consolidation step
```

This is not Optuna/TPE over optimizer knobs. The system is high-dimensional but
data-rich: each event can produce derivative observations. The right stack is
structured local modeling, empirical-Bayes or residual shrinkage, held-out
validation, and stochastic trust-region control.

## What It Is Not

This is not a rescue of Double Fan / cloud momentum.

The old line used workers aligned at the same point. Same-point worker clouds
estimate noise. That signal is real, but by itself it did not produce a
surviving optimizer action against schedules, tuned scalar learning-rate scale,
or Adam-like controls. Keep that as a boundary condition, not as the title of
this project.

This project needs worker misalignment, local paths, probes, heterogeneous
channels, or async tags to create action-relevant evidence.

## Current Evidence

Positive or live:

- E101 extends the adaptive-HPO breadth with six additional domain shapes:
  multiclass tabular classification, sparse text-like bags, pairwise ranking,
  graph/node summaries, sequence classification, and matrix-interaction
  regression.  On 64 held-out seeds, same-round probe selection beats
  validation-selected external fixed-LR HPO by `+0.085 +/- 0.016` at `33.33%`
  probe overhead, and a probe-batch-16 rerun still clears by
  `+0.077 +/- 0.016` at `8.33%`; all six domains clear individually.
- E100 adds five-domain adaptive-HPO breadth.  On 64 held-out seeds across
  tabular classification, tabular regression, PFN-style tabular in-context
  classification, time-series forecasting, and synthetic 8x8 vision, same-round
  probe aggregation beats validation-selected external fixed-LR HPO by
  `+0.112 +/- 0.020` at `33.33%` probe overhead.  Four domains clear
  individually; synthetic vision is label-content only.
- E099 is the usable tiny-LM adaptive tuning result: a tiny causal-LM stream
  shifts from off-domain prose to code-like text, and same-round probe
  selection beats validation-selected external fixed-LR HPO by
  `+0.043 +/- 0.026`.
- E093/E094/E095 are the strongest fixed-compute outer-control packet:
  64-seed main replication, longer-round 64-seed win, and large-K support,
  each beating validation-selected external HPO in the deliberately diverse
  outer-control setting.

- Local path/probe geometry can expose information that server-side temporal
  secants miss on synthetic beds.
- The local-SGD correction must respect the K-step dynamics. A fresh Newton or
  L-BFGS correction double-applies curvature already used by the local steps.
- Noise-controlled random probes plus residual shrinkage are live on the CPU
  synthetic local-map beds.
- E028 is a clean positive for the broader framing: tagged heterogeneous async
  worker evidence beats homogeneous averaging while tying it when workers are
  truly homogeneous.

Negative or cautioned:

- Same-point cloud momentum is frozen negative.
- E027 says a one-step greedy global query policy is explained by a fixed probe
  schedule on its CPU bed.
- E026 says delayed gamma does not replicate the same-round dense validation
  gain. Same-round validation may be a safety layer, not a reusable policy.
- Real-path GPU gates must be harvested or requeued before making claims.

## Fair Baselines

The project should be compared to useful distributed training methods, not to
impossible oracle schedules as the final verdict. Oracle and tuned schedules are
diagnostics.

Important competitors include:

- FedAvg / local SGD with tuned scalar scale;
- SlowMo / BMUF;
- DiLoCo-style outer optimization;
- FedAvgM / FedAdam / FedOpt;
- SCAFFOLD / FedProx when heterogeneity is active;
- gradient tracking, gossip, and EASGD variants;
- server-side temporal secants and classical quasi-Newton controls;
- best-worker or elitist fallback.

The useful claim is tuning reduction, transfer, or matched-cost improvement in a
regime where the baseline cannot cheaply sweep every action.

## Active Operations

Use `AGENT_PROMPT.md` after every context clear. It contains:

- how to scan and assign project workers;
- how to use the Manifold queues and source bundle;
- how to use Mr. Gitties for GitHub and Overleaf;
- how to handle GPU/Buck2 worker failures;
- the current experiment status and next actions.
