# Local GP Research Control Board

## Prime Directive

Do not repeat the Double Fan failure mode.

The goal is not to invent another controller and then explain away a loss. The
goal is first to ask whether local worker paths contain held-out predictive
information that simple schedules and scalar baselines do not contain.

## Project Boundary

This project is separate from:

- same-point cloud momentum;
- dense centered-R preconditioning;
- EB-to-zero gradient shrinkage;
- SNR compression;
- worker fault sentinels.

Those branches produced useful diagnostics but no optimizer product. Local GP
gets a clean board because it uses different observations: K-step worker paths,
not only same-point gradient scatter.

## Hypothesis

For K>1 local SGD, worker paths reveal local geometry. A GP or kernel predictor
over path/probe descriptors may predict:

- which proposal direction is safer;
- how much of local history to trust;
- which scalar action from a small action grid transfers;
- whether a round-only schedule is about to fail.

The first claim is prediction. The optimizer claim comes later.

## Evidence Required

Before closed loop, require an offline held-out prediction win:

- source-only fit;
- held-out target seeds/settings;
- paired confidence intervals;
- shuffled-feature control;
- round-only and path-norm baselines;
- target oracle headroom diagnostic.

## Current Status

E001 has run and returned `SCHEDULE_EXPLAINS`.

Summary:

- existing E78 K>1 ledger only; no fresh training and no controller;
- target K=8 oracle headroom over frozen source-selected constant:
  0.001031 [0.000504, 0.001615];
- selected `rbf_path_krr` regret: 0.001031 [0.000513, 0.001618];
- round-only regret: 0.001031;
- GP/KRR advantage over round-only: 0.000000 [0.000000, 0.000000].

Interpretation: the cheap scalar path descriptors in E78 do not add held-out
prediction over round shape. This blocks a closed-loop optimizer based on E001.
The broader local-GP idea remains open only if we collect richer information:
fuller per-worker path vectors or shared-probe function-space features.

Operational note: `devvm23211` can run GPU code if tasks use Buck2
`gpu_python_binary(..., keep_gpu_sections = True)` or
`-c fbcode.enable_gpu_sections=true`. Bare `nvidia-smi` can fail even when the
Buck2 GPU path works.

E002 has run locally and returned `TUNED_GAMMA_EXPLAINS`.

Summary:

- the diagonal-quadratic local-trajectory curvature headroom is real;
- oracle-C remains around 1.20x tuned-scale FedAvg;
- the measured within-round readout is 1.124/1.141/1.167x at N=8/32/128;
- a one-scalar tuned gamma control applied to the same measured correction
  shape scores 1.127/1.146/1.172x.

Interpretation: intermediate local points expose a useful correction shape, but
the current posterior readout is explained by tuned shrinkage. This blocks a
closed-loop method claim from the current readout. The next gate must show that
a principled Bayesian/GP evidence rule can set shrinkage at least as well as
tuned gamma without target tuning.

E003 built the first optimizer artifact:
`local_gp.optimizer.LocalGPAggregator`.

Synthetic diagonal-quadratic smoke:

- N=8: local GP/base 1.063, oracle-C/base 1.190;
- N=32: local GP/base 1.132, oracle-C/base 1.233;
- N=128: local GP/base 1.148, oracle-C/base 1.207.

Interpretation: the aggregator is now concrete and positive on the synthetic
bed, but it remains gamma-tuned and synthetic. It is an object to test, not a
real-data method claim.

E004 returned `INVALID`.

The worker did not run workload code: the external repository / Mr. Gitties
legitimacy gate failed on the worker. This is a plumbing failure, not evidence
against the optimizer. The replacement real-path gate must be self-contained,
for example via a Manifold source bundle, and must not depend on worker GitHub
access.

E005 has run locally and returned `AUTO_GAMMA_LIVE_ON_SYNTHETIC`.

Summary:

- `gamma=None` now estimates shrinkage from split-half reliability of the
  correction vector on each round;
- held-out synthetic ratios over FedAvg:
  - N=8: tuned gamma 1.045, auto-gamma 1.065, oracle 1.171;
  - N=32: tuned gamma 1.143, auto-gamma 1.142, oracle 1.247;
  - N=128: tuned gamma 1.168, auto-gamma 1.163, oracle 1.248.

Interpretation: the first promising small Bayesian object is not a full GP over
parameters. It is a low-dimensional decision layer over aggregation actions:
start with scalar `gamma`, then maybe move to a small number of group-wise
gammas. On the synthetic bed, source-free auto-gamma removes the pilot grid
without losing the effect.

E006 has run locally and returned `SCALAR_ACTION_MVP`.

Summary:

- compared scalar auto-gamma to 4-bin and 8-bin group-wise auto-gamma;
- held-out synthetic ratios over FedAvg:
  - N=8: tuned 1.054, auto1 1.072, auto4 1.081, auto8 1.084;
  - N=32: tuned 1.129, auto1 1.129, auto4 1.127, auto8 1.126;
  - N=128: tuned 1.138, auto1 1.135, auto4 1.135, auto8 1.135.

Interpretation: group-wise shrinkage gives a small low-N bump, but the clean
MVP is scalar action-space Bayes. High-dimensional worker paths estimate a
correction shape; a one-dimensional reliability posterior decides how much to
trust it.

## Success Labels

- `PATH_SIGNAL_CONFIRMED`: GP/kernel predictor beats round-only and path-norm
  baselines on held-out proposal-quality or oracle-action prediction.
- `TRANSFER_CONFIRMED`: predictor trained on source setting improves held-out
  target prediction and preserves performance under a K/model/batch shift.
- `CLOSED_LOOP_READY`: offline prediction win is large enough to justify one
  constrained optimizer run.

## Failure Labels

- `SCHEDULE_EXPLAINS`: round-only schedule matches the GP.
- `PATH_NORM_EXPLAINS`: simple path length/norm features match the GP.
- `NO_ORACLE_HEADROOM`: even the target oracle action barely beats the frozen
  baseline.
- `NO_HELDOUT_SIGNAL`: source fit works but held-out prediction fails.
- `INVALID`: target leakage, no held-out split, no target oracle diagnostic, or
  no baseline stronger than random.

## Near-Term Plan

1. Do not run a closed-loop optimizer from E001.
2. Do not run a closed-loop optimizer from E002's measured readout.
3. Do not attempt a full parameter-space GP. Use the scalar action-space
   Bayesian layer first: source-free auto-gamma from split-half reliability.
   Group-wise gamma is a secondary low-N variant only.
4. Replace E004 with a self-contained real-path gate that does not depend on
   worker GitHub/Mr. Gitties access.
5. In the real-path gate, compare FedAvg/source-tuned scale, pilot-tuned gamma,
   and source-free auto-gamma. A tie with tuned gamma is already useful because
   it reduces tuning cost; a win is an adaptation claim.
6. Richer real-data logging remains needed: full per-worker paths or
   shared-probe function-space features, not only scalar endpoint summaries.
7. Do not run a closed-loop optimizer before a held-out prediction/control gate
   passes.
