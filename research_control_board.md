# Local GP Research Control Board

## Prime Directive

Do not repeat the Double Fan failure mode.

## 2026-10-12 Paper Claim Lock

The live paper is now **Deliberately Diverse Aggregation for Distributed
Optimization**.

Anonymous artifact URL for submission:
https://anonymous.4open.science/r/local_gp-9F83/

Do not split the story into "HPO" on one side and "aggregation" on the other.
The invention is their coupling: aggregation becomes an HPO mechanism because
the distributed round deliberately creates optimizer-arm diversity and then
uses probe evidence to aggregate the candidates.

```text
distributed workers create optimizer/hyperparameter arms;
probes estimate arm value from the same base point;
the server reinvests exploration into the shared trajectory by selecting or
weighting candidate updates.
```

This is the paper claim.  Separate external HPO trials spend workers on
trajectories that are later discarded or only copied from.  It is also not a
claim that probe-weighted averaging of exchangeable IID workers is generically
better than FedAvg.  Under identical IID arms the labels are exchangeable:
realized probe weights can move because update/probe noise is finite, but there
is no persistent hyperparameter signal to exploit.  That null is a control and
a possible stochastic-update-filter diagnostic, not the headline.

The HPO win must be reported in HPO terms:

- default or stale default;
- best fixed hyperparameter / schedule in hindsight;
- naive fixed-compute parallel HPO;
- shuffled-probe controller;
- probe-weighted internal HPO;
- oracle arm or oracle mixture.

Primary metrics are final score, regret to best fixed/schedule, oracle gap
closed, improvement over shuffled probes, and value per probe overhead.

The goal is not to invent another controller and then explain away a loss. The
old same-point cloud-momentum line is frozen in
`double_fan_freeze_20261003.md`: reliable measurements, no surviving optimizer
product. The live paper starts from the different observation available in
K>1 local SGD: workers move through a neighborhood and thereby produce local
map data.

The first question is whether local worker paths or explicit probe pairs contain
held-out predictive information that simple schedules, scalar FedAvg-scale
baselines, server/time secants, and best-worker fallback do not contain.

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

The one sentence bridge is:

```text
alignment estimates noise; misalignment can estimate curvature.
```

That bridge is motivation only. It is not evidence for an optimizer claim.

## Hypothesis

For K>1 local SGD, worker paths reveal local geometry. A GP or kernel predictor
over path/probe descriptors may predict:

- which proposal direction is safer;
- how much of local history to trust;
- which scalar action from a small action grid transfers;
- whether a round-only schedule is about to fail.

The first claim is prediction. The optimizer claim comes later.

Current paper hypothesis, after E014-E025: local path/probe observations should
be treated as a noisy estimate of the K-step update map, not as a fresh Newton
step and not as a GP over all parameters. The corrected quadratic operator is
\[
  [I-(I-\eta H)^K]^{-1}\Delta_K.
\]
The outside controller is a stochastic trust-region method with a ratio test,
not Optuna/TPE over high-dimensional actions.

## Evidence Required

Before closed loop, require an offline held-out prediction win:

- source-only fit;
- held-out target seeds/settings;
- paired confidence intervals;
- shuffled-feature control;
- round-only and path-norm baselines;
- target oracle headroom diagnostic.

## Current Status

### Internal-HPO Busy Loop - 2026-10-06

E082-E085 completed the guarded internal-HPO agenda:

- E082, `experiments/dfw_e082_lr_skew_summary_diagnostic.py`, returned
  `LR_SKEW_SUMMARY_NO_SAFE_COLLAPSE_TRACE_NEEDED` by reanalyzing the durable
  E060/E061 real-GPU summaries.  Across 10 nontrivial LR-spread cells,
  pilot-best minus fixed was `+0.245+/-0.137` and pilot-best minus shuffled was
  `+0.447+/-0.231`.  This does not prove low/default/high arm mass, so free LR
  search remains gated behind a trajectory/arm-mass trace.
- E083, `experiments/dfw_e083_gpu_batch_value_per_flop_hpo.py`, ran on
  `devvm52518` as `REAL_GPU` and returned `BATCH_VALUE_PER_FLOP_NEGATIVE`.
  Batch-size arms have oracle headroom, but value/FLOP probe minus shuffled was
  only `+0.054+/-0.112` at K=8 and `+0.287+/-0.405` at K=16.  The simple
  default-anchored value/FLOP selector is not promoted.
- E084, `experiments/dfw_e084_gpu_outer_control_trajectory_diagnostic.py`,
  ran on `devvm52518` as `REAL_GPU` and returned
  `OUTER_TRAJECTORY_STABLE_SOFT_INTERIOR`.  The E080/E081 low-K controller
  has high entropy (`~0.976`), low alpha-edge mass (`~0.08-0.09`), and
  negligible phase span (`<=0.008`), so moving-arm GP machinery is not
  triggered.
- E085, `experiments/dfw_e085_gpu_stale_default_guarded_hp_arm.py`, ran on
  `devvm52518` as `REAL_GPU` and returned
  `STALE_DEFAULT_GUARDED_PROBE_LIVE`.  A guarded abandonment rule for a stale
  low-LR default (`lr_multiplier=0.25`) improves over stale-only by
  `+0.997+/-0.739`, `+0.395+/-0.654`, and `+0.727+/-1.114` for K=8/16/32,
  while keeping the stale fallback about 80% of rounds.
- E086, `experiments/dfw_e086_gpu_marginal_value_group_probe_replication.py`,
  implements the stricter marginal-value gate against the frozen E063 raw
  grouped LR probe.  It is not scientifically run yet.  The first worker-route
  attempts were bad routes: direct worker Python lacked `torch`; static Buck
  targets built and imported Meta PyTorch, but the checked worker processes on
  `devvm40229`, `devvm24439`, and `devvm14473` did not expose CUDA device
  nodes.  Treat this as failed route telemetry, not a project-level GPU
  blockage and not a scientific negative.

Current interpretation: preserve E080/E081 as the method core; do not expand
the outer-control grid without new regime-shift evidence; do not promote the
simple batch-size selector; do promote stale-default recovery only in the
guarded abandon-or-fallback form; keep the marginal-value-vs-raw E086 gate
queued for a correct real-GPU run.

### HPO-Strengthening Queue - 2026-10-07

The next experiments should make the HPO claim impossible to miss:

1. **IID exchangeability/null audit.** Identical arms should show no persistent
   arm advantage even though probe weights fluctuate round by round.  Report
   weight entropy/L1, selected-arm identity persistence, selector minus uniform,
   selector minus shuffled, and target/probe correlation.  This separates true
   internal HPO from stochastic update filtering.
2. **Fixed-compute naive-HPO baseline.** Compare the same total worker budget
   spent on independent parallel HPO trajectories against probe-weighted
   recombination on one shared trajectory.  The desired win is not "we tried
   arms"; it is "the exploration contributed to the trained model instead of
   being discarded."
3. **Stale-default HPO table.** Promote E085 by adding best fixed/fresh,
   stale-only, naive chooser, shuffled guard, guarded internal HPO, and oracle.
   This is the most legible customer-facing HPO win.
4. **E086 only if scientifically run.** The marginal-value scorer can enter
   the paper only after a correct real-GPU lane produces raw-vs-gradient rows.

Update: split finishable worker runs landed on 2026-10-08.

- E087, `experiments/dfw_e087_iid_exchangeability_noise_audit.py`, ran through
  the Buck/PyTorch path on `devvm24439` as `TORCH_CPU` and returned
  `IID_STOCHASTIC_FILTERING_POSITIVE`.  Artificial IID fake arms had selector
  minus uniform `+0.621 +/- 0.320`, selector minus shuffled
  `+0.476 +/- 0.222`, mean weight L1 `0.137`, entropy `0.991`, and
  probe/target Spearman `0.406`.  This is not HPO; it proves the user's point
  that IID probe weights can move and can even help by stochastic-update
  filtering.  The paper must separate this from designed hyperparameter signal.
- E088, `experiments/dfw_e088_fixed_compute_parallel_hpo_baseline.py`, ran an
  ultra-reduced fixed-compute sanity table through Buck/PyTorch on `devvm14473`
  as `TORCH_CPU` and returned `FIXED_COMPUTE_BASELINE_NOT_CLEARED`.  At
  `K=8`, `rounds=8`, `heldout=2`, internal probe minus external
  validation-selected HPO was `-0.006 +/- 0.002`, internal minus shuffled was
  `+0.003 +/- 0.003`, and internal minus parallel-arm average was
  `0.000 +/- 0.001`.  This is a useful negative/sanity check, not a final
  high-power table.
- E089, `experiments/dfw_e089_fixed_compute_parallel_hpo_headroom.py`, ran the
  serious fixed-compute comparison in the E063 grouped-LR spread-16 regime
  through Buck/PyTorch on `devvm44191` as `TORCH_CPU`.  It used
  `K in {8,16,32}`, 36 rounds, 8 held-out seeds, LR arms `(0.25,1.0,4.0)`,
  and worker counts `(2,4,2)`.  Decision:
  `INTERNAL_HPO_CONTENT_LIVE_EXTERNAL_HPO_TIE`.  Pooled internal minus
  external validation-selected HPO was `-0.006 +/- 0.003`; internal minus
  shuffled was `+0.004 +/- 0.002`; internal minus parallel-arm average was
  `+0.001 +/- 0.001`; probe cost ran from `1.34%` at K=8 to `0.33%` at K=32.
  A supplemental pilot-pair run tightened but did not clear the external-HPO
  gap (`-0.004 +/- 0.003`).  Interpretation: E089 is now the real
  fixed-compute negative/partial result, not a tiny smoke; it preserves live
  internal HPO label content but says validation-selected naive parallel HPO is
  still the binding comparator in this grouped-LR final-trajectory setting.
- E090, `experiments/dfw_e090_fixed_compute_stale_default_recovery.py`, ran
  the intended stale-default fixed-compute final-trajectory audit on
  Buck/PyTorch `TORCH_CPU`.  Decision:
  `FIXED_COMPUTE_STALE_DEFAULT_NOT_CLEARED`.  Pooled guarded internal minus
  stale full-worker default was `-0.008 +/- 0.002`; guarded minus shuffled was
  `+6.117 +/- 1.841`; guarded minus external validation-selected HPO was
  `-0.007 +/- 0.003`.  Keep E085 as the practical guarded-recovery result, but
  do not claim stale-LR recovery clears this stricter fixed-compute ledger.
- E090B, `experiments/dfw_e090b_fixed_compute_outer_control_hpo.py`, ran the
  fixed-compute comparator for the E080/E081 outer-control method on
  Buck/PyTorch `TORCH_CPU`.  Decision:
  `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`.  Pooled internal minus stale/default
  server pair was `+3.789 +/- 0.515`, internal minus shuffled was
  `+0.161 +/- 0.061`, internal minus external validation-selected HPO was
  `+1.354 +/- 0.463`, and internal minus external oracle was
  `+0.605 +/- 0.340`.  This is the TMLR main fixed-compute demonstration of
  deliberately diverse aggregation: several server-control arms are evaluated
  from the same worker-update pool and reinvested into one shared update.
- E091, `bandha_e091_outer_control_replication_20261013/`, reran the E090B
  fixed-compute outer-control audit with fresh held-out seeds `998100..998131`.
  The Buck/PyTorch run again returned `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`.
  Pooled internal minus stale/default was `+3.878 +/- 0.488`, internal minus
  shuffled was `+0.093 +/- 0.073`, internal minus external validation-selected
  HPO was `+1.148 +/- 0.511`, and internal minus external oracle was
  `+0.413 +/- 0.345`.  This strengthens the E090B table as a reproducible
  fixed-compute HPO win rather than a one-seed artifact.
- E093, `bandha_e093_outer_control_64seed_replication_20261013/`, reran the
  E090B fixed-compute audit with 64 fresh held-out seeds `998300..998363`.
  Decision: `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`.  Pooled internal minus
  stale/default was `+3.829 +/- 0.378`, internal minus shuffled was
  `+0.140 +/- 0.058`, internal minus external validation-selected HPO was
  `+1.224 +/- 0.313`, and internal minus external oracle was
  `+0.343 +/- 0.255`.  This is the strongest replication packet for the TMLR
  fixed-compute HPO claim.
- E094, `bandha_e094_outer_control_longer_64seed_20261013/`, reran the
  longer `rounds=16` outer-control audit with 64 fresh held-out seeds
  `998400..998463`.  Decision: `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`.  Pooled
  internal minus stale/default was `+4.403 +/- 0.417`, internal minus
  fresh/theory was `+0.194 +/- 0.072`, internal minus shuffled was
  `+0.233 +/- 0.076`, internal minus external validation-selected HPO was
  `+1.275 +/- 0.239`, and internal minus external oracle was
  `+0.465 +/- 0.222`.  This upgrades the longer-round story from E092's
  high-variance live result to a CI-cleared fixed-compute HPO win.
- E095, `bandha_e095_outer_control_k16_64seed_20261013/`, extended the
  outer-control fixed-compute audit to `K=16`, `rounds=16`, and 64 fresh
  held-out seeds `998500..998563`.  Decision:
  `FIXED_COMPUTE_OUTER_CONTROL_HPO_WIN`.  Pooled internal minus stale/default
  was `+0.094 +/- 0.061`, internal minus fresh/theory was `+0.115 +/- 0.056`,
  internal minus shuffled was `+0.075 +/- 0.048`, internal minus external
  validation-selected HPO was `+0.516 +/- 0.262`, and internal minus external
  oracle was `-0.067 +/- 0.277`.  This is the large-K support row: smaller
  absolute margins, lower probe cost, and still a fixed-compute external-HPO
  win.
- E096, `bandha_e096_tiny_lm_probe_hpo_20261014/main/`, added a lightweight
  tiny causal-LM tuning audit.  Soft probe-weighted LR-arm aggregation beat the
  default fixed LR by `+0.202 +/- 0.011` and shuffled probes by
  `+0.033 +/- 0.021`, but trailed validation-selected external HPO by
  `-0.096 +/- 0.015`.  Treat this as LM label-content and a soft-averaging
  boundary, not as the LM headline.
- E097, `bandha_e097_tiny_lm_argmin_hpo_20261014/main/`, reran the tiny
  causal-LM audit with one-hot probe selection.  Decision:
  `TINY_LM_PROBE_LABEL_CONTENT`.  Internal probe selection beat the default
  fixed LR by `+0.298 +/- 0.011`, beat shuffled probes by
  `+0.177 +/- 0.015`, and tied validation-selected external HPO
  (`-0.000 +/- 0.000`).  This is the honest LM tuning row: probe labels matter,
  but the tiny-LM fixed-LR grid is simple enough that external HPO can match it.
- E098/E099 add the missing adaptive LM regime: a tiny causal-LM training
  stream shifts from off-domain prose to code-like text, while probe/select and
  target text are code-like.  E098 cleared on 16 seeds; E099,
  `bandha_e099_tiny_lm_domain_shift_32seed_20261014/main/`, reran 32 seeds and
  returned `TINY_LM_DOMAIN_SHIFT_HPO_WIN`.  Pooled internal minus default was
  `+0.452 +/- 0.022`, internal minus shuffled was `+0.466 +/- 0.029`, and
  internal minus validation-selected external fixed-LR HPO was
  `+0.043 +/- 0.026` at `16.67%` probe overhead.  This is the usable LM tuning
  win: same-round probe selection adapts across the domain shift, whereas
  fixed external HPO must choose one LR for the whole trajectory.
- E102, `bandha_e102_lm_family_domain_hpo_20261015/main32_cipher_transformer_gru_code_math/`,
  expands the LM evidence from one tiny Transformer/code shift to two LM model
  families and two adversarial textual shifts.  Transformer-small and GRU
  character-level causal LMs train on a substitution-cipher version of the
  target text for the early rounds and real target-domain text for later
  rounds; domains are code and theorem/math text.  On 32 held-out seeds per
  cell, same-round one-hot probe selection returns `LM_FAMILY_DOMAIN_HPO_WIN`:
  internal minus default `+0.661 +/- 0.025`, internal minus shuffled
  `+0.605 +/- 0.030`, and internal minus validation-selected external fixed-LR
  HPO `+0.098 +/- 0.011` at `16.67%` probe overhead.  All four
  family/domain cells clear individually.  The low-probe rerun,
  `bandha_e102_lm_family_domain_hpo_20261015/main32_probe6_cipher_transformer_gru_code_math/`,
  keeps the result at `8.33%` overhead: internal minus default
  `+0.662 +/- 0.027`, internal minus shuffled `+0.603 +/- 0.032`, and
  internal minus validation-selected external fixed-LR HPO
  `+0.098 +/- 0.012`, again with all four cells clearing.  Treat this as a
  useful adversarial stress test, not the LM headline: it is character-level
  and intentionally shifted by a substitution cipher.
- E103, `bandha_e103_token_lm_scale_hpo_20261015/main32_core_natural/`,
  moves the LM evidence to word/punctuation token prediction and natural text
  domain shifts.  Transformer and GRU causal LMs train early on off-domain
  natural text and later on target-domain text; probe/select and target-eval
  splits are held-out target-domain tokens.  With LR arms
  `(0,0.25,0.5,1,2,4)`, one-hot same-round probe selection, four shifts
  (`prose_to_code`, `dialogue_to_json`, `manual_to_sql`,
  `recipe_to_math`), and 32 held-out seeds per family/domain cell, the run
  returns `TOKEN_LM_SCALE_HPO_WIN`: pooled internal minus default
  `+1.331 +/- 0.068`, internal minus shuffled `+1.632 +/- 0.076`, and
  internal minus validation-selected external fixed-LR HPO
  `+0.482 +/- 0.042` at `8.33%` probe overhead.  All eight
  Transformer/GRU natural-shift cells clear individually.
- E103 broad-family audit,
  `bandha_e103_token_lm_scale_hpo_20261015/main8_all4_natural/`, adds LSTM
  and causal-conv token-level causal LMs at 8 seeds per cell.  The pooled row
  remains positive against external HPO (`+0.205 +/- 0.071`) and shuffled
  probes (`+1.519 +/- 0.115`), but the family slices are the honest boundary:
  Transformer and GRU clear all cells, LSTM clears 2/4, and causal-conv clears
  0/4 against external HPO.  Use this as robustness and limitation evidence,
  not as an all-family headline.
- E100, `bandha_e100_multidomain_shift_hpo_20261014/main64_5domain_sgd_lr0.02_probe64/`,
  broadens the adaptive-HPO win to five small domains: tabular classification,
  tabular regression, PFN-style tabular in-context classification, time-series
  forecasting, and synthetic 8x8 vision.  The stream is deliberately shifted:
  early rounds are off-target or label-corrupted, while probe/select and target
  batches are target-domain.  With SGD LR arms `(0,0.25,0.5,1,2,4)`, one-hot
  same-round probe aggregation, 64 held-out seeds, and `33.33%` probe overhead,
  the run returned `MULTIDOMAIN_SHIFT_HPO_WIN`.  Pooled internal minus default
  was `+0.194 +/- 0.026`, internal minus shuffled was `+0.206 +/- 0.029`, and
  internal minus validation-selected external fixed-LR HPO was
  `+0.112 +/- 0.020`.  Four domains clear individually: tabular
  classification (`+0.013 +/- 0.001` vs external), tabular regression
  (`+0.450 +/- 0.025`), PFN-style tabular context (`+0.090 +/- 0.005`), and
  time-series (`+0.017 +/- 0.008`); vision is a label-content row that clears
  default and shuffled but not external HPO.  This is now the broad synthetic
  domain-family support row.
- E101, `bandha_e101_more_domains_shift_hpo_20261014/main64_antitarget/`,
  adds six more adaptive-HPO domains without changing the controller:
  multiclass tabular classification, sparse text-like bags, pairwise ranking,
  graph/node summaries, sequence classification, and matrix-interaction
  regression.  The corrected anti-target stream keeps early training genuinely
  label-corrupted instead of accidentally target-compatible.  With the same SGD
  LR arms `(0,0.25,0.5,1,2,4)`, one-hot same-round probe selection, 64 held-out
  seeds, and `33.33%` probe overhead, the run returned
  `MORE_DOMAINS_SHIFT_HPO_WIN`.  Pooled internal minus default was
  `+0.125 +/- 0.024`, internal minus shuffled was `+0.137 +/- 0.026`, and
  internal minus validation-selected external fixed-LR HPO was
  `+0.085 +/- 0.016`.  All six added domains clear individually, with
  external-HPO margins from `+0.005 +/- 0.001` on sparse text to
  `+0.425 +/- 0.029` on matrix interaction.  The lower-cost probe-batch-16
  rerun, `bandha_e101_more_domains_shift_hpo_20261014/main64_probe16_antitarget/`,
  also returned `MORE_DOMAINS_SHIFT_HPO_WIN`: pooled internal minus default
  `+0.118 +/- 0.024`, internal minus shuffled `+0.133 +/- 0.028`, and internal
  minus validation-selected external fixed-LR HPO `+0.077 +/- 0.016` at only
  `8.33%` probe overhead, again with all six domains clearing individually.
- E092, `bandha_e092_outer_control_longer_rounds_20261013/`, reran the same
  fixed-compute outer-control audit with `rounds=16` and fresh held-out seeds
  `998200..998231`.  Decision:
  `FIXED_COMPUTE_OUTER_CONTROL_HPO_LIVE`.  Pooled internal minus stale/default
  was `+2.838 +/- 1.979`, internal minus fresh/theory was
  `+0.619 +/- 0.472`, internal minus shuffled was `+0.771 +/- 0.632`, and
  internal minus external validation-selected HPO was `+0.330 +/- 2.151`.
  Interpretation: longer slices preserve probe label content and control
  improvement, but the external-HPO comparator is high-variance and not
  cleared at this budget.

### Round-Aligned Probe-Control Branch - 2026-10-04

This is now the primary live branch.

The structural move is: keep aligned rounds and the distributed
momentum/DiLoCo/FedOpt outer-optimizer skeleton. Use probe sets to choose
low-dimensional outer control variables, especially `(alpha,beta)`, not to
infer dense local geometry or score unaligned async arrivals.

`ushmoo_rho_spectrum_check.py` established the theory target. Local SGD maps
each Hessian direction through

```text
rho_j = 1 - (1 - eta lambda_j)^K
```

so the outer optimizer acts on a preconditioned pseudo-gradient spectrum. On
the E021 bed, the predicted optimal outer momentum falls with K: about `0.610`
at `K=8` and `0.140` at `K=128`.

X1, `bandha_alpha_beta_ceiling_20261004.py`, returned
`ALPHA_BETA_CONTROL_LIVE`.

At `N=32`, per-round oracle `(alpha,beta)` versus best fixed `(alpha,beta)`:

| K | best fixed `(alpha,beta)` | per-round oracle | gap |
|---:|---:|---:|---:|
| 8 | `1.2690 +/- 0.0010` | `6.7264 +/- 0.1184` | `5.4574 +/- 0.1181` |
| 16 | `1.3404 +/- 0.0013` | `3.9036 +/- 0.0681` | `2.5632 +/- 0.0681` |
| 32 | `1.3262 +/- 0.0024` | `2.3522 +/- 0.0387` | `1.0259 +/- 0.0375` |
| 64 | `1.1195 +/- 0.0022` | `1.6047 +/- 0.0300` | `0.4852 +/- 0.0287` |

At `K=8`, per-round `(alpha,beta)` gets close to `oracle_H`
(`6.7264` vs `7.6823`), while the per-worker span oracle is only `1.3738`.
The main headroom on this bed is outer control, not per-worker reweighting.

X2, `bandha_probe_fit_vs_argmax_20261004.py`, returned
`PROBE_ARGMAX_SUFFICIENT`.

Correction to the wording in the original result artifact: this is not a dense
1220-probe grid. It is a 12-point sparse probe design; the 1220-point grid is
only a cheap search over the fitted surface. At `K=8`, sparse argmax over
`(alpha,beta)` captures almost all of the X1 oracle gap:

| P | argmax | argmax-fixed | argmax gap captured | shuffled argmax |
|---:|---:|---:|---:|---:|
| 3 | `6.5858 +/- 0.1240` | `5.3168 +/- 0.1238` | `0.9742 +/- 0.0051` | `1.8807 +/- 0.8894` |
| 9 | `6.6787 +/- 0.1254` | `5.4097 +/- 0.1252` | `0.9912 +/- 0.0042` | `-7.5554 +/- 13.4051` |
| 33 | `6.7119 +/- 0.1192` | `5.4429 +/- 0.1189` | `0.9973 +/- 0.0016` | `-3.2393 +/- 6.9826` |

The quadratic-fit selector consistently trailed argmax on this first grid
design. Do not claim fit yet. The deployable object for the next pass is probe
argmax/shrinkage over a structured `(alpha,beta)` proposal set.

X3, `bandha_alpha_beta_schedule_baseline_20261004.py`, returned
`SCHEDULE_EXPLAINS_ALPHA_BETA`.

This is the harsh zero-probe baseline: greedily tune one `(alpha_t,beta_t)`
pair per round index on pilot repetitions, then freeze that schedule on heldout
repetitions. It captures nearly all of the X1 oracle gap when tuned at the same
target `K`:

| K | target-tuned schedule capture |
|---:|---:|
| 8 | `0.9867 +/- 0.0046` |
| 16 | `0.9914 +/- 0.0032` |
| 32 | `0.9815 +/- 0.0045` |
| 64 | `0.9606 +/- 0.0063` |

Interpretation: "probes beat constant `(alpha,beta)`" is not a paper claim.
The baseline to beat is a tuned time schedule.

X4, `bandha_alpha_beta_schedule_transfer_20261004.py`, returned
`SCHEDULE_NEEDS_RETUNING`.

The schedule tuned at `K=8` transfers to `K=8`, then fails badly under K shift:

| target K | source-K=8 schedule capture | target-retuned schedule capture |
|---:|---:|---:|
| 8 | `0.9926 +/- 0.0032` | `0.9910 +/- 0.0043` |
| 16 | `-0.0884 +/- 0.1007` | `0.9926 +/- 0.0018` |
| 32 | `-9.5110 +/- 1.0704` | `0.9802 +/- 0.0059` |
| 64 | `-58.8088 +/- 4.8647` | `0.9541 +/- 0.0086` |

Interpretation: schedules are strong, but not portable across K. This is the
actual opening for probes: cheaply re-measure the low-dimensional control
surface instead of retuning an offline schedule for every K/regime.

X5, `bandha_probe_fit_vs_argmax_k_sweep_20261004.py`, returned
`PROBE_K_PARTIAL`.

The conservative label is partial because the shuffled-label controls have huge
variance, so the CI bar is not clean. The main signal is still strong: the same
12-point sparse design, reused unchanged across `K=8,16,32,64`, captures
85-96% of the oracle gap with sub-percent overhead:

| K | overhead at P=3 | argmax gap captured |
|---:|---:|---:|
| 8 | `0.41%` | `0.9510 +/- 0.0118` |
| 16 | `0.20%` | `0.9263 +/- 0.0146` |
| 32 | `0.10%` | `0.8541 +/- 0.0195` |
| 64 | `0.05%` | `0.8531 +/- 0.0224` |

Interpretation: the live paper shape is now `probe-controlled outer schedule
adaptation`, not per-worker reweighting and not dense map recovery.

X7, `bandha_probe_trust_shrinkage_k_sweep_20261004.py`, returned
`TRUST_SHRINKAGE_PARTIAL`.

The unconstrained quadratic fit is still unsafe. Trust-region fitting around
the observed sparse winner repairs the worst extrapolation at small K, but
decays at larger K. Half-shrinkage between the observed winner and the
trust-region fit is broadly positive across the K sweep, while still trailing
sparse argmax:

| K | P=3 argmax capture | P=3 fit capture | P=3 trust capture | P=3 half-shrink capture |
|---:|---:|---:|---:|---:|
| 8 | `0.9453 +/- 0.0150` | `0.0815 +/- 0.0550` | `0.8657 +/- 0.0188` | `0.9338 +/- 0.0124` |
| 16 | `0.9418 +/- 0.0185` | `-0.3630 +/- 0.2231` | `0.8171 +/- 0.0263` | `0.9256 +/- 0.0156` |
| 32 | `0.8318 +/- 0.0236` | `-3.8576 +/- 2.5524` | `0.5729 +/- 0.0416` | `0.7969 +/- 0.0182` |
| 64 | `0.8866 +/- 0.0294` | `-5.1808 +/- 4.6905` | `0.1480 +/- 0.0931` | `0.7479 +/- 0.0258` |

Interpretation: do not claim an unconstrained learned quadratic selector.
Sparse observed argmax remains the default deployable object. Half-shrinkage is
a conservative safety variant, not a stronger result.

X8, `bandha_continuous_spectrum_robustness_20261004.py`, returned
`CONTINUOUS_SPECTRUM_PARTIAL`.

The smoother log-spaced spectrum preserves the schedule story but weakens the
source-frozen sparse probe claim. Target-tuned schedules still capture about
91-96% of the oracle-fixed gap across K. A source-K=8 schedule transfers to K=8
and partly to K=16, then fails at K=32/64. The frozen 12-point sparse probe
design is positive at K=16/32/64 but misses K=8:

| K | target schedule cap | source-K=8 schedule cap | P=3 argmax cap | P=3 half-shrink cap |
|---:|---:|---:|---:|---:|
| 8 | `0.9077 +/- 0.0257` | `0.9077 +/- 0.0257` | `0.0840 +/- 0.1750` | `-0.0165 +/- 0.2125` |
| 16 | `0.9559 +/- 0.0100` | `0.7612 +/- 0.0588` | `0.5393 +/- 0.0874` | `0.5300 +/- 0.0837` |
| 32 | `0.9530 +/- 0.0145` | `0.0666 +/- 0.1708` | `0.5957 +/- 0.1168` | `0.6253 +/- 0.0930` |
| 64 | `0.9488 +/- 0.0193` | `-3.1240 +/- 0.8848` | `0.7666 +/- 0.0316` | `0.7513 +/- 0.0430` |

The focused K=8 audit,
`bandha_continuous_k8_probe_budget_20261004.py`, returned
`K8_DESIGN_OR_SIGNAL_MISS`: raising P to 9 and 33 only lifted argmax capture to
`0.3399 +/- 0.0870` and `0.3945 +/- 0.0790`, with overhead rising to 1.22% and
4.46%. The next fix must be a principled source-frozen design rule or probe
shard change, not simply more P on the old hand design.

X9a, `bandha_principled_probe_design_20261004.py`, returned
`ACTION_MAXIMIN_K8_PARTIAL`.

The pure action-space maximin rule covers normalized update coefficients
`(alpha, alpha * beta)` instead of raw `(alpha,beta)`. It repaired the
continuous-spectrum K=8 miss (`0.7852 +/- 0.0520` argmax capture at P=9), but
it dropped the old low-alpha spine and failed the original spike/bulk K=64
cell (`-0.2938 +/- 0.0808` at P=9). Interpretation: high-alpha/low-momentum
coverage is needed, but not by replacing the old source design wholesale.

X9b, `bandha_augmented_probe_design_20261004.py`, returned
`AUGMENTED_DESIGN_LIVE`.

The repair keeps the old 12 source-frozen design points and appends two
low-momentum action-space anchors, `(3.50,0.15)` and `(6.00,0.00)`. The
14-point design fixes the smooth-spectrum K=8 miss and preserves the
spike/bulk sweep:

| K | smooth P=3 cap | smooth P=9 cap | spike/bulk P=3 cap | spike/bulk P=9 cap |
|---:|---:|---:|---:|---:|
| 8 | `0.5937 +/- 0.0762` | `0.7737 +/- 0.0371` | `0.9631 +/- 0.0104` | `0.9769 +/- 0.0089` |
| 16 | `0.6988 +/- 0.0787` | `0.8339 +/- 0.0108` | `0.9478 +/- 0.0044` | `0.9551 +/- 0.0043` |
| 32 | `0.6907 +/- 0.0636` | `0.8169 +/- 0.0132` | `0.8906 +/- 0.0174` | `0.9304 +/- 0.0098` |
| 64 | `0.7118 +/- 0.0756` | `0.8014 +/- 0.0284` | `0.8904 +/- 0.0199` | `0.9220 +/- 0.0101` |

At P=9, overhead is 1.42% at K=8 and falls to 0.18% at K=64. Shuffled-label
controls remain heavy-tailed, so report them honestly; do not use this as a
real-path claim.

DFW-E032, `experiments/dfw_e032_gpu_probe_outer_control_real_path.py`, returned
`REAL_SCHEDULE_OR_FIXED_EXPLAINS` on a real PyTorch/CUDA path
(`torch=2.14.1+cu130`, NVIDIA PG509-210).

Setup: E030 two-layer binary classifier, actual local-SGD worker paths,
`N in {8,16}`, `K in {4,8,16}`, pilot-tuned fixed `(alpha,beta)`,
DiLoCo-like `beta=0.9` with tuned alpha, target-retuned time schedule,
source-K schedule transfer diagnostic, frozen 12-point sparse probe argmax,
shuffled probe labels, and target oracle.

Key result:

| N | K | fixed | target schedule | probe | oracle | probe-fixed |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 4 | `10.845 +/- 2.004` | `10.658 +/- 2.278` | `10.231 +/- 2.735` | `13.108 +/- 3.224` | `-0.614 +/- 0.910` |
| 8 | 8 | `5.889 +/- 0.933` | `5.882 +/- 1.078` | `5.527 +/- 1.859` | `7.429 +/- 1.973` | `-0.361 +/- 1.079` |
| 8 | 16 | `3.380 +/- 0.497` | `3.363 +/- 0.546` | `2.578 +/- 0.758` | `4.339 +/- 0.940` | `-0.803 +/- 0.309` |
| 16 | 4 | `10.888 +/- 2.382` | `10.868 +/- 2.427` | `9.351 +/- 2.153` | `13.152 +/- 2.812` | `-1.537 +/- 1.247` |
| 16 | 8 | `5.945 +/- 1.122` | `5.997 +/- 1.189` | `4.897 +/- 1.290` | `7.261 +/- 1.575` | `-1.048 +/- 0.830` |
| 16 | 16 | `3.304 +/- 0.674` | `3.381 +/- 0.703` | `2.500 +/- 1.240` | `4.386 +/- 0.944` | `-0.803 +/- 0.882` |

Interpretation: the old frozen 12-point probe controller is not a deployable
real-path result. The real-path workload still has oracle headroom, and
source-K schedules can fail, but fixed/target-retuned controls explain or beat
the probe rule. This blocks `PAPER_READY` on a positive real-path claim.

DFW-E032 large-K stress was attempted with `K in {32,64}` and source `K=8`, but
the remote SSH job terminated after 806 seconds with exit code 143 before a
result file was written. Treat that as `INVALID`, not evidence.

DFW-E033 triage, `experiments/dfw_e033_gpu_probe_design_real_path.py` with
`K=8`, `N=8`, select batches `{96,384}`, returned
`REAL_REPAIRED_PROBE_STILL_LOSES`. This was a bounded triage, not the full
paper gate. It compared old12, the low-momentum synthetic augmentation,
K-scaled low-momentum points, and the hybrid. The best repaired probes still
trailed the fixed/schedule controls and had negative gap capture:
best select=96 was `scaled` with probe `2.832 +/- 2.923` vs fixed
`5.062 +/- 1.542`; best select=384 was `low_aug` with probe
`2.810 +/- 4.497` vs fixed `5.062 +/- 1.542`.

Interpretation: the real-path failure is not obviously repaired by adding the
synthetic low-momentum anchors or by simply increasing the select batch. The
next repair needs a clearer real-path design diagnosis or a different control
family/cost ledger.

DFW-E034, `experiments/dfw_e034_gpu_select_target_rank_diagnostic.py`,
returned `REAL_SELECT_TARGET_MISRANK_CONFIRMED` on the same real PyTorch/CUDA
host and the same `K=8,N=8` real-path profile.

It compared the deployed select-loss winner with a target oracle constrained to
the same old12, low-augmented, scaled, and hybrid design sets. The target
oracles inside those design sets beat the best fixed/DiLoCo/schedule control,
but the select-loss winner gave away the headroom. At select batch 384, old12
scored `5.505 +/- 1.018` while its constrained target oracle scored
`6.562 +/- 1.152`; hybrid scored `5.664 +/- 1.037` while its constrained target
oracle scored `6.748 +/- 1.126`. The oracle-control margins were
`0.759 +/- 0.312` for old12 and `0.945 +/- 0.323` for hybrid, but top-1 target
hit rates were only `0.333` and `0.083`.

Interpretation: the real-path candidate family contains target-good actions,
but the select shard is not a reliable target proxy. This motivated a direct
selector/proxy repair before spending on more synthetic point additions.

DFW-E035, `experiments/dfw_e035_gpu_selector_proxy_repair.py`, returned
`REAL_SELECTOR_PROXY_REPAIR_STILL_MISRANKS` on the same real PyTorch/CUDA host
and the same `K=8,N=8` real-path profile.

It directly tested the obvious selector/proxy repairs: split-select top-k
reranking, mean loss over two or four independent select shards,
conservative UCB-on-loss selection, and schedule-guarded proxy selection. The
target batch was used only for scoring and for the constrained target-design
oracle. None of the deployable selectors beat the best fixed/DiLoCo/schedule
controls. The schedule-guarded policies reduced damage by falling back most of
the time, but still trailed controls: old12 best proxy was
`5.553 +/- 1.035` with `-0.250 +/- 0.154` selector-control, while hybrid best
proxy was `5.591 +/- 1.050` with `-0.212 +/- 0.155` selector-control. The
target-design oracles remained high (`6.562 +/- 1.152` for old12 and
`6.748 +/- 1.126` for hybrid).

Interpretation: E034's misranking diagnosis survives simple selector repairs.
The next real-path gate should change the proxy data source, control family, or
cost ledger; do not spend another run merely reranking the same select
distribution.

DFW-E036, `experiments/dfw_e036_gpu_target_proxy_control_gate.py`, returned
`REAL_TARGET_PROXY_STILL_NOT_ENOUGH` on the same real PyTorch/CUDA host and the
same `K=8,N=8` real-path profile.

It changed the proxy data source rather than reranking the same select split:
candidate actions were selected using target-distribution validation proxy
batches of 24, 48, or 96 examples, then scored on a disjoint target batch. This
was priced as extra validation-proxy forward work. The best old12 target-proxy
selector was `target_guard_b96`, scoring `4.884 +/- 1.396` with
`-0.296 +/- 0.520` selector-control at 0.73% proxy cost. The hybrid target
oracle remained `5.903 +/- 0.842`, but its best target-proxy selector scored
`4.858 +/- 1.342` with `-0.323 +/- 0.448` selector-control at 1.90% proxy cost.

Interpretation: the easiest proxy-source repair is also negative. A small
target-distribution validation proxy improves some old12 rank diagnostics, but
it still does not beat fixed/DiLoCo/schedule controls at the tested costs. The
next real-path gate should change the control family or cost ledger more
substantially, not simply add another select/target mini-batch proxy around the
same `(alpha,beta)` candidate sets.

DFW-E037,
`experiments/dfw_e037_gpu_schedule_residual_control_family.py`, returned
`REAL_SCHEDULE_RESIDUAL_SELECTOR_BLOCKED` on the same real PyTorch/CUDA host
and the same `K=8,N=8` real-path profile.

It changed the control family rather than only changing the proxy source:
candidate actions were schedule-neighborhood residuals around the pilot-tuned
time schedule, and the schedule action was always an explicit fallback
candidate. This tested whether a lower-dimensional family would make
real-path proxy selection easier. It did not. The alpha-only family target
oracle scored `6.723 +/- 1.433`, which was `0.951 +/- 0.402` above the best
control, but its best deployed selector scored `5.598 +/- 1.634`, or
`-0.174 +/- 0.644` below control. The beta-only family came closest, with
selector `5.706 +/- 1.564` and `-0.066 +/- 0.544` below control. The full
residual family had the largest target-oracle margin (`1.059 +/- 0.512` above
control) but the deployed selector still scored `5.521 +/- 1.706`, or
`-0.251 +/- 0.718` below control.

Interpretation: the blocker survives both a proxy-source change and a
schedule-neighborhood control-family change. The next real-path gate should be
a substantially different cost ledger, workload, or representation test, not
another mini-batch selector or nearby `(alpha,beta)` residual family.

DFW-E038,
`experiments/dfw_e038_gpu_large_k_probe_transfer_gate.py`, returned
`REAL_LARGE_K_PROBE_ONLY_BEATS_SOURCE_SCHEDULE` on the same real PyTorch/CUDA
host.

It narrowed the invalid broad E032 large-K attempt to a focused workload shift:
tune a source time schedule at `K=8`, then evaluate `K=32,N=8` with select
batch 384. The source schedule collapsed at `-16.006 +/- 3.896`, while the
best target-tuned control was fixed `(alpha,beta)=(4.0,0.35)` at
`1.613 +/- 1.053` and the full target oracle was `2.645 +/- 0.833`. Repaired
probe designs beat the stale source schedule by large paired margins, but not
the target-tuned controls with paired confidence: scaled scored
`1.842 +/- 0.195` with `0.228 +/- 0.917` probe-control, and hybrid scored
`1.841 +/- 0.196` with the same probe-control margin. Target-design oracles
inside the scaled/hybrid families remained high (`2.567 +/- 0.779` and
`2.570 +/- 0.784`), but this focused gate still does not produce a deployable
real-path probe win.

Interpretation: the real CUDA path does show the source-schedule transfer
failure that motivates probes at larger `K`, but E038 does not overturn the
E032--E037 boundary. This motivated a direct high-cost validation-ledger test
rather than another simple large-`K` rerun.

DFW-E039,
`experiments/dfw_e039_gpu_validation_cost_ledger_gate.py`, returned
`REAL_HIGH_COST_TARGET_VALIDATION_STILL_MISRANKS` on the same real
PyTorch/CUDA host and the same `K=8,N=8` real-path profile.

It changed the validation/cost ledger rather than the point design: candidate
actions in old12, hybrid, and scaled families were selected with large
target-distribution validation proxies of 192 and 320 examples, then scored on
a disjoint target batch of 192 examples. Target-design oracles still beat the
best fixed/DiLoCo/schedule controls: old12 oracle-control was
`0.551 +/- 0.225`, hybrid was `0.732 +/- 0.240`, and scaled was
`0.728 +/- 0.239`. The deployed selectors still did not clear controls. The
best old12 selector was `target_guard_b320` at `5.215 +/- 1.174`, or
`-0.343 +/- 0.480` versus control; the best hybrid selector remained
`select384` at `5.436 +/- 0.952`, or `-0.123 +/- 0.228`; and the best scaled
selector remained `select384` at `5.551 +/- 0.909`, or
`-0.007 +/- 0.204`.

Interpretation: the E036 small-target-proxy negative is not obviously repaired
by spending a much larger disjoint target-validation proxy. At the tested
validation costs, the real-path target-good actions still do not become a
deployable controller. The next real-path gate should change representation or
workload, not only target-proxy batch size.

DFW-E040,
`experiments/dfw_e040_gpu_worker_quality_burst_gate.py`, returned
`REAL_BURSTY_TARGET_ORACLE_SELECTOR_BLOCKED` on the same real PyTorch/CUDA
host.

It changed the workload instead of another selector repair: at `K=8,N=8`, each
round had probability `0.50` of selecting a random `0.50` fraction of workers
as bad workers whose local steps used label-flipped minibatches. Select and
target scoring stayed on clean held-out splits, and fixed/DiLoCo/schedule
controls were tuned on bursty pilot repetitions. Held-out rounds had active
bursts in `0.54` of rounds with `4.00` bad workers per active round. The
clean-source schedule scored `6.699 +/- 4.056`, the best target-tuned control
scored `10.409 +/- 4.399`, and the full oracle scored `16.430 +/- 3.540`.
The best deployed hybrid probe scored `12.859 +/- 0.829`, with
`2.450 +/- 4.820` versus target controls and `6.160 +/- 4.755` versus the
clean-source schedule. Its target-design oracle scored `14.886 +/- 2.328`,
or `4.477 +/- 2.808` above target controls.

Interpretation: worker-quality bursts are a materially different real-path
workload and create large target-oracle headroom, but the deployed probe margin
over target-tuned controls is not paired-positive. This is a promising
replication/representation lead, not a deployable real-path win.

DFW-E041,
`experiments/dfw_e041_gpu_burst_tagged_aggregate_gate.py`, returned
`REAL_BURST_TAGGED_DESIGN_ORACLE_SELECTOR_BLOCKED` on the same real
PyTorch/CUDA host.

It tested the most direct representation follow-up to E040: score every worker
endpoint on the clean select shard, keep the best half of workers, and average
only those updates. Fixed, DiLoCo-like, and time-schedule controls were then
tuned on both the raw aggregate and this same tagged aggregate, so a probe win
had to beat a fair zero-probe tagged baseline. The clean-select tag selected no
injected bad workers in held-out rounds (`selected_bad_frac=0.000`,
`bad_kept_frac=0.000`). The best raw zero-probe control scored
`9.262 +/- 2.972`; the best tagged zero-probe control rose to
`14.509 +/- 6.373`; the tagged full oracle scored `16.403 +/- 6.929`; and the
unavailable clean-worker oracle scored `16.486 +/- 6.972`. The best deployed
hybrid tagged probe scored `14.578 +/- 6.418`, only `0.069 +/- 0.661` above
the fair tagged-control baseline, while its tagged-design oracle remained
`1.644 +/- 0.785` above controls.

Interpretation: worker-quality tags are a real representation lead under the
burst workload, but they do not rescue the sparse `(alpha,beta)` probe claim.
Once fixed/schedule controls get the same tagged aggregate, the deployed probe
selector no longer has a paired-positive margin.

DFW-E042,
`experiments/dfw_e042_gpu_worker_quality_burst_replication.py`, returned
`REAL_BURSTY_TARGET_ORACLE_SELECTOR_BLOCKED` on the same real PyTorch/CUDA
host.

It reused the E040 raw burst workload but increased held-out repetitions from
6 to 12 and restricted the reported designs to `old12,hybrid`. At `K=8,N=8`,
held-out rounds had active bursts in `0.55` of rounds with `4.00` bad workers
per active round. The best retuned fixed/DiLoCo/schedule control scored
`12.474 +/- 5.685`, the clean-source schedule scored `2.986 +/- 5.524`, and
the full oracle scored `19.206 +/- 8.980`. The best deployed hybrid probe
scored `12.328 +/- 4.338`, or `-0.147 +/- 3.265` versus the best retuned
control, while its target-design oracle scored `15.052 +/- 3.734`, or
`2.578 +/- 2.475` above controls.

Interpretation: the raw burst workload remains a useful real-path stressor
because it creates oracle headroom and punishes stale clean schedules, but the
E040 positive deployed mean does not survive as a stronger result under the
wider held-out replication. Do not spend the next gate simply repeating the
same raw burst `(alpha,beta)` candidate set.

DFW-E043,
`experiments/dfw_e043_gpu_data_order_drift_gate.py`, returned
`REAL_DATA_ORDER_DRIFT_TARGET_ORACLE_SELECTOR_BLOCKED` on the same real
PyTorch/CUDA host.

It changed the workload without corrupting labels: at `K=8,N=8`, each round
sampled local worker minibatches from one random clean data-order stratum
(`balanced`, high/low first feature, or label-one/label-zero-heavy pools),
while select and target scoring stayed on clean held-out splits. Held-out mode
rates were `balanced=0.31`, `x0_high=0.17`, `x0_low=0.25`,
`label_one=0.17`, and `label_zero=0.10`. The clean-source schedule collapsed
to `-31.354 +/- 45.173`, the target-tuned time schedule scored
`-4.924 +/- 16.752`, and the best retuned fixed/DiLoCo/schedule control
scored `3.945 +/- 3.484`. The deployed hybrid probe scored
`11.477 +/- 10.531`, or `7.532 +/- 8.323` versus the best retuned control,
while its target-design oracle scored `12.997 +/- 10.922`, or
`9.052 +/- 8.752` above controls.

Interpretation: clean data-order drift is a better real-path stressor than a
simple rerun: it destroys stale clean schedules and creates large target-oracle
headroom inside the current probe designs. But the deployed probe margin still
crosses zero, so this is another selector-blocked boundary rather than a
deployable real-path win.

DFW-E044,
`experiments/dfw_e044_gpu_data_order_drift_replication.py`, returned
`REAL_DATA_ORDER_DRIFT_PROBE_BEATS_TARGET_CONTROLS` on the same real
PyTorch/CUDA host.

It widened E043 to 12 held-out seeds while keeping the clean data-order drift
workload, E043 source/drift pilots, select batch 384, and `old12,hybrid`
design families. Held-out mode rates were `balanced=0.30`, `x0_high=0.15`,
`x0_low=0.22`, `label_one=0.20`, and `label_zero=0.14`. The clean-source
schedule scored `-19.694 +/- 23.590`, the target-tuned time schedule scored
`-4.600 +/- 8.897`, and the best retuned fixed/DiLoCo/schedule control scored
`1.123 +/- 2.939`. The deployed hybrid probe scored `6.874 +/- 5.923`, or
`5.751 +/- 5.270` versus the best retuned control, while its target-design
oracle scored `8.335 +/- 6.437`, or `7.212 +/- 6.094` above controls.

Interpretation: E044 is the first deployed real-path probe win, but it is
narrow. The result is specific to the clean data-order drift workload at
`K=8,N=8`, and the hybrid design costs 7.59% overhead. E032--E043 remain
important boundary conditions rather than obsolete negatives.

DFW-E045,
`experiments/dfw_e045_gpu_balanced_data_order_drift_confirmation.py`, returned
`REAL_DATA_ORDER_DRIFT_TARGET_ORACLE_SELECTOR_BLOCKED` on the same real
PyTorch/CUDA host.

It reused the E043/E044 clean data-order drift harness on independent seeds and
replaced random mode draws with a deterministic balanced clean-strata cycle:
`(seed + round_idx) mod 5` over `balanced`, `x0_high`, `x0_low`,
`label_one`, and `label_zero`. With 15 held-out repetitions and eight rounds,
held-out mode rates were exactly `0.20` for every mode. The clean-source
schedule collapsed to `-65.484 +/- 69.148`, the target-tuned schedule scored
`9.400 +/- 9.505`, and the best retuned fixed/DiLoCo/schedule control scored
`13.100 +/- 14.846`. The deployed hybrid probe scored
`16.212 +/- 17.243`, or `3.112 +/- 2.622` versus that control, while its
target-design oracle scored `17.825 +/- 18.103`, or
`4.725 +/- 3.422` above controls; hybrid capture was only
`0.437 +/- 0.283`.

Interpretation: the balanced independent-seed clean-drift confirmation does
not support treating E044 as a stable real-path optimizer win. It preserves
source-schedule transfer pressure and target-good probe actions, and it even
keeps a positive paired probe-control margin, but the deployed selector fails
the capture gate and remains below its constrained target-design oracle. The
clean-drift evidence is fragile and workload-specific.

X6, `bandha_nonstationary_schedule_probe_gate_20261004.py`, returned
`NONSTATIONARY_PROBE_NO_SIGNAL`.

This is the sharper gate from the ushmoo/wanger diagnosis: perturb the bed with
random per-round curvature regimes so `(alpha*_t,beta*_t)` is not purely a
function of round index. The result is mixed but not a clean positive:

| K | schedule capture | probe capture | probe - schedule |
|---:|---:|---:|---:|
| 8 | `0.8201 +/- 0.1308` | `0.9314 +/- 0.0165` | `0.4441 +/- 0.4864` |
| 32 | `0.5912 +/- 0.3948` | `0.8617 +/- 0.0284` | `0.1467 +/- 0.2206` |
| 64 | `0.7634 +/- 0.1154` | `0.7235 +/- 0.1036` | `-0.0274 +/- 0.0402` |

Interpretation: probes remain robust in absolute gap capture, but they do not
beat the frozen schedule with paired CI, and they lose slightly at K=64. This
does **not** support the stronger claim "probes track nonstationarity a schedule
cannot" on this first synthetic perturbation. The paper can still claim
cross-K retuning replacement, but the stronger nonstationary story is not yet
proven.

X10, `bandha_inner_lr_phase_shift_probe_gate_20261004.py`, returned
`INNER_LR_PHASE_SCHEDULE_TRANSFERS`.

This was a principled follow-up to X6 using a training-shaped perturbation:
the local-step inner learning-rate profile had mean-one cosine modulation, the
source pilot schedule used phase `0`, and heldout target rounds cycled phases
`pi/2`, `pi`, and `3pi/2`. The augmented 14-point source-frozen design stayed
useful, but the source-phase time schedule transferred strongly:

| K | source schedule cap | target schedule cap | augmented P=9 cap | augmented-source |
|---:|---:|---:|---:|---:|
| 8 | `0.9752 +/- 0.0067` | `0.9853 +/- 0.0041` | `0.9838 +/- 0.0061` | `0.0435 +/- 0.0568` |
| 32 | `0.9755 +/- 0.0046` | `0.9834 +/- 0.0049` | `0.9343 +/- 0.0066` | `-0.0452 +/- 0.0092` |
| 64 | `0.9462 +/- 0.0098` | `0.9532 +/- 0.0100` | `0.8960 +/- 0.0184` | `-0.0262 +/- 0.0104` |

Interpretation: inner-LR phase shifts do not repair the nonstationarity claim.
They are a useful cautionary negative: not every realistic-looking training
drift invalidates a round-indexed schedule. Future nonstationary tests should
use a different mechanism, such as data-order drift, worker-quality bursts, or
a real workload shift.

Next gates:

1. Turn 19 made the submission-framing decision, turn 20 compressed the
   abstract/opening scope, and turn 21 completed the final claim/table sanity
   pass: the synthetic mechanism plus E032--E045 real-path boundary and
   fragile clean-drift evidence are defensible only as a short workshop
   mechanism-and-boundary paper, not as a broad optimizer/product claim. The
   boundary is now specific: target-good actions exist in several candidate
   families, and a K=8 source schedule can fail badly at real-path K=32. E040
   adds that random worker-quality bursts create a large positive mean for the
   deployed hybrid probe but still do not clear target-tuned
   fixed/DiLoCo/schedule controls with paired confidence. E041 adds that
   clean-select worker tags identify the injected bad workers but tagged
   zero-probe controls explain the gain. E042 narrows the raw burst uncertainty
   and moves the deployed hybrid probe mean slightly below the best retuned
   control. E043 adds that clean data-order drift collapses the clean-source
   schedule and gives the deployed hybrid probe a large positive mean, while
   E044 widens that gate to a paired-positive deployed margin over retuned
   controls. E045 uses independent seeds and balanced clean strata, then returns
   to a selector-blocked verdict because capture remains below the positive
   threshold. Cheap and high-cost target-proxy deployable selection remain
   blocked outside the fragile clean-drift signal. The remaining paper-ready
   action is a successful build or Overleaf compile check; local TeX tools and
   the checked remote host's TeX tools are unavailable, so the busy-wait is
   blocked on compile infrastructure.
2. Complete or narrow the full DFW-E033 real-path repair only after a concrete
   change that is not another selector around the same `(alpha,beta)` candidate
   sets or a schedule-neighborhood residual family.
3. Strengthen/repair X6 only if there is a principled nonstationarity not
   already explained by schedules. Inner-LR phase shifts have now transferred,
   E044 tested a widened clean data-order drift replication on the real path
   with a narrow-positive deployed margin, and E045's balanced independent-seed
   confirmation returned selector-blocked. Any follow-up should change
   representation or workload materially rather than hand-tune perturbations
   merely to make probes win.
4. Stress the augmented 14-point source-frozen design on another synthetic
   regime only if it addresses a credible reviewer objection; do not keep
   hand-tuning probe points.

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

E007 has one completed real-path shard and returned `FEDAVG_SCALE_EXPLAINS`.

Summary:

- workload: CUDA MNIST local-SGD paths, N=8, K=8, eta=0.2, linear classifier;
- selected source gamma: 0.050;
- selected source FedAvg scale: 1.900;
- target one-step validation loss:
  - FedAvg scale: 0.396898;
  - tuned gamma: 0.430577;
  - auto gamma: 0.456850;
  - full gamma: 0.506164;
  - raw FedAvg: 0.431406.

Interpretation: synthetic auto-gamma did not transfer. More strongly, tuned
FedAvg scale beat the gamma arms on held-out target loss. This is a one-shard
real-path gate, not a closed-loop verdict; the remaining E007 worker is still
running at time of recording.

E005 has run locally and returned `GAMMA_FLAT_ON_SYNTHETIC`.

Summary:

- `gamma=None` now estimates shrinkage from split-half reliability of the
  correction vector on each round;
- held-out synthetic ratios over FedAvg:
  - N=8: tuned gamma 1.045, auto-gamma 1.065, oracle 1.171;
  - N=32: tuned gamma 1.143, auto-gamma 1.142, oracle 1.247;
  - N=128: tuned gamma 1.168, auto-gamma 1.163, oracle 1.248.

Interpretation: auto-gamma matching tuned gamma is not a contribution on this
bed because hardcoded `gamma=1` also ties. The gamma objective is flat enough
that automating the knob is not meaningful evidence.

E006 has run locally and returned `GROUP_GAMMA_FLAT_ON_SYNTHETIC`.

Summary:

- compared scalar auto-gamma to 4-bin and 8-bin group-wise auto-gamma;
- held-out synthetic ratios over FedAvg:
  - N=8: tuned 1.054, auto1 1.072, auto4 1.081, auto8 1.084;
  - N=32: tuned 1.129, auto1 1.129, auto4 1.127, auto8 1.126;
  - N=128: tuned 1.138, auto1 1.135, auto4 1.135, auto8 1.135.

Interpretation: group-wise shrinkage also mostly inherits the flat gamma axis.
It gives a small low-N bump but does not justify expanding the action layer.
The gamma line should stop unless a real-path result shows that `full_gamma`
fails.

The durable positive signal is elsewhere: intermediate local points carry
curvature information that server-side secants did not capture on the synthetic
bed. That secant gap, not gamma automation, is the next target.

E008 has run locally and returned `FULL_CORRECTION_UNSTABLE`.

Summary on a nonlinear/heterogeneous synthetic bed:

- local full correction/FedAvg:
  - N=8: 0.019 +/- 0.282;
  - N=32: -1.490 +/- 0.642;
  - N=128: -8.054 +/- 3.262;
- auto-gamma/FedAvg:
  - N=8: 0.746 +/- 0.170;
  - N=32: -0.134 +/- 0.337;
  - N=128: -7.268 +/- 3.270;
- server secant/FedAvg:
  - N=8: 0.880 +/- 0.146;
  - N=32: 1.002 +/- 0.114;
  - N=128: 1.106 +/- 0.091.

Interpretation: raw exact-quadratic correction does not survive nonlinearity /
heterogeneity. It overcorrects badly.

E009 has run locally and returned `CONSERVATIVE_GAMMA_NEEDED`.

Summary on the same nonlinear/heterogeneous bed:

- source-tuned conservative gamma is 0.40/0.35/0.30 at N=8/32/128;
- tuned gamma/FedAvg is 1.460/1.473/1.501;
- auto-gamma/FedAvg is 1.124/0.387/-4.449;
- full gamma/FedAvg is 0.598/-0.398/-4.871;
- server secant/FedAvg is 0.936/0.954/1.066.

Interpretation: the secant gap is real on a bed that can tell actions apart.
Intermediate local paths contain useful correction shape that server-side
secants do not recover. But the current auto-gamma rule is wrong: it measures
estimator agreement, not model-bias risk. The next method must shrink based on
nonlinearity / heterogeneity risk, not only split-half reliability.

E010 has run locally and returned `LOCAL_PATH_SHAPE_BEATS_SCALAR`.

Summary on the same nonlinear/heterogeneous bed:

- N=8:
  - tuned scalar/FedAvg: 1.089 +/- 0.052;
  - local-path shape/FedAvg: 1.391 +/- 0.164;
  - shape minus scalar: 0.302 +/- 0.115;
  - server secant/FedAvg: 0.878 +/- 0.182.
- N=32:
  - tuned scalar/FedAvg: 1.100 +/- 0.032;
  - local-path shape/FedAvg: 1.674 +/- 0.243;
  - shape minus scalar: 0.574 +/- 0.211;
  - server secant/FedAvg: 0.897 +/- 0.183.
- N=128:
  - tuned scalar/FedAvg: 1.066 +/- 0.026;
  - local-path shape/FedAvg: 1.468 +/- 0.099;
  - shape minus scalar: 0.402 +/- 0.074;
  - server secant/FedAvg: 1.043 +/- 0.131.

Interpretation: this is the first clean positive in the local-GP lane. The
local-path correction shape beats both tuned scalar scale and server-side
secants on a nonlinear/heterogeneous bed. The live method is not full GP and
not auto-gamma. It is local-path shape plus conservative, bias-aware shrinkage.

E011-E013 opened a separate event-triggered multisecant branch.

Motivation: under a pay-by-wire cost model, workers should run silently until a
drift/quorum event, then send full useful state. The controller forms centered
worker secant pairs
\[
  s_i=x_i-\bar x,\qquad y_i=g_i-\bar g
\]
and computes the consolidation step. No \(d^2\) object is formed.

Results:

- E011 (`PASSIVE_DRIFT_SECANTS_WEAK`): passive stochastic drift did not produce
  useful spatial secants. One-pair server BB beat spatial arms.
- E012 (`SEEDED_PROBES_NOT_ENOUGH_WITH_SCALAR_BB`): controller-seeded probe
  offsets gave a small spatial-BB bump over scalar extrapolation, but still lost
  to server BB.
- E013 (`MULTISECANT_WORKS_WITH_LOW_NOISE_ONLY`): actual spatial L-BFGS
  multisecants on a rotated block quadratic failed with ordinary stochastic
  secants, but worked in the noiseless sanity check:
  - N=32: space L-BFGS 1.170 vs time L-BFGS 1.098;
  - N=128: space L-BFGS 1.259 vs time L-BFGS 1.103.

Interpretation: the architecture is mechanically sound only when secant SNR is
high enough. Ordinary independent stochastic gradients make the spatial secant
bank too noisy. The next event-triggered experiment must use noise-controlled
secants: same-batch probes, repeated event gradients, a shared probe set, or a
large-batch/large-N regime.

E014 corrected the event-branch operator and returned
`DYNAMICS_AWARE_PROBES_WIN_ON_EXACT_QUADRATIC`.

E013's plain L-BFGS step was the wrong consolidation: K local steps already
apply part of the Newton correction. On a quadratic with no momentum, the
FedAvg displacement is
\[
  \Delta_K=-H^{-1}[I-(I-\eta H)^K]g_0,
\]
so the remaining correction is
\[
  [I-(I-\eta H)^K]^{-1}\Delta_K.
\]

Summary on the rotated-block quadratic with noise-controlled `N*K` probes:

- N=8: tuned scalar 1.054, directed probe 1.332, random probe 1.358;
- N=32: tuned scalar 1.047, directed probe 1.301, random probe 1.301;
- N=128: tuned scalar 1.051, directed probe 1.326, random probe 1.326.

The centered local-trajectory estimate is weaker but improves with worker count
(1.061/1.097/1.214). This keeps the rank/noise concern active while validating
the corrected operator. E014 is a Stage-0 mechanism positive, not a product
claim: it relies on an exact quadratic, known 2D block structure, extra probe
points, and repeated gradients.

Current local-GP state:

- gamma automation is dead;
- plain BB/L-BFGS consolidation is dead;
- local-path shape is live from E010;
- dynamics-aware correction is live from E014;
- the next useful gate is the corrected operator on the nonlinear/heterogeneous
  E008-E010 bed, then a real-path gate only if that survives tuned scalar scale
  and server/time secants.

E015 ran that CPU gate and returned
`DYNAMICS_AWARE_PROBES_SURVIVE_NONLINEAR_BED`.

Summary on the nonlinear/heterogeneous bed:

- N=8: tuned scalar 1.069, random probe 1.306, coordinate probe 1.134;
- N=32: tuned scalar 1.053, random probe 1.445, coordinate probe 1.424;
- N=128: tuned scalar 1.101, random probe 2.009, coordinate probe 2.034.

Server/time secants remain weak (0.911/0.974/1.113), and path-only centered
trajectory pairs select gamma zero and tie FedAvg. The CPU-positive mechanism is
therefore probe-fed local-map correction, not plain worker-scatter
multisecants.

E016 ran the rotation gate and returned
`DYNAMICS_AWARE_SURVIVES_ROTATION`.

Summary on rotated 2D nonlinear blocks:

- N=8: tuned scalar 1.032, random probe 1.254, directed probe 1.268;
- N=32: tuned scalar 1.057, random probe 1.497, directed probe 1.497;
- N=128: tuned scalar 1.132, random probe 2.124, directed probe 2.122.

The best-worker oracle fallback does not explain the result (0.729/0.799/0.876),
server/time secants are unstable and strongly negative, and path-only pairs
again select gamma zero. The E015 gain is therefore not a diagonal-coordinate
artifact on this CPU bed.

E017 ran the shrinkage/fallback gate and returned `RISK_SHRINKAGE_WORKS`.

Summary on the E015 nonlinear/heterogeneous bed:

- N=8: tuned scalar 1.075, tuned probe 1.288, risk probe 1.226;
- N=32: tuned scalar 1.070, tuned probe 1.554, risk probe 1.460;
- N=128: tuned scalar 1.127, tuned probe 2.039, risk probe 1.773.

Risk-aware gamma is selected from split-probe residuals, not an arbitrary gamma
grid, and stays within roughly 5%, 6%, and 13% of the tuned-gamma probe arm. Full
gamma is destructive at N=8, so the residual rule is doing real shrinkage. The
best-worker oracle fallback again loses.

E020 ran the probe-budget / active-design gate and returned
`RANDOM_BUDGET_REQUIRED`.

Summary on the rotated nonlinear block bed with residual shrinkage:

- full `N*K` random probes still beat tuned scalar scale:
  - N=8: full random 1.252 vs scalar 1.061;
  - N=32: full random 1.480 vs scalar 1.072;
  - N=128: full random 1.941 vs scalar 1.115.
- the current `value_weighted` active policy usually underperforms matched
  random probes;
- `uncertainty_value` is the better active variant and sometimes ties matched
  random, but it does not provide a robust budget reduction;
- the cheapest plausible policy is still random probes, around `2N` when
  `N >= 32`, with full `N*K` for very small `N`.

Interpretation: random probe budget is a real product cost. The live mechanism
does not require active design v1, and active design v1 should not replace
random probes.

E023 ran the random-probe budget confirmation gate and returned
`TWO_N_RANDOM_CONFIRMED`.

Summary on the same rotated nonlinear block bed, now with eight held-out reps
and `probe_reps in {1, 4}`:

- at N=32, `2N` random probes beat tuned scalar by 0.331 +/- 0.048 with
  `probe_reps=1` and by 0.394 +/- 0.045 with `probe_reps=4`;
- at N=128, `2N` random probes beat tuned scalar by 0.478 +/- 0.089 with
  `probe_reps=1` and by 0.621 +/- 0.115 with `probe_reps=4`;
- matched against full `N*K` random probes, the `2N` gaps are within the
  requested 10-15% band for N=32/128;
- at N=8, `2N` is still not clearly enough, while full `N*K` remains stable;
- `probe_reps=1` does not fail, although `probe_reps=4` improves the absolute
  ratio when extra gradient repeats are cheap.

Interpretation: the default CPU-bed random budget is now `2N` for `N >= 32`,
with full `N*K` reserved for very small worker counts. Probe repeats are a
quality/variance knob, not a requirement for the basic budget result.

E021 ran the dense low-rank / correlated-noise stress and returned
`DENSE_GEOMETRY_BREAKS_METHOD`.

Summary:

- tuned scalar remains near 1.04-1.05 on the dense bed;
- oracle dense-H correction reaches 2.658/3.304/2.778 at N=8/32/128, so the
  bed has large non-scalar headroom;
- diagonal, random low-rank, learned low-rank, and true-subspace low-rank arms
  do not beat tuned scalar with positive confidence;
- learned low-rank-plus-diagonal with E017 shrinkage is unsafe and can be
  destructive;
- pilot-tuned combined gamma nearly zeros the combined learned update, which is
  itself evidence that the representation needs a stricter trust gate.

Interpretation: dense low-rank-plus-diagonal geometry is the next hard problem.
The current residual shrinkage is not enough under correlated low-rank noise.
Do not ship the combined estimator without held-out loss/probe validation or an
equivalent trust layer that can reject the correction.

E022 ran the dense trust-layer gate and returned
`TRUST_LAYER_RECOVERS_DENSE`.

Summary on the same dense low-rank correlated-noise bed:

- the residual-shrunk learned combo remains destructive:
  - N=8: -6.028 +/- 3.875;
  - N=32: -5.464 +/- 8.616;
  - N=128: -1.793 +/- 2.381.
- rowwise `combo_val_gamma`, selected by held-out mini-loss proxy, beats tuned
  scalar with positive CIs:
  - N=8: +0.038 +/- 0.015;
  - N=32: +0.103 +/- 0.039;
  - N=128: +0.074 +/- 0.020.
- the accept/reject gate mostly prevents harm but lands near scalar;
- oracle-H is still much larger at 2.476/2.661/2.421, so most dense headroom is
  not recovered by the current representation;
- best-worker fallback and server/time secants do not explain the trusted gain.

Interpretation: dense geometry is not intrinsically dead, but it is only safe
behind rowwise held-out validation. The next dense work should improve
subspace/model identification and explicitly account for validation-proxy cost.

E024 ran the causal delayed-gamma control and returned `DELAYED_GAMMA_WEAK`.

Summary:

- same-round `combo_val_gamma` reproduced E022's table;
- applying round `t`'s selected gamma to round `t+1` weakened the effect:
  - N=8: delayed - scalar `+0.010 +/- 0.012`;
  - N=32: delayed - scalar `+0.048 +/- 0.036`;
  - N=128: delayed - scalar `-0.003 +/- 0.080`;
- delayed gamma still avoids the E021 destructive combo, but it does not yet
  preserve the same-round gain across all N;
- the E022 validation grid costs about 17.23 full-forward equivalents per row,
  or 5.74 full-gradient-call equivalents if one backward costs two forwards.

Interpretation: E022 remains a quality positive for same-round validation, but
winner's curse is now an active gate. Do not cite E022 as a settled
dense-method claim until E026 delayed/stale/shuffled validation lands.

E025 ran the validation-cost pricing gate and returned
`VALIDATION_COST_SURVIVES`.

Summary:

- the full E022 8-rep validation grid costs about 17.23 full-forward
  equivalents per row, or 5.74 gradient-call equivalents if one backward costs
  two forwards;
- same-round `combo_val_gamma_reps_8` beats tuned scalar and the `N+1`
  extra-worker scalar rival:
  - N=8: val8 - scalar `+0.049 +/- 0.033`;
  - N=32: val8 - scalar `+0.116 +/- 0.023`;
  - N=128: val8 - scalar `+0.134 +/- 0.021`;
- spending the validation-equivalent budget on extra fit probes does not
  explain the gain; that arm remains destructive on the dense bed;
- one or two mini-loss validation reps recover most of the same-round benefit.

Interpretation: the same-round trust gate is not merely buying its win by
spending compute that scalar FedAvg or extra fit probes would use better. This
settles the cost objection for the same-round synthetic bed. It does not settle
winner's curse or causal persistence; E026 remains the binding dense gate.

E026 ran the higher-power delayed-gamma replication and returned
`SAME_ROUND_SELECTION_EXPLAINS`.

Summary:

- same-round validation-gamma beats scalar and shuffled on the drop-two subset;
- lag-1 is weaker than same-round everywhere, does not beat shuffled with a
  positive CI, and only clearly beats scalar at N=32;
- lag-2 is weaker still and does not clear scalar/shuffled;
- gamma autocorrelation is small or negative.

Interpretation: dense geometry still has oracle headroom, and same-round
validation can act as a safety/selection layer. But E022/E025 do not establish a
causal reusable trust signal across rounds.

E027 ran the global information query-policy gate and returned
`FIXED_SCHEDULE_EXPLAINS`.

Summary:

- the greedy global VoI policy beat fixed FedAvg by a large paired margin;
- it lost to a fixed round-robin exploit/probe/replicate schedule by
  `0.0221 +/- 0.0049` final objective;
- oracle policy still had headroom.

Interpretation: shared information state is still a useful framing, but this
one-step greedy query allocator mostly rediscovered "probe a lot" and did not
beat a simple fixed schedule.

E027 follow-up `ushmoo_e027_noise_sweep_oracle.py` returned
`PROBE_CROSSOVER_EXISTS`.

Summary:

- when curvature must be learned from the same probe primitive as E027, a
  probe-bearing fixed mix beats the best no-probe mix with positive paired CI
  at every swept noise scale from `0.05` to `2.50`;
- best mixes usually spend about 24-49% of actions on probes, depending on
  noise scale;
- no-probe arms keep curvature RMSE at the prior error (`0.90496`), while
  probe-bearing arms reduce it sharply.

Interpretation: E027 is not a bed where probes are intrinsically worthless. The
original adaptive policy failed because its acquisition was wrong. Also, in the
E027 implementation, `exploit` and `replicate` are both base-gradient
observations within a round; do not overinterpret their split. The meaningful
axis is base-gradient budget versus probe budget.

E027 follow-up `ushmoo_e027_bayes_risk_acquisition.py` returned
`BAYES_RISK_FIXES_GREEDY_NOT_FIXED`.

Summary:

- Bayes-risk acquisition beats original greedy at higher noise:
  - noise `1.00`: `+0.01517 +/- 0.00528`;
  - noise `2.50`: `+0.03317 +/- 0.01621`.
- It does not beat the fixed-mix frontier:
  - noise `1.00`: `-0.01277 +/- 0.00655`;
  - noise `2.50`: `-0.04134 +/- 0.01261`.

Interpretation: replacing entropy/D-optimality with a consolidation-risk proxy
is directionally right but not enough. The next query-control bar is not
"improve on original greedy"; it is "beat the fixed base/probe frontier."

`ushmoo_validation_lottery_20261004.py` ran the first cheap-validation toy and
returned `SELF_DROP_EXPLAINS_VALIDATION_LOTTERY`.

Setup:

- workers propose ordinary training-gradient updates;
- each reports a biased/noisy local self-drop;
- each randomly samples validation budget `m in {0, 1, 4, 16}` with expected
  `0.840` validation forwards per worker;
- matched-cost scalar gets `41` workers instead of `32`, using the rule one
  training step costs about three validation forwards.

Result:

- raw self-drop vs sixteen-forward diagnostic correlation: `0.724`;
- after lottery-tail per-tag calibration, self posterior vs true value
  correlation: `0.999`;
- lottery posterior vs true value correlation: `0.999`;
- calibrated self-drop beats matched-cost scalar by `0.00768 +/- 0.00320`;
- ongoing per-update lottery does not beat calibrated self-drop:
  `-0.00007 +/- 0.00007`.

Interpretation: on this toy, the value of randomized validation is calibration,
not per-update measurement. A small validation tail can make the nearly-free
training self-drop usable, after which most workers should probably run
`m=0`. The next version must add calibration drift or hidden contexts; otherwise
ongoing validation is explained by calibrated self-drop.

`ushmoo_span_oracle_check.py` measured the missing rung-2 ceiling on the E026
bed and returned `SIMPLEX_SPAN_ORACLE_LIVE`.

Summary:

- at `N=32`, tuned scalar was `1.2306 +/- 0.0012`;
- row-oracle scalar was `3.2658 +/- 0.0413`;
- simplex worker reweighting was `4.1203 +/- 0.0323`;
- simplex minus row-oracle scalar was `0.8545 +/- 0.0717`.

Interpretation: same-round action-value headroom is real, and per-worker
reweighting adds headroom beyond a rowwise scalar. But much of the rung-2
ceiling is step-size/action-value selection, not dense geometry.

`ushmoo_linearization_gap_check.py` tested whether one shared validation
gradient explains per-worker value scoring and returned
`SECOND_ORDER_VALUE_NEEDED`.

Summary at `N=32, alpha=1`:

- pooled correlation: `0.9014`;
- mean within-round correlation: `0.5259 +/- 0.0241`;
- top-20 overlap: `0.4880 +/- 0.0227`;
- linear-best/true-best: `0.4327 +/- 0.1669`.

Interpretation: pooled first-order correlation is misleading. The server needs
within-round worker ranking, and the shared-gradient score misses too much of
that ranking. A forward probe set is buying nonlinear action-value evidence on
this bed.

`bandha_probe_set_rung2_20261004.py` ran the direct probe-set action-value
experiment and returned `PROBE_SET_RUNG2_LIVE`.

Setup:

- workers propose ordinary K-step local-SGD endpoint updates on the E026 bed;
- a shared cheap probe set scores
  `L_probe(theta) - L_probe(theta + alpha_score u_i)` for each worker update;
- the server softmax-reweights worker updates from those scalar value labels;
- controls are tuned scalar FedAvg, a shared-gradient linearization scorer,
  shuffled probe labels, and a true-value softmax diagnostic.

Result:

- at `N=32`, tuned scalar was `1.2300 +/- 0.0009`;
- shared-gradient softmax was `1.2353 +/- 0.0105`;
- shuffled `m=4` probe labels scored `1.4521 +/- 0.0022`;
- best probe arm `probe_mean_m16` scored `1.5738 +/- 0.0245`;
- best probe minus scalar: `0.3438 +/- 0.0241`;
- best probe minus shared-gradient: `0.3384 +/- 0.0234`;
- best probe minus shuffled: `0.1217 +/- 0.0237`.

Interpretation: the probe-set idea is now the live positive branch. The probe
set is not reconstructing a dense local map; it is buying nonlinear scalar
action-value evidence. The remaining burden is cost accounting and transfer:
prove the same value labels beat matched-cost extra SGD and survive real-path
workloads.

E028 ran the heterogeneous async query-channel gate and returned
`TAGGED_GLOBAL_POSTERIOR_LIVE`.

Summary:

- homogeneous regime tied averaging exactly (`0.04634` vs `0.04634`);
- heterogeneous final objective improved from equal averaging `0.33736` to
  tagged posterior `0.08990`, with oracle at `0.05158`;
- tagged-vs-true reliability correlation in the heterogeneous regime was
  `0.941`, compared with `0.487` for noise-only reliability.

Interpretation: tagged async/global evidence can beat homogeneous averaging
under strong channel heterogeneity while degrading gracefully to averaging when
workers are actually homogeneous. This is the current clean positive.

E029 first returned `BOOTSTRAP_LEGITIMACY_FAIL`, then its source-bundle requeue
returned `NO_GPU_HARNESS`.

The first worker lacked `/data/users/atavory/local_gp` and the Mr.
Gitties/context bootstrap needed to clone it. The requeue used the staged source
bundle but had no usable GPU harness: ad hoc Python lacked `torch`, and the
bundle had no `BUCK`, `BUCK.v2`, or `TARGETS` file. Its CPU fallback is
diagnostic only.

E030 first returned `BOOTSTRAP_LEGITIMACY_FAIL`, then its source-bundle requeue
ran on real PyTorch CUDA and returned `SCALAR_EXPLAINS_REAL_PATH`.

Summary:

- compute: `REAL_GPU`, `torch=2.14.1+cu130`, device `NVIDIA PG509-210`;
- workload: torch two-layer binary classifier, real local-SGD worker paths,
  `D=97`, input 10, hidden 8, `K=4`, `eta=0.04`;
- main local-map arms used `2N` seeded one-sided probes;
- residual-shrunk local map minus tuned scalar: `-5.460 +/- 1.570`;
- validation-shrunk local map minus tuned scalar: `-0.497 +/- 0.909`;
- validation-shrunk local map minus server/time secant: `0.413 +/- 0.985`;
- validation-shrunk minus best of scalar/server: `-0.654 +/- 0.964`.

Interpretation: the real-GPU local-map gate did not support a product claim.
Tuned scalar FedAvg matched or beat the local-map arms. This strengthens the
case for action-value/probe-set scoring rather than dense local-map estimation
on real paths.

E031 first failed the same bootstrap gate, then its source-bundle requeue ran
and returned `REAL_TRUST_NO_SIGNAL` with `CPU_FALLBACK`.

Summary:

- the bundle had no Buck target and ad hoc Python lacked `torch`;
- a pure-Python two-layer classifier fallback was run with real local-SGD paths;
- same-round minus scalar was `-9.880 +/- 2.083`;
- lag-1 minus scalar was `-9.913 +/- 2.147`;
- lag-1 minus shuffled was `-0.052 +/- 0.152`.

Interpretation: on this CPU fallback, neither same-round nor delayed path-derived
trust cleared scalar/shuffled controls. This is useful negative evidence for the
fallback workload, but it is not a GPU/Buck2 real-path verdict.

Updated local-GP state:

- gamma automation from the old aggregator line is dead;
- plain BB/L-BFGS consolidation is dead;
- path-only worker scatter is not enough;
- probe-fed dynamics-aware local-map correction is alive across exact,
  nonlinear, and rotated nonlinear CPU beds;
- risk-aware residual shrinkage has replaced target gamma tuning on the current
  CPU bed;
- random probe budget remains material; active design v1 is not enough to cut
  it reliably, but E023 confirms `2N` random probes as the default for
  `N >= 32` on the rotated CPU bed;
- dense low-rank-plus-diagonal geometry has oracle headroom; E022/E025 show a
  held-out validation-gamma layer can recover a small same-round safe gain at
  priced cost, but E026 says the delayed/reusable trust signal does not
  replicate;
- tagged heterogeneous async evidence is live from E028 and should now face
  stronger distributed-training controls;
- the controller should use empirical-Bayes local-map fitting plus stochastic
  trust-region adaptation, not Optuna/TPE over high-dimensional actions;
- E030 is the first real-GPU local-map verdict and is negative against tuned
  scalar;
- E029 remains a harness failure on its worker; E031 produced only a CPU
  fallback result;
- the next useful work is hardening the live same-round action-value/probe-set
  branch, stronger E028 controls, and costed validation/probe ledgers.

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
3. Do not continue the gamma line unless `full_gamma` fails. Any E007 positive
   must beat or at least improve over `full_gamma`, not merely tuned gamma.
4. Replace E004 with a self-contained real-path gate that does not depend on
   worker GitHub/Mr. Gitties access.
5. In the real-path gate, compare FedAvg/source-tuned scale, pilot-tuned gamma,
   source-free auto-gamma, and full gamma. `FULL_GAMMA_EXPLAINS` is a null for
   gamma estimation.
6. E010/E015-E017 show the method target: noise-controlled probes estimate a
   dynamics-aware local-map correction that beats tuned scalar scale on CPU
   synthetic beds.
7. Richer real-data logging remains needed: probe gradients or shared-probe
   function-space features, not only scalar endpoint summaries.
8. Event-triggered multisecant branch: stop plain BB/L-BFGS consolidation; use
   the K-step local-map operator.
9. Treat E030's real-GPU `SCALAR_EXPLAINS_REAL_PATH` result as the current
   local-map real-path verdict; do not spend more time on dense local-map gamma
   polishing without a new observation model.
10. Use residual-shrunk random probes as the default budget policy for now:
    around `2N` when `N >= 32`, full `N*K` for very small `N`.
11. For dense low-rank-plus-diagonal geometry, keep the held-out validation
    gate as a same-round safety/selection layer. E026 says not to claim delayed
    causal trust from it.
12. Do not run a closed-loop optimizer before a held-out real-path/control gate
   passes against tuned scalar scale and best-worker fallback.
13. Harden the same-round action-value / probe-set rung-2 experiment: matched
    extra-SGD cost accounting, calibration drift, and real-path transfer.
14. Rewrite the paper around the reset thesis:
    - Double Fan / same-point cloud momentum is archived as a failed approach;
    - FedAvg is the infinite-lengthscale / averaging baseline;
    - local drift/probes estimate local-map curvature unavailable to
      server/time secants;
    - the product path is dynamics-aware local-map correction plus a
      trust-region controller;
    - dense correlated geometry is the current failure boundary.
