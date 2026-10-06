# Parallel Hyperparameter Probes

Status: live synthesis, 2026-10-06.

This note supersedes the narrow "rank workers with probes" framing.  The useful
object is not a set of exchangeable workers.  The useful object is a parallel
experiment over optimizer hyperparameters.

## One Sentence

Use aligned workers as a live experimental design over inner optimizer
hyperparameters, use cheap probe sets to estimate the value of the returned
candidate updates, and aggregate through a conservative posterior rule that
shrinks back to averaging when the evidence is weak.

## Deployment Claim

The useful target is **not** "beat a free tuned default."  A tuned default is
not free.  It is a sweep over whole training runs, repeated whenever `K`,
model, data, batch size, optimizer, or training phase changes.

The useful target is:

> Match the per-regime tuned schedule without running a new tuning sweep, at
> sub-1% to low-single-digit probe overhead, and collapse to the default when
> the probe signal is uninformative.

This already has one strong synthetic support point from the earlier
round-level `(alpha,beta)` probe-control branch:

| K | K=8 schedule transferred | per-K retuned schedule | sparse probe argmax, no retuned design | probe overhead |
|---:|---:|---:|---:|---:|
| 8 | 6.676 | 6.668 | 6.308-6.357 | 0.41%-1.22% |
| 16 | 0.998 | 3.918 | 3.676-3.733 | 0.20%-0.61% |
| 32 | -9.745 | 2.406 | 2.204-2.234 | 0.10%-0.30% |
| 64 | -28.857 | 1.613 | 1.545-1.564 | 0.05%-0.15% |

Readout:

- the schedule tuned at `K=8` fails catastrophically when transferred to
  `K=16,32,64`;
- the per-K retuned schedule recovers, but it costs a new sweep for each K;
- the sparse probe controller uses the same 12 design points across K and
  recovers roughly 85%-96% of the per-round oracle gap at tiny per-run cost.

That is the clean "you should use this" claim: probes buy online retuning
inside the training run.  E071-E078 now give the first real-GPU support for the
same story.  The important update is E078: after isolating only the cells where
the selector actually uses probes (`K<=8`), the rule beats both shuffled probes
and fixed controls with CIs excluding zero:

```text
selector - shuffled = +0.242 +/- 0.081
selector - fixed    = +0.246 +/- 0.080
```

That is the current actionable positive.  It is not a large-K result yet, and
it is not a claim that probe content beats every tuned control family.  It says
the probe labels carry deployable value in the probe-active low-K regime.  The
next job is to reduce/measure the cost frontier and then decide whether this
becomes a budgeted component of the optimizer.

## Why This Is Different

The failed stationary setup was:

```text
same theta_t
same K
same inner LR
same optimizer
N nearly exchangeable worker updates
probe tries to rank them
```

That is a bad statistical problem.  Uniform averaging is already close to the
right answer, and the gaps between workers are mostly minibatch luck.

The new setup is:

```text
same theta_t
aligned round
worker groups run different inner hyperparameters
probe scores the resulting group-mean candidate updates
server learns/uses a value surface over hyperparameters
```

The worker differences are now first-order design differences, not accidental
sampling noise.  The probe asks "which hyperparameter arm is valuable in this
training phase?" rather than "which worker got lucky?"

## Favorable Results In Hand

### E059: Individual LR Arms Create Headroom But Do Not Extract

E059 deliberately spread inner learning rates across individual workers.  It
created large oracle headroom, but the per-worker probe rule did not extract it:
the proxy noise was still roughly four times the adjacent candidate gap.

Interpretation: non-exchangeability is necessary but not sufficient.  Ranking
individual workers is still too noisy.

### E060: Grouped LR Arms Give A Real Stationary Positive

E060 grouped workers by inner LR and probed the group means.

Key held-out rows:

| spread | groups | fixed | oracle subset | probe policy | minus fixed | minus shuffled | cost |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `1` | 5.654 | 5.654 | 5.654 | 0.000+/-0.000 | 0.000+/-0.000 | 0.22% |
| 4 | `0.5,1,2` | 5.018 | 8.597 | 5.185 | 0.167+/-0.153 | 0.228+/-0.129 | 1.34% |
| 16 | `0.25,1,4` | 3.226 | 7.644 | 3.531 | 0.304+/-0.175 | 0.691+/-0.325 | 1.34% |

The important control is spread 1: the method collapses exactly to no effect
when the arms are identical.  It is not hallucinating gains under
exchangeability.

### E061: K Sweep Confirms The Mechanism, With Fragility

E061 repeats the grouped-LR admission across `K in {4,8,16,32}`.

Key held-out rows:

| K | spread | fixed | oracle subset | pilot-best | minus fixed | minus shuffled | cost |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 7.774 | 12.407 | 8.446 | 0.672+/-0.634 | 1.034+/-0.898 | 2.68% |
| 4 | 16 | 5.547 | 9.756 | 6.143 | 0.597+/-0.610 | 1.103+/-1.070 | 2.68% |
| 8 | 16 | 3.209 | 5.826 | 3.441 | 0.231+/-0.066 | 0.460+/-0.226 | 1.34% |
| 16 | 16 | 1.057 | 2.706 | 1.214 | 0.156+/-0.027 | 0.237+/-0.088 | 0.67% |
| 32 | 4 | 1.112 | 2.653 | 1.187 | 0.075+/-0.058 | 0.157+/-0.132 | 0.33% |
| 32 | 16 | 1.335 | 4.185 | 1.332 | -0.002+/-0.011 | 0.029+/-0.025 | 0.17% |

Read precisely:

- Positive at `K=8` and `K=16`, especially with spread 16.
- Correct zero effect when spread is 1.
- `K=32, spread=16` exposes policy fragility: the pilot-chosen damped rule
  fails even though the best sweep remains positive (`+0.183+/-0.159`).

So the mechanism is real but under-extracted.  This is not yet a polished
method.

### E062: Online Tracker Does Not Improve Raw Group Probes

E062 tested a quadratic/EMA response tracker over the same LR groups.  This was
the first attempt to turn the grouped arms into an online hyperparameter
surface model.

Decision: `FROZEN_GROUP_PROBE_LIVE_TRACKER_TIES`.

Readout:

- raw grouped probes remained positive;
- the tracker mostly selected the same `group_mean2_l40_t2` rule as raw;
- when it differed at `K=32`, the improvement over raw was tiny
  (`+0.006+/-0.024`, `+0.007+/-0.062`);
- therefore GP/response-surface machinery is not justified yet by these rows.

The important correction is that "fit a surface" is not automatically better
than "score three frozen group means."  The next model has to beat raw grouped
probes, not merely beat the fixed default arm.

### E063: Frozen Group Probes Replicate Across K

E063 froze the E060/E061 raw rule (`group_mean2_l40_t2`) and repeated it with
12 held-out seeds.  No pilot-selected policy was allowed to move.

Decision: `RAW_GROUP_PROBE_REPLICATES`.

| spread | K | fixed | oracle | raw | raw minus fixed | raw minus shuffled | capture | cost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 5.055 | 5.055 | 5.055 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.45% |
| 16 | 8 | 2.988 | 5.931 | 3.239 | 0.251+/-0.153 | 0.447+/-0.207 | 0.072+/-0.041 | 1.34% |
| 1 | 16 | 2.832 | 2.832 | 2.832 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.22% |
| 16 | 16 | 1.685 | 4.570 | 1.847 | 0.162+/-0.118 | 0.439+/-0.130 | 0.058+/-0.040 | 0.67% |
| 1 | 32 | 1.554 | 1.554 | 1.554 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.11% |
| 16 | 32 | 1.090 | 5.898 | 1.361 | 0.271+/-0.147 | 0.824+/-0.303 | 0.053+/-0.027 | 0.33% |

This is the cleanest favorable stationary result so far:

- exact zero effect under exchangeability (`spread=1`);
- positive held-out gains at `K=8,16,32` when the LR arms are deliberately
  non-exchangeable;
- positive shuffled-control gaps;
- sub-1.5% probe cost.

It is still a small-capture result, not a "smash everything" result.  The value
is repeatability and collapse behavior: the signal is real, conservative, and
not fabricated by noisy ranking.

### E064: Sub-1% Probe Cost Frontier Is Live

E064 kept the E063 method fixed (`lambda=0.4`, `tau=2.0`) and swept only the
probe budget: examples per forward slice and number of independent slices.

Decision: `SUB1_PERCENT_COST_FRONTIER_LIVE`.

Representative rows:

| K | batch | slices | cost | raw minus fixed | raw minus shuffled | capture |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 48 | 1 | 0.08% | -0.008+/-0.166 | 0.405+/-0.262 | -0.007+/-0.044 |
| 8 | 96 | 4 | 0.67% | 0.190+/-0.161 | 0.647+/-0.336 | 0.051+/-0.043 |
| 8 | 384 | 1 | 0.67% | 0.185+/-0.163 | 0.611+/-0.322 | 0.051+/-0.043 |
| 16 | 48 | 1 | 0.04% | -0.030+/-0.101 | 0.273+/-0.147 | -0.008+/-0.041 |
| 16 | 48 | 4 | 0.17% | 0.168+/-0.121 | 0.525+/-0.255 | 0.048+/-0.034 |
| 16 | 192 | 4 | 0.67% | 0.207+/-0.148 | 0.535+/-0.273 | 0.059+/-0.037 |
| 32 | 48 | 1 | 0.02% | -0.027+/-0.116 | 0.260+/-0.143 | -0.011+/-0.047 |
| 32 | 48 | 4 | 0.08% | 0.183+/-0.121 | 0.483+/-0.216 | 0.044+/-0.034 |
| 32 | 192 | 4 | 0.33% | 0.210+/-0.132 | 0.497+/-0.228 | 0.048+/-0.044 |

This is the cost shape the project needed:

- one tiny forward slice is too noisy and can be negative;
- several sub-1% budgets recover the E063 edge;
- increasing beyond that gives only small incremental returns;
- the shuffled-control gap stays positive, so the probe is not merely adding
  variance.

The strongest practical default from this table is **4 slices of 48-96 examples
per group**, depending on K.  That lives around `0.08%-0.67%` of a round in
this setup.

### E065: LR x Momentum Creates Headroom, Extraction Still Thin

E065 added a second designed hyperparameter axis.  Worker groups ran the 9-arm
grid:

```text
inner_lr_multiplier in {0.5, 1.0, 2.0}
inner_momentum in {0.0, 0.5, 0.9}
```

All workers used the same `K`; only the optimizer hyperparameters varied.

Decision: `SECOND_DIM_GROUP_PROBE_BEATS_PARALLEL_AVERAGE`.

| K | arms | cost | fixed all | default only | oracle subset | raw | raw minus fixed | raw minus default | raw minus shuffled | capture |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 9 | 0.89% | 1.924+/-0.736 | 1.686+/-1.893 | 9.599+/-3.385 | 2.157+/-0.682 | 0.233+/-0.280 | 0.471+/-1.906 | 0.283+/-0.293 | 0.036+/-0.038 |
| 16 | 9 | 0.45% | 1.127+/-0.220 | 1.296+/-1.131 | 9.366+/-3.923 | 1.445+/-0.282 | 0.318+/-0.274 | 0.150+/-1.215 | 0.400+/-0.316 | 0.041+/-0.026 |

Read this carefully:

- the 2D arm grid creates very large oracle headroom;
- the frozen grouped-probe rule beats averaging the parallel arms;
- the captured fraction is still only about 4%;
- it does **not** cleanly beat the tuned default-only arm because that control
  is very noisy and overlaps the raw arm.

So the "use parallelism as hyperparameter search" idea is live, but the current
extractor is still too primitive for the 2D case.  A GP/response model is now
motivated by a measured failure mode: there is a smooth arm space with large
oracle headroom, and same-round soft weighting captures only a thin slice.

### E066: Default-Only Control Explains The Stationary LR Win

E066 asked the control question that E063/E064 did not answer: if the default
inner LR arm is known and separately tuned, does the grouped probe still win?

Decision: `LR_GROUP_PROBE_BEATS_PARALLEL_AVERAGE_ONLY`.

| K | cost | fixed all | default only | oracle subset | raw | raw minus fixed | raw minus default | raw minus shuffled | capture |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.67% | 3.482+/-0.596 | 5.313+/-1.185 | 7.056+/-1.290 | 3.719+/-0.487 | 0.238+/-0.160 | -1.593+/-1.415 | 0.467+/-0.415 | 0.072+/-0.035 |
| 16 | 0.33% | 1.451+/-0.755 | 5.884+/-4.853 | 7.283+/-6.143 | 1.763+/-0.469 | 0.312+/-0.295 | -4.122+/-5.293 | 0.798+/-1.016 | 0.070+/-0.027 |
| 32 | 0.17% | 1.000+/-0.000 | 4.362+/-4.419 | 4.020+/-2.004 | 1.100+/-0.041 | 0.100+/-0.041 | -3.262+/-4.428 | 0.337+/-0.219 | 0.049+/-0.020 |

Readout:

- grouped probes consistently rescue the intentionally bad parallel average;
- grouped probes do **not** beat a separately tuned default-only arm;
- the positive stationary LR result is therefore not "better than a known good
  optimizer setting";
- it is evidence that probes can manage exploratory parallel arms when we do
  not want to spend whole runs tuning each arm.

This is a useful boundary, not a method win.  The next extractor has to be
default-anchored: keep the default arm unless probe evidence says an exploratory
arm is better.  Softly averaging all exploratory arms is exactly the wrong
thing when one default arm is already strong.

### E067: Default-Anchored Admission Is Safer, But Still Negative

E067 tested the direct repair for E066.  Instead of shrinking toward the
all-worker prior, it used the default LR group as a one-hot fallback and only
mixed toward exploratory LR groups through the same frozen probe damping.

Decision: `DEFAULT_ANCHORED_PROBE_NEGATIVE`.

| K | cost | default only | all-prior raw | anchored | anchored minus default | anchored minus shuffled | capture vs default |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.67% | 2.854+/-2.343 | 1.303+/-3.063 | 2.159+/-2.358 | -0.694+/-2.172 | 2.481+/-3.386 | -0.021+/-0.186 |
| 16 | 0.33% | 1.331+/-1.423 | 1.114+/-1.096 | 1.244+/-1.733 | -0.087+/-0.956 | 1.323+/-2.358 | -0.224+/-0.342 |
| 32 | 0.17% | 11.280+/-19.975 | 0.549+/-1.249 | 8.892+/-14.249 | -2.387+/-5.748 | 3.039+/-5.560 | -0.267+/-0.369 |

Readout:

- anchoring reduces the damage from bad exploratory arms, especially at K=32
  (`all-prior - default = -10.731`, anchored `-2.387`);
- it does not beat default-only at any K;
- the shuffled-control gaps remain positive but very noisy;
- the oracle gap remains large, so the problem is extraction, not absence of
  better candidate arms.

This closes the stationary default-known branch for the current same-round
probe extractor.  In a stationary regime with a strong default, probes are not
yet good enough to justify departures from that default.  The viable roles are
now narrower: unknown/default-stale hyperparameter search, drift/nonstationary
tracking, or a substantially better value proxy.

### E068: Doubling Probe Budget Does Not Rescue Default-Anchored Admission

E068 repeated E067 with the largest clean probe budget available in the current
two-select-draw bed: `probe_batch=192`, `slices=4`.

Decision: `DEFAULT_ANCHORED_PROBE_NEGATIVE`.

| K | cost | default only | all-prior raw | anchored | anchored minus default | capture vs default |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.34% | 2.854+/-2.343 | 1.277+/-3.066 | 2.235+/-2.305 | -0.619+/-2.026 | -0.024+/-0.179 |
| 16 | 0.67% | 1.331+/-1.423 | 1.149+/-0.981 | 1.344+/-1.555 | 0.013+/-0.774 | -0.224+/-0.341 |
| 32 | 0.33% | 11.280+/-19.975 | 0.559+/-1.185 | 8.851+/-14.270 | -2.429+/-5.725 | -0.273+/-0.371 |

Compared with E067 (`96 x 4`), the doubled budget barely changes the conclusion:

- K=8 improves only from `-0.694` to `-0.619` versus default;
- K=16 moves from `-0.087` to `+0.013`, a tie;
- K=32 remains strongly below default.

So the stationary/default-known failure is not primarily "probes were too
cheap."  It is a proxy/extraction failure: the held-out probe signal can see
that bad exploratory arms exist, but it does not reliably identify departures
from a strong default that improve final held-out value.

### E069: Bad/Stale Default Does Not Rescue The Current Extractor

E069 inverted the E066-E068 control: instead of anchoring on the known default
LR group (`1.0`), it anchored on a deliberately stale/wrong low-LR group
(`0.25`) and asked whether probes could justify leaving it.

Decision: `DEFAULT_ANCHORED_PROBE_NEGATIVE`.

| K | cost | stale-anchor only | oracle subset | anchored | anchored minus stale anchor | capture vs stale anchor |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.67% | 2.049+/-1.330 | 9.697+/-3.126 | -1.610+/-7.113 | -3.659+/-7.755 | -0.233+/-0.516 |
| 16 | 0.33% | 2.171+/-3.213 | 9.035+/-6.251 | 1.611+/-1.134 | -0.561+/-2.980 | 0.034+/-0.212 |
| 32 | 0.17% | 7.549+/-14.555 | 24.863+/-33.751 | 6.171+/-9.208 | -1.378+/-5.527 | 0.117+/-0.102 |

This kills the naive "bad default" rescue for the existing softmax probe
extractor.  The branch still has oracle headroom, but the probe forward
differences do not extract it.  The next test should change the measurement,
not the anchoring rule: use a validation-gradient marginal-value score
`<g_val, u_i>` so one backward pass gives scores for all arms.

### E070: Validation-Gradient Marginal Value Does Not Rescue Extraction

E070 made that measurement change.  It replaced per-arm forward finite
differences with one validation backward per slice:

```text
g_V = grad L_V(theta)
score_i = <g_V, alpha * (u_i + beta v)>
```

Decision: `VALIDATION_GRADIENT_PROBE_NEGATIVE`.

| K | cost | default only | oracle subset | forward anchor | forward minus default | gradient anchor | gradient minus default | gradient minus forward | gradient capture |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.67% | 3.189+/-2.177 | 8.116+/-1.474 | 3.845+/-1.755 | 0.656+/-0.914 | 2.923+/-2.632 | -0.266+/-2.173 | -0.922+/-1.414 | 0.057+/-0.292 |
| 16 | 0.33% | 1.627+/-1.681 | 6.804+/-2.712 | 2.249+/-1.258 | 0.622+/-0.611 | 1.761+/-1.375 | 0.134+/-1.305 | -0.488+/-0.841 | 0.120+/-0.133 |
| 32 | 0.17% | 0.197+/-1.357 | 6.026+/-2.451 | 1.309+/-0.761 | 1.112+/-0.830 | 1.317+/-0.689 | 1.120+/-0.826 | 0.008+/-0.169 | 0.171+/-0.058 |

Readout:

- the first-order marginal-value score does **not** beat the forward extractor;
- it does not produce a robust default-beating result at K=8 or K=16;
- K=32 is directionally good for both forward and gradient anchors, but the
  paired CI still includes zero;
- therefore the shared blocker is not simply "we should have used one
  validation gradient."  The bottleneck is the decision/extraction rule for
  turning weak same-round value evidence into departures from the default.

This matters because it closes the last cheap measurement swap.  The live
method cannot just change the probe statistic; it needs either a better
decision problem (unknown/stale defaults, drift, mandatory exploration) or a
proper response model over the hyperparameter surface.

### E071-E073: Real-GPU Robust Outer-Control Probes Replicate With K-Awareness

E039 found a useful failure mode: the target-good outer-control action was
often in the top few probe-ranked candidates, but brittle top-1 selection still
lost.  E071 tested robust selectors on the real PyTorch/CUDA path: top-k means,
soft losses, and guarded variants over the `scaled` and `hybrid` candidate
designs.

E071 decision: `TOPK_PROBE_BEATS_TARGET_CONTROLS`.

Best E071 rows:

| K | best frozen design/policy | best control | selector | selector minus control | selector minus shuffled | capture |
|---:|---|---:|---:|---:|---:|---:|
| 8 | `scaled/soft_std2` | 2.801 | 4.753+/-1.046 | 1.952+/-1.324 | 0.205+/-0.432 | 0.518+/-0.251 |
| 32 | `scaled/top3_mean` | 0.323 | 1.359+/-0.497 | 1.036+/-1.049 | 1.242+/-1.388 | 0.185+/-0.522 |

That result was encouraging but post-hoc over selector family, so E072 froze
the single `scaled/soft_std2` rule and repeated on fresh seeds.

E072 decision: `TOPK_PROBE_STILL_LOSES`.

| K | frozen policy | best control | selector | selector minus control | selector minus shuffled |
|---:|---|---:|---:|---:|---:|
| 8 | `scaled/soft_std2` | 4.882+/-1.433 | 5.425+/-0.781 | 0.543+/-0.688 | 0.093+/-0.157 |
| 32 | `scaled/soft_std2` | 1.152+/-1.340 | 1.134+/-1.200 | -0.017+/-0.195 | 0.387+/-0.534 |

This said the selector cannot be K-invariant.  That is not fatal: `K` is known
to the outer optimizer.  E073 therefore froze the K-aware rule that E071
identified:

```text
K <= 8:  scaled/soft_std2
K >  8:  scaled/top3_mean
```

E073 decision: `K_AWARE_TOPK_PROBE_REPLICATES`.

| K | frozen K-aware policy | best control | selector | selector minus control | selector minus shuffled | capture |
|---:|---|---:|---:|---:|---:|---:|
| 8 | `soft_std2` | 5.184+/-0.804 | 5.638+/-0.495 | 0.454+/-0.329 | 0.098+/-0.112 | 0.156+/-0.306 |
| 32 | `top3_mean` | 1.691+/-0.327 | 1.720+/-0.285 | 0.029+/-0.144 | 0.975+/-0.446 | 0.063+/-0.177 |

Readout:

- this is the first real-GPU, fresh-seed, frozen-rule positive for the
  probe-control branch;
- the win is modest: K=8 clears the control comparison, while K=32 is a
  positive-mean tie against controls;
- the shuffled-control gap is positive at both K values and clean at K=32;
- one K-invariant selector failed, so the method needs known-regime
  conditioning.  That is acceptable for `K`, because `K` is an explicit control
  variable, not hidden state.

This is now the live method claim:

> Use round-aligned probe sets to perform low-cost online retuning of outer
> optimizer controls, with K-aware robust selection over a fixed candidate
> design.

## Current Interpretation

The favorable result is:

> Probes can extract useful value when the action space is deliberately
> structured and low-dimensional: group-level optimizer arms in synthetic
> stationary beds, and round-level outer-control actions on the real GPU path.

The limitation is:

> Same-round worker/arm weighting still captures only a small fraction of
> oracle headroom and does not beat a known, separately tuned default arm.  The
> successful real-path selector is K-aware and modest, not a universal adaptive
> optimizer.

E062 says the first online tracker does not solve that limitation.  E064 says
the frozen grouped-probe method is cheap enough to matter.  E065 says a second
hyperparameter dimension creates much more headroom but needs a better
extractor.  E066 says the extractor must be default-anchored rather than a
free soft average over all exploratory arms.  E067 says default anchoring is
safer but still not enough to beat a known strong default.  E068 says merely
buying more same-round probe precision does not rescue that branch.  E069 says
even a stale/wrong default does not rescue the current forward-difference
extractor.  E070 says a validation-gradient marginal-value proxy also does not
rescue the default-anchored extractor.  E071-E073 say robust K-aware
outer-control selection can work on the real GPU path.  E074 says the same
selector transfers over a wider K/N grid as a small pooled win, with useful
control gains at `K<=8` and large-K cells mostly tying or slightly losing to
the best controls while beating shuffled probes.  E077 then showed that the
pooled K-aware/fallback story was diluted by large-K fallback and schedule
overfit.  E078 fixes that audit by testing only probe-active `K<=8` cells and
replicates real probe content: `+0.242+/-0.081` over shuffled and
`+0.246+/-0.080` over fixed.  So the current live branch is:

> use aligned probes for low-dimensional optimizer-control decisions where the
> candidate set is deliberately structured, the fallback is explicit, and the
> selector is robust to noisy top-1 rankings.

The immediate next question is fallback and extraction:

- budget: can the E078 `K<=8` probe-content win survive at lower probe cost
  and across a small budget frontier?
- fallback: after the `K<=8` branch is costed, can a genuinely tuning-free
  large-K fallback avoid importing per-K tuned `fixed` values?
- extraction: can a default-anchored response model, not a same-round softmax,
  capture more than 4% of E065's large 2D oracle headroom without losing to the
  default-only control?
- regime: can probes win when the default is stale or the best hyperparameter
  changes, rather than when a tuned default is already known?
- measurement: E070 says the simple validation-gradient marginal-value score
  does not improve capture over forward-difference probes.
- real path: E078 says K-aware robust round-level outer control has proven
  probe content at `K<=8`; E079 is the cost frontier for that claim.

## New Program: Track The Hyperparameter Surface

Each worker group gets a hyperparameter setting:

```text
h_i = (inner_lr_multiplier, K, momentum, weight_decay, optimizer_type, ...)
```

All groups start from the same `theta_t`, run their local optimizer, and return
candidate updates:

```text
u_i = local_update(theta_t; h_i)
```

The probe set returns noisy values:

```text
y_i ~= V_t(h_i) = improvement(theta_t + alpha * u_i)
```

The server should not independently argmin one round's noisy values.  It should
fit a low-dimensional response model:

```text
V_t(h) = smooth function of log_lr, K, momentum, ...
```

First model to try:

```text
quadratic in log(inner_lr_multiplier)
with exponential forgetting over rounds
```

Then, if useful:

```text
small GP / Bayesian linear model over 1-3 hyperparameter dimensions
Thompson sampling or UCB to place next round's worker arms
```

The parallelism is the point.  Full-run HPO spends one whole training run per
point.  This method gets several same-base hyperparameter observations per
round and evaluates them with a tiny probe set.

## Why A GP Might Be Right, But Not First

A GP is attractive because the response surface over `log_lr` and `log_K` is
smooth but not necessarily quadratic.  However, the immediate bottleneck is not
expressiveness.  It is extraction under noisy probes.

Start with:

```text
Bayesian quadratic over log_lr
forgetting factor for drift
mandatory default arm
random exploration arms
conservative shrinkage toward uniform/FedAvg
```

Move to GP only after the quadratic beats:

- fixed grouped arms;
- shuffled probes;
- best fixed default arm;
- offline tuned schedule;
- oracle diagnostic gap.

## Next Experiments

### E062: Online Quadratic LR Tracker  [DONE]

Use aligned rounds.  Worker groups are placed at three or five
`log_lr_multiplier` points around a center.  Each round:

1. run group workers;
2. probe group-mean candidate updates;
3. update a Bayesian quadratic model of value versus `log_lr_multiplier`;
4. choose the next center/radius;
5. aggregate with damped weights around the model's posterior optimum.

Controls:

- uniform default LR;
- fixed LR group design from E060/E061;
- shuffled probes;
- oracle group choice;
- best fixed LR group chosen in hindsight.

Outcome:

- tracker tied raw grouped probes;
- GP/response-surface work is premature until it beats raw grouped probes.

### E063: Frozen Group-Probe Replication  [DONE]

Freeze `group_mean2_l40_t2`, repeat across `K in {8,16,32}`, and verify
collapse at spread 1.

Outcome: replicated.  This is now the baseline to beat.

### E064: Cost/Noise Sweep  [DONE]

Vary probe batch/slices:

```text
select_batch in {96, 192, 384, 768}
slices in {1, 2, 4}
```

Measure the capture/cost frontier.  The method only matters if it keeps a
positive edge below roughly 1-3% round overhead.

Outcome: sub-1% frontier is live.  Several budgets in `0.08%-0.67%` preserve
the edge; single-slice ultra-cheap budgets are too noisy.

### E065: Add A Second Hyperparameter Dimension  [DONE]

After the cost/noise frontier, test whether the same probe signal scales beyond
one designed hyperparameter:

```text
log_lr_multiplier x K_multiplier
```

or

```text
log_lr_multiplier x inner_momentum
```

This is where a GP becomes more natural.

Outcome: 2D headroom is large, and probes beat the parallel-arm average, but
raw soft weighting captures only about 4% of the oracle gap and does not
clearly beat the tuned default-only arm.

### E066: Default-Only LR Control  [DONE]

Run the E063/E064 frozen grouped-probe setting with an explicit default-only
arm and a separately tuned default-only outer `(alpha,beta)`.

Outcome: grouped probes beat the parallel-arm average but lose to default-only.
This changes the method target: do not use probes to average a bad exploration
population; use probes to decide when to depart from a strong default.

### E067: Default-Anchored Probe Admission

Keep the default LR group as the baseline action.  Run exploratory groups in
parallel, probe group-mean candidate updates, and use a conservative rule:

```text
if posterior(value(explorer) - value(default)) > threshold:
    mix toward the explorer
else:
    use default
```

Controls:

- default-only;
- fixed all-arm average;
- raw E063 grouped-probe soft average;
- shuffled probes;
- oracle default-or-explorer choice.

Outcome: safer but negative.  It reduced harm from exploratory arms but did not
beat default-only.  This closes the current stationary/default-known branch.

### E068: High-Budget Default-Anchored Probe  [DONE]

Double the E067 probe budget to test whether the negative was only a cheap-probe
noise floor.

Outcome: negative.  The high-budget rule ties default at K=16 but still loses at
K=8 and K=32.  Same-round probe precision is not enough.

### E069: Stale-Default Anchor Probe  [DONE]

Anchor on the deliberately stale/wrong LR group (`0.25`) and ask whether
same-round probes can justify leaving it.

Outcome: negative.  The current forward-difference softmax extractor still
does not extract the oracle gap.

### E070: Validation-Gradient Marginal-Value Score  [DONE]

Replace per-arm forward finite differences with one validation backward:

```text
g_V = grad L_V(theta)
score_i = <g_V, alpha * (u_i + beta v)>
```

This gives first-order scores for all arms for roughly one backward pass rather
than one forward per arm.  It directly attacks the shared extraction failure
instead of opening another anchor/schedule variant.

Outcome: negative.  It ties/improves only at K=32 and does not beat the forward
extractor or default-only control robustly.

### E071: Top-k / Soft Probe Outer Control On Real GPU  [DONE]

Replace brittle top-1 argmin over probe losses with robust selectors over the
same real-path `(alpha,beta)` candidate designs:

```text
top3_mean
top5_mean
soft_std1
soft_std2
top3_guard_fixed
```

Outcome: positive screen.  `scaled/soft_std2` was best at K=8 and
`scaled/top3_mean` was best at K=32, both against fixed, DiLoCo, and
target-retuned schedule controls.

### E072: Single-Policy Frozen Replication  [DONE]

Freeze `scaled/soft_std2` across K and repeat on fresh seeds.

Outcome: negative.  It directionally improved K=8 but tied/lost at K=32.  The
policy cannot be K-invariant.

### E073: K-aware Frozen Replication  [DONE]

Freeze the K-aware rule:

```text
K <= 8:  scaled/soft_std2
K >  8:  scaled/top3_mean
```

Outcome: replicated on fresh seeds in the pooled decision.  It beat the best
fixed/DiLoCo/schedule control by `0.454+/-0.329` at K=8; at K=32 it tied the
control (`0.029+/-0.144`) but strongly beat shuffled (`0.975+/-0.446`).

### E074: K/N Transfer  [DONE]

E074 kept the E073 frozen K-aware rule and expanded the real-GPU grid:

```text
K in {4, 8, 16, 32}
N in {8, 16}
selector batch = 384
K <= 8: scaled/soft_std2
K >  8: scaled/top3_mean
```

Decision: `K_N_TRANSFER_REPLICATES`.

| N | K | policy | best control | selector | selector minus control | selector minus shuffled | capture |
|---:|---:|---|---:|---:|---:|---:|---:|
| 8 | 4 | `soft_std2` | 10.595+/-0.977 | 10.917+/-1.408 | 0.322+/-0.458 | 0.267+/-0.280 | 0.084+/-0.295 |
| 8 | 8 | `soft_std2` | 5.687+/-0.392 | 5.900+/-0.665 | 0.213+/-0.297 | 0.145+/-0.170 | 0.116+/-0.214 |
| 8 | 16 | `top3_mean` | 3.276+/-0.230 | 3.286+/-0.526 | 0.010+/-0.338 | 0.445+/-0.337 | -0.275+/-0.402 |
| 8 | 32 | `top3_mean` | 2.043+/-0.142 | 2.025+/-0.413 | -0.017+/-0.303 | 0.601+/-0.330 | -0.346+/-0.501 |
| 16 | 4 | `soft_std2` | 10.735+/-1.044 | 11.322+/-1.627 | 0.587+/-0.617 | 0.585+/-0.324 | 0.167+/-0.267 |
| 16 | 8 | `soft_std2` | 5.828+/-0.643 | 6.022+/-0.833 | 0.194+/-0.232 | 0.339+/-0.191 | 0.086+/-0.229 |
| 16 | 16 | `top3_mean` | 3.186+/-0.316 | 3.183+/-0.663 | -0.002+/-0.436 | 0.379+/-0.387 | -0.621+/-1.008 |
| 16 | 32 | `top3_mean` | 1.913+/-0.254 | 1.853+/-0.419 | -0.060+/-0.292 | 0.985+/-0.953 | -0.547+/-0.734 |

Pooled:

```text
selector minus best control: 0.156+/-0.139
selector minus shuffled:     0.468+/-0.157
capture:                    -0.167+/-0.189
```

Readout:

- the transfer result is positive pooled, but not a smash;
- the control win is driven by `K <= 8`, where outer-control headroom is still
  large;
- at `K >= 16`, the selector mostly ties or slightly loses to the best control
  while still beating shuffled probes, which says the probe signal is real but
  the best action is often the conservative fallback;
- negative capture at large K is a warning against using capture as the sole
  metric in thin-headroom cells;
- the next method should make the fallback explicit rather than forcing probe
  selection where measured headroom has mostly vanished.

### E075: Fallback-Aware K Rule  [DONE]

E075 froze three conservative rules that only give the probe selector authority
where E073/E074 showed actual control headroom:

```text
e073_k_aware:
  K <= 8: scaled/soft_std2
  K >  8: scaled/top3_mean

hard_large_k_fallback:
  K <= 8: scaled/soft_std2
  K >  8: fixed-control fallback, no probes

guard_large_k_fallback:
  K <= 8: scaled/soft_std2
  K >  8: scaled/top3_guard_fixed
```

Decision: `FALLBACK_AWARE_K_RULE_PARTIAL`.

| rule | pooled selector minus best control | selector minus shuffled | large-K selector minus best control | capture |
|---|---:|---:|---:|---:|
| `e073_k_aware` | 0.095+/-0.128 | 0.541+/-0.263 | -0.086+/-0.166 | -0.084+/-0.159 |
| `guard_large_k_fallback` | 0.096+/-0.127 | 0.497+/-0.248 | -0.084+/-0.165 | -0.085+/-0.160 |
| `hard_large_k_fallback` | 0.116+/-0.097 | 0.082+/-0.061 | -0.045+/-0.024 | 0.041+/-0.068 |

Readout:

- the hard fallback improves the pooled control comparison enough to exclude
  zero (`+0.116+/-0.097`);
- the same hard fallback also gives a positive shuffled-control gap
  (`+0.082+/-0.061`), despite doing no probe work at large K;
- the strict large-K bar still fails (`-0.045+/-0.024`) because the fallback
  was plain fixed while the comparator is per-seed best-of
  fixed/DiLoCo/schedule;
- the guarded probe fallback does not improve on the E073 rule;
- therefore the next repair is not "more probing at large K."  It is an
  implementable control fallback: pick the large-K fallback from pilot data
  among fixed, DiLoCo, and schedule, then only let probes override that when
  the margin is strong.

### E076: Pilot-Best Control Fallback  [DONE]

E076 repeated E075, but for `K > 8` fell back to the pilot-selected control
family rather than to plain fixed.  This kept the action implementable and
asked whether E075's remaining large-K loss was just the wrong fallback.

```text
K <= 8:  scaled/soft_std2
K >  8:  pilot-best among fixed, DiLoCo, and schedule
```

Decision: `PILOT_BEST_FALLBACK_FAILS`.

| rule | pooled selector minus best control | selector minus shuffled | large-K selector minus best control | capture |
|---|---:|---:|---:|---:|
| `e073_k_aware` | 0.032+/-0.180 | 0.209+/-0.105 | -0.113+/-0.123 | -0.064+/-0.133 |
| `pilot_best_large_k_fallback` | 0.012+/-0.172 | 0.060+/-0.046 | -0.153+/-0.060 | -0.135+/-0.117 |

Pilot fallback chose `schedule` in every cell:

```text
select=384 N=8  K=4:  schedule
select=384 N=8  K=8:  schedule
select=384 N=8  K=16: schedule
select=384 N=8  K=32: schedule
select=384 N=16 K=4:  schedule
select=384 N=16 K=8:  schedule
select=384 N=16 K=16: schedule
select=384 N=16 K=32: schedule
```

Readout:

- E076 fails the method gate: pooled control gap does not exclude zero, and
  large-K fallback is worse than the strict comparator;
- the pilot schedule choice overfit the two pilot seeds and transferred worse
  than plain fixed in the large-K held-out rows;
- small-K probe rows were also weak on these fresh seeds, so E074's small
  positive is not yet a stable deployment result;
- the strict `best control` column is a useful ceiling but not an implementable
  baseline, because it chooses the best of fixed, DiLoCo, and schedule per
  held-out seed;
- the next result table must separate **implementable controls** from
  **oracle-over-control ceilings**, otherwise the experiment keeps chasing an
  unshippable baseline.

### E077: Implementable-Control Audit  [DONE]

E077 re-scored the E071-E076 style rows with three separate comparisons:

```text
1. fixed control                 # implementable
2. pilot-selected control        # implementable, but may overfit
3. held-out best-of controls     # oracle ceiling, not a baseline
```

Decision: `IMPLEMENTABLE_CONTROL_AUDIT_FAILS`.  Keep this label.  The gate was
pre-declared as a positive CI over fixed and pilot-best implementable controls,
and the rule did not clear it.

| rule | minus fixed | minus schedule / pilot-best | minus held-out-best ceiling | minus shuffled |
|---|---:|---:|---:|---:|
| `e073_k_aware` | -0.127+/-0.196 | 0.165+/-0.181 | -0.205+/-0.202 | 0.216+/-0.306 |
| `small_k_probe_fixed_large_k` | 0.062+/-0.089 | 0.353+/-0.138 | -0.017+/-0.084 | 0.044+/-0.077 |
| `small_k_probe_pilot_large_k` | -0.091+/-0.130 | 0.200+/-0.115 | -0.170+/-0.119 | 0.044+/-0.077 |

Pilot-best again chose `schedule` in every cell.  The apparent win over
schedule/pilot is therefore not automatically probe evidence; it may be
plain fixed beating an overfit schedule.

Readout:

- `small_k_probe_fixed_large_k` beats the schedule/pilot baseline by
  `0.353+/-0.138`, but it only ties fixed (`0.062+/-0.089`);
- it does **not** beat shuffled (`0.044+/-0.077`);
- it nearly matches the held-out-best control ceiling (`-0.017+/-0.084`), but
  that is because the control ceiling has collapsed toward fixed, not because
  probe extraction is strong;
- the strict `heldout-best` column remains useful as a ceiling but is not a
  deployable baseline;
- the deployable conclusion is: fixed is extremely hard to beat in this
  stationary control family, and the probe content is not yet proven.

Reanalysis, using the E077 reported cell means and reconstructing pooled CIs
from the per-cell `n=12` summaries:

| component | minus fixed | minus schedule / pilot-best | minus held-out-best ceiling | minus shuffled |
|---|---:|---:|---:|---:|
| K<=8 probe-active cells | 0.124+/-0.176 | 0.400+/-0.216 | 0.031+/-0.155 | 0.089+/-0.154 |
| K>8 fixed-fallback cells | 0.000+/-0.000 | 0.306+/-0.172 | -0.065+/-0.062 | 0.000+/-0.000 |

This split changes the interpretation:

- the actual probe content lives only in the K<=8 rows;
- the large-K improvement over schedule/pilot is not probe information, it is
  fixed fallback beating a pilot-selected schedule that overfit;
- the pooled shuffled comparison is diluted because K>8 uses fixed for both
  the rule and the shuffled control;
- after removing the dilution, the K<=8 probe-vs-shuffled contrast is still
  null (`0.089+/-0.154`);
- therefore E077 is evidence for schedule-overfit avoidance, not for a clean
  probe-value signal.

The column-4 claim still needs a **genuinely tuning-free large-K fallback**:
a frozen single pair, a K=8-transferred pair, or a simple theory-derived pair.
Using `cell.fixed` at large K imports per-K pilot tuning, so it cannot be the
final no-retuning story.

### E078: K<=8 Probe-Content Replication  [DONE]

Before any GP/bandit model, replicate the only place probes actually act:
`K<=8`.  The gate is the shuffled contrast, not the schedule contrast.

```text
probe-active rows only:
  selector - shuffled > 0 with CI excluding zero
  selector - fixed >= 0
```

Decision: `PROBE_CONTENT_REPLICATES`.

Compute: `REAL_GPU` on `devvm52518`, PyTorch CUDA
`2.14.1+cu130`, `NVIDIA PG509-210`.

Frozen configuration:

```text
design = scaled
policy = soft_std2
K      = {4, 8}
N      = {8, 16}
select = 384 batches
pilot  = 2 seeds
heldout = 33 fresh seeds
```

Pooled result:

| comparison | value |
|---|---:|
| selector - fixed | `+0.246+/-0.080` |
| selector - schedule | `+0.387+/-0.100` |
| selector - best implementable control | `+0.117+/-0.109` |
| selector - shuffled | `+0.242+/-0.081` |
| capture vs oracle gap | `+0.025+/-0.074` |

Cell-level `selector - shuffled` was positive in every probe-active cell:

| N | K | selector - shuffled |
|---:|---:|---:|
| 8 | 4 | `+0.243+/-0.203` |
| 8 | 8 | `+0.168+/-0.127` |
| 16 | 4 | `+0.353+/-0.186` |
| 16 | 8 | `+0.204+/-0.112` |

Readout:

- this is the first recent **actionable positive** by the stricter bar:
  probe content beats shuffled labels and fixed controls with pooled CIs
  excluding zero;
- E077 was correctly labeled a failure because it mixed probe-active cells with
  fallback cells; E078 removes that dilution and shows the probe labels carry
  signal where they are used;
- the claim is restricted to `K<=8` and this probe budget.  It is not yet a
  large-K claim, and the cost frontier is still open.

Artifacts:

```text
experiments/dfw_e078_gpu_kle8_probe_content_replication.py
experiments/dfw_e078_gpu_kle8_probe_content_replication_result_20261006.md
experiments/dfw_e078_gpu_kle8_probe_content_replication_run_20261006.txt
experiments/dfw_e078_gpu_kle8_probe_content_replication_LEGITIMACY_CHECK_20261006.md
```

### E079: K<=8 Probe-Budget Content Sweep  [DONE]

Now that E078 proves probe content at `K<=8`, sweep the validation/probe budget
on the same frozen selector:

```text
K      = {4, 8}
N      = {8, 16}
select = {384, 768, 1536}
policy = soft_std2
design = scaled
gate   = selector - shuffled > 0 and selector - fixed >= 0
```

Purpose:

- locate the cheapest batch budget that preserves the E078 content result;
- check whether larger probe batches increase capture or only tighten the
  same small effect;
- keep GP/bandit work blocked until the current selector's cost frontier is
  measured.

The first queued worker attempt on `assafcohen__devvm23211` returned
`NO_GPU_HARNESS` because that host had a GPU but no importable `torch` in the
requested Python.  The real run was executed directly on the known-good
`devvm52518` GPU/PyTorch host.

Decision: `PROBE_CONTENT_BUDGET_NULL`.

Compute: `REAL_GPU` on `devvm52518`, PyTorch CUDA
`2.14.1+cu130`, `NVIDIA PG509-210`.

Pooled result by probe batch:

| select | selector - fixed | selector - best control | selector - shuffled | capture |
|---:|---:|---:|---:|---:|
| 384 | `+1.521+/-0.924` | `+0.779+/-0.525` | `+0.090+/-0.142` | `+0.171+/-0.126` |
| 768 | `+1.487+/-0.902` | `+0.746+/-0.515` | `+0.040+/-0.126` | `+0.156+/-0.134` |
| 1536 | `+1.526+/-0.917` | `+0.785+/-0.537` | `+0.069+/-0.137` | `+0.164+/-0.137` |

Readout:

- the selector/action family beat fixed and best implementable controls on
  these seeds, with positive CIs;
- the probe labels did **not** beat shuffled labels at any tested budget;
- increasing probe batch size from 384 to 1536 did not materially increase
  capture or shuffled contrast;
- therefore E079 is not an actionable probe-content positive.  It says the
  control family is useful on this seed slice, but the labels are not yet
  proven to be the reason.

This does not erase E078; it makes the immediate ambiguity sharper:

```text
E078, select=384, 33 heldout seeds: selector - shuffled = +0.242 +/- 0.081
E079, select=384, 13 heldout seeds: selector - shuffled = +0.090 +/- 0.142
```

The next run should be high-power at the single cheapest budget (`select=384`),
not a wider budget sweep.

Artifacts:

```text
experiments/dfw_e079_gpu_kle8_probe_budget_content.py
experiments/dfw_e079_gpu_kle8_probe_budget_content_result_20261006.md
experiments/dfw_e079_gpu_kle8_probe_budget_content_run_20261006.txt
experiments/dfw_e079_gpu_kle8_probe_budget_content_LEGITIMACY_CHECK_20261006.md
```

### E080: K<=8 Probe-Content Power Replication  [NEXT]

Repeat only the cheapest/proven configuration with fresh pilot seeds and many
held-out seeds:

```text
K      = {4, 8}
N      = {8, 16}
select = 384
policy = soft_std2
design = scaled
pilot  = fresh 2-seed pilot
heldout = >= 60 fresh seeds
gate   = selector - shuffled > 0 and selector - fixed >= 0
```

Purpose:

- decide whether E078 was real and E079 was underpowered/noisy;
- test whether the probe labels remain useful under a fresh pilot;
- keep the action-space/GP branch blocked until label content survives this
  high-power replication.

### E081: Moving-Arm / GP Bandit  [BLOCKED UNTIL E080]

Use the workers as a live design over hyperparameters:

```text
arms_t = GP/UCB/Thompson points over log_lr, K, momentum
probes_t = cheap values of group-mean candidate updates
server_t = conservative aggregate around posterior-best arms
```

The baseline is not "same fixed hyperparameters."  The baseline is E064's
frozen grouped arms, the default-only arm, and offline tuned schedules.  E065
is the reason to run this: 2D headroom exists and raw extraction is weak.  E066
adds the constraint: the GP must be default-anchored or it will simply pay to
include bad exploratory arms.  E078 adds the permission to continue: probe
labels can beat shuffled labels in the restricted low-K regime.  E079 says
that result is not stable enough from a 13-seed budget sweep, so E080 must
replicate label content at high power before adding a GP layer.

## Paper Shape

The paper should not claim "Bayesian probes beat averaging" in full generality.

Better claim:

> Distributed training has parallel experimental bandwidth that standard
> optimizers waste.  In aligned rounds, cheap probe sets can retune
> low-dimensional outer optimizer controls online, matching or exceeding
> fixed/scheduled controls without a new per-regime sweep.  For worker/arm
> weighting, the same evidence must be pooled and conservatively shrunk; it
> collapses when arms are exchangeable.

That claim fits the favorable evidence:

- E060/E061 show the stationary grouped-LR mechanism is live.
- E063 replicates the frozen group-probe rule across K with exact collapse.
- E064 shows the edge survives below 1% probe overhead.
- E065 shows the 2D hyperparameter arm space has large oracle headroom, while
  raw same-round weighting under-extracts it.
- E066 shows the stationary LR win does not beat a known good default arm, so
  the correct method is default-anchored exploration rather than averaging all
  parallel arms.
- E067 shows default anchoring is not enough on the stationary/default-known
  bed; the current extractor still cannot justify departures from the default.
- E068 shows doubling the probe budget still does not fix this, pointing to
  value-proxy/extraction rather than simple probe cheapness.
- E069 shows a bad/stale anchor is not enough for the existing extractor; the
  measurement itself must improve.
- E070 shows the obvious validation-gradient measurement swap is not enough;
  the next improvement has to be a response model or a different regime, not
  another same-round scalar selector.
- E071 shows robust top-k/soft probe selection can beat real-path controls, but
  only as a selector-family screen.
- E072 shows a single K-invariant selector does not replicate.
- E073 shows the frozen K-aware selector does replicate on fresh real-GPU
  seeds in the pooled decision: a K=8 control win plus a K=32 shuffled-control
  win/control tie.
- E074 shows the K-aware selector transfers over `K in {4,8,16,32}` and
  `N in {8,16}` as a small pooled win (`+0.156+/-0.139` over best controls),
  but the win is driven by K<=8 and the large-K cells should fall back.
- E075 shows hard fallback improves the pooled control comparison
  (`+0.116+/-0.097`) but plain fixed is not the right large-K fallback against
  a best-of fixed/DiLoCo/schedule comparator.
- E076 shows naive pilot-selected fallback also fails: it picked schedule in
  every cell and transferred worse than plain fixed at large K.  It also
  exposes that held-out best-of controls are an oracle ceiling, not a deployable
  baseline.
- E077 separates implementable controls from that ceiling.  The best simple
  rule, `small_k_probe_fixed_large_k`, beats the schedule/pilot baseline
  (`+0.353+/-0.138`), ties fixed (`+0.062+/-0.089`), and nearly matches the
  held-out control ceiling (`-0.017+/-0.084`).
- E078 isolates the probe-active `K<=8` cells and gives the actionable
  positive that E077 lacked: `selector - shuffled = +0.242+/-0.081` and
  `selector - fixed = +0.246+/-0.080`, both pooled CIs excluding zero.
- E079 sweeps budgets on fresh seeds and fails the shuffled-label content gate:
  the action family beats fixed/best-control, but not shuffled labels.
- E080 is the deciding high-power replication for the K<=8 label-content
  claim.
- Spread-1 collapse shows the rule does not invent signal.
- E059 explains why individual-worker ranking failed.
- E062 shows online surface tracking is not yet useful beyond raw grouped
  probes.
- Earlier schedule-transfer failures motivate online tracking instead of
  offline retuning.
