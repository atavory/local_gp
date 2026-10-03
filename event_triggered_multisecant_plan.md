# Event-Triggered Multisecant Synchronization

## Core Reframe

The workers are not independent optimizers whose final parameters should simply
be averaged. They are probes. Between communication events, each worker runs a
local heavy-ball process from the last broadcast point. The controller uses the
resulting cloud of positions and gradients as a local experimental design.

This flips the usual FedAvg picture:

- workers generate information;
- the controller owns the optimizer state;
- synchronization is an event-triggered consolidation decision, not a fixed
  period;
- worker-side autonomy is only about reducing wire use, not choosing the
  optimization action.

## Cost Model

For the current target, we pay primarily by wire/round, not by bytes. Therefore:

- no continuous sketch streaming in the first design;
- workers run silently after a broadcast;
- a worker may send a tiny ready signal when its local drift crosses a budget;
- the controller fires an event on quorum or timeout;
- at an event, send the full useful state, because the round is already paid.

Per event, gather full \((x_i, g_i)\), and optionally \(v_i\). No \(d^2\)
object is ever formed. The centered secant pairs are
\[
  s_i = x_i - \bar x,\qquad y_i = g_i - \bar g .
\]
All-pairs differences are redundant: if \(B s_i \approx y_i\), then pairwise
differences \(x_i-x_j\) are already covered.

## What Survives From Momentum / Acceleration

Momentum remains part of the worker probe dynamics; it controls the local paths
that generate secants. A separate acceleration buffer should not be the product.
Temporal acceleration is a noisy rank-one finite difference. The spatial secant
bank is a multi-direction, same-event estimate of curvature. Once the bank
exists, model-based consolidation should dominate Taylor extrapolation from a
per-worker acceleration buffer.

## Minimal Algorithm

1. Controller broadcasts \((\bar x, \bar v, \delta)\).
2. Workers run local heavy-ball silently.
3. Worker \(i\) sends a ready bit when \(\|x_i-\bar x\|\ge\delta\).
4. Controller triggers on quorum or timeout.
5. Workers send full \((x_i, g_i)\), and optionally \(v_i\).
6. Controller forms centered secant pairs \((s_i,y_i)\).
7. Controller chooses a consolidation:
   - FedAvg: \(\bar x\);
   - scalar extrapolation: \(\bar x_0 + c(\bar x-\bar x_0)\);
   - BB scalar: \(\bar x-\alpha \bar g\);
   - diagonal / limited-memory multisecant step.
8. Controller broadcasts the next \((\bar x,\bar v)\).

## First Theory Tasks

1. Difference process: \(m=2\), noisy quadratic, heavy-ball, commuting \(H\)
   and \(\Sigma\). Solve the stationary covariance of \(x_1-x_2\) and \(v_1-v_2\).
2. Secant SNR: derive per-eigendirection error of
   \(\hat\lambda_j=\sum_i s_{ij}y_{ij}/\sum_i s_{ij}^2\).
3. Trigger rule: information gain from more drift versus trust-region/model
   error risk.
4. Non-quadratic guardrail: Hessian-Lipschitz error grows with drift radius, so
   the event threshold is also a trust-region radius.

## First Experimental Gate

Stage 0 should be synthetic and CPU:

- workers run heavy-ball until a drift-threshold quorum fires;
- compare FedAvg, tuned scalar extrapolation, server/time secants, and centered
  worker multisecants;
- score actual one-event objective decrease;
- report secant SNR and drift PCA;
- kill criterion: if multisecant does not beat tuned scalar extrapolation on a
  controlled quadratic/nonlinear bed, stop.

This is separate from the gamma line. The first product claim is not "we choose
gamma"; it is "event-time worker spread supplies curvature shape that scalar
extrapolation and server-side secants miss."

## Stage-0 Findings

E011 tested passive stochastic drift. It failed: the spatial secants did not
beat tuned scalar extrapolation or a one-pair server BB step. Passive minibatch
noise is not a good experimental design.

E012 added controller-seeded probe offsets. Spatial BB improved slightly over
scalar extrapolation, but still lost to the one-pair server BB step. Probe
diversity alone was not enough.

E013 tested actual spatial L-BFGS multisecants on a rotated block quadratic.
With ordinary stochastic gradients, spatial L-BFGS selected zero damping and
lost. With noiseless secants, it beat one-pair time L-BFGS at N=32 and N=128.

E014 corrected the consolidation operator. Plain quasi-Newton consolidation
double-applies curvature, because K local steps have already interpolated
between a gradient step and a Newton step. On a quadratic with no momentum,
\[
  \Delta_K=-H^{-1}[I-(I-\eta H)^K]g_0,
\]
so the remaining correction is
\[
  [I-(I-\eta H)^K]^{-1}\Delta_K.
\]

With this dynamics-aware operator, noise-controlled probe estimates beat a tuned
scalar FedAvg scale on the rotated-block quadratic:

- N=8: directed probe 1.332 vs scalar 1.054;
- N=32: directed probe 1.301 vs scalar 1.047;
- N=128: directed probe 1.326 vs scalar 1.051.

The path-only centered trajectory estimate is weaker (1.061/1.097/1.214),
which keeps the rank/noise critique live. The successful probe arms use an
`N*K` probe budget and repeated gradients, so this is a corrected Stage-0
mechanism result, not a product claim.

The architecture is therefore conditional and more specific:

- worker spread can provide useful multi-direction secant information;
- but ordinary independent stochastic gradients make the secant equations too
  noisy;
- the consolidation must model the K-step local dynamics rather than applying a
  fresh quasi-Newton step;
- the next viable version must control secant noise using same-batch probes,
  repeated event gradients, a shared probe set, or large-batch/large-N regimes.

This updates the theory target. The first theorem should not assume the spatial
bank is useful; it should derive the SNR threshold under which spatial
multisecants beat scalar/time secants, and the update theorem should be written
for \([I-(I-\eta H)^K]^{-1}\Delta_K\), not for \(H^{-1}g\).

E015 moved the corrected operator to the nonlinear/heterogeneous CPU bed. The
probe-fed arms survived the tuned scalar null:

- N=8: random probe 1.306 vs scalar 1.069;
- N=32: random probe 1.445 vs scalar 1.053;
- N=128: random probe 2.009 vs scalar 1.101.

Path-only centered trajectory pairs selected zero and tied FedAvg, while
server/time secants stayed weak. The useful architecture is therefore not
"worker drift alone"; it is controller-owned probing or noise-controlled local
geometry plus a dynamics-aware map correction.

E016 replaced the diagonal nonlinear bed with rotated 2D nonlinear blocks. The
probe-fed local-map correction still beat tuned scalar scale:

- N=8: random probe 1.254 vs scalar 1.032;
- N=32: random probe 1.497 vs scalar 1.057;
- N=128: random probe 2.124 vs scalar 1.132.

This rules out the simplest coordinate-artifact explanation. E017 then replaced
target gamma tuning with split-probe residual shrinkage:

- N=8: risk probe 1.226 vs scalar 1.075;
- N=32: risk probe 1.460 vs scalar 1.070;
- N=128: risk probe 1.773 vs scalar 1.127.

The oracle best-worker fallback loses in both gates. Path-only worker scatter
also fails in both gates. So the usable design is narrower but real:
noise-controlled probes feed a dynamics-aware local-map correction, with
residual shrinkage guarding model risk.

E020 tested whether active design can reduce the probe budget. It did not:
full random probes remain strong, current value-weighted probes often
underperform matched random probes, and `uncertainty_value` only ties random
in the stronger cases. For now, the practical rule is random probes near `2N`
when `N >= 32` and full `N*K` for very small `N`.

E023 confirmed that rule with more held-out reps and a direct probe-repeat
check. `2N` random probes beat tuned scalar and stayed close to matched full
random for N=32/128; `probe_reps=1` is usable when gradient-call budget binds,
while `probe_reps=4` is a quality knob.

E021 tested dense low-rank curvature with correlated low-rank noise. Oracle
dense-H correction has large headroom, but diagonal, low-rank, and combined
estimated corrections do not clear tuned scalar. The learned combined update is
destructive under E017-style residual shrinkage, so dense geometry requires a
held-out validation or trust layer before applying a low-rank-plus-diagonal
correction.

E022 tested that trust layer. Rowwise validation-gamma selected by held-out
mini-loss proxy prevents the E021 crash and beats tuned scalar on the dense bed,
but the gain is small relative to oracle-H. Dense correction should therefore
remain trust-gated while the representation and validation-budget accounting
improve.

The next event-triggered work is real-path plumbing plus trust-gated dense
subspace/model identification and function-space probes.
