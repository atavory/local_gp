# ushmoo handoff — local_gp

Single file. Replaces `ushmoo_problem_statement.md`, `ushmoo_guardrails.md`,
`ushmoo_audit_bandha_20261003.md`, `ushmoo_theory_localsgd_as_gp_20261003.md`,
`ushmoo_shadow_framing_20261004.md`, all deleted. The `ushmoo_*.py` scripts stay
— they are the evidence and each reproduces its numbers in seconds on CPU.

Lane rule: ushmoo writes only `ushmoo_*` files. Read everyone's, write mine.

---

## 1. What this project is

Local SGD and FedAvg run `K` local steps on each of `N` workers and average the
endpoint displacements. That averaging is **exactly the Bayes-optimal read-out
under a prior in which the gradient field is constant over the region the
workers explored.** The workers' trajectories are derivative observations of a
latent function; the aggregation rule is a posterior read-out; and the standard
rule is the infinite-lengthscale case.

So the question is not whether averaging is naive. It is: *what does the
posterior say when the lengthscale is finite, and how much is that worth?*

This framing makes FedAvg a special case rather than an opponent, and it turns a
scattered literature of filtered and weighted aggregation into one question
about prior misspecification.

## 2. What is established

Numbers are from the scripts named; bed is a diagonal quadratic (d=120, 10
sharp λ~U[8,12] + 110 flat λ~U[0.1,1]), K=8, η=0.05, scored as one-step gain on
a real FedAvg trajectory against a **tuned-scale** FedAvg baseline.

**The displacement law.** `Δ̄_j = −g*_j(1−a_j^K)/λ_j`, `a_j = 1−ηλ_j`.
Verified 1.0218 sharp / 0.9346 flat (`ushmoo_why_precond_fails.py`). K local
steps interpolate gradient step → Newton step with crossover at `ηλ_jK ≈ 1`.
Measured implied step: sharp λ=10 → **0.0996** against Newton `1/λ=0.1000`.

> **Consequence, and the best single result here:** sharp directions reach the
> Newton step *before* aggregation, so preconditioning `Δ̄` by the inverse
> Hessian double-applies. With the **exact** Hessian it scores **0.945×** — worse
> than doing nothing, with perfect knowledge. The correct operator is
> `[I−(I−ηH)^K]^{-1}`, worth **1.310×**.

**The duality.** With `C_j = [1−(1−ηλ_j)^K]^{-1}` the correction owed, benefit is
`(1−1/C_j)²` and relative estimation error is `ε_j ∝ C_j` **in both design
branches**. The gain and the measurability are the same quantity in opposition.
Measured `ε`: 0.069 sharp, 3.080 flat, 41/200 coordinates non-invertible. Value
× precision peaks at `C=3`, i.e. `ρ = ηλK ≈ 0.4` (measured peak 0.24), so
**benefit is non-monotone in K**.

**We are at the achievable frontier** (`ushmoo_frontier_and_ksweep_20261004.py`):

| | /base |
|---|---|
| ours, prior fitted from data | **1.113** |
| frontier: posterior under the TRUE prior | 1.103 |
| Newton oracle | 1.221 |

**109.5% of the frontier, 51% of Newton.** The Newton gap is *unidentifiable*
from a drift design, not unestimated.

**Why nothing improves it.** In the worker-noise-driven branch (the flat
directions, where all the gain is), `Sxx ∝ NK(ησ)²K` and `ε/C = 1/√(c'N)` —
**σ cancels**, because the design spread and the measurement noise are the same
noise. Three measured consequences: more workers doesn't help (flat in N, 8→128),
round-pooling doesn't help (stale = fresh: 1.522 vs 1.526 late-window), and
heterogeneous K doesn't help (3.4–3.7× worse; it has *more* design spread but
the within-round O(1/n) bias scales with group size).

**The lever that does move it.** A *deliberate* probe of chosen magnitude `r`
gives `ε/C = σ·η·√K/(r·√N)`, which is not σ-free. Condition: `r ≫ ησ√K` (the
noise floor is ~0.02 on this bed). Cost ≈ `½λ_j r_j²` excess loss per
coordinate. That trade has a closed-form optimum — the explore/exploit knob with
a formula instead of a threshold to tune. Untested.

**What makes it not L-BFGS.** Server-side temporal secants — the pairs an L-BFGS
on the aggregated iterates would use, free, no extra logging — capture ~**1.00**
of the headroom. Within-round worker geometry captures most of it. Without this
control the work reads as reinvented quasi-Newton.

**Implementation facts that are load-bearing.** Read out `E[λC]/E[λ]`, never
`E[C]`: `λC → 1/(ηK)` is bounded, `C` is not, and the unbounded form is
catastrophic (0.091×). Put the prior on `log λ` so the posterior cannot reach
zero. Cross-fitting across worker halves preserves the full gain (1.111 vs
1.119), so the effect is estimation, not same-sample correlation.

## 3. The competitive axis — and a correction to my own standard

The incumbents are **DiLoCo**, SlowMo/BMUF, FedAvgM/FedAdam, SCAFFOLD/FedProx,
EASGD and decentralized SGD with gradient tracking.

**DiLoCo's headline is *matching* fully-synchronous training at ~500× less
communication.** My standard — beat a tuned scalar on loss with a CI excluding
zero — scores "match" as a null, and would therefore have rejected DiLoCo. The
axis was wrong: the incumbent competes on **communication**, not on loss.

Our equivalent claim is available and has never been measured. The
dynamics-aware operator exactly compensates the K-step distortion, so corrected
aggregation should hold quality at **larger K**, i.e. fewer synchronizations.
"Same loss, fewer rounds" is the DiLoCo-shaped claim; "11% better loss per
round" is not.

### Evaluation tiers

1. **Sanity** — tuned scalar scale, shuffled-feature, lag-1, best-worker,
   server/time secant. These *are* kill criteria: they catch artifacts. Two
   results died here correctly (a degenerate KRR whose shuffled control tied it
   exactly; a γ-grid whose lag-1 arm was indistinguishable from shuffled).
2. **Incumbents** — DiLoCo, SlowMo, FedAvgM/FedAdam, SCAFFOLD. The real bar.
   Beat *or match* at matched communication and compute.
3. **Oracles** — per-regime best schedule, true curvature, the Bayes frontier.
   **Map, not executioner.** Never a survival bar.

I was using tier 3 as the survival bar, and the frontier measurement is the
clean demonstration of why that is wrong: against Newton we look like 51%;
against the ceiling the design can actually reach, we are at 109%.

Separately, the stated scope must stay honest: everything above is a diagonal
quadratic with homogeneous workers. The displacement law is exact algebra there
and inherits that limit. No real-network result exists.

## 4. Next actions, ranked

1. **Rounds-to-target as a function of K.** Does the correction hold quality at
   larger K? This is the only framing that competes with DiLoCo on its own axis,
   and it follows directly from the displacement law. Nothing else should go
   first.
2. **Probe-magnitude sweep.** Vary `r` from the noise floor (≈0.02) up by
   decades; check the interior optimum lands where `½λr²` balances the curvature
   gain. If the closed form predicts the measured optimum, that is the central
   figure.
3. **Re-measure the frontier under a probe design.** The 51%-of-Newton ceiling
   is for drift designs only; how much of the remaining 49% probes buy is the
   value proposition stated honestly.
4. **Break the quadratic.** Everything inherits exact-quadratic algebra.
5. **Heterogeneous workers** (multi-task GP; SCAFFOLD's control variate is the
   posterior mean of the client offset under an infinite-lengthscale prior on
   that offset — same misspecification one level up).

Prior art to verify before writing: **Hennig & Kiefel, "Quasi-Newton Methods: A
New Direction," JMLR 2013** (quasi-Newton as Gaussian regression on secants) and
**CMA-ES** (its covariance converges to H⁻¹ from population spread; sep-CMA-ES
and LM-CMA are our diagonal and low-rank factorizations). Both are novelty
threats. The defensible delta is the *parallel common-origin design* — quasi-
Newton accumulates secants sequentially along one path, local SGD generates them
in parallel from one point, already paid for — plus the duality and the secant
null.

## 5. Operational

**Repo.** `github.com/atavory/local_gp`, private, cloned at
`/data/users/atavory/local_gp`.

**Push.** Direct `git push` fails (`fwdproxy` CONNECT 403). Route through Mr.
Gitties:

```python
import json, socket, os
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(os.path.expanduser('~/.mrgitties/mrgitties.sock'))
s.sendall(json.dumps({'cmd': 'git push origin main',
                      'cwd': '/data/users/atavory/local_gp',
                      'timeout': 240}).encode() + b'\n')
s.shutdown(socket.SHUT_WR)
buf = b''
while True:
    c = s.recv(65536)
    if not c: break
    buf += c
print(json.loads(buf))
```

Other agents' loops write to the repo continuously, so `git pull --rebase` often
fails on their unstaged files. Commit only `ushmoo_*` paths, then push without
rebasing; retry with a pull if the remote moved.

**Overleaf checkouts** (local git clones, one directory per project):

| project | path |
|---|---|
| local_gp draft (`local_gp_from_scratch.tex`) | `/data/users/atavory/overleaf/6a65a8d1af99ca2536f4a218/` |
| double_fan (frozen negative) | `/data/users/atavory/overleaf/6aa106bb39f566201f723916/` |
| DR-estimation paper | `/data/users/atavory/overleaf/69edff47028a983c95b7fcc2/` |
| wanger shadow | `/data/users/atavory/overleaf/6aa106ce53a4d0bbe789b369/` |

**Manifold** (`ROSETTA_DISABLE_AOT=1 manifold get|put|ls`), bucket
`aai_research_tlv`:

| what | path |
|---|---|
| worker queue | `tree/double_fan/messages/to_workers/assafcohen/` |
| worker results | `tree/double_fan/messages/from_workers/assafcohen/` |
| agent mailbox | `tree/double_fan/message_box/{inbox,outbox,thread}/` |
| local_gp pings | `tree/local_gp/cpu_pings/<date>/` |

Dispatch a worker by `manifold put`-ing a prompt named
`<UTC timestamp>_<name>.md` into the queue path.

**Busy-wait loop.** `/data/users/atavory/busy_wait/busy_wait.py run <dir>
--agent-cmd ...`, with `state.md` / `agenda.md` / `log.md` in the run dir.
`agenda.md` is the operational file. Two cautions learned the hard way:
`state.md` must be *overwritten*, not appended (a per-minute heartbeat buried it
at 323 lines), and an agenda whose last item is "keep monitoring" can never
satisfy its own stop condition — give polling loops a deadline and a liveness
check, not just a wall-clock cap.

**No numpy on this host.** All scripts are pure stdlib and run in seconds.

**Never** use `find`/`grep`/`rg` or the Grep/Glob tools inside `fbsource` — they
traverse a virtual filesystem and time out. `local_gp` and the overleaf
checkouts are ordinary directories and are fine.
