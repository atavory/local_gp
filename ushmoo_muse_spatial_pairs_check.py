"""muse's unbiasedness argument, and what it costs. Pre-registered predictions.

muse's thread (independent of this repo) arrived at the same object and made one
argument this repo had not: in stochastic L-BFGS the step s_k depends on the
same noise as the gradients, which biases the secant equation, whereas pairs
taken at sync time use fresh noise and so satisfy E[y_i | positions] = H s_i
exactly, for free.

That is directly relevant, because this repo's §6 finding is that its estimator
IS the biased construction (within a round, s_{i,k} for k>=1 contains the noise
from steps k'<k, which the round-centred gbar also contains), and that the bias
is apparently doing the work: a less-biased causal estimator with more data
scores 1.00 where the biased one scores 1.13.

But muse's construction changes the DESIGN, not just the bias, and that is the
part the thread has not noticed. Mean-centring the workers at a fixed local step
removes the common drift exactly, because the drift is identical across workers
and lives entirely in the mean. What remains is the worker-to-worker scatter,
which wanger's Lyapunov argument and this repo's measurement both put in the
FLAT directions (5x the gap variance, top-12 overlap with the sharp set 0.000).
The trajectory estimator keeps the drift, and the drift is largest in the SHARP
directions, because those are the ones that actually travel.

So the two constructions probe different subspaces:
    spatial, mean-centred  -> worker scatter -> flat-concentrated, unbiased
    trajectory, pooled     -> drift retained -> sharp-concentrated, biased

PREDICTIONS, recorded before running:
  P-A  spatial pairs have much lower bias (muse is right about that).
  P-B  spatial pairs have much WORSE relative error in the sharp directions,
       because mean-centring discarded the only well-conditioned part of the
       design.
  P-C  therefore the spatial arm scores worse than the trajectory arm despite
       being unbiased. If instead it scores BETTER, §6's bias story is wrong
       and muse's fix is simply the right answer.

Arms, all on the same FedAvg trajectory, scored as one-step gain vs a
TUNED-SCALE FedAvg baseline. Correction factor is 1/(1-a^K), not 1/lam -- see
ushmoo_why_precond_fails.py for why 1/lam double-applies.
"""

import math
import random

D, N_OUT, K, ETA, SIG, T, REPS = 120, 10, 8, 0.05, 1.0, 30, 5
NS = (8, 32)
LMAX = 1.5 / ETA
ZG = [math.log(1e-3) + i * (math.log(LMAX) - math.log(1e-3)) / 39 for i in range(40)]
LG = [math.exp(z) for z in ZG]
LCG = [l / (1.0 - (1.0 - ETA * l) ** K) for l in LG]


def gain(u, gstar, lam):
    num = sum(a * b for a, b in zip(gstar, u))
    den = sum(l * x * x for l, x in zip(lam, u))
    return num * num / (2.0 * den) if den > 0 else 0.0


def Cof(l):
    return 1.0 / (1.0 - (1.0 - ETA * min(max(l, 1e-9), LMAX)) ** K)


def ratio(bhat, v, m, s2):
    lw = [-((bhat - l) ** 2) / (2.0 * v) - ((z - m) ** 2) / (2.0 * s2)
          for l, z in zip(LG, ZG)]
    mx = max(lw)
    w = [math.exp(a - mx) for a in lw]
    sl = sum(wi * li for wi, li in zip(w, LG))
    return (sum(wi * lc for wi, lc in zip(w, LCG)) / sl) if sl > 0 else 1.0 / (ETA * K)


def coeffs(xx, xg, nvar, D_):
    ols, vv = [], []
    for j in range(D_):
        if xx[j] > 1e-300:
            ols.append(xg[j] / xx[j]); vv.append(nvar / xx[j])
        else:
            ols.append(0.0); vv.append(1e9)
    pos = [b for b in ols if 1e-3 < b < LMAX] or [1.0]
    m = sum(math.log(b) for b in pos) / len(pos)
    s2 = max(1e-2, sum((math.log(b) - m) ** 2 for b in pos) / len(pos))
    return [ratio(ols[j], vv[j], m, s2) for j in range(D_)], ols


def run(seed, lam, N):
    rng = random.Random(seed)
    x = [rng.gauss(0, 1.0) for _ in range(D)]
    txx = [0.0] * D; txg = [0.0] * D       # trajectory, round-centred (biased)
    sxx = [0.0] * D; sxg = [0.0] * D       # spatial, (round,k)-centred (unbiased)
    keys = ("base", "traj", "spatial", "oracleC")
    acc = {k: 0.0 for k in keys}
    est = {"traj": [0.0] * D, "spatial": [0.0] * D, "n": 0}
    nk = N * K
    for t in range(T):
        gstar = [l * xj for l, xj in zip(lam, x)]
        dev = [[0.0] * N for _ in range(D)]
        pts = [[] for _ in range(D)]
        grd = [[] for _ in range(D)]
        for k in range(K):
            # snapshot this step's (position, gradient) across workers
            sk = [[0.0] * N for _ in range(D)]
            gk = [[0.0] * N for _ in range(D)]
            for j in range(D):
                lj, xj, dj = lam[j], x[j], dev[j]
                pj, qj = pts[j], grd[j]
                for i in range(N):
                    s = dj[i]
                    g = lj * (xj + s) + rng.gauss(0, SIG)
                    pj.append(s); qj.append(g)
                    sk[j][i] = s; gk[j][i] = g
                    dj[i] = s - ETA * g
            # spatial pairs: centre ACROSS WORKERS at this fixed k.
            # removes the common drift exactly; noise in g is fresh wrt position
            for j in range(D):
                ms = sum(sk[j]) / N
                mg = sum(gk[j]) / N
                sxx[j] += sum((v - ms) ** 2 for v in sk[j])
                sxg[j] += sum((v - ms) * (w - mg) for v, w in zip(sk[j], gk[j]))
        dbar = [sum(dev[j]) / N for j in range(D)]
        for j in range(D):
            ms = sum(pts[j]) / nk
            mg = sum(grd[j]) / nk
            txx[j] += sum((p - ms) ** 2 for p in pts[j])
            txg[j] += sum((p - ms) * (q - mg) for p, q in zip(pts[j], grd[j]))

        ct, ot = coeffs(txx, txg, SIG * SIG, D)
        cs, os_ = coeffs(sxx, sxg, SIG * SIG, D)
        acc["base"] += gain(dbar, gstar, lam) / T
        acc["traj"] += gain([dbar[j] * ct[j] for j in range(D)], gstar, lam) / T
        acc["spatial"] += gain([dbar[j] * cs[j] for j in range(D)], gstar, lam) / T
        acc["oracleC"] += gain([dbar[j] * Cof(lam[j]) for j in range(D)],
                               gstar, lam) / T
        if t >= 5:
            est["n"] += 1
            for j in range(D):
                est["traj"][j] += (ot[j] - lam[j]) / lam[j]
                est["spatial"][j] += (os_[j] - lam[j]) / lam[j]
        x = [x[j] + dbar[j] for j in range(D)]
    return acc, est


def main():
    rb = random.Random(0)
    lam = [rb.uniform(8, 12) for _ in range(N_OUT)] + \
          [rb.uniform(0.1, 1.0) for _ in range(D - N_OUT)]
    flat = sorted(range(N_OUT, D), key=lambda j: lam[j])
    h = len(flat) // 2
    buckets = [("sharp", list(range(N_OUT))), ("flat low", flat[:h]),
               ("flat high", flat[h:])]
    print(f"d={D} sharp={N_OUT} K={K} eta={ETA} rounds={T} reps={REPS}")
    print("ratios to tuned-scale FedAvg at the same N; correction = 1/(1-a^K)")
    print()
    for N in NS:
        acc = {k: 0.0 for k in ("base", "traj", "spatial", "oracleC")}
        bt = [0.0] * D; bs = [0.0] * D; nn = 0
        for r in range(REPS):
            a, e = run(55_000 + r, lam, N)
            for k in acc:
                acc[k] += a[k] / REPS
            nn += e["n"]
            for j in range(D):
                bt[j] += e["traj"][j]; bs[j] += e["spatial"][j]
        b = acc["base"]
        print(f"N={N}:  base {b:.3f} | traj {acc['traj']/b:.3f}  "
              f"spatial {acc['spatial']/b:.3f}  oracleC {acc['oracleC']/b:.3f}")
        print(f"   {'bucket':<11} {'lam':>7} | {'rel bias traj':>14} "
              f"{'rel bias spatial':>17}")
        for name, idx in buckets:
            ml = sum(lam[j] for j in idx) / len(idx)
            vt = sum(bt[j] for j in idx) / len(idx) / nn
            vs = sum(bs[j] for j in idx) / len(idx) / nn
            print(f"   {name:<11} {ml:>7.3f} | {vt:>+14.3f} {vs:>+17.3f}")
        print()
    print("P-A: spatial bias << traj bias.  P-B: spatial much worse on sharp.")
    print("P-C: spatial scores worse overall despite being unbiased.")


if __name__ == "__main__":
    main()
