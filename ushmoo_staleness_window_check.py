"""Is stale curvature usable? This decides whether async is safe.

Section 6 of the theory note found a strictly causal estimator (pooled over
rounds < t, more data, lower error) scoring 1.00 where a same-round estimator
scored 1.13. I attributed it to estimator bias; ushmoo_muse_spatial_pairs_check
refuted that (an 8-10x less biased estimator ties, costing ~1%).

Remaining hypothesis, stated there and untested: it is a WEIGHTING artifact.
Per-round gain is far larger early, because x starts far from the optimum and
the sharp directions converge within a few rounds. The causal arm is uninformed
exactly during the rounds that dominate the average, so its deficit is a
cold-start penalty, not a staleness penalty.

This matters well beyond the bookkeeping. The whole case for asynchrony rests on
"secant pairs are timeless within the trust region" -- stale pairs are still
valid, unlike stale gradients. If stale curvature genuinely does not work, that
argument is dead and async buys nothing. If the 1.00 was cold start, stale
curvature is fine and async is safe.

Test: score the same arms over a late window only (rounds >= WARM), where every
arm has a mature estimate, and report the per-round profile so the weighting is
visible rather than inferred.
"""

import math
import random

D, N_OUT, K, ETA, SIG, T, REPS = 120, 10, 8, 0.05, 1.0, 40, 5
N, WARM = 32, 15
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


def coeffs(xx, xg, nvar):
    ols, vv = [], []
    for j in range(D):
        if xx[j] > 1e-300:
            ols.append(xg[j] / xx[j]); vv.append(nvar / xx[j])
        else:
            ols.append(0.0); vv.append(1e9)
    pos = [b for b in ols if 1e-3 < b < LMAX] or [1.0]
    m = sum(math.log(b) for b in pos) / len(pos)
    s2 = max(1e-2, sum((math.log(b) - m) ** 2 for b in pos) / len(pos))
    return [ratio(ols[j], vv[j], m, s2) for j in range(D)]


def run(seed, lam):
    rng = random.Random(seed)
    x = [rng.gauss(0, 1.0) for _ in range(D)]
    wxx = [0.0] * D; wxg = [0.0] * D
    per = []
    nk = N * K
    for t in range(T):
        cc = coeffs(wxx, wxg, SIG * SIG)          # causal: rounds < t
        gstar = [l * xj for l, xj in zip(lam, x)]
        dev = [[0.0] * N for _ in range(D)]
        pts = [[] for _ in range(D)]
        grd = [[] for _ in range(D)]
        for _k in range(K):
            for j in range(D):
                lj, xj, dj = lam[j], x[j], dev[j]
                pj, qj = pts[j], grd[j]
                for i in range(N):
                    s = dj[i]
                    g = lj * (xj + s) + rng.gauss(0, SIG)
                    pj.append(s); qj.append(g)
                    dj[i] = s - ETA * g
        dbar = [sum(dev[j]) / N for j in range(D)]
        for j in range(D):
            ms = sum(pts[j]) / nk
            mg = sum(grd[j]) / nk
            wxx[j] += sum((p - ms) ** 2 for p in pts[j])
            wxg[j] += sum((p - ms) * (q - mg) for p, q in zip(pts[j], grd[j]))
        cs = coeffs(wxx, wxg, SIG * SIG)          # same-round: rounds <= t
        per.append((
            gain(dbar, gstar, lam),
            gain([dbar[j] * cs[j] for j in range(D)], gstar, lam),
            gain([dbar[j] * cc[j] for j in range(D)], gstar, lam),
            gain([dbar[j] * Cof(lam[j]) for j in range(D)], gstar, lam),
        ))
        x = [x[j] + dbar[j] for j in range(D)]
    return per


def main():
    rb = random.Random(0)
    lam = [rb.uniform(8, 12) for _ in range(N_OUT)] + \
          [rb.uniform(0.1, 1.0) for _ in range(D - N_OUT)]
    agg = [[0.0] * 4 for _ in range(T)]
    for r in range(REPS):
        for t, row in enumerate(run(77_000 + r, lam)):
            for q in range(4):
                agg[t][q] += row[q] / REPS

    def window(lo, hi):
        s = [0.0] * 4
        for t in range(lo, hi):
            for q in range(4):
                s[q] += agg[t][q]
        return s

    print(f"d={D} N={N} K={K} rounds={T} reps={REPS}")
    print()
    print("per-round gain profile (shows the weighting directly)")
    print(f"  {'round':>6} {'base gain':>11} {'share of total':>15} "
          f"{'same/base':>10} {'causal/base':>12}")
    tot = sum(agg[t][0] for t in range(T))
    for t in [0, 1, 2, 3, 5, 10, 15, 20, 30, 39]:
        b = agg[t][0]
        print(f"  {t:>6} {b:>11.3f} {100*b/tot:>14.1f}% "
              f"{agg[t][1]/b:>10.3f} {agg[t][2]/b:>12.3f}")
    print()
    for tag, lo, hi in (("ALL rounds", 0, T), (f"rounds >= {WARM}", WARM, T)):
        s = window(lo, hi)
        print(f"{tag:<18} base {s[0]:>8.3f} | same-round {s[1]/s[0]:.3f}  "
              f"causal {s[2]/s[0]:.3f}  oracleC {s[3]/s[0]:.3f}")
    print()
    print("If causal/base rises to meet same-round in the late window, the 1.00")
    print("was COLD START and stale curvature is fine -> async is safe.")
    print("If it stays at ~1.00 late, stale curvature genuinely fails.")


if __name__ == "__main__":
    main()
