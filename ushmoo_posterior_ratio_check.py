"""The right posterior read-out, and the N at which it pays.

ushmoo_bayes_step_check.py left a clean gap on a real FedAvg trajectory:

    TRUE curvature x Wiener shrink        1.235   <- the idea is worth ~24%
    MAP curvature  x Wiener shrink        0.972   <- the estimator cannot reach it
    plug-in C(lamhat)                     0.144   <- unregularised is catastrophic

Shrinking for Delta-bar's noise cost almost nothing (1.249 -> 1.235), so that
obstruction is not binding. Estimating lam in the flat directions is.

But C(lam_MAP) is the wrong functional. The quantity the posterior should
supply is not a plug-in correction; it is the Bayes step itself. Minimising
E[f(x+delta)] = sum_j (E[g_j] delta_j + 0.5 E[lam_j] delta_j^2) gives

    delta_j = -E[g_j]/E[lam_j],

and since the displacement law says g*_j = -dbar_j lam_j C_j exactly,

    delta_j  =  dbar_j * E[lam_j C_j] / E[lam_j].

This matters because of a structural fact: lam*C(lam) = lam/(1-(1-eta lam)^K)
is BOUNDED and smooth -- it tends to 1/(eta K) as lam -> 0 and to lam as lam
grows. C(lam) alone is unbounded at lam -> 0, and E[C] under any posterior with
mass near zero is infinite. So the ratio of posterior expectations is finite
where the plug-in is not, and no clipping, flooring or Tikhonov knob is needed.
The regularisation is supplied by the prior, which is what a GP does.

Posterior computed on a grid in z = log lam (positivity by construction),
likelihood lamhat ~ N(lam, v) from the pooled trajectory regression, prior
log lam ~ N(m, s^2) fitted by pooling across coordinates -- the one-dimensional
instance of ARD hyperparameter learning by marginal likelihood.

Then: sweep N. The single number that decides the lane is the N at which the
posterior arm crosses the tuned-scale baseline.
"""

import math
import random

D, N_OUT, K, ETA, SIG, T, REPS = 150, 10, 8, 0.05, 1.0, 25, 6
NS = (8, 16, 32, 64, 128)
LMAX = 1.5 / ETA
ZG = [math.log(1e-3) + i * (math.log(LMAX) - math.log(1e-3)) / 39 for i in range(40)]
LG = [math.exp(z) for z in ZG]
CG = [1.0 / (1.0 - (1.0 - ETA * l) ** K) for l in LG]
LCG = [l * c for l, c in zip(LG, CG)]


def gain(u, gstar, lam):
    num = sum(a * b for a, b in zip(gstar, u))
    den = sum(l * x * x for l, x in zip(lam, u))
    return num * num / (2.0 * den) if den > 0 else 0.0


def Cof(l):
    l = min(max(l, 1e-9), LMAX)
    return 1.0 / (1.0 - (1.0 - ETA * l) ** K)


def post_ratio(bhat, v, m, s2):
    """E[lam*C]/E[lam] under the log-normal-prior posterior, on the z grid."""
    lw = [-((bhat - l) ** 2) / (2.0 * v) - ((z - m) ** 2) / (2.0 * s2)
          for l, z in zip(LG, ZG)]
    mx = max(lw)
    w = [math.exp(a - mx) for a in lw]
    sl = sum(wi * li for wi, li in zip(w, LG))
    slc = sum(wi * lc for wi, lc in zip(w, LCG))
    return slc / sl if sl > 0 else 1.0 / (ETA * K)


def run(seed, lam, N):
    rng = random.Random(seed)
    x = [rng.gauss(0, 1.0) for _ in range(D)]
    sx = [0.0] * D; sg = [0.0] * D; sxx = [0.0] * D; sxg = [0.0] * D
    n = 0
    keys = ("base", "oracleC_shr", "mapC_shr", "post", "post_shr", "ceiling")
    acc = {k: 0.0 for k in keys}
    for t in range(T):
        gstar = [l * xj for l, xj in zip(lam, x)]
        dev = [[0.0] * N for _ in range(D)]
        for _k in range(K):
            for j in range(D):
                lj, xj, dj = lam[j], x[j], dev[j]
                for i in range(N):
                    s = dj[i]
                    g = lj * (xj + s) + rng.gauss(0, SIG)
                    sx[j] += s; sg[j] += g; sxx[j] += s * s; sxg[j] += s * g
                    dj[i] = s - ETA * g
        n += N * K
        dbar = [sum(dev[j]) / N for j in range(D)]
        nu2 = [sum((v - dbar[j]) ** 2 for v in dev[j]) / (N - 1) / N for j in range(D)]
        shr = [max(0.0, 1.0 - nu2[j] / (dbar[j] ** 2)) if dbar[j] else 0.0
               for j in range(D)]

        ols, vv = [], []
        for j in range(D):
            cxx = sxx[j] - sx[j] * sx[j] / n
            cxg = sxg[j] - sx[j] * sg[j] / n
            if cxx > 1e-300:
                ols.append(cxg / cxx); vv.append(SIG * SIG / cxx)
            else:
                ols.append(0.0); vv.append(1e9)
        pos = [b for b in ols if 1e-3 < b < LMAX] or [1.0]
        m = sum(math.log(b) for b in pos) / len(pos)
        s2 = max(1e-2, sum((math.log(b) - m) ** 2 for b in pos) / len(pos))
        pr = [post_ratio(ols[j], vv[j], m, s2) for j in range(D)]
        mp = [Cof(math.exp(m)) for _ in range(D)]  # placeholder, replaced below
        # MAP arm, for continuity with the previous script
        mp = []
        for j in range(D):
            lo, hi = m - 12 * math.sqrt(s2), m + 12 * math.sqrt(s2)

            def dgf(z, b=ols[j], v=vv[j]):
                l = math.exp(z)
                return (b - l) * l / v - (z - m) / s2
            flo = dgf(lo)
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if (dgf(mid) >= 0) == (flo >= 0):
                    lo = mid
                else:
                    hi = mid
            mp.append(math.exp(0.5 * (lo + hi)))

        acc["base"] += gain(dbar, gstar, lam) / T
        acc["oracleC_shr"] += gain([dbar[j] * Cof(lam[j]) * shr[j]
                                    for j in range(D)], gstar, lam) / T
        acc["mapC_shr"] += gain([dbar[j] * Cof(mp[j]) * shr[j]
                                 for j in range(D)], gstar, lam) / T
        # posterior ratio arm: coefficient is E[lam C]/E[lam], no clipping
        acc["post"] += gain([dbar[j] * pr[j] / max(ols[j], 1e-9) if False else
                             dbar[j] * pr[j] for j in range(D)], gstar, lam) / T
        acc["post_shr"] += gain([dbar[j] * pr[j] * shr[j] for j in range(D)],
                                gstar, lam) / T
        acc["ceiling"] += 0.5 * sum(g * g / l for g, l in zip(gstar, lam)) / T
        x = [x[j] + dbar[j] for j in range(D)]
    return acc


def main():
    rb = random.Random(0)
    lam = [rb.uniform(8, 12) for _ in range(N_OUT)] + \
          [rb.uniform(0.1, 1.0) for _ in range(D - N_OUT)]
    keys = ("base", "oracleC_shr", "mapC_shr", "post", "post_shr", "ceiling")
    print(f"d={D} sharp={N_OUT} K={K} eta={ETA} rounds={T} reps={REPS}")
    print(f"C_true: sharp={Cof(10.0):.3f} flat(0.5)={Cof(0.5):.3f} "
          f"flattest(0.1)={Cof(0.1):.3f}")
    print("all ratios are to the tuned-scale FedAvg baseline at the same N")
    print()
    print(f"  {'N':>5} {'base':>8} | {'oracleC':>8} {'mapC':>8} {'post':>8} "
          f"{'post+shr':>9} {'ceiling':>8}")
    for N in NS:
        acc = {k: 0.0 for k in keys}
        for r in range(REPS):
            o = run(9000 + r, lam, N)
            for k in keys:
                acc[k] += o[k] / REPS
        b = acc["base"]
        print(f"  {N:>5} {b:>8.3f} | {acc['oracleC_shr']/b:>8.3f} "
              f"{acc['mapC_shr']/b:>8.3f} {acc['post']/b:>8.3f} "
              f"{acc['post_shr']/b:>9.3f} {acc['ceiling']/b:>8.3f}")
    print()
    print("oracleC = TRUE curvature, the idea's ceiling. post = Bayes ratio")
    print("E[lam C]/E[lam], no clipping. Crossing 1.000 is the whole question.")


if __name__ == "__main__":
    main()
