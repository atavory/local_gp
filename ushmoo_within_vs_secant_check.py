"""Corrected pooling, plus the control this lane actually has to beat.

TWO FIXES to ushmoo_posterior_ratio_check.py, which reported the posterior arm
pinned at 1.00 +- 0.02 for every N from 8 to 128.

FIX 1 (bug). That script pooled rounds into one regression with a single
intercept. The true per-round model is g = lam*(x_t + s) + xi, so each round has
its OWN intercept lam*x_t, and the within-round mean displacement is
mean(s_t) ~ -eta*K*lam*x_t -- perfectly anti-correlated with that intercept.
Pooling with a common intercept is textbook omitted-variable bias, and the
between-round cluster means trace a line of slope ~ -lam/(eta K): the WRONG
SIGN. The bias does not shrink with N, which is very likely why the arm sat at
parity for every N. Corrected here by centring within each round before
accumulating -- the fixed-effects estimator.

FIX 2 (missing control). Curvature has a second source that is FREE and
CLASSICAL, and the project never armed it. Workers evaluate the gradient at
x_t at local step 0, so the server already has gbar_t = mean_i g_i0 at every
round. Across consecutive rounds:

    gbar_{t+1} - gbar_t  =  H (x_{t+1} - x_t) + noise

which is the ordinary quasi-Newton secant equation. This needs no extra logging,
no extra communication, and no intermediate local points. It is what L-BFGS on
the server would use.

So the project's thesis -- that the INTERMEDIATE local points carry curvature
worth having -- is only interesting if the within-round estimator adds something
over the across-round secant. That comparison is the real gate, and it is the
analogue of the tuned-constant control that killed five Double Fan arms.

Arms, all scored as one-step gain on the SAME FedAvg trajectory:
  base          tuned-scale FedAvg
  within        curvature from within-round local trajectories (the thesis)
  secant        curvature from across-round server secants (free, classical)
  both          the two designs pooled
  oracleC       true curvature, the idea's ceiling
  ceiling       diagonal Newton
Every estimator arm uses the same log-normal-prior posterior read-out
E[lam C]/E[lam], so the arms differ ONLY in which design feeds them.
"""

import math
import random

D, N_OUT, K, ETA, SIG, T, REPS = 120, 10, 8, 0.05, 1.0, 30, 5
NS = (8, 32, 128)
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
    return 1.0 / (1.0 - (1.0 - ETA * min(max(l, 1e-9), LMAX)) ** K)


def ratio(bhat, v, m, s2):
    """E[lam*C]/E[lam]: finite even when the posterior has mass at lam -> 0,
    because lam*C is bounded (-> 1/(eta K)) while C alone is not."""
    lw = [-((bhat - l) ** 2) / (2.0 * v) - ((z - m) ** 2) / (2.0 * s2)
          for l, z in zip(LG, ZG)]
    mx = max(lw)
    w = [math.exp(a - mx) for a in lw]
    sl = sum(wi * li for wi, li in zip(w, LG))
    return (sum(wi * lc for wi, lc in zip(w, LCG)) / sl) if sl > 0 \
        else 1.0 / (ETA * K)


def coeffs(acc_xx, acc_xg, nvar, D_, s2fix=None):
    """OLS + sampling variance per coordinate, then pooled-prior posterior."""
    ols, vv = [], []
    for j in range(D_):
        if acc_xx[j] > 1e-300:
            ols.append(acc_xg[j] / acc_xx[j]); vv.append(nvar / acc_xx[j])
        else:
            ols.append(0.0); vv.append(1e9)
    pos = [b for b in ols if 1e-3 < b < LMAX] or [1.0]
    m = sum(math.log(b) for b in pos) / len(pos)
    s2 = max(1e-2, sum((math.log(b) - m) ** 2 for b in pos) / len(pos))
    if s2fix is not None:
        s2 = s2fix
    return [ratio(ols[j], vv[j], m, s2) for j in range(D_)]


def run(seed, lam, N):
    rng = random.Random(seed)
    x = [rng.gauss(0, 1.0) for _ in range(D)]
    wxx = [0.0] * D; wxg = [0.0] * D          # within-round, round-centred
    axx = [0.0] * D; axg = [0.0] * D          # worker half A only
    bxx = [0.0] * D; bxg = [0.0] * D          # worker half B only
    sxx = [0.0] * D; sxg = [0.0] * D          # across-round secants
    prev_x = prev_g = None
    keys = ("base", "within", "within_causal", "within_split", "secant",
            "both", "oracleC", "ceiling", "causal_fixprior")
    acc = {k: 0.0 for k in keys}
    nk = N * K
    HA = list(range(N // 2)); HB = list(range(N // 2, N))
    for t in range(T):
        # strictly causal coefficients: fitted on rounds < t only, so they
        # share no noise with this round's dbar
        cw_c = coeffs(wxx, wxg, SIG * SIG, D)
        cw_cf = coeffs(wxx, wxg, SIG * SIG, D, s2fix=0.25)
        gstar = [l * xj for l, xj in zip(lam, x)]
        dev = [[0.0] * N for _ in range(D)]
        pts = [[] for _ in range(D)]
        grd = [[] for _ in range(D)]
        g0 = [0.0] * D
        for k in range(K):
            for j in range(D):
                lj, xj, dj = lam[j], x[j], dev[j]
                pj, qj = pts[j], grd[j]
                for i in range(N):
                    s = dj[i]
                    g = lj * (xj + s) + rng.gauss(0, SIG)
                    pj.append(s); qj.append(g)
                    if k == 0:
                        g0[j] += g / N
                    dj[i] = s - ETA * g
        dbar = [sum(dev[j]) / N for j in range(D)]

        # FIX 1: centre within this round before pooling (fixed effects)
        for j in range(D):
            ms = sum(pts[j]) / nk
            mg = sum(grd[j]) / nk
            wxx[j] += sum((p - ms) ** 2 for p in pts[j])
            wxg[j] += sum((p - ms) * (q - mg) for p, q in zip(pts[j], grd[j]))
        # FIX 3: cross-fitting. pts/grd are laid out [k][j][i] flattened as
        # index k*N + i within pts[j], so worker i's points are at stride N.
        for j in range(D):
            pa = [pts[j][k * N + i] for k in range(K) for i in HA]
            qa = [grd[j][k * N + i] for k in range(K) for i in HA]
            pb = [pts[j][k * N + i] for k in range(K) for i in HB]
            qb = [grd[j][k * N + i] for k in range(K) for i in HB]
            for P, Q, XX, XG in ((pa, qa, axx, axg), (pb, qb, bxx, bxg)):
                mp = sum(P) / len(P); mq = sum(Q) / len(Q)
                XX[j] += sum((v - mp) ** 2 for v in P)
                XG[j] += sum((v - mp) * (w - mq) for v, w in zip(P, Q))
        dbarA = [sum(dev[j][i] for i in HA) / len(HA) for j in range(D)]
        dbarB = [sum(dev[j][i] for i in HB) / len(HB) for j in range(D)]

        acc["base"] += gain(dbar, gstar, lam) / T
        acc["oracleC"] += gain([dbar[j] * Cof(lam[j]) for j in range(D)],
                               gstar, lam) / T
        acc["ceiling"] += 0.5 * sum(g * g / l for g, l in zip(gstar, lam)) / T

        # --- across-round secants ---
        if prev_x is not None:
            for j in range(D):
                dxj = x[j] - prev_x[j]
                dgj = g0[j] - prev_g[j]
                sxx[j] += dxj * dxj
                sxg[j] += dxj * dgj
        prev_x, prev_g = list(x), list(g0)

        cw = coeffs(wxx, wxg, SIG * SIG, D)
        cs = coeffs(sxx, sxg, 2.0 * SIG * SIG / N, D)
        cb = coeffs([wxx[j] + sxx[j] for j in range(D)],
                    [wxg[j] + sxg[j] for j in range(D)], SIG * SIG, D)
        acc["within"] += gain([dbar[j] * cw[j] for j in range(D)], gstar, lam) / T
        acc["within_causal"] += gain([dbar[j] * cw_c[j] for j in range(D)],
                                     gstar, lam) / T
        cA = coeffs(axx, axg, SIG * SIG, D)
        cB = coeffs(bxx, bxg, SIG * SIG, D)
        acc["within_split"] += gain(
            [0.5 * (cA[j] * dbarB[j] + cB[j] * dbarA[j]) for j in range(D)],
            gstar, lam) / T
        acc["causal_fixprior"] += gain([dbar[j] * cw_cf[j] for j in range(D)],
                                       gstar, lam) / T
        acc["secant"] += gain([dbar[j] * cs[j] for j in range(D)], gstar, lam) / T
        acc["both"] += gain([dbar[j] * cb[j] for j in range(D)], gstar, lam) / T

        x = [x[j] + dbar[j] for j in range(D)]
    return acc


def main():
    rb = random.Random(0)
    lam = [rb.uniform(8, 12) for _ in range(N_OUT)] + \
          [rb.uniform(0.1, 1.0) for _ in range(D - N_OUT)]
    keys = ("base", "within", "within_causal", "within_split", "secant",
            "both", "oracleC", "ceiling", "causal_fixprior")
    print(f"d={D} sharp={N_OUT} K={K} eta={ETA} rounds={T} reps={REPS}")
    print(f"C_true: sharp={Cof(10.0):.3f} flat(0.5)={Cof(0.5):.3f} "
          f"flattest(0.1)={Cof(0.1):.3f}")
    print("ratios to the tuned-scale FedAvg baseline at the same N")
    print()
    print(f"  {'N':>5} {'base':>8} | {'within':>8} {'wthn_caus':>10} "
          f"{'wthn_split':>11} {'caus_fixpr':>11} {'oracleC':>8} {'ceiling':>8}")
    for N in NS:
        acc = {k: 0.0 for k in keys}
        for r in range(REPS):
            o = run(31_000 + r, lam, N)
            for k in keys:
                acc[k] += o[k] / REPS
        b = acc["base"]
        print(f"  {N:>5} {b:>8.3f} | {acc['within']/b:>8.3f} "
              f"{acc['within_causal']/b:>10.3f} {acc['within_split']/b:>11.3f} "
              f"{acc['causal_fixprior']/b:>11.3f} {acc['oracleC']/b:>8.3f} "
              f"{acc['ceiling']/b:>8.3f}")
    print()
    print("'within' is the project's thesis. 'secant' is free and classical.")
    print("If secant >= within, the intermediate local points add nothing and")
    print("the right answer is server-side quasi-Newton, which already exists.")


if __name__ == "__main__":
    main()
