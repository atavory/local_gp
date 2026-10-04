"""Two constructive moves. Stop measuring against the wrong ceiling; pull the
one free design lever.

MOVE 1 -- the reference is wrong.
Every result in this project is reported as a fraction of the NEWTON oracle,
and every one therefore looks like a failure ("captures 6-10% of headroom").
But the duality says the Newton step in the flat directions is not merely hard
to estimate, it is unidentifiable from this design: eps_j is proportional to
C_j, so the correction owed and the error in estimating it scale together.
If that is right, Newton is not an achievable ceiling and the gap to it is not
a defect.

The achievable ceiling is the BAYES-OPTIMAL action under the true prior and the
actual likelihood -- the best any estimator can do with the information the
design contains. Measure that and the same numbers may read as "at the
frontier" rather than "6% of oracle". Computed here with the TRUE mixture prior
over lam (which the simulation knows), against our empirical-Bayes version
which has to fit the prior from data.

MOVE 2 -- heterogeneous K is the free lever.
The information-optimal local-step count is PER COORDINATE:
    rho_j = eta lam_j K = 0.4   =>   K*_j = 0.4 / (eta lam_j)
At eta=0.05 that is K*=0.8 for lam=10, K*=16 for lam=0.5, K*=80 for lam=0.1.
A spectrum spanning 100x in lam wants K spanning 100x, so NO SINGLE K serves
it, and homogeneous-K local SGD is necessarily mis-designed for most of the
spectrum. Running workers at geometrically spaced K puts every curvature scale
near some worker's sweet spot.

This costs nothing in communication (same number of syncs) and the config below
uses 25% FEWER gradient evaluations than the homogeneous baseline. Note the
inversion: heterogeneous local steps are a known NUISANCE for FedAvg (objective
inconsistency; FedNova), and here they are the design variable that makes
curvature estimable.
"""

import math
import random

D, N_OUT, ETA, SIG, T, REPS = 120, 10, 0.05, 1.0, 25, 6
KHOMO, N = 8, 32
# (count, K) groups: 192 worker-steps vs homogeneous 32*8 = 256
HET = [(16, 2), (8, 4), (4, 8), (2, 16), (2, 32)]
LMAX = 1.5 / ETA
ZG = [math.log(1e-3) + i * (math.log(LMAX) - math.log(1e-3)) / 49 for i in range(50)]
LG = [math.exp(z) for z in ZG]


def Cof(l, K):
    l = min(max(l, 1e-9), LMAX)
    return 1.0 / (1.0 - (1.0 - ETA * l) ** K)


def gain(u, gstar, lam):
    num = sum(a * b for a, b in zip(gstar, u))
    den = sum(l * x * x for l, x in zip(lam, u))
    return num * num / (2.0 * den) if den > 0 else 0.0


def true_prior_logpdf():
    """Mixture the sim actually draws from: 10/120 U[8,12], 110/120 U[0.1,1]."""
    w_s, w_f = N_OUT / D, (D - N_OUT) / D
    out = []
    for l in LG:
        p = 0.0
        if 8.0 <= l <= 12.0:
            p += w_s / 4.0
        if 0.1 <= l <= 1.0:
            p += w_f / 0.9
        out.append(math.log(p) if p > 0 else -1e30)
    return out


TRUE_LP = true_prior_logpdf()


def readout(bhat, v, logprior, K):
    """E[lam*C]/E[lam]: bounded, because lam*C -> 1/(eta K) as lam -> 0."""
    lw = [-((bhat - l) ** 2) / (2.0 * v) + lp for l, lp in zip(LG, logprior)]
    mx = max(lw)
    w = [math.exp(a - mx) for a in lw]
    sl = sum(wi * li for wi, li in zip(w, LG))
    if sl <= 0:
        return 1.0 / (ETA * K)
    return sum(wi * li * Cof(li, K) for wi, li in zip(w, LG)) / sl


def eb_logprior(ols):
    pos = [b for b in ols if 1e-3 < b < LMAX] or [1.0]
    m = sum(math.log(b) for b in pos) / len(pos)
    s2 = max(1e-2, sum((math.log(b) - m) ** 2 for b in pos) / len(pos))
    return [-((z - m) ** 2) / (2 * s2) for z in ZG]


def local_run(rng, lam, x, N_, K_, xx, xg):
    """N_ workers x K_ local steps from x; accumulate round-centred stats."""
    dev = [[0.0] * N_ for _ in range(D)]
    pts = [[] for _ in range(D)]
    grd = [[] for _ in range(D)]
    for _k in range(K_):
        for j in range(D):
            lj, xj, dj = lam[j], x[j], dev[j]
            pj, qj = pts[j], grd[j]
            for i in range(N_):
                s = dj[i]
                g = lj * (xj + s) + rng.gauss(0, SIG)
                pj.append(s); qj.append(g)
                dj[i] = s - ETA * g
    n = N_ * K_
    for j in range(D):
        ms = sum(pts[j]) / n
        mg = sum(grd[j]) / n
        xx[j] += sum((p - ms) ** 2 for p in pts[j])
        xg[j] += sum((p - ms) * (q - mg) for p, q in zip(pts[j], grd[j]))
    return [sum(dev[j]) / N_ for j in range(D)]


def ols_of(xx, xg):
    o, v = [], []
    for j in range(D):
        if xx[j] > 1e-300:
            o.append(xg[j] / xx[j]); v.append(SIG * SIG / xx[j])
        else:
            o.append(0.0); v.append(1e9)
    return o, v


# ----------------------------------------------------- MOVE 1: the frontier
def frontier(lam):
    keys = ("base", "eb", "frontier", "oracleC")
    acc = {k: 0.0 for k in keys}
    for r in range(REPS):
        rng = random.Random(90_000 + r)
        x = [rng.gauss(0, 1.0) for _ in range(D)]
        xx = [0.0] * D; xg = [0.0] * D
        for t in range(T):
            gstar = [l * xj for l, xj in zip(lam, x)]
            dbar = local_run(rng, lam, x, N, KHOMO, xx, xg)
            o, v = ols_of(xx, xg)
            lp_eb = eb_logprior(o)
            c_eb = [readout(o[j], v[j], lp_eb, KHOMO) for j in range(D)]
            c_fr = [readout(o[j], v[j], TRUE_LP, KHOMO) for j in range(D)]
            acc["base"] += gain(dbar, gstar, lam) / T / REPS
            acc["eb"] += gain([dbar[j] * c_eb[j] for j in range(D)],
                              gstar, lam) / T / REPS
            acc["frontier"] += gain([dbar[j] * c_fr[j] for j in range(D)],
                                    gstar, lam) / T / REPS
            acc["oracleC"] += gain([dbar[j] * Cof(lam[j], KHOMO)
                                    for j in range(D)], gstar, lam) / T / REPS
            x = [x[j] + dbar[j] for j in range(D)]
    b = acc["base"]
    print("=" * 72)
    print("MOVE 1: what is actually achievable with this design's information?")
    print("=" * 72)
    print(f"  {'arm':<46} {'gain':>9} {'/base':>8}")
    for k, nm in (("base", "tuned-scale FedAvg [BASELINE]"),
                  ("eb", "ours: posterior, prior FITTED from data"),
                  ("frontier", "FRONTIER: posterior, TRUE prior  <-- ceiling"),
                  ("oracleC", "Newton oracle (knows lam exactly)")):
        print(f"  {nm:<46} {acc[k]:>9.3f} {acc[k]/b:>8.3f}")
    fr, eb, orc = acc["frontier"]/b, acc["eb"]/b, acc["oracleC"]/b
    print()
    print(f"  fraction of FRONTIER captured : {(eb-1)/(fr-1):>6.1%}")
    print(f"  fraction of NEWTON captured   : {(eb-1)/(orc-1):>6.1%}")
    print("  If the first is high and the second low, the Newton gap is")
    print("  INFORMATION, not estimator quality, and we have been reporting")
    print("  against a ceiling the design cannot reach.")


# ------------------------------------------------- MOVE 2: K-diversity lever
def ksweep(lam):
    flat = sorted(range(N_OUT, D), key=lambda j: lam[j])
    h = len(flat) // 2
    buckets = [("sharp", list(range(N_OUT))), ("flat high lam", flat[h:]),
               ("flat LOW lam", flat[:h])]
    res = {}
    for tag in ("homo", "het"):
        err = [0.0] * D
        nn = 0
        for r in range(REPS):
            rng = random.Random(91_000 + r)
            x = [rng.gauss(0, 1.0) for _ in range(D)]
            xx = [0.0] * D; xg = [0.0] * D
            for t in range(T):
                if tag == "homo":
                    dbar = local_run(rng, lam, x, N, KHOMO, xx, xg)
                else:
                    ds = [local_run(rng, lam, x, cnt, kk, xx, xg)
                          for cnt, kk in HET]
                    # FedNova-style normalised mean so the step is comparable
                    tot = sum(c for c, _ in HET)
                    dbar = [sum(c * ds[q][j] / kk for q, (c, kk) in enumerate(HET))
                            / tot * KHOMO for j in range(D)]
                o, _ = ols_of(xx, xg)
                if t >= 5:
                    nn += 1
                    for j in range(D):
                        err[j] += abs(o[j] - lam[j]) / lam[j]
                x = [x[j] + dbar[j] for j in range(D)]
        res[tag] = [e / nn for e in err]

    print()
    print("=" * 72)
    print("MOVE 2: heterogeneous K. K*_j = 0.4/(eta lam_j) is PER COORDINATE.")
    print(f"  homogeneous: {N} workers x K={KHOMO}  = {N*KHOMO} worker-steps")
    hs = sum(c * k for c, k in HET)
    print(f"  heterogeneous: {HET}  = {hs} worker-steps "
          f"({100*hs/(N*KHOMO):.0f}% of homogeneous)")
    print("=" * 72)
    print(f"  {'bucket':<16} {'mean lam':>9} {'K* wanted':>10} "
          f"{'|err| homo':>11} {'|err| het':>10} {'ratio':>8}")
    for nm, idx in buckets:
        ml = sum(lam[j] for j in idx) / len(idx)
        ks = 0.4 / (ETA * ml)
        eh = sum(res["homo"][j] for j in idx) / len(idx)
        ht = sum(res["het"][j] for j in idx) / len(idx)
        print(f"  {nm:<16} {ml:>9.3f} {ks:>10.1f} {eh:>11.3f} {ht:>10.3f} "
              f"{ht/eh:>8.3f}")
    print("  ratio < 1 means heterogeneous K estimates that bucket BETTER,")
    print("  on 25% fewer gradients and the same number of synchronisations.")


def main():
    rb = random.Random(0)
    lam = [rb.uniform(8, 12) for _ in range(N_OUT)] + \
          [rb.uniform(0.1, 1.0) for _ in range(D - N_OUT)]
    print(f"d={D} sharp={N_OUT} eta={ETA} rounds={T} reps={REPS}\n")
    frontier(lam)
    ksweep(lam)


if __name__ == "__main__":
    main()
