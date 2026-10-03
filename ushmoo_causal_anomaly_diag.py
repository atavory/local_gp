"""Where do the causal and cross-fitted estimators actually differ?

Two conjectures already dead:
  - same-sample correlation: refuted, cross-fitting preserves the full gain
  - miscalibrated shrinkage: refuted, forcing s2=0.25 leaves causal at ~1.02

lam is exactly constant across rounds here, so a causal estimator pooling
rounds < t has strictly more data about the same target than a cross-fitted
estimator using half the workers of round t alone. It still loses, 1.00 vs 1.13.

So either (a) the causal lamhat is worse despite more data -- i.e. the pooled
estimator is biased -- or (b) both lamhat are fine and the gain does not come
from lamhat at all. Print the estimates and find out. Reported per curvature
bucket because the duality says the two ends behave differently.
"""

import math
import random

D, N_OUT, N, K, ETA, SIG, T, REPS = 120, 10, 32, 8, 0.05, 1.0, 30, 4
LMAX = 1.5 / ETA


def Cof(l):
    return 1.0 / (1.0 - (1.0 - ETA * min(max(l, 1e-9), LMAX)) ** K)


def run(seed, lam):
    rng = random.Random(seed)
    x = [rng.gauss(0, 1.0) for _ in range(D)]
    wxx = [0.0] * D; wxg = [0.0] * D
    axx = [0.0] * D; axg = [0.0] * D
    HA = list(range(N // 2))
    nk = N * K
    rows = []
    for t in range(T):
        # causal estimate from rounds < t
        caus = [wxg[j] / wxx[j] if wxx[j] > 0 else float("nan") for j in range(D)]
        dev = [[0.0] * N for _ in range(D)]
        pts = [[] for _ in range(D)]
        grd = [[] for _ in range(D)]
        for k in range(K):
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
            pa = [pts[j][k * N + i] for k in range(K) for i in HA]
            qa = [grd[j][k * N + i] for k in range(K) for i in HA]
            mp = sum(pa) / len(pa); mq = sum(qa) / len(qa)
            axx[j] += sum((v - mp) ** 2 for v in pa)
            axg[j] += sum((v - mp) * (w - mq) for v, w in zip(pa, qa))
        split = [axg[j] / axx[j] if axx[j] > 0 else float("nan") for j in range(D)]
        if t >= 5:
            rows.append((caus[:], split[:]))
        x = [x[j] + dbar[j] for j in range(D)]
    return rows


def main():
    rb = random.Random(0)
    lam = [rb.uniform(8, 12) for _ in range(N_OUT)] + \
          [rb.uniform(0.1, 1.0) for _ in range(D - N_OUT)]
    sc, ss, nn = [0.0] * D, [0.0] * D, 0
    bc, bs = [0.0] * D, [0.0] * D
    for r in range(REPS):
        for caus, split in run(777 + r, lam):
            nn += 1
            for j in range(D):
                if caus[j] == caus[j]:
                    sc[j] += abs(caus[j] - lam[j]) / lam[j]
                    bc[j] += (caus[j] - lam[j]) / lam[j]
                if split[j] == split[j]:
                    ss[j] += abs(split[j] - lam[j]) / lam[j]
                    bs[j] += (split[j] - lam[j]) / lam[j]

    flat = sorted(range(N_OUT, D), key=lambda j: lam[j])
    q = len(flat) // 3
    buckets = [("sharp lam~10", list(range(N_OUT))),
               ("flat  lam low", flat[:q]),
               ("flat  lam mid", flat[q:2 * q]),
               ("flat  lam high", flat[2 * q:])]
    print(f"d={D} N={N} K={K} rounds={T} reps={REPS}  (rounds>=5 scored)")
    print("causal = pooled over rounds<t, all N workers (MORE data)")
    print("split  = round t only, N/2 workers                (LESS data)")
    print()
    print(f"  {'bucket':<16} {'lam':>7} {'C_true':>8} | "
          f"{'|err| caus':>11} {'|err| split':>12} | "
          f"{'bias caus':>10} {'bias split':>11}")
    for name, idx in buckets:
        ml = sum(lam[j] for j in idx) / len(idx)
        mc = sum(Cof(lam[j]) for j in idx) / len(idx)
        ec = sum(sc[j] for j in idx) / len(idx) / nn
        es = sum(ss[j] for j in idx) / len(idx) / nn
        bbc = sum(bc[j] for j in idx) / len(idx) / nn
        bbs = sum(bs[j] for j in idx) / len(idx) / nn
        print(f"  {name:<16} {ml:>7.3f} {mc:>8.3f} | {ec:>11.3f} {es:>12.3f} | "
              f"{bbc:>+10.3f} {bbs:>+11.3f}")
    print()
    print("If 'bias caus' is large and 'bias split' is not, pooling across")
    print("rounds is biased and the extra data is being spent on a wrong target.")


if __name__ == "__main__":
    main()
