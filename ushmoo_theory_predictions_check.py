"""Numerical test of four predictions from the local-SGD <-> GP duality.

DERIVATION (tested below; do not trust until the numbers agree).

On a quadratic, K local steps from a common x_t give mean displacement
    dbar_j = -g*_j (1 - a_j^K)/lam_j,    a_j = 1 - eta lam_j
so the step the server already has is the Newton step divided by

    C_j := 1 / (1 - (1-eta lam_j)^K)        "correction still owed"

C_j -> 1 when rho_j := eta lam_j K >> 1 (direction already converged to Newton)
C_j -> 1/rho_j when rho_j << 1 (direction barely moved, still a gradient step).

BENEFIT of applying C_j, per direction, as a fraction of that direction's
remaining loss:  b_j = (1 - 1/C_j)^2.  Zero at C=1, ->1 as C grows.

COST: C_j must be estimated. Two branches for the design spread Sxx_j.
  drift-dominated (|g_j| >> sigma sqrt(12/K)):
      eps_j := sd(lamhat_j)/lam_j ~ sqrt(12) sigma / (sqrt(N) |g_j| eta lam_j K^1.5)
      so eps_j = sqrt(12) C_j / (|g_j| sqrt(NK) / sigma)
  worker-noise-dominated (the flat directions, and the FL-relevant branch):
      Sxx_j ~ c' N K * (eta^2 sigma^2 K)  =>  eps_j ~ C_j / sqrt(c' N)

In BOTH branches eps_j is PROPORTIONAL TO C_j. The relative error of the
estimate and the size of the correction owed scale identically, so their ratio
does not depend on the direction's curvature at all. That is the duality:

    the directions with gain to give are exactly the directions you cannot measure.

PREDICTIONS
  P1  eps_j / C_j is constant across the spectrum (no lam dependence).
  P2  worker-noise branch: usable needs eps_j <~ 1, i.e. N >~ C_j^2 / c'.
      With c' ~ 0.4 this is N >~ 2.5 C_j^2. Sweep N, expect the corrected arm
      to cross the tuned-scale baseline near there.
  P3  pooling T rounds multiplies the design by T, so N*T is the governing
      product: 8 workers x 10 rounds should behave like 80 workers x 1 round.
  P4  benefit b_j = (1-1/C)^2 rises with C while precision 1/eps_j ~ 1/C falls.
      Their product (1-1/C)^2 / C is maximised at C = 3, i.e. rho = eta lam K
      ~ 0.405. So realised gain should be NON-MONOTONE in rho with an interior
      peak -- not "flatter is always better".
"""

import math
import random

ETA, SIG, K_DEF = 0.05, 1.0, 8


def gain(u, gstar, lam):
    num = sum(a * b for a, b in zip(gstar, u))
    den = sum(l * x * x for l, x in zip(lam, u))
    return num * num / (2.0 * den) if den > 0 else 0.0


def Ctrue(l, eta=ETA, K=K_DEF):
    return 1.0 / (1.0 - (1.0 - eta * l) ** K)


def simulate(rng, lam, N, K=K_DEF, eta=ETA, sig=SIG, T=1):
    """T independent rounds, each from a fresh x_t. Pools design stats over T.

    Returns the LAST round's (dbar, gstar) -- the step we must choose -- with
    curvature statistics pooled over all T rounds (curvature is static).
    """
    d = len(lam)
    sx = [0.0] * d; sg = [0.0] * d; sxx = [0.0] * d; sxg = [0.0] * d
    n = 0
    dbar = gstar = None
    for _ in range(T):
        xt = [rng.gauss(0, 1.0) for _ in range(d)]
        gstar = [l * x for l, x in zip(lam, xt)]
        dev = [[0.0] * N for _ in range(d)]
        for _k in range(K):
            for j in range(d):
                lj, xtj = lam[j], xt[j]
                dj = dev[j]
                for i in range(N):
                    x = dj[i]
                    g = lj * (xtj + x) + rng.gauss(0, sig)
                    sx[j] += x; sg[j] += g; sxx[j] += x * x; sxg[j] += x * g
                    dj[i] = x - eta * g
        n += N * K
        dbar = [sum(dev[j]) / N for j in range(d)]

    ols = []
    for j in range(d):
        cxx = sxx[j] - sx[j] * sx[j] / n
        cxg = sxg[j] - sx[j] * sg[j] / n
        ols.append((cxg / cxx, sig * sig / cxx) if cxx > 1e-300 else (0.0, 1e9))
    return dbar, gstar, ols


def eb_shrink(ols):
    """Pooled empirical Bayes = the simplest honest instance of GP marginal
    likelihood: one Gaussian prior on lam, hyperparameters from the evidence."""
    d = len(ols)
    b = [o[0] for o in ols]
    v = [o[1] for o in ols]
    mu0 = sum(b) / d
    tau2 = max(1e-9, sum((x - mu0) ** 2 for x in b) / d - sum(v) / d)
    return [(bj / vj + mu0 / tau2) / (1.0 / vj + 1.0 / tau2) for bj, vj in zip(b, v)]


def Chat(lh, eta=ETA, K=K_DEF):
    lh = min(max(lh, 1e-9), (2.0 - 1e-6) / eta)
    den = 1.0 - (1.0 - eta * lh) ** K
    return 1.0 / den if den > 1e-9 else 1e9


def arms(dbar, gstar, lam, ols, K=K_DEF):
    d = len(lam)
    out = {"base": gain(dbar, gstar, lam)}
    out["plugin"] = gain([dbar[j] * Chat(ols[j][0], K=K) for j in range(d)], gstar, lam)
    eb = eb_shrink(ols)
    out["eb"] = gain([dbar[j] * Chat(eb[j], K=K) for j in range(d)], gstar, lam)
    out["oracle"] = gain([dbar[j] * Ctrue(lam[j], K=K) for j in range(d)], gstar, lam)
    return out


def bed(seed=0, d=200, n_out=12):
    rng = random.Random(seed)
    return [rng.uniform(8, 12) for _ in range(n_out)] + \
           [rng.uniform(0.1, 1.0) for _ in range(d - n_out)]


# ---------------------------------------------------------------- P1 and P2
def exp_A(reps=25):
    lam = bed()
    d, n_out = len(lam), 12
    print("=" * 74)
    print("P1/P2  sweep N on the sharp+flat bed (d=200, 12 sharp, K=8, eta=0.05)")
    print(f"       C_true: sharp={Ctrue(10.0):.3f}  flat={Ctrue(0.5):.3f}")
    print(f"       P2 says usable at N >~ 2.5*C^2 = {2.5*Ctrue(0.5)**2:.0f} for the flat dirs")
    print("=" * 74)
    print(f"  {'N':>5} {'base':>9} {'plugin':>9} {'eb':>9} {'oracle':>9} "
          f"{'eb/base':>9} {'orc/base':>9}")
    for N in (8, 16, 32, 64, 128, 256):
        acc = {k: 0.0 for k in ("base", "plugin", "eb", "oracle")}
        for r in range(reps):
            rng = random.Random(10_000 + r)
            dbar, gstar, ols = simulate(rng, lam, N)
            a = arms(dbar, gstar, lam, ols)
            for k in acc:
                acc[k] += a[k] / reps
        print(f"  {N:>5} {acc['base']:>9.2f} {acc['plugin']:>9.2f} {acc['eb']:>9.2f} "
              f"{acc['oracle']:>9.2f} {acc['eb']/acc['base']:>9.3f} "
              f"{acc['oracle']/acc['base']:>9.3f}")

    # P1: eps_j / C_j constant across the spectrum?
    print()
    print("P1  eps_j = sd(lamhat)/lam, measured per curvature bucket, N=32")
    reps2 = 40
    err = [0.0] * d
    for r in range(reps2):
        rng = random.Random(777 + r)
        _, _, ols = simulate(rng, lam, 32)
        for j in range(d):
            err[j] += abs(ols[j][0] - lam[j]) / lam[j] / reps2
    buckets = [("sharp  lam~10", range(n_out))]
    flat = sorted(range(n_out, d), key=lambda j: lam[j])
    q = len(flat) // 3
    buckets += [("flat   lam low", flat[:q]), ("flat   lam mid", flat[q:2 * q]),
                ("flat   lam high", flat[2 * q:])]
    print(f"  {'bucket':<16} {'mean lam':>9} {'eps':>9} {'C_true':>9} {'eps/C':>9}")
    for name, idx in buckets:
        idx = list(idx)
        ml = sum(lam[j] for j in idx) / len(idx)
        me = sum(err[j] for j in idx) / len(idx)
        mc = sum(Ctrue(lam[j]) for j in idx) / len(idx)
        print(f"  {name:<16} {ml:>9.3f} {me:>9.3f} {mc:>9.3f} {me/mc:>9.3f}")
    print("  P1 holds if the last column is roughly constant down the rows.")


# ---------------------------------------------------------------------- P3
def exp_B(reps=25):
    lam = bed()
    print()
    print("=" * 74)
    print("P3  do pooled rounds substitute for workers?  N=8 fixed, sweep T")
    print("    P2/P3 predict the governing product is N*T, so T=10 at N=8")
    print(f"    should behave like N={8*10} at T=1 (compare with the P2 table)")
    print("=" * 74)
    print(f"  {'T':>4} {'N*T':>6} {'base':>9} {'plugin':>9} {'eb':>9} {'eb/base':>9}")
    for T in (1, 2, 4, 8, 16, 32):
        acc = {k: 0.0 for k in ("base", "plugin", "eb", "oracle")}
        for r in range(reps):
            rng = random.Random(20_000 + r)
            dbar, gstar, ols = simulate(rng, lam, 8, T=T)
            a = arms(dbar, gstar, lam, ols)
            for k in acc:
                acc[k] += a[k] / reps
        print(f"  {T:>4} {8*T:>6} {acc['base']:>9.2f} {acc['plugin']:>9.2f} "
              f"{acc['eb']:>9.2f} {acc['eb']/acc['base']:>9.3f}")


# ---------------------------------------------------------------------- P4
def exp_C(reps=20):
    print()
    print("=" * 74)
    print("P4  sweep the spectrum CENTRE to sweep rho = eta*lam*K.")
    print("    Interior peak predicted near C=3  <=>  rho ~ 0.405.")
    print("    (spread held at 20x so there is a shape correction to make at all)")
    print("=" * 74)
    print(f"  {'centre lam':>11} {'rho':>7} {'C(centre)':>10} {'base':>9} "
          f"{'eb':>9} {'oracle':>9} {'eb/base':>9} {'orc/base':>9}")
    d, N = 100, 32
    for c in (0.1, 0.25, 0.6, 1.5, 4.0, 10.0, 25.0):
        rngb = random.Random(5)
        r = math.sqrt(20.0)
        lam = [c * math.exp(rngb.uniform(-math.log(r), math.log(r))) for _ in range(d)]
        if max(lam) * ETA >= 1.9:
            print(f"  {c:>11.2f}  (skipped: eta*lam_max={max(lam)*ETA:.2f} unstable)")
            continue
        acc = {k: 0.0 for k in ("base", "plugin", "eb", "oracle")}
        for rep in range(reps):
            rng = random.Random(30_000 + rep)
            dbar, gstar, ols = simulate(rng, lam, N)
            a = arms(dbar, gstar, lam, ols)
            for k in acc:
                acc[k] += a[k] / reps
        print(f"  {c:>11.2f} {ETA*c*K_DEF:>7.3f} {Ctrue(c):>10.3f} "
              f"{acc['base']:>9.2f} {acc['eb']:>9.2f} {acc['oracle']:>9.2f} "
              f"{acc['eb']/acc['base']:>9.3f} {acc['oracle']/acc['base']:>9.3f}")
    print("  orc/base is the HEADROOM curve (peaks where correction is worth most);")
    print("  eb/base is the REALISED curve (peaks where it is also estimable).")


if __name__ == "__main__":
    exp_A()
    exp_B()
    exp_C()
