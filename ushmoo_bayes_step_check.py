"""The decisive experiment: is there gain left after Delta-bar's own noise?

Three things the first prediction sweep (ushmoo_theory_predictions_check.py)
forced me to fix.

1. P2 FAILED outright: more workers made the C-corrected arm WORSE
   (eb/base 0.058 -> 0.008 from N=8 to N=256). Cause: C_j = 1/(1-(1-eta lam_j)^K)
   is unbounded as lam_j -> 0, a Gaussian prior lets the posterior mean of lam_j
   reach zero, and a single blown-up coordinate captures the whole step. The
   binding constraint is min_j lam_j, not typical lam_j. Real Hessians have
   near-zero and negative eigenvalues, so an unregularised 1/C is never safe.

2. P4's headroom curve came out BELOW 1 at rho=0.04 (orc/base = 0.890) -- the
   TRUE-curvature correction losing to plain rescaling. That cannot be an
   estimation problem, and it exposes a second obstruction I had missed:

   Delta-bar_j is itself noisy, with standard error nu_j = sd_i(dev_ij)/sqrt(N).
   Multiplying by C_j scales signal and noise together, so it does not change
   per-coordinate SNR -- but it REWEIGHTS the step toward the coordinates with
   the lowest SNR, because a flat direction's displacement is small precisely
   by the factor 1/C_j. The correction systematically up-weights the noisiest
   components.

   So the Bayes-optimal coefficient is the correction TIMES a Wiener shrinkage:

       c_j = [ 1/(1-(1-eta lam_j)^K) ]  x  [ SNR_j^2 / (1 + SNR_j^2) ]
             \_____ debias local steps ____/  \___ shrink Delta-bar noise ___/

   with SNR_j^2 = max(0, dbar_j^2/nu_j^2 - 1) estimable FROM THE WORKER SPREAD
   ALONE. Note c_j = C_j/(1 + C_j^2/A^2)-shaped: the Tikhonov form that
   ushmoo_field_repair_check.py had to tune on the test set is what the
   posterior gives for free, with the knob set by the measured noise.

   This is also the first principled use this programme has found for the
   worker cloud. The cloud carries error SCALE, never direction -- 112
   experiments established that. Scale is exactly what the shrinkage factor
   needs. The cloud does not tell you where to step; it tells you how far to
   trust the curvature correction.

3. P3's magnitude was an artifact. exp_B drew a FRESH x_t ~ N(0,I) each round,
   so pooling rounds injected design spread of order 1 against a within-round
   spread of order eta^2 sigma^2 K ~ 0.02 -- a 50x free gift that consecutive
   real rounds do not give. Here the iterate follows an actual FedAvg
   trajectory, x_{t+1} = x_t + dbar_t.

All arms are scored on the SAME FedAvg trajectory (one-step gain at each round),
so no arm gets a different bed. Curvature stats are pooled causally: at round t
only rounds <= t are used.
"""

import math
import random

D, N_OUT, N, K, ETA, SIG, T, REPS = 200, 12, 16, 8, 0.05, 1.0, 40, 10


def gain(u, gstar, lam):
    num = sum(a * b for a, b in zip(gstar, u))
    den = sum(l * x * x for l, x in zip(lam, u))
    return num * num / (2.0 * den) if den > 0 else 0.0


def Cof(l, eta=ETA, K=K):
    l = min(max(l, 1e-9), (2.0 - 1e-6) / eta)
    den = 1.0 - (1.0 - eta * l) ** K
    return 1.0 / den if den > 1e-12 else 1e12


def log_prior_map(bhat, v, m, s2):
    """Posterior mode of lam under lam>0: lam_hat ~ N(lam, v), log lam ~ N(m, s2).

    Positivity is the whole point -- a Gaussian prior on lam lets the posterior
    reach 0 and 1/C explodes. Parameterising in log space is what every GP
    package does for lengthscales, and it is exactly the fix for this pathology.
    Solved by bisection on the stationarity condition in z = log lam.
    """
    def dg(z):
        l = math.exp(z)
        return (bhat - l) * l / v - (z - m) / s2
    lo, hi = m - 12.0 * math.sqrt(s2), m + 12.0 * math.sqrt(s2)
    flo = dg(lo)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if (dg(mid) >= 0) == (flo >= 0):
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def run(seed, lam):
    rng = random.Random(seed)
    x = [rng.gauss(0, 1.0) for _ in range(D)]
    sx = [0.0] * D; sg = [0.0] * D; sxx = [0.0] * D; sxg = [0.0] * D
    n = 0
    acc = {k: 0.0 for k in ("base", "oracleC", "oracleC_shr", "plugC",
                            "mapC_shr", "oracle_full")}
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
        # standard error of the mean displacement, from the worker spread alone
        nu2 = [sum((v - dbar[j]) ** 2 for v in dev[j]) / (N - 1) / N for j in range(D)]
        shr = [max(0.0, 1.0 - nu2[j] / (dbar[j] ** 2)) if dbar[j] != 0 else 0.0
               for j in range(D)]

        ols, vv = [], []
        for j in range(D):
            cxx = sxx[j] - sx[j] * sx[j] / n
            cxg = sxg[j] - sx[j] * sg[j] / n
            if cxx > 1e-300:
                ols.append(cxg / cxx); vv.append(SIG * SIG / cxx)
            else:
                ols.append(0.0); vv.append(1e9)
        pos = [b for b in ols if b > 0] or [1.0]
        m = sum(math.log(b) for b in pos) / len(pos)
        s2 = max(1e-3, sum((math.log(b) - m) ** 2 for b in pos) / len(pos))
        mapl = [log_prior_map(ols[j], vv[j], m, s2) for j in range(D)]

        acc["base"] += gain(dbar, gstar, lam) / T
        acc["oracleC"] += gain([dbar[j] * Cof(lam[j]) for j in range(D)],
                               gstar, lam) / T
        acc["oracleC_shr"] += gain([dbar[j] * Cof(lam[j]) * shr[j] for j in range(D)],
                                   gstar, lam) / T
        acc["plugC"] += gain([dbar[j] * Cof(ols[j]) for j in range(D)],
                             gstar, lam) / T
        acc["mapC_shr"] += gain([dbar[j] * Cof(mapl[j]) * shr[j] for j in range(D)],
                                gstar, lam) / T
        acc["oracle_full"] += 0.5 * sum(g * g / l for g, l in zip(gstar, lam)) / T

        x = [x[j] + dbar[j] for j in range(D)]
    return acc


def main():
    rb = random.Random(0)
    lam = [rb.uniform(8, 12) for _ in range(N_OUT)] + \
          [rb.uniform(0.1, 1.0) for _ in range(D - N_OUT)]
    keys = ["base", "oracleC", "oracleC_shr", "plugC", "mapC_shr", "oracle_full"]
    acc = {k: 0.0 for k in keys}
    for r in range(REPS):
        o = run(4000 + r, lam)
        for k in keys:
            acc[k] += o[k] / REPS

    b = acc["base"]
    print(f"d={D} sharp={N_OUT} N={N} K={K} eta={ETA} rounds={T} reps={REPS}")
    print(f"C_true: sharp={Cof(10.0):.3f}  flat(lam=0.5)={Cof(0.5):.3f}  "
          f"flattest(lam=0.1)={Cof(0.1):.3f}")
    print("mean one-step gain on a real FedAvg trajectory, causal pooling")
    print()
    print(f"  {'arm':<40} {'gain':>10} {'/base':>9}")
    lbl = [("base", "FedAvg dbar, best-scaled [BASELINE]"),
           ("plugC", "x C(lamhat), plug-in, no shrinkage"),
           ("oracleC", "x C(lam) TRUE, no shrinkage"),
           ("oracleC_shr", "x C(lam) TRUE x Wiener shrink"),
           ("mapC_shr", "x C(lam_MAP, log-prior) x Wiener shrink  <-- method"),
           ("oracle_full", "diagonal Newton oracle [CEILING]")]
    for k, name in lbl:
        print(f"  {name:<40} {acc[k]:>10.3f} {acc[k]/b:>9.3f}")
    print()
    print("Decisive row is 'TRUE x Wiener shrink': the ceiling of the whole idea")
    print("once Delta-bar's own noise is charged. If that is ~1.000 the lane is")
    print("closed by noise, not by estimation, and no estimator can rescue it.")


if __name__ == "__main__":
    main()
