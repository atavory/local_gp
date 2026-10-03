"""Mechanism: K local steps ALREADY apply the curvature correction.

ushmoo_field_repair_check.py produced the decisive row: preconditioning the mean
proposal by the TRUE inverse curvature scores 0.945x the plain scale baseline --
worse than doing nothing, with perfect knowledge. That cannot be an estimation
problem. Proposed mechanism:

On a quadratic, K local steps from x_t with step eta contract as a_j = 1-eta*lam_j,
so the noiseless displacement is a geometric sum:

    dbar_j  ~  -g*_j * (1 - a_j^K) / lam_j        [predicted]

Read the limits:
  eta*lam_j << 1/K  ->  (1-a^K)/lam -> eta*K     : a GRADIENT step
  eta*lam_j >> 1/K  ->  (1-a^K)/lam -> 1/lam     : a NEWTON step

So local SGD interpolates between gradient and Newton on its own, with the
crossover at eta*lam ~ 1/K. In the sharp directions dbar is ALREADY the Newton
step; dividing it by lam again double-applies the correction. That predicts both
observed facts: raw 1/lamhat is catastrophic, and TRUE 1/lam still loses.

If the mechanism holds, the correction that actually reaches the oracle is
multiplication by 1/(1 - a_j^K) -- which is ~1 in sharp directions (no help
needed) and ~1/(eta*K*lam_j) in flat ones. That puts the entire available gain in
exactly the directions where the design's relative curvature error was 3.080.

Three checks below: (1) the displacement law, (2) the correct correction with
TRUE lam, (3) the same correction with the ESTIMATED lam.
"""

import random

D, N_OUT, N, K, ETA, SIG, REPS = 200, 12, 8, 8, 0.05, 1.0, 60


def gain(u, gstar, lam):
    num = sum(a * b for a, b in zip(gstar, u))
    den = sum(l * x * x for l, x in zip(lam, u))
    return num * num / (2.0 * den) if den > 0 else 0.0


def one_rep(rng, lam):
    xt = [rng.gauss(0, 1.0) for _ in range(D)]
    gstar = [l * x for l, x in zip(lam, xt)]
    dev = [[0.0] * N for _ in range(D)]
    sx = [0.0] * D; sg = [0.0] * D; sxx = [0.0] * D; sxg = [0.0] * D
    n = N * K
    for _ in range(K):
        for j in range(D):
            lj = lam[j]
            for i in range(N):
                x = dev[j][i]
                g = lj * (xt[j] + x) + rng.gauss(0, SIG)
                sx[j] += x; sg[j] += g; sxx[j] += x * x; sxg[j] += x * g
                dev[j][i] = x - ETA * g
    dbar = [sum(dev[j]) / N for j in range(D)]
    lamhat = []
    for j in range(D):
        cxx = sxx[j] - sx[j] * sx[j] / n
        cxg = sxg[j] - sx[j] * sg[j] / n
        lamhat.append(cxg / cxx if cxx > 0 else 0.0)

    # (1) displacement law
    pred = [-gstar[j] * (1.0 - (1.0 - ETA * lam[j]) ** K) / lam[j] for j in range(D)]
    ratio = [dbar[j] / pred[j] if abs(pred[j]) > 1e-12 else float("nan")
             for j in range(D)]

    out = {"scale": gain(dbar, gstar, lam),
           "newton": 0.5 * sum(g * g / l for g, l in zip(gstar, lam)),
           "ratio_sharp": sum(ratio[j] for j in range(N_OUT)) / N_OUT,
           "ratio_flat": sum(ratio[j] for j in range(N_OUT, D)) / (D - N_OUT)}

    # (2) correct correction, TRUE lam
    u = [dbar[j] / (1.0 - (1.0 - ETA * lam[j]) ** K) for j in range(D)]
    out["fix_true"] = gain(u, gstar, lam)
    # (3) correct correction, ESTIMATED lam (clipped to the stable range)
    u = []
    for j in range(D):
        lh = min(max(lamhat[j], 1e-6), (2.0 - 1e-6) / ETA)
        u.append(dbar[j] / (1.0 - (1.0 - ETA * lh) ** K))
    out["fix_hat"] = gain(u, gstar, lam)
    # (4) correct correction, TRUE lam, applied ONLY to flat directions
    u = [dbar[j] / (1.0 - (1.0 - ETA * lam[j]) ** K) if j >= N_OUT else dbar[j]
         for j in range(D)]
    out["fix_true_flatonly"] = gain(u, gstar, lam)
    return out


def main():
    rng = random.Random(0)
    lam = [rng.uniform(8, 12) for _ in range(N_OUT)] + \
          [rng.uniform(0.1, 1.0) for _ in range(D - N_OUT)]
    keys = ["scale", "newton", "fix_true", "fix_hat", "fix_true_flatonly",
            "ratio_sharp", "ratio_flat"]
    acc = {k: 0.0 for k in keys}
    for r in range(REPS):
        o = one_rep(random.Random(1000 + r), lam)
        for k in keys:
            acc[k] += o[k] / REPS

    print(f"d={D} sharp={N_OUT} N={N} K={K} eta={ETA} reps={REPS}")
    print()
    print("=== (1) displacement law  dbar_j ~ -g*_j (1-a_j^K)/lam_j ===")
    print(f"  observed/predicted, sharp dirs : {acc['ratio_sharp']:.4f}")
    print(f"  observed/predicted, flat  dirs : {acc['ratio_flat']:.4f}")
    print("  (1.0000 = law holds exactly in expectation)")
    print()
    eff_s = (1 - (1 - ETA * 10.0) ** K) / 10.0
    eff_f = (1 - (1 - ETA * 0.5) ** K) / 0.5
    print(f"  implied effective step, sharp lam=10  : {eff_s:.4f}  "
          f"(Newton 1/lam = {1/10.0:.4f}, gradient eta*K = {ETA*K:.4f})")
    print(f"  implied effective step, flat  lam=0.5 : {eff_f:.4f}  "
          f"(Newton 1/lam = {1/0.5:.4f}, gradient eta*K = {ETA*K:.4f})")
    print("  -> sharp dirs are already AT the Newton step; flat dirs are not.")
    print()
    base, ceil = acc["scale"], acc["newton"]
    print("=== (2-4) the correction the law says is the right one ===")
    print(f"  {'arm':<34} {'gain':>9} {'/ceiling':>9} {'/baseline':>10}")
    for k, tag in [("scale", "best-scaled dbar (baseline)"),
                   ("fix_true", "x 1/(1-a^K), TRUE lam"),
                   ("fix_hat", "x 1/(1-a^K), ESTIMATED lam"),
                   ("fix_true_flatonly", "x 1/(1-a^K), TRUE lam, flat only"),
                   ("newton", "diagonal oracle (ceiling)")]:
        v = acc[k]
        print(f"  {tag:<34} {v:>9.4f} {v/ceil:>9.3f} {v/base:>10.3f}")
    print()
    print("Read: if TRUE-lam recovers the ceiling but ESTIMATED-lam does not, the")
    print("idea is right and the DESIGN is too weak -- a different failure from")
    print("'the idea is redundant', and a different fix.")


if __name__ == "__main__":
    main()
