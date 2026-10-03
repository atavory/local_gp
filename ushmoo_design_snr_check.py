"""Two questions bandha's E001 spec does not ask, measured on the same sim.

Q1 (audit of E001's action space). The E001 worker prompt scores a finite grid
    action = scale * mean_local_proposal,   scale in {0, .25, .5, .75, 1, 1.25, 1.5}
and every feature it logs is a dispersion scalar (norms, p10/p50/p90, spreads,
cosines-to-mean). That is a PURE GAIN action space: it can rescale the mean
proposal, never rotate it. Question: what fraction of the available headroom
can a pure-gain action space reach at all? If it is small, E001 cannot show
value no matter how good the predictor is, and the gate is uninformative before
it runs.

Q2 (wanger's regime point, §2c). wanger showed endpoint-gap VARIANCE is
allocated to FLAT directions in every stable regime, concluding the design is
misaligned for curvature estimation. Variance allocation is the right diagnostic
for "which direction does the top gap eigenvector point at". But the quantity
that governs a field-fitted STEP is the RELATIVE curvature error |lamhat-lam|/lam,
because the step is -g/lamhat. Those are different questions and they need not
have the same answer: flat directions have more spread (smaller absolute slope
error) but also smaller lam (bigger relative error). Which wins is an arithmetic
question, so measure it.

Model: diagonal quadratic f(x) = 0.5 * sum_j lam_j x_j^2, exact per-coordinate.
N workers take K local SGD steps from a common x_t with iid gradient noise.
"""

import math
import random

D, N_OUT, N, K, ETA, SIG, REPS = 200, 12, 8, 8, 0.05, 1.0, 60


def gain(u, gstar, lam):
    """Loss decrease from the best-scaled step along direction u on a quadratic."""
    num = sum(a * b for a, b in zip(gstar, u))
    den = sum(l * x * x for l, x in zip(lam, u))
    return num * num / (2.0 * den) if den > 0 else 0.0


def one_rep(rng, lam):
    xt = [rng.gauss(0, 1.0) for _ in range(D)]
    gstar = [l * x for l, x in zip(lam, xt)]

    # N workers, K local steps each, per coordinate.
    dev = [[0.0] * N for _ in range(D)]          # x_i,k - x_t
    xs = [[] for _ in range(D)]                   # design points (centred on x_t)
    gs = [[] for _ in range(D)]                   # gradients observed there
    for _ in range(K):
        for j in range(D):
            for i in range(N):
                g = lam[j] * (xt[j] + dev[j][i]) + rng.gauss(0, SIG)
                xs[j].append(dev[j][i])
                gs[j].append(g)
                dev[j][i] -= ETA * g

    dbar = [sum(dev[j]) / N for j in range(D)]
    gapvar = [sum((v - dbar[j]) ** 2 for v in dev[j]) / (N - 1) for j in range(D)]

    # Per-coordinate least-squares slope of g on x: identifies lam_j.
    lamhat, neg = [], 0
    for j in range(D):
        mx = sum(xs[j]) / len(xs[j])
        mg = sum(gs[j]) / len(gs[j])
        sxx = sum((x - mx) ** 2 for x in xs[j])
        sxg = sum((x - mx) * (g - mg) for x, g in zip(xs[j], gs[j]))
        b = sxg / sxx if sxx > 0 else 0.0
        if b <= 0:
            neg += 1
        lamhat.append(b)

    # Arms, all evaluated at their own optimal step size (gain/shape separated).
    u_scale = dbar                                         # E001's space: rescale dbar
    floor = 1e-3
    u_fit = [dbar[j] / max(lamhat[j], floor) for j in range(D)]   # rotate by fitted field
    g_newton = 0.5 * sum(g * g / l for g, l in zip(gstar, lam))   # diagonal oracle

    return {
        "scale": gain(u_scale, gstar, lam),
        "fit": gain(u_fit, gstar, lam),
        "newton": g_newton,
        "gapvar": gapvar,
        "lamhat": lamhat,
        "neg": neg,
    }


def main():
    rng = random.Random(0)
    lam = [rng.uniform(8, 12) for _ in range(N_OUT)] + \
          [rng.uniform(0.1, 1.0) for _ in range(D - N_OUT)]
    sharp, flat = list(range(N_OUT)), list(range(N_OUT, D))

    acc = {"scale": 0.0, "fit": 0.0, "newton": 0.0}
    rel_s, rel_f, gv_s, gv_f, negs, ovl = [], [], [], [], [], []
    for r in range(REPS):
        o = one_rep(random.Random(1000 + r), lam)
        for k in acc:
            acc[k] += o[k] / REPS
        rel_s.append(sum(abs(o["lamhat"][j] - lam[j]) / lam[j] for j in sharp) / len(sharp))
        rel_f.append(sum(abs(o["lamhat"][j] - lam[j]) / lam[j] for j in flat) / len(flat))
        gv_s.append(sum(o["gapvar"][j] for j in sharp) / len(sharp))
        gv_f.append(sum(o["gapvar"][j] for j in flat) / len(flat))
        negs.append(o["neg"])
        top = sorted(range(D), key=lambda j: o["gapvar"][j], reverse=True)[:N_OUT]
        ovl.append(sum(1 for j in top if j < N_OUT) / N_OUT)

    def m(v):
        return sum(v) / len(v)

    print(f"d={D} sharp={N_OUT} (lam~U[8,12]) flat={D-N_OUT} (lam~U[0.1,1]) "
          f"N={N} K={K} eta={ETA} reps={REPS}")
    print()
    print("=== Q2: wanger's variance allocation, reproduced ===")
    print(f"  mean endpoint-gap variance, sharp dirs : {m(gv_s):.3e}")
    print(f"  mean endpoint-gap variance, flat  dirs : {m(gv_f):.3e}")
    print(f"  top-{N_OUT} gap-variance overlap with sharp set : {m(ovl):.3f}")
    print("  -> confirmed: gap variance sits in the FLAT directions, overlap ~0.")
    print()
    print("=== Q2': but relative curvature error is the quantity a step uses ===")
    print(f"  mean |lamhat-lam|/lam, sharp dirs : {m(rel_s):.3f}")
    print(f"  mean |lamhat-lam|/lam, flat  dirs : {m(rel_f):.3f}")
    print(f"  coords with lamhat <= 0 (uninvertible): {m(negs):.1f} / {D}")
    print("  -> the two diagnostics disagree: least spread, best relative estimate.")
    print()
    print("=== Q1: ceiling of E001's pure-gain action space ===")
    print(f"  best-scaled mean proposal   (E001's space) : {acc['scale']:.4f}")
    print(f"  field-rotated step          (needs lamhat) : {acc['fit']:.4f}")
    print(f"  diagonal oracle             (ceiling)      : {acc['newton']:.4f}")
    print()
    print(f"  fraction of oracle reachable by ANY scale rule : "
          f"{acc['scale']/acc['newton']:.4f}")
    print(f"  fraction reached by rotating with the fitted field : "
          f"{acc['fit']/acc['newton']:.4f}")
    print()
    print("  The scale row is an upper bound on E001 with a PERFECT predictor:")
    print("  no predictor of the optimal scale can exceed the best scale.")


if __name__ == "__main__":
    main()
