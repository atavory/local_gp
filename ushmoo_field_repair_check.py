"""Can the field-rotated step be repaired? (follow-up to ushmoo_design_snr_check.py)

That check produced a result against my own proposal: on a diagonal quadratic,
a PURE SCALE rule on the mean proposal reaches 73.9% of the diagonal oracle,
while rotating by the raw fitted curvature reaches 6.9% -- ten times worse than
doing nothing clever. Cause is visible in the same table: relative curvature
error is 0.069 in sharp directions but 3.080 in flat ones, and 41/200 coordinates
come back with lamhat <= 0. Dividing by a badly-estimated small lamhat amplifies
exactly the directions that are already worst.

So the question is not "does the trajectory carry curvature information" (it does,
7% relative error on the sharp directions) but "can a step use it without being
destroyed by the directions where it does not". Tikhonov on the inverse:

    u_j = dbar_j * lamhat_j / (lamhat_j^2 + delta^2)

-> 1/lamhat_j for lamhat_j >> delta, -> lamhat_j/delta^2 (i.e. damped to zero
correction) for lamhat_j << delta. One knob, swept.

delta is swept and the BEST value reported, so every number here is an
ORACLE-TUNED CEILING, not an achievable method. A ceiling that still loses is
decisive; a ceiling that wins only licenses the real experiment, with the
tuned-constant control that guardrail 2 requires.
"""

import math
import random

D, N_OUT, N, K, ETA, SIG, REPS = 200, 12, 8, 8, 0.05, 1.0, 60
DELTAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def gain(u, gstar, lam):
    num = sum(a * b for a, b in zip(gstar, u))
    den = sum(l * x * x for l, x in zip(lam, u))
    return num * num / (2.0 * den) if den > 0 else 0.0


def one_rep(rng, lam):
    xt = [rng.gauss(0, 1.0) for _ in range(D)]
    gstar = [l * x for l, x in zip(lam, xt)]
    dev = [[0.0] * N for _ in range(D)]
    sx = [0.0] * D
    sg = [0.0] * D
    sxx = [0.0] * D
    sxg = [0.0] * D
    n = N * K
    for _ in range(K):
        for j in range(D):
            lj = lam[j]
            for i in range(N):
                x = dev[j][i]
                g = lj * (xt[j] + x) + rng.gauss(0, SIG)
                sx[j] += x
                sg[j] += g
                sxx[j] += x * x
                sxg[j] += x * g
                dev[j][i] = x - ETA * g

    dbar = [sum(dev[j]) / N for j in range(D)]
    lamhat = []
    for j in range(D):
        cxx = sxx[j] - sx[j] * sx[j] / n
        cxg = sxg[j] - sx[j] * sg[j] / n
        lamhat.append(cxg / cxx if cxx > 0 else 0.0)

    out = {"scale": gain(dbar, gstar, lam),
           "newton": 0.5 * sum(g * g / l for g, l in zip(gstar, lam))}
    for dl in DELTAS:
        if dl == 0.0:
            u = [dbar[j] / max(lamhat[j], 1e-3) for j in range(D)]
        else:
            u = [dbar[j] * lamhat[j] / (lamhat[j] * lamhat[j] + dl * dl)
                 for j in range(D)]
        out[f"d{dl}"] = gain(u, gstar, lam)
    # true-curvature preconditioner: isolates estimation error from the idea
    out["truelam"] = gain([dbar[j] / lam[j] for j in range(D)], gstar, lam)
    # sharp-only correction with TRUE lam: how much is reachable if you only
    # ever touch the directions the design can actually estimate
    u = [dbar[j] / lam[j] if j < N_OUT else dbar[j] for j in range(D)]
    out["sharponly_true"] = gain(u, gstar, lam)
    return out


def main():
    rng = random.Random(0)
    lam = [rng.uniform(8, 12) for _ in range(N_OUT)] + \
          [rng.uniform(0.1, 1.0) for _ in range(D - N_OUT)]
    keys = ["scale", "newton", "truelam", "sharponly_true"] + [f"d{d}" for d in DELTAS]
    acc = {k: 0.0 for k in keys}
    for r in range(REPS):
        o = one_rep(random.Random(1000 + r), lam)
        for k in keys:
            acc[k] += o[k] / REPS

    base, ceil = acc["scale"], acc["newton"]
    print(f"d={D} sharp={N_OUT} N={N} K={K} eta={ETA} reps={REPS}")
    print(f"ceiling (diagonal oracle)        = {ceil:.4f}")
    print(f"baseline (best-scaled dbar)      = {base:.4f}  "
          f"({base/ceil:.3f} of ceiling)")
    print()
    print(f"  {'arm':<26} {'gain':>10} {'/ceiling':>10} {'/baseline':>11}")
    for dl in DELTAS:
        v = acc[f"d{dl}"]
        tag = "raw 1/lamhat" if dl == 0.0 else f"tikhonov delta={dl}"
        print(f"  {tag:<26} {v:>10.4f} {v/ceil:>10.3f} {v/base:>11.3f}")
    print()
    for k, tag in [("truelam", "TRUE 1/lam precond"),
                   ("sharponly_true", "TRUE 1/lam, sharp dirs only")]:
        v = acc[k]
        print(f"  {tag:<26} {v:>10.4f} {v/ceil:>10.3f} {v/base:>11.3f}")
    print()
    best = max(DELTAS, key=lambda d: acc[f"d{d}"])
    bv = acc[f"d{best}"]
    print(f"best oracle-tuned delta = {best} -> {bv/base:.3f}x the scale baseline")
    print("Any value <= 1.000 in the /baseline column means the fitted field is")
    print("worse than doing nothing, even with its knob tuned on the test set.")


if __name__ == "__main__":
    main()
