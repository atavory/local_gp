"""LGP-E002: tuned-gamma control for the local-SGD curvature correction.

This is the control called out in `ushmoo_theory_localsgd_as_gp_20261003.md`.
The measured within-round arm wins on the diagonal-quadratic bed, but its gain
appears to come from a beneficial upward bias in lambda_hat, not from estimator
accuracy. The decisive control is:

    c_j(gamma) = 1 + gamma * (c_j - 1)

where c_j is the per-coordinate correction shape and gamma is one tuned scalar.
If this matches the measured arm, the live posterior readout is mostly a
tuned-shrinkage schedule over a curvature shape, not a method yet.

Pure stdlib; runs in seconds on CPU.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable


D = 120
N_OUT = 10
K = 8
ETA = 0.05
SIG = 1.0
T = 30
NS = (8, 32, 128)
PILOT_REPS = 4
HELDOUT_REPS = 8
LMAX = 1.5 / ETA
GAMMAS = [i / 20 for i in range(0, 41)]  # 0.00 .. 2.00

ZG = [math.log(1e-3) + i * (math.log(LMAX) - math.log(1e-3)) / 39 for i in range(40)]
LG = [math.exp(z) for z in ZG]
CG = [1.0 / (1.0 - (1.0 - ETA * l) ** K) for l in LG]
LCG = [l * c for l, c in zip(LG, CG)]


def cof(lam: float) -> float:
    lam = min(max(lam, 1e-9), LMAX)
    return 1.0 / (1.0 - (1.0 - ETA * lam) ** K)


def gain(u: list[float], gstar: list[float], lam: list[float]) -> float:
    num = sum(a * b for a, b in zip(gstar, u))
    den = sum(l * x * x for l, x in zip(lam, u))
    return num * num / (2.0 * den) if den > 0 else 0.0


def posterior_ratio(bhat: float, var: float, mean_log: float, var_log: float) -> float:
    """E[lambda*C(lambda)] / E[lambda] under a log-normal prior grid."""
    lw = [
        -((bhat - lam) ** 2) / (2.0 * var) - ((z - mean_log) ** 2) / (2.0 * var_log)
        for lam, z in zip(LG, ZG)
    ]
    mx = max(lw)
    w = [math.exp(a - mx) for a in lw]
    denom = sum(wi * lam for wi, lam in zip(w, LG))
    if denom <= 0:
        return 1.0 / (ETA * K)
    return sum(wi * lc for wi, lc in zip(w, LCG)) / denom


def posterior_coeffs(acc_xx: list[float], acc_xg: list[float]) -> list[float]:
    ols = []
    var = []
    for xx, xg in zip(acc_xx, acc_xg):
        if xx > 1e-300:
            ols.append(xg / xx)
            var.append(SIG * SIG / xx)
        else:
            ols.append(0.0)
            var.append(1e9)
    pos = [b for b in ols if 1e-3 < b < LMAX] or [1.0]
    mean_log = sum(math.log(b) for b in pos) / len(pos)
    var_log = max(1e-2, sum((math.log(b) - mean_log) ** 2 for b in pos) / len(pos))
    return [posterior_ratio(b, v, mean_log, var_log) for b, v in zip(ols, var)]


@dataclass
class RoundRecord:
    lam: list[float]
    gstar: list[float]
    dbar: list[float]
    dbar_a: list[float]
    dbar_b: list[float]
    coeff_a: list[float]
    coeff_b: list[float]


def make_records(seed: int, lam: list[float], n_workers: int) -> list[RoundRecord]:
    rng = random.Random(seed)
    x = [rng.gauss(0, 1.0) for _ in range(D)]
    half_a = list(range(n_workers // 2))
    half_b = list(range(n_workers // 2, n_workers))
    axx = [0.0] * D
    axg = [0.0] * D
    bxx = [0.0] * D
    bxg = [0.0] * D
    records: list[RoundRecord] = []

    for _t in range(T):
        gstar = [l * xj for l, xj in zip(lam, x)]
        dev = [[0.0] * n_workers for _ in range(D)]
        pts = [[] for _ in range(D)]
        grd = [[] for _ in range(D)]

        for _k in range(K):
            for j in range(D):
                lj = lam[j]
                xj = x[j]
                for i in range(n_workers):
                    s = dev[j][i]
                    g = lj * (xj + s) + rng.gauss(0, SIG)
                    pts[j].append(s)
                    grd[j].append(g)
                    dev[j][i] = s - ETA * g

        # Cross-fit accumulators using this round's local trajectories.
        for j in range(D):
            pa = [pts[j][k * n_workers + i] for k in range(K) for i in half_a]
            qa = [grd[j][k * n_workers + i] for k in range(K) for i in half_a]
            pb = [pts[j][k * n_workers + i] for k in range(K) for i in half_b]
            qb = [grd[j][k * n_workers + i] for k in range(K) for i in half_b]
            for pvals, qvals, xx, xg in ((pa, qa, axx, axg), (pb, qb, bxx, bxg)):
                mp = sum(pvals) / len(pvals)
                mq = sum(qvals) / len(qvals)
                xx[j] += sum((p - mp) ** 2 for p in pvals)
                xg[j] += sum((p - mp) * (q - mq) for p, q in zip(pvals, qvals))

        dbar = [sum(dev[j]) / n_workers for j in range(D)]
        dbar_a = [sum(dev[j][i] for i in half_a) / len(half_a) for j in range(D)]
        dbar_b = [sum(dev[j][i] for i in half_b) / len(half_b) for j in range(D)]
        records.append(
            RoundRecord(
                lam=lam,
                gstar=gstar,
                dbar=dbar,
                dbar_a=dbar_a,
                dbar_b=dbar_b,
                coeff_a=posterior_coeffs(axx, axg),
                coeff_b=posterior_coeffs(bxx, bxg),
            )
        )
        x = [xj + dj for xj, dj in zip(x, dbar)]
    return records


def score_records(records: Iterable[RoundRecord], gamma: float | None = None) -> dict[str, float]:
    out = {k: 0.0 for k in ("base", "within_split", "gamma_split", "oracleC")}
    count = 0
    for rec in records:
        count += 1
        out["base"] += gain(rec.dbar, rec.gstar, rec.lam)
        split_step = [
            0.5 * (ca * db + cb * da)
            for ca, cb, da, db in zip(rec.coeff_a, rec.coeff_b, rec.dbar_a, rec.dbar_b)
        ]
        out["within_split"] += gain(split_step, rec.gstar, rec.lam)
        if gamma is not None:
            gamma_step = [
                0.5 * ((1.0 + gamma * (ca - 1.0)) * db + (1.0 + gamma * (cb - 1.0)) * da)
                for ca, cb, da, db in zip(rec.coeff_a, rec.coeff_b, rec.dbar_a, rec.dbar_b)
            ]
            out["gamma_split"] += gain(gamma_step, rec.gstar, rec.lam)
        out["oracleC"] += gain([d * cof(l) for d, l in zip(rec.dbar, rec.lam)], rec.gstar, rec.lam)
    return {k: v / count for k, v in out.items()}


def main() -> None:
    rb = random.Random(0)
    lam = [rb.uniform(8, 12) for _ in range(N_OUT)] + [rb.uniform(0.1, 1.0) for _ in range(D - N_OUT)]

    print(f"d={D} sharp={N_OUT} K={K} eta={ETA} rounds={T}")
    print("gamma tuned on pilot reps, evaluated on held-out reps")
    print()
    print(f"{'N':>5} {'gamma*':>7} {'base':>8} {'within/base':>12} {'gamma/base':>11} {'oracleC/base':>12}")

    for n_workers in NS:
        pilot = []
        heldout = []
        for r in range(PILOT_REPS):
            pilot.extend(make_records(50_000 + 100 * n_workers + r, lam, n_workers))
        for r in range(HELDOUT_REPS):
            heldout.extend(make_records(60_000 + 100 * n_workers + r, lam, n_workers))

        pilot_scores = []
        for gamma in GAMMAS:
            s = score_records(pilot, gamma)
            pilot_scores.append((s["gamma_split"] / s["base"], gamma))
        _best_ratio, best_gamma = max(pilot_scores)

        held = score_records(heldout, best_gamma)
        base = held["base"]
        print(
            f"{n_workers:5d} {best_gamma:7.2f} {base:8.3f} "
            f"{held['within_split'] / base:12.3f} "
            f"{held['gamma_split'] / base:11.3f} "
            f"{held['oracleC'] / base:12.3f}"
        )

    print()
    print("Interpretation: if gamma/base matches within/base, the live posterior")
    print("readout is not yet a method; a one-scalar shrinkage control explains it.")


if __name__ == "__main__":
    main()
