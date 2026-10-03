"""Gate for a small action-space layer over local-GP coefficients.

Full parameter-space GP is too large. E005 showed that a scalar source-free
shrinkage weight can match tuned gamma on the synthetic bed. E006 asks whether
the next smaller-than-full object is useful: a few reliability bins over the
correction vector.

This is still a synthetic gate. It is meant to decide whether the action space
should be scalar gamma or group-wise gamma before spending real-path GPU time.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_gp import LocalGPAggregator, LocalGPConfig


D = 120
N_OUT = 10
K = 8
ETA = 0.05
SIG = 1.0
T = 20
NS = (8, 32, 128)
PILOT_REPS = 2
HELDOUT_REPS = 4
GAMMAS = [i / 20 for i in range(0, 41)]


def gain(update, gstar, lam):
    num = sum(a * b for a, b in zip(gstar, update))
    den = sum(l * x * x for l, x in zip(lam, update))
    return num * num / (2.0 * den) if den > 0 else 0.0


def cof(lam):
    return 1.0 / (1.0 - (1.0 - ETA * lam) ** K)


def make_rounds(seed, lam, n_workers):
    rng = random.Random(seed)
    x = [rng.gauss(0, 1.0) for _ in range(D)]
    rounds = []
    for _t in range(T):
        gstar = [l * xj for l, xj in zip(lam, x)]
        dev = [[0.0] * D for _ in range(n_workers)]
        step_displacements = [[None for _ in range(K)] for _ in range(n_workers)]
        step_gradients = [[None for _ in range(K)] for _ in range(n_workers)]

        for k in range(K):
            for i in range(n_workers):
                disp = list(dev[i])
                grad = [l * (xj + s) + rng.gauss(0, SIG) for l, xj, s in zip(lam, x, disp)]
                step_displacements[i][k] = disp
                step_gradients[i][k] = grad
                dev[i] = [s - ETA * g for s, g in zip(dev[i], grad)]

        rounds.append((gstar, step_displacements, step_gradients, [list(v) for v in dev]))
        fedavg = [sum(worker[j] for worker in dev) / n_workers for j in range(D)]
        x = [xj + dj for xj, dj in zip(x, fedavg)]
    return rounds


def score(rounds, gamma, groups=1):
    agg = LocalGPAggregator(
        LocalGPConfig(
            eta=ETA,
            local_steps=K,
            gamma=gamma,
            shrinkage_groups=groups,
            crossfit=True,
            readout="posterior_ratio",
        )
    )
    acc = {k: 0.0 for k in ("fedavg", "local_gp", "oracle_c")}
    diag = {"mean_gamma": 0.0, "mean_coeff": 0.0}
    for gstar, steps, grads, endpoints in rounds:
        result = agg.aggregate(steps, grads, endpoints)
        acc["fedavg"] += gain(result.fedavg_update, gstar, lam_global)
        acc["local_gp"] += gain(result.update, gstar, lam_global)
        oracle = [u * cof(l) for u, l in zip(result.fedavg_update, lam_global)]
        acc["oracle_c"] += gain(oracle, gstar, lam_global)
        diag["mean_gamma"] += result.gamma
        diag["mean_coeff"] += result.mean_coefficient
    count = len(rounds)
    for k in acc:
        acc[k] /= count
    for k in diag:
        diag[k] /= count
    return acc, diag


def select_gamma(pilot):
    best_gamma = None
    best_score = None
    for gamma in GAMMAS:
        acc, _diag = score(pilot, gamma)
        ratio = acc["local_gp"] / acc["fedavg"]
        if best_score is None or ratio > best_score:
            best_score = ratio
            best_gamma = gamma
    return best_gamma


def main():
    global lam_global
    rb = random.Random(0)
    lam_global = [rb.uniform(8, 12) for _ in range(N_OUT)] + [
        rb.uniform(0.1, 1.0) for _ in range(D - N_OUT)
    ]

    print(f"d={D} sharp={N_OUT} K={K} eta={ETA} rounds={T}")
    print("small action layer: scalar auto-gamma vs grouped auto-gamma")
    print()
    print(
        f"{'N':>5} {'gamma*':>7} {'tuned':>8} {'auto1':>8} {'auto4':>8} "
        f"{'auto8':>8} {'oracle':>8} {'g1':>6} {'g4':>6} {'g8':>6}"
    )

    for n_workers in NS:
        pilot = []
        heldout = []
        for r in range(PILOT_REPS):
            pilot.extend(make_rounds(92_000 + 100 * n_workers + r, lam_global, n_workers))
        for r in range(HELDOUT_REPS):
            heldout.extend(make_rounds(93_000 + 100 * n_workers + r, lam_global, n_workers))

        best_gamma = select_gamma(pilot)
        tuned, _ = score(heldout, best_gamma)
        auto1, d1 = score(heldout, None, groups=1)
        auto4, d4 = score(heldout, None, groups=4)
        auto8, d8 = score(heldout, None, groups=8)
        base = tuned["fedavg"]
        print(
            f"{n_workers:5d} {best_gamma:7.2f} {tuned['local_gp'] / base:8.3f} "
            f"{auto1['local_gp'] / base:8.3f} {auto4['local_gp'] / base:8.3f} "
            f"{auto8['local_gp'] / base:8.3f} {tuned['oracle_c'] / base:8.3f} "
            f"{d1['mean_gamma']:6.3f} {d4['mean_gamma']:6.3f} {d8['mean_gamma']:6.3f}"
        )


if __name__ == "__main__":
    lam_global = []
    main()
