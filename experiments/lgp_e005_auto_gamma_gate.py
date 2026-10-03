"""Gate for source-free shrinkage in `LocalGPAggregator`.

E003 proved that the public optimizer can reproduce the synthetic local-GP
effect with a tuned scalar gamma. This gate asks the product question:

    Can the optimizer choose that shrinkage from the worker cloud itself?

The candidate is `gamma=None`, which uses split-half reliability of the
coefficient vector on each round. The controls are FedAvg, tuned-gamma selected
on pilot reps, and the unshrunk coefficient vector.
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
T = 24
NS = (8, 32, 128)
GAMMAS = [i / 20 for i in range(0, 41)]
PILOT_REPS = 3
HELDOUT_REPS = 4


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


def score(rounds, gamma):
    agg = LocalGPAggregator(
        LocalGPConfig(
            eta=ETA,
            local_steps=K,
            gamma=gamma,
            crossfit=True,
            readout="posterior_ratio",
        )
    )
    acc = {k: 0.0 for k in ("fedavg", "local_gp", "oracle_c")}
    diag = {"mean_coeff": 0.0, "mean_gamma": 0.0, "fallback": 0.0}
    for gstar, steps, grads, endpoints in rounds:
        result = agg.aggregate(steps, grads, endpoints)
        acc["fedavg"] += gain(result.fedavg_update, gstar, lam_global)
        acc["local_gp"] += gain(result.update, gstar, lam_global)
        oracle = [u * cof(l) for u, l in zip(result.fedavg_update, lam_global)]
        acc["oracle_c"] += gain(oracle, gstar, lam_global)
        diag["mean_coeff"] += result.mean_coefficient
        diag["mean_gamma"] += result.gamma
        diag["fallback"] += result.fallback_fraction
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
    print("auto-gamma is split-half reliability; tuned gamma uses pilot reps")
    print()
    print(
        f"{'N':>5} {'gamma*':>7} {'auto_g':>7} {'tuned/base':>12} "
        f"{'auto/base':>11} {'full/base':>10} {'oracle/base':>12}"
    )

    for n_workers in NS:
        pilot = []
        heldout = []
        for r in range(PILOT_REPS):
            pilot.extend(make_rounds(90_000 + 100 * n_workers + r, lam_global, n_workers))
        for r in range(HELDOUT_REPS):
            heldout.extend(make_rounds(91_000 + 100 * n_workers + r, lam_global, n_workers))

        best_gamma = select_gamma(pilot)
        tuned, _ = score(heldout, best_gamma)
        auto, auto_diag = score(heldout, None)
        full, _ = score(heldout, 1.0)
        base = tuned["fedavg"]
        print(
            f"{n_workers:5d} {best_gamma:7.2f} {auto_diag['mean_gamma']:7.3f} "
            f"{tuned['local_gp'] / base:12.3f} {auto['local_gp'] / base:11.3f} "
            f"{full['local_gp'] / base:10.3f} {tuned['oracle_c'] / base:12.3f}"
        )


if __name__ == "__main__":
    lam_global = []
    main()
