"""Smoke test for the `LocalGPAggregator` optimizer object.

This test answers a narrow engineering question:

    Does the exported optimizer module reproduce the E002 tuned-gamma effect
    when used through its public API?

It is still synthetic. It is not a closed-loop real-network result.
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
T = 30
NS = (8, 32, 128)
GAMMAS = [i / 20 for i in range(0, 41)]
PILOT_REPS = 3
HELDOUT_REPS = 6


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


def score(rounds, n_workers, gamma):
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
    diag = {"mean_coeff": 0.0, "rms_coeff": 0.0, "fallback": 0.0}
    for gstar, steps, grads, endpoints in rounds:
        result = agg.aggregate(steps, grads, endpoints)
        acc["fedavg"] += gain(result.fedavg_update, gstar, lam_global)
        acc["local_gp"] += gain(result.update, gstar, lam_global)
        oracle = [u * cof(l) for u, l in zip(result.fedavg_update, lam_global)]
        acc["oracle_c"] += gain(oracle, gstar, lam_global)
        diag["mean_coeff"] += result.mean_coefficient
        diag["rms_coeff"] += result.rms_coefficient
        diag["fallback"] += result.fallback_fraction
    count = len(rounds)
    for k in acc:
        acc[k] /= count
    for k in diag:
        diag[k] /= count
    return acc, diag


def main():
    global lam_global
    rb = random.Random(0)
    lam_global = [rb.uniform(8, 12) for _ in range(N_OUT)] + [
        rb.uniform(0.1, 1.0) for _ in range(D - N_OUT)
    ]

    print(f"d={D} sharp={N_OUT} K={K} eta={ETA} rounds={T}")
    print("public LocalGPAggregator API, gamma selected on pilot reps")
    print()
    print(
        f"{'N':>5} {'gamma*':>7} {'fedavg':>8} {'local_gp/base':>14} "
        f"{'oracle/base':>12} {'mean_c':>8} {'fallback':>9}"
    )

    for n_workers in NS:
        pilot = []
        heldout = []
        for r in range(PILOT_REPS):
            pilot.extend(make_rounds(70_000 + 100 * n_workers + r, lam_global, n_workers))
        for r in range(HELDOUT_REPS):
            heldout.extend(make_rounds(80_000 + 100 * n_workers + r, lam_global, n_workers))

        best_gamma = None
        best_score = None
        for gamma in GAMMAS:
            acc, _diag = score(pilot, n_workers, gamma)
            ratio = acc["local_gp"] / acc["fedavg"]
            if best_score is None or ratio > best_score:
                best_score = ratio
                best_gamma = gamma

        acc, diag = score(heldout, n_workers, best_gamma)
        base = acc["fedavg"]
        print(
            f"{n_workers:5d} {best_gamma:7.2f} {base:8.3f} "
            f"{acc['local_gp'] / base:14.3f} {acc['oracle_c'] / base:12.3f} "
            f"{diag['mean_coeff']:8.3f} {diag['fallback']:9.3f}"
        )


if __name__ == "__main__":
    lam_global = []
    main()
