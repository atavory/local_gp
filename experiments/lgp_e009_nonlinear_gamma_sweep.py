"""E009: gamma sweep on the nonlinear/heterogeneous secant bed.

E008 shows full local-path correction is unstable once the quadratic bed is
broken. This gate asks whether the correction shape is still useful after
proper shrinkage, or whether FedAvg/server secants dominate.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_gp import LocalGPAggregator, LocalGPConfig


D = 80
N_SHARP = 8
K = 8
ETA = 0.025
N_WORKERS = (8, 32, 128)
ROUNDS = 40
PILOT_REPS = 4
HELDOUT_REPS = 4
SIGMA = 0.35
HET = 0.35
GAMMAS = [i / 20 for i in range(0, 21)]


def loss(x, lam, quartic):
    return sum(0.5 * l * z * z + 0.25 * q * z**4 for z, l, q in zip(x, lam, quartic))


def mean_rows(rows):
    return [sum(row[j] for row in rows) / len(rows) for j in range(len(rows[0]))]


def coeff_from_lam(lhat):
    max_lam = (2.0 - 1e-9) / ETA
    out = []
    for l in lhat:
        if l <= 1e-9 or l >= max_lam:
            out.append(1.0)
            continue
        denom = 1.0 - (1.0 - ETA * l) ** K
        out.append(1.0 if denom <= 1e-12 else min(25.0, max(1.0, 1.0 / denom)))
    return out


def make_rounds(seed, n_workers, lam, quartic):
    rng = random.Random(seed)
    x = [rng.gauss(0.0, 0.35) for _ in range(D)]
    rounds = []
    prev_x = None
    prev_g_obs = None
    for _round in range(ROUNDS):
        start_x = list(x)
        start_loss = loss(x, lam, quartic)
        dev = [[0.0] * D for _ in range(n_workers)]
        step_displacements = [[None for _ in range(K)] for _ in range(n_workers)]
        step_gradients = [[None for _ in range(K)] for _ in range(n_workers)]
        raw_bias = [[rng.gauss(0.0, HET) for _ in range(D)] for _ in range(n_workers)]
        bias_mean = mean_rows(raw_bias)
        worker_bias = [[b - mb for b, mb in zip(row, bias_mean)] for row in raw_bias]

        first_step_grads = []
        for k in range(K):
            for i in range(n_workers):
                disp = list(dev[i])
                z = [xj + sj for xj, sj in zip(x, disp)]
                grad = [
                    l * zj + q * zj**3 + worker_bias[i][j] + rng.gauss(0.0, SIGMA)
                    for j, (zj, l, q) in enumerate(zip(z, lam, quartic))
                ]
                if k == 0:
                    first_step_grads.append(grad)
                step_displacements[i][k] = disp
                step_gradients[i][k] = grad
                dev[i] = [s - ETA * g for s, g in zip(dev[i], grad)]

        endpoints = [list(v) for v in dev]
        fedavg = mean_rows(endpoints)
        g_obs = mean_rows(first_step_grads)
        if prev_x is None:
            secant = fedavg
        else:
            lhat = []
            for xj, pxj, gj, pgj in zip(start_x, prev_x, g_obs, prev_g_obs):
                dx = xj - pxj
                lhat.append((gj - pgj) / dx if abs(dx) > 1e-10 else 0.0)
            secant = [c * u for c, u in zip(coeff_from_lam(lhat), fedavg)]
        rounds.append((start_loss, start_x, step_displacements, step_gradients, endpoints, fedavg, secant))
        x = [xj + uj for xj, uj in zip(x, fedavg)]
        prev_x = start_x
        prev_g_obs = g_obs
    return rounds


def score(rounds, gamma, lam, quartic):
    if gamma == "secant":
        totals = {"fedavg": 0.0, "candidate": 0.0}
        for start_loss, start_x, _steps, _grads, _endpoints, fedavg, secant in rounds:
            totals["fedavg"] += start_loss - loss([x + u for x, u in zip(start_x, fedavg)], lam, quartic)
            totals["candidate"] += start_loss - loss([x + u for x, u in zip(start_x, secant)], lam, quartic)
        return totals["candidate"] / totals["fedavg"]

    agg = LocalGPAggregator(LocalGPConfig(eta=ETA, local_steps=K, gamma=gamma, crossfit=True))
    totals = {"fedavg": 0.0, "candidate": 0.0}
    gammas = []
    for start_loss, start_x, steps, grads, endpoints, fedavg, _secant in rounds:
        result = agg.aggregate(steps, grads, endpoints)
        totals["fedavg"] += start_loss - loss([x + u for x, u in zip(start_x, fedavg)], lam, quartic)
        totals["candidate"] += start_loss - loss([x + u for x, u in zip(start_x, result.update)], lam, quartic)
        gammas.append(result.gamma)
    return totals["candidate"] / totals["fedavg"], sum(gammas) / len(gammas)


def mean_ci(vals):
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
    return m, 1.96 * (var / len(vals)) ** 0.5


def main():
    rng = random.Random(0)
    lam = [rng.uniform(8, 12) for _ in range(N_SHARP)] + [
        rng.uniform(0.1, 1.0) for _ in range(D - N_SHARP)
    ]
    quartic = [rng.uniform(4.0, 10.0) for _ in range(N_SHARP)] + [
        rng.uniform(0.1, 1.0) for _ in range(D - N_SHARP)
    ]
    print(
        f"nonlinear gamma sweep: d={D} K={K} eta={ETA} "
        f"sigma={SIGMA} het={HET} rounds={ROUNDS}"
    )
    print()
    print(
        f"{'N':>5} {'gamma*':>7} {'tuned':>12} {'auto':>12} "
        f"{'full':>12} {'secant':>12} {'auto_g':>8}"
    )
    for n_workers in N_WORKERS:
        pilot = []
        heldout = []
        for r in range(PILOT_REPS):
            pilot.extend(make_rounds(130_000 + 1000 * n_workers + r, n_workers, lam, quartic))
        for r in range(HELDOUT_REPS):
            heldout.append(make_rounds(131_000 + 1000 * n_workers + r, n_workers, lam, quartic))

        best_gamma = max(GAMMAS, key=lambda g: score(pilot, g, lam, quartic)[0])
        tuned_vals = [score(rows, best_gamma, lam, quartic)[0] for rows in heldout]
        auto_pairs = [score(rows, None, lam, quartic) for rows in heldout]
        auto_vals = [p[0] for p in auto_pairs]
        auto_g = sum(p[1] for p in auto_pairs) / len(auto_pairs)
        full_vals = [score(rows, 1.0, lam, quartic)[0] for rows in heldout]
        sec_vals = [score(rows, "secant", lam, quartic) for rows in heldout]
        cells = []
        for vals in (tuned_vals, auto_vals, full_vals, sec_vals):
            m, hw = mean_ci(vals)
            cells.append(f"{m:6.3f}±{hw:5.3f}")
        print(
            f"{n_workers:5d} {best_gamma:7.2f} {cells[0]:>12} {cells[1]:>12} "
            f"{cells[2]:>12} {cells[3]:>12} {auto_g:8.3f}"
        )


if __name__ == "__main__":
    main()
