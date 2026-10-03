"""E008: does the local-path/secant gap survive nonlinearity + heterogeneity?

The durable signal from the local-GP theory note is not gamma selection. It is
the secant gap: server-side across-round secants behave like the classical
quasi-Newton information source, while K-step local worker paths contain
within-round curvature observations from a common origin.

This synthetic bed breaks the exact quadratic assumptions:

- diagonal but nonlinear global objective: 0.5 lam x^2 + 0.25 q x^4;
- per-round worker heterogeneity: centered linear worker biases;
- stochastic gradient noise.

It scores actual one-step global loss decrease, not just quadratic gain.
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
ROUNDS = 50
REPS = 8
SIGMA = 0.35
HET = 0.35


def loss(x, lam, quartic):
    return sum(0.5 * l * z * z + 0.25 * q * z**4 for z, l, q in zip(x, lam, quartic))


def grad_global(x, lam, quartic):
    return [l * z + q * z**3 for z, l, q in zip(x, lam, quartic)]


def coeff_from_lam(lhat):
    max_lam = (2.0 - 1e-9) / ETA
    out = []
    for l in lhat:
        if l <= 1e-9 or l >= max_lam:
            out.append(1.0)
            continue
        denom = 1.0 - (1.0 - ETA * l) ** K
        if denom <= 1e-12:
            out.append(1.0)
        else:
            out.append(min(25.0, max(1.0, 1.0 / denom)))
    return out


def mean_rows(rows):
    return [sum(row[j] for row in rows) / len(rows) for j in range(len(rows[0]))]


def simulate(seed, n_workers, lam, quartic):
    rng = random.Random(seed)
    x = [rng.gauss(0.0, 0.35) for _ in range(D)]
    agg_full = LocalGPAggregator(
        LocalGPConfig(eta=ETA, local_steps=K, gamma=1.0, crossfit=True)
    )
    agg_auto = LocalGPAggregator(
        LocalGPConfig(eta=ETA, local_steps=K, gamma=None, crossfit=True)
    )
    totals = {
        "fedavg": 0.0,
        "local_full": 0.0,
        "local_auto": 0.0,
        "server_secant": 0.0,
    }
    prev_x = None
    prev_g_obs = None
    secant_ready = 0

    for _round in range(ROUNDS):
        start_x = list(x)
        start_loss = loss(x, lam, quartic)
        dev = [[0.0] * D for _ in range(n_workers)]
        step_displacements = [[None for _ in range(K)] for _ in range(n_workers)]
        step_gradients = [[None for _ in range(K)] for _ in range(n_workers)]

        # Center worker biases so the global objective is unchanged while local
        # paths still fan out heterogeneously.
        raw_bias = [[rng.gauss(0.0, HET) for _ in range(D)] for _ in range(n_workers)]
        bias_mean = mean_rows(raw_bias)
        worker_bias = [
            [b - mb for b, mb in zip(row, bias_mean)]
            for row in raw_bias
        ]

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
        local_full = agg_full.aggregate(step_displacements, step_gradients, endpoints).update
        local_auto = agg_auto.aggregate(step_displacements, step_gradients, endpoints).update

        g_obs = mean_rows(first_step_grads)
        if prev_x is None:
            secant = fedavg
        else:
            lhat = []
            for xj, pxj, gj, pgj in zip(x, prev_x, g_obs, prev_g_obs):
                dx = xj - pxj
                lhat.append((gj - pgj) / dx if abs(dx) > 1e-10 else 0.0)
            c = coeff_from_lam(lhat)
            secant = [cj * uj for cj, uj in zip(c, fedavg)]
            secant_ready += 1

        updates = {
            "fedavg": fedavg,
            "local_full": local_full,
            "local_auto": local_auto,
            "server_secant": secant,
        }
        for name, update in updates.items():
            after = [xj + uj for xj, uj in zip(x, update)]
            totals[name] += start_loss - loss(after, lam, quartic)

        x = [xj + uj for xj, uj in zip(x, fedavg)]
        prev_x = start_x
        prev_g_obs = g_obs

    base = totals["fedavg"]
    return {
        name: totals[name] / base if base > 0 else 1.0
        for name in totals
    }


def mean_ci(vals):
    m = sum(vals) / len(vals)
    if len(vals) <= 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
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
        f"nonlinear diagonal bed: d={D} K={K} eta={ETA} "
        f"sigma={SIGMA} het={HET} rounds={ROUNDS} reps={REPS}"
    )
    print()
    print(f"{'N':>5} {'local_full':>12} {'local_auto':>12} {'server_secant':>14}")
    for n_workers in N_WORKERS:
        rows = [simulate(120_000 + 1000 * n_workers + r, n_workers, lam, quartic) for r in range(REPS)]
        parts = []
        for name in ("local_full", "local_auto", "server_secant"):
            m, hw = mean_ci([row[name] for row in rows])
            parts.append(f"{m:6.3f}±{hw:5.3f}")
        print(f"{n_workers:5d} {parts[0]:>12} {parts[1]:>12} {parts[2]:>14}")


if __name__ == "__main__":
    main()
