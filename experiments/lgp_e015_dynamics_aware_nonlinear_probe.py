"""E015: dynamics-aware probes on the nonlinear/heterogeneous bed.

E014 fixed the local-map operator on an exact rotated-block quadratic. This
gate moves the corrected operator to the E008-E010 nonlinear/heterogeneous CPU
bed and separates three information sources:

  - centered local trajectory pairs;
  - noise-controlled random probe points around the round start;
  - coordinate-directed probe points around the round start.

All shape arms use the same K-step dynamics correction

    [I - (I - eta Hhat)^K]^{-1} Delta_K,

then source-select one scalar gamma against a tuned scalar FedAvg-scale null.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.lgp_e009_nonlinear_gamma_sweep as e9


PROBE_REPS = 4
PROBE_RADIUS = 0.25
RIDGE = 1e-4
LAMBDA_MAX = (2.0 - 1e-9) / e9.ETA
SCALES = [i / 20 for i in range(0, 81)]
GAMMAS = [i / 20 for i in range(0, 21)]


def grad_global(x, lam, quartic):
    return [l * z + q * z**3 for z, l, q in zip(x, lam, quartic)]


def noisy_global_grad(x, lam, quartic, rng):
    return [g + rng.gauss(0.0, e9.SIGMA) for g in grad_global(x, lam, quartic)]


def avg_noisy_global_grad(x, lam, quartic, rng):
    acc = [0.0] * e9.D
    for _ in range(PROBE_REPS):
        g = noisy_global_grad(x, lam, quartic, rng)
        for j, val in enumerate(g):
            acc[j] += val
    return [val / PROBE_REPS for val in acc]


def mean_rows(rows):
    return [sum(row[j] for row in rows) / len(rows) for j in range(len(rows[0]))]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def mul(c, a):
    return [c * x for x in a]


def coeff_from_lam(lhat):
    out = []
    for l in lhat:
        if l <= 1e-9 or l >= LAMBDA_MAX:
            out.append(1.0)
            continue
        denom = 1.0 - (1.0 - e9.ETA * l) ** e9.K
        out.append(1.0 if denom <= 1e-12 else min(25.0, max(1.0, 1.0 / denom)))
    return out


def estimate_diag_from_pairs(pairs):
    num = [0.0] * e9.D
    den = [RIDGE] * e9.D
    for s, y in pairs:
        for j, (sj, yj) in enumerate(zip(s, y)):
            num[j] += sj * yj
            den[j] += sj * sj
    return [max(0.0, min(LAMBDA_MAX, n / d)) for n, d in zip(num, den)]


def path_pairs(steps, grads):
    pairs = []
    for k in range(e9.K):
        disps = [steps[i][k] for i in range(len(steps))]
        gs = [grads[i][k] for i in range(len(grads))]
        dbar = mean_rows(disps)
        gbar = mean_rows(gs)
        for d, g in zip(disps, gs):
            pairs.append((sub(d, dbar), sub(g, gbar)))
    return pairs


def random_offsets(rng, count):
    rows = []
    scale = PROBE_RADIUS / (e9.D**0.5)
    for _ in range(count):
        rows.append([scale * rng.choice((-1.0, 1.0)) for _ in range(e9.D)])
    mean = mean_rows(rows)
    return [sub(row, mean) for row in rows]


def coordinate_offsets(count):
    rows = []
    for i in range(count):
        row = [0.0] * e9.D
        row[(i // 2) % e9.D] = PROBE_RADIUS if i % 2 == 0 else -PROBE_RADIUS
        rows.append(row)
    mean = mean_rows(rows)
    return [sub(row, mean) for row in rows]


def probe_pairs(start_x, offsets, lam, quartic, rng):
    xs = [add(start_x, off) for off in offsets]
    grads = [avg_noisy_global_grad(x, lam, quartic, rng) for x in xs]
    xbar = mean_rows(xs)
    gbar = mean_rows(grads)
    return [(sub(x, xbar), sub(g, gbar)) for x, g in zip(xs, grads)]


def corrected_update(fedavg, lhat):
    c = coeff_from_lam(lhat)
    return [cj * uj for cj, uj in zip(c, fedavg)]


def augment_rounds(seed, n_workers, lam, quartic):
    rows = []
    raw = e9.make_rounds(seed, n_workers, lam, quartic)
    for idx, (start_loss, start_x, steps, grads, endpoints, fedavg, secant) in enumerate(raw):
        rng = random.Random(seed * 1009 + idx)
        budget = n_workers * e9.K
        path_lam = estimate_diag_from_pairs(path_pairs(steps, grads))
        rand_lam = estimate_diag_from_pairs(
            probe_pairs(start_x, random_offsets(rng, budget), lam, quartic, rng)
        )
        coord_lam = estimate_diag_from_pairs(
            probe_pairs(start_x, coordinate_offsets(budget), lam, quartic, rng)
        )
        true_lam = [l + 3.0 * q * z * z for z, l, q in zip(start_x, lam, quartic)]
        rows.append(
            {
                "start_loss": start_loss,
                "start_x": start_x,
                "fedavg": fedavg,
                "secant": secant,
                "path": corrected_update(fedavg, path_lam),
                "random_probe": corrected_update(fedavg, rand_lam),
                "coord_probe": corrected_update(fedavg, coord_lam),
                "oracle": corrected_update(fedavg, true_lam),
            }
        )
    return rows


def candidate(row, arm, param):
    if arm == "scale":
        return mul(param, row["fedavg"])
    if arm == "secant":
        return row["secant"]
    return add(row["fedavg"], mul(param, sub(row[arm], row["fedavg"])))


def score(rows, arm, param, lam, quartic):
    base = 0.0
    total = 0.0
    for row in rows:
        start_x = row["start_x"]
        fedavg = row["fedavg"]
        base += row["start_loss"] - e9.loss(add(start_x, fedavg), lam, quartic)
        update = candidate(row, arm, param)
        total += row["start_loss"] - e9.loss(add(start_x, update), lam, quartic)
    return total / base if base > 0 else 1.0


def select(rows, arm, lam, quartic):
    if arm == "secant":
        return 1.0
    grid = SCALES if arm == "scale" else GAMMAS
    return max(grid, key=lambda p: score(rows, arm, p, lam, quartic))


def mean_ci(vals):
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
    return m, 1.96 * (var / len(vals)) ** 0.5


def main():
    rng = random.Random(0)
    lam = [rng.uniform(8, 12) for _ in range(e9.N_SHARP)] + [
        rng.uniform(0.1, 1.0) for _ in range(e9.D - e9.N_SHARP)
    ]
    quartic = [rng.uniform(4.0, 10.0) for _ in range(e9.N_SHARP)] + [
        rng.uniform(0.1, 1.0) for _ in range(e9.D - e9.N_SHARP)
    ]
    print(
        f"dynamics-aware nonlinear probe: d={e9.D} K={e9.K} eta={e9.ETA} "
        f"sigma={e9.SIGMA} het={e9.HET} probe_reps={PROBE_REPS} radius={PROBE_RADIUS}"
    )
    print()
    arms = ("scale", "oracle", "path", "random_probe", "coord_probe", "secant")
    print(
        f"{'N':>5} {'scale*':>7} {'oracle*':>8} {'path*':>7} {'rand*':>7} {'coord*':>7} "
        f"{'scalar':>12} {'oracle':>12} {'path':>12} {'random':>12} {'coord':>12} "
        f"{'secant':>12} {'coord-scale':>13} {'rand-scale':>13}"
    )
    for n_workers in e9.N_WORKERS:
        pilot = []
        heldout = []
        for r in range(e9.PILOT_REPS):
            pilot.extend(augment_rounds(220_000 + 1000 * n_workers + r, n_workers, lam, quartic))
        for r in range(e9.HELDOUT_REPS):
            heldout.append(augment_rounds(221_000 + 1000 * n_workers + r, n_workers, lam, quartic))
        params = {arm: select(pilot, arm, lam, quartic) for arm in arms}
        vals = {arm: [score(rows, arm, params[arm], lam, quartic) for rows in heldout] for arm in arms}
        coord_diff = [a - b for a, b in zip(vals["coord_probe"], vals["scale"])]
        rand_diff = [a - b for a, b in zip(vals["random_probe"], vals["scale"])]
        cells = []
        for arm in arms:
            m, hw = mean_ci(vals[arm])
            cells.append(f"{m:6.3f}±{hw:5.3f}")
        cm, chw = mean_ci(coord_diff)
        rm, rhw = mean_ci(rand_diff)
        print(
            f"{n_workers:5d} {params['scale']:7.2f} {params['oracle']:8.2f} "
            f"{params['path']:7.2f} {params['random_probe']:7.2f} {params['coord_probe']:7.2f} "
            f"{cells[0]:>12} {cells[1]:>12} {cells[2]:>12} {cells[3]:>12} {cells[4]:>12} "
            f"{cells[5]:>12} {cm:6.3f}±{chw:5.3f} {rm:6.3f}±{rhw:5.3f}"
        )


if __name__ == "__main__":
    main()
