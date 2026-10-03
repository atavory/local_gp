"""E010: does the local correction shape beat a tuned scalar scale?

E009 showed conservative gamma beats FedAvg and server secants on the
nonlinear/heterogeneous bed. This control asks whether that is actually a shape
win, or just another tuned learning-rate/scale win.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.lgp_e009_nonlinear_gamma_sweep as e9


SCALES = [i / 20 for i in range(0, 81)]


def score_scale(rounds, scale, lam, quartic):
    totals = {"fedavg": 0.0, "candidate": 0.0}
    for start_loss, start_x, _steps, _grads, _endpoints, fedavg, _secant in rounds:
        update = [scale * u for u in fedavg]
        totals["fedavg"] += start_loss - e9.loss([x + u for x, u in zip(start_x, fedavg)], lam, quartic)
        totals["candidate"] += start_loss - e9.loss([x + u for x, u in zip(start_x, update)], lam, quartic)
    return totals["candidate"] / totals["fedavg"]


def main():
    rng = e9.random.Random(0)
    lam = [rng.uniform(8, 12) for _ in range(e9.N_SHARP)] + [
        rng.uniform(0.1, 1.0) for _ in range(e9.D - e9.N_SHARP)
    ]
    quartic = [rng.uniform(4.0, 10.0) for _ in range(e9.N_SHARP)] + [
        rng.uniform(0.1, 1.0) for _ in range(e9.D - e9.N_SHARP)
    ]
    print(
        f"shape-vs-scalar nonlinear control: d={e9.D} K={e9.K} "
        f"eta={e9.ETA} sigma={e9.SIGMA} het={e9.HET}"
    )
    print()
    print(
        f"{'N':>5} {'scale*':>7} {'gamma*':>7} {'scalar':>12} "
        f"{'shape':>12} {'shape-scalar':>14} {'secant':>12}"
    )
    for n_workers in e9.N_WORKERS:
        pilot = []
        heldout = []
        for r in range(e9.PILOT_REPS):
            pilot.extend(e9.make_rounds(140_000 + 1000 * n_workers + r, n_workers, lam, quartic))
        for r in range(e9.HELDOUT_REPS):
            heldout.append(e9.make_rounds(141_000 + 1000 * n_workers + r, n_workers, lam, quartic))

        best_scale = max(SCALES, key=lambda s: score_scale(pilot, s, lam, quartic))
        best_gamma = max(e9.GAMMAS, key=lambda g: e9.score(pilot, g, lam, quartic)[0])

        scalar_vals = [score_scale(rows, best_scale, lam, quartic) for rows in heldout]
        shape_vals = [e9.score(rows, best_gamma, lam, quartic)[0] for rows in heldout]
        diff_vals = [a - b for a, b in zip(shape_vals, scalar_vals)]
        sec_vals = [e9.score(rows, "secant", lam, quartic) for rows in heldout]

        cells = []
        for vals in (scalar_vals, shape_vals, diff_vals, sec_vals):
            m, hw = e9.mean_ci(vals)
            cells.append(f"{m:6.3f}±{hw:5.3f}")
        print(
            f"{n_workers:5d} {best_scale:7.2f} {best_gamma:7.2f} "
            f"{cells[0]:>12} {cells[1]:>12} {cells[2]:>14} {cells[3]:>12}"
        )


if __name__ == "__main__":
    main()
