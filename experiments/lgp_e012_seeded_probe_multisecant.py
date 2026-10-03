"""E012: controller-seeded probe diversity for event multisecants.

E011 showed passive stochastic drift is too noisy. Here the controller broadcasts
small centered probe offsets at the start of the silent worker phase. Workers
still run local heavy-ball and the event still fires by drift quorum, but the
spatial secant design is no longer left to minibatch noise.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.lgp_e011_event_multisecant_stage0 as e11


PROBE_RADIUS = 0.18


def centered_probe_offsets(rng, n_workers):
    rows = []
    for _ in range(n_workers):
        row = [rng.choice((-1.0, 1.0)) for _ in range(e11.D)]
        scale = PROBE_RADIUS / math.sqrt(e11.D)
        rows.append([scale * z for z in row])
    mean = e11.mean_rows(rows)
    return [e11.sub(row, mean) for row in rows]


def run_event(seed, n_workers, lam):
    rng = random.Random(seed)
    x0 = [rng.gauss(0.0, 0.7) for _ in range(e11.D)]
    offsets = centered_probe_offsets(rng, n_workers)
    xs = [e11.add(x0, off) for off in offsets]
    vs = [[0.0] * e11.D for _ in range(n_workers)]
    radius = e11.DELTA * max(1.0, e11.norm(x0))
    quorum_count = max(1, math.ceil(e11.QUORUM * n_workers))
    local_steps = 0

    for step in range(1, e11.MAX_LOCAL_STEPS + 1):
        for i in range(n_workers):
            g = e11.noisy_grad(xs[i], lam, rng)
            vs[i] = e11.add(e11.mul(e11.BETA, vs[i]), e11.mul(-e11.ETA, g))
            xs[i] = e11.add(xs[i], vs[i])
        local_steps = step
        ready = sum(1 for x in xs if e11.norm(e11.sub(x, x0)) >= radius)
        if ready >= quorum_count:
            break

    xbar = e11.mean_rows(xs)
    g_workers = [e11.noisy_grad(x, lam, rng) for x in xs]
    gbar = e11.mean_rows(g_workers)

    g0_workers = [e11.noisy_grad(x0, lam, rng) for _ in range(n_workers)]
    g0 = e11.mean_rows(g0_workers)
    dbar = e11.sub(xbar, x0)
    l_time = []
    for dy, dx in zip(e11.sub(gbar, g0), dbar):
        l_time.append(dy / dx if abs(dx) > 1e-10 else 0.0)
    alpha_time = e11.bb_inverse_scalar([dbar], [e11.sub(gbar, g0)])

    l_space = []
    for j in range(e11.D):
        num = 0.0
        den = 0.0
        for x, g in zip(xs, g_workers):
            s = x[j] - xbar[j]
            y = g[j] - gbar[j]
            num += s * y
            den += s * s
        l_space.append(num / (den + 1e-12))
    s_rows = [e11.sub(x, xbar) for x in xs]
    y_rows = [e11.sub(g, gbar) for g in g_workers]
    alpha_space = e11.bb_inverse_scalar(s_rows, y_rows)

    return {
        "x0": x0,
        "xbar": xbar,
        "gbar": gbar,
        "dbar": dbar,
        "l_time": l_time,
        "alpha_time": alpha_time,
        "l_space": l_space,
        "alpha_space": alpha_space,
        "steps": local_steps,
        "base_improvement": e11.loss(x0, lam) - e11.loss(xbar, lam),
    }


def main():
    lam = e11.lam_spectrum()
    print(
        f"seeded-probe multisecant: d={e11.D} eta={e11.ETA} beta={e11.BETA} "
        f"sigma={e11.SIGMA} delta={e11.DELTA} probe={PROBE_RADIUS}"
    )
    print()
    print(
        f"{'N':>5} {'steps':>7} {'scale*':>7} {'tbb*':>7} {'tdiag*':>7} "
        f"{'sdiag*':>7} {'sbb*':>7} {'scale':>12} {'timeBB':>12} "
        f"{'timeDiag':>12} {'spaceDiag':>12} {'spaceBB':>12} "
        f"{'sdiag-scale':>14} {'sdiag-timeBB':>14} {'sbb-scale':>14}"
    )
    for n_workers in e11.N_WORKERS:
        pilot = [
            run_event(160_000 + 1000 * n_workers + i, n_workers, lam)
            for i in range(e11.PILOT_EVENTS)
        ]
        heldout = [
            run_event(161_000 + 1000 * n_workers + i, n_workers, lam)
            for i in range(e11.HELDOUT_EVENTS)
        ]
        p_scale = e11.select(pilot, lam, "scale")
        p_time_bb = e11.select(pilot, lam, "time_bb")
        p_time = e11.select(pilot, lam, "time")
        p_space = e11.select(pilot, lam, "space")
        p_bb = e11.select(pilot, lam, "space_bb")
        chunks = e11.split_batches(heldout, 8)
        scale_vals = [e11.score(c, lam, "scale", p_scale) for c in chunks]
        time_bb_vals = [e11.score(c, lam, "time_bb", p_time_bb) for c in chunks]
        time_vals = [e11.score(c, lam, "time", p_time) for c in chunks]
        space_vals = [e11.score(c, lam, "space", p_space) for c in chunks]
        bb_vals = [e11.score(c, lam, "space_bb", p_bb) for c in chunks]
        diff_vals = [a - b for a, b in zip(space_vals, scale_vals)]
        diff_time_bb_vals = [a - b for a, b in zip(space_vals, time_bb_vals)]
        bb_diff_vals = [a - b for a, b in zip(bb_vals, scale_vals)]
        steps = sum(ev["steps"] for ev in heldout) / len(heldout)
        cells = []
        for vals in (
            scale_vals,
            time_bb_vals,
            time_vals,
            space_vals,
            bb_vals,
            diff_vals,
            diff_time_bb_vals,
            bb_diff_vals,
        ):
            m, hw = e11.mean_ci(vals)
            cells.append(f"{m:6.3f}±{hw:5.3f}")
        print(
            f"{n_workers:5d} {steps:7.2f} {p_scale:7.2f} {p_time_bb:7.2f} "
            f"{p_time:7.2f} {p_space:7.2f} {p_bb:7.2f} {cells[0]:>12} "
            f"{cells[1]:>12} {cells[2]:>12} {cells[3]:>12} {cells[4]:>12} "
            f"{cells[5]:>14} {cells[6]:>14} {cells[7]:>14}"
        )


if __name__ == "__main__":
    main()
