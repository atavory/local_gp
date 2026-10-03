"""E011: event-triggered centered-worker secants, Stage 0.

Workers start from a common point and run local heavy-ball silently until a
drift-threshold quorum fires. At the event, the controller gathers full
positions and gradients and compares consolidation choices:

- FedAvg at the worker mean;
- tuned scalar extrapolation along the mean displacement;
- diagonal server/time secant;
- diagonal centered-worker spatial secant.

This tests the Muse architecture directly under the "pay by wire" model: no
streaming sketches, no fixed sync period, and no d^2 object.
"""

from __future__ import annotations

import math
import random


D = 60
N_SHARP = 8
ETA = 0.025
BETA = 0.75
SIGMA = 0.20
DELTA = 0.50
QUORUM = 0.5
MAX_LOCAL_STEPS = 80
N_WORKERS = (8, 32, 128)
PILOT_EVENTS = 80
HELDOUT_EVENTS = 160
SCALES = [i / 20 for i in range(0, 81)]
GAMMAS = [i / 20 for i in range(0, 41)]
MIN_LAM = 1e-4
MAX_LAM = 80.0
STEP_CAP = 2.0


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def mul(c, a):
    return [c * x for x in a]


def mean_rows(rows):
    return [sum(row[j] for row in rows) / len(rows) for j in range(len(rows[0]))]


def loss(x, lam):
    return 0.5 * sum(l * z * z for l, z in zip(lam, x))


def exact_grad(x, lam):
    return [l * z for l, z in zip(lam, x)]


def noisy_grad(x, lam, rng):
    return [l * z + rng.gauss(0.0, SIGMA) for l, z in zip(lam, x)]


def clip_lam(l):
    if not math.isfinite(l) or l <= MIN_LAM:
        return None
    return min(MAX_LAM, max(MIN_LAM, l))


def capped_add(xbar, step, ref_norm):
    step_norm = norm(step)
    cap = STEP_CAP * max(ref_norm, 1e-12)
    if step_norm > cap:
        step = mul(cap / step_norm, step)
    return add(xbar, step)


def apply_diag_step(xbar, gbar, lhat, gamma, ref_norm):
    step = []
    for x, g, l in zip(xbar, gbar, lhat):
        lc = clip_lam(l)
        if lc is None:
            step.append(0.0)
        else:
            step.append(-gamma * g / lc)
    return capped_add(xbar, step, ref_norm)


def bb_inverse_scalar(s_rows, y_rows):
    sy = 0.0
    yy = 0.0
    for s, y in zip(s_rows, y_rows):
        sy += dot(s, y)
        yy += dot(y, y)
    if yy <= 1e-12 or sy <= 0.0:
        return 0.0
    return sy / yy


def run_event(seed, n_workers, lam):
    rng = random.Random(seed)
    x0 = [rng.gauss(0.0, 0.7) for _ in range(D)]
    xs = [list(x0) for _ in range(n_workers)]
    vs = [[0.0] * D for _ in range(n_workers)]
    radius = DELTA * max(1.0, norm(x0))
    quorum_count = max(1, math.ceil(QUORUM * n_workers))
    local_steps = 0

    for step in range(1, MAX_LOCAL_STEPS + 1):
        for i in range(n_workers):
            g = noisy_grad(xs[i], lam, rng)
            vs[i] = add(mul(BETA, vs[i]), mul(-ETA, g))
            xs[i] = add(xs[i], vs[i])
        local_steps = step
        ready = sum(1 for x in xs if norm(sub(x, x0)) >= radius)
        if ready >= quorum_count:
            break

    xbar = mean_rows(xs)
    g_workers = [noisy_grad(x, lam, rng) for x in xs]
    gbar = mean_rows(g_workers)

    # A server-side/time secant gets only the common-origin mean displacement.
    g0_workers = [noisy_grad(x0, lam, rng) for _ in range(n_workers)]
    g0 = mean_rows(g0_workers)
    dbar = sub(xbar, x0)
    l_time = []
    for dy, dx in zip(sub(gbar, g0), dbar):
        l_time.append(dy / dx if abs(dx) > 1e-10 else 0.0)
    alpha_time = bb_inverse_scalar([dbar], [sub(gbar, g0)])

    # Centered worker secants use m directions from the same event.
    l_space = []
    s_rows = []
    y_rows = []
    for j in range(D):
        num = 0.0
        den = 0.0
        for x, g in zip(xs, g_workers):
            s = x[j] - xbar[j]
            y = g[j] - gbar[j]
            num += s * y
            den += s * s
        l_space.append(num / (den + 1e-12))
    for x, g in zip(xs, g_workers):
        s_rows.append(sub(x, xbar))
        y_rows.append(sub(g, gbar))
    alpha_space = bb_inverse_scalar(s_rows, y_rows)

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
        "base_improvement": loss(x0, lam) - loss(xbar, lam),
    }


def score(events, lam, arm, param):
    total_base = 0.0
    total = 0.0
    for ev in events:
        total_base += ev["base_improvement"]
        if arm == "scale":
            x = add(ev["x0"], mul(param, ev["dbar"]))
        elif arm == "time":
            x = apply_diag_step(ev["xbar"], ev["gbar"], ev["l_time"], param, norm(ev["dbar"]))
        elif arm == "time_bb":
            step = mul(-param * ev["alpha_time"], ev["gbar"])
            x = capped_add(ev["xbar"], step, norm(ev["dbar"]))
        elif arm == "space":
            x = apply_diag_step(ev["xbar"], ev["gbar"], ev["l_space"], param, norm(ev["dbar"]))
        elif arm == "space_bb":
            step = mul(-param * ev["alpha_space"], ev["gbar"])
            x = capped_add(ev["xbar"], step, norm(ev["dbar"]))
        else:
            raise ValueError(arm)
        total += loss(ev["x0"], lam) - loss(x, lam)
    return total / total_base if total_base > 0 else 1.0


def select(events, lam, arm):
    grid = SCALES if arm == "scale" else GAMMAS
    return max(grid, key=lambda p: score(events, lam, arm, p))


def mean_ci(vals):
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
    return m, 1.96 * math.sqrt(var / len(vals))


def split_batches(events, parts):
    size = len(events) // parts
    return [events[i * size : (i + 1) * size] for i in range(parts)]


def lam_spectrum():
    rng = random.Random(0)
    return [rng.uniform(8, 12) for _ in range(N_SHARP)] + [
        rng.uniform(0.15, 1.0) for _ in range(D - N_SHARP)
    ]


def main():
    lam = lam_spectrum()
    print(
        f"event multisecant stage0: d={D} eta={ETA} beta={BETA} "
        f"sigma={SIGMA} delta={DELTA} quorum={QUORUM}"
    )
    print()
    print(
        f"{'N':>5} {'steps':>7} {'scale*':>7} {'tbb*':>7} {'tdiag*':>7} "
        f"{'sdiag*':>7} {'sbb*':>7} {'scale':>12} {'timeBB':>12} "
        f"{'timeDiag':>12} {'spaceDiag':>12} {'spaceBB':>12} "
        f"{'sdiag-scale':>14} {'sdiag-timeBB':>14} {'sbb-scale':>14}"
    )
    for n_workers in N_WORKERS:
        pilot = [
            run_event(150_000 + 1000 * n_workers + i, n_workers, lam)
            for i in range(PILOT_EVENTS)
        ]
        heldout = [
            run_event(151_000 + 1000 * n_workers + i, n_workers, lam)
            for i in range(HELDOUT_EVENTS)
        ]
        p_scale = select(pilot, lam, "scale")
        p_time = select(pilot, lam, "time")
        p_time_bb = select(pilot, lam, "time_bb")
        p_space = select(pilot, lam, "space")
        p_bb = select(pilot, lam, "space_bb")
        chunks = split_batches(heldout, 8)
        scale_vals = [score(c, lam, "scale", p_scale) for c in chunks]
        time_vals = [score(c, lam, "time", p_time) for c in chunks]
        time_bb_vals = [score(c, lam, "time_bb", p_time_bb) for c in chunks]
        space_vals = [score(c, lam, "space", p_space) for c in chunks]
        bb_vals = [score(c, lam, "space_bb", p_bb) for c in chunks]
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
            m, hw = mean_ci(vals)
            cells.append(f"{m:6.3f}±{hw:5.3f}")
        print(
            f"{n_workers:5d} {steps:7.2f} {p_scale:7.2f} {p_time_bb:7.2f} "
            f"{p_time:7.2f} {p_space:7.2f} {p_bb:7.2f} {cells[0]:>12} "
            f"{cells[1]:>12} {cells[2]:>12} {cells[3]:>12} {cells[4]:>12} "
            f"{cells[5]:>14} {cells[6]:>14} {cells[7]:>14}"
        )


if __name__ == "__main__":
    main()
