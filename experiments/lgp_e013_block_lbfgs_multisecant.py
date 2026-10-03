"""E013: spatial L-BFGS multisecants on a rotated block quadratic.

E011/E012 used diagonal inverse estimates. On a diagonal quadratic, a single
time displacement is already an unusually strong coordinate-wise diagnostic.
This test moves to a block-rotated SPD Hessian and uses actual L-BFGS two-loop
recursion from secant pairs.

Question: does the centered worker secant bank beat one-pair server/time L-BFGS
and tuned scalar extrapolation at the same event?
"""

from __future__ import annotations

import math
import random


D = 40
ETA = 0.018
BETA = 0.75
SIGMA = 0.10
DELTA = 0.45
PROBE_RADIUS = 0.20
QUORUM = 0.5
MAX_LOCAL_STEPS = 90
N_WORKERS = (8, 32, 128)
PILOT_EVENTS = 80
HELDOUT_EVENTS = 160
SCALES = [i / 20 for i in range(0, 81)]
DAMPS = [i / 20 for i in range(0, 41)]
STEP_CAP = 2.5


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


def make_blocks():
    rng = random.Random(0)
    blocks = []
    for b in range(D // 2):
        if b < 4:
            l1, l2 = rng.uniform(8, 14), rng.uniform(1.5, 4)
        else:
            l1, l2 = rng.uniform(0.2, 1.2), rng.uniform(0.05, 0.4)
        theta = rng.uniform(0.2, 1.2)
        c, s = math.cos(theta), math.sin(theta)
        h00 = c * c * l1 + s * s * l2
        h11 = s * s * l1 + c * c * l2
        h01 = c * s * (l1 - l2)
        blocks.append((h00, h01, h11))
    return blocks


def hmul(x, blocks):
    out = [0.0] * D
    for b, (h00, h01, h11) in enumerate(blocks):
        j = 2 * b
        x0, x1 = x[j], x[j + 1]
        out[j] = h00 * x0 + h01 * x1
        out[j + 1] = h01 * x0 + h11 * x1
    return out


def loss(x, blocks):
    return 0.5 * dot(x, hmul(x, blocks))


def noisy_grad(x, blocks, rng):
    return [g + rng.gauss(0.0, SIGMA) for g in hmul(x, blocks)]


def centered_probe_offsets(rng, n_workers):
    rows = []
    for _ in range(n_workers):
        row = [rng.choice((-1.0, 1.0)) for _ in range(D)]
        scale = PROBE_RADIUS / math.sqrt(D)
        rows.append([scale * z for z in row])
    mean = mean_rows(rows)
    return [sub(row, mean) for row in rows]


def capped_add(x, step, ref_norm):
    step_norm = norm(step)
    cap = STEP_CAP * max(ref_norm, 1e-12)
    if step_norm > cap:
        step = mul(cap / step_norm, step)
    return add(x, step)


def lbfgs_inverse_apply(pairs, g):
    good = [(s, y) for s, y in pairs if dot(s, y) > 1e-10]
    if not good:
        return list(g)
    q = list(g)
    alphas = []
    rhos = []
    for s, y in reversed(good):
        rho = 1.0 / dot(s, y)
        alpha = rho * dot(s, q)
        q = sub(q, mul(alpha, y))
        alphas.append(alpha)
        rhos.append(rho)
    sy = sum(dot(s, y) for s, y in good)
    yy = sum(dot(y, y) for _s, y in good)
    h0 = sy / yy if yy > 1e-12 and sy > 0 else 1.0
    r = mul(h0, q)
    for (s, y), alpha, rho in zip(good, reversed(alphas), reversed(rhos)):
        beta = rho * dot(y, r)
        r = add(r, mul(alpha - beta, s))
    return r


def run_event(seed, n_workers, blocks):
    rng = random.Random(seed)
    x0 = [rng.gauss(0.0, 0.5) for _ in range(D)]
    offsets = centered_probe_offsets(rng, n_workers)
    xs = [add(x0, off) for off in offsets]
    vs = [[0.0] * D for _ in range(n_workers)]
    radius = DELTA * max(1.0, norm(x0))
    quorum_count = max(1, math.ceil(QUORUM * n_workers))
    local_steps = 0
    for step in range(1, MAX_LOCAL_STEPS + 1):
        for i in range(n_workers):
            g = noisy_grad(xs[i], blocks, rng)
            vs[i] = add(mul(BETA, vs[i]), mul(-ETA, g))
            xs[i] = add(xs[i], vs[i])
        local_steps = step
        ready = sum(1 for x in xs if norm(sub(x, x0)) >= radius)
        if ready >= quorum_count:
            break

    xbar = mean_rows(xs)
    g_workers = [noisy_grad(x, blocks, rng) for x in xs]
    gbar = mean_rows(g_workers)
    g0 = mean_rows([noisy_grad(x0, blocks, rng) for _ in range(n_workers)])
    dbar = sub(xbar, x0)
    time_pairs = [(dbar, sub(gbar, g0))]
    space_pairs = [(sub(x, xbar), sub(g, gbar)) for x, g in zip(xs, g_workers)]
    return {
        "x0": x0,
        "xbar": xbar,
        "gbar": gbar,
        "dbar": dbar,
        "time_pairs": time_pairs,
        "space_pairs": space_pairs,
        "steps": local_steps,
        "base_improvement": loss(x0, blocks) - loss(xbar, blocks),
    }


def score(events, blocks, arm, param):
    base = 0.0
    total = 0.0
    for ev in events:
        base += ev["base_improvement"]
        if arm == "scale":
            x = add(ev["x0"], mul(param, ev["dbar"]))
        elif arm == "time_lbfgs":
            step = mul(-param, lbfgs_inverse_apply(ev["time_pairs"], ev["gbar"]))
            x = capped_add(ev["xbar"], step, norm(ev["dbar"]))
        elif arm == "space_lbfgs":
            step = mul(-param, lbfgs_inverse_apply(ev["space_pairs"], ev["gbar"]))
            x = capped_add(ev["xbar"], step, norm(ev["dbar"]))
        else:
            raise ValueError(arm)
        total += loss(ev["x0"], blocks) - loss(x, blocks)
    return total / base if base > 0 else 1.0


def select(events, blocks, arm):
    grid = SCALES if arm == "scale" else DAMPS
    return max(grid, key=lambda p: score(events, blocks, arm, p))


def mean_ci(vals):
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
    return m, 1.96 * math.sqrt(var / len(vals))


def split_batches(events, parts):
    size = len(events) // parts
    return [events[i * size : (i + 1) * size] for i in range(parts)]


def main():
    blocks = make_blocks()
    print(
        f"block L-BFGS multisecant: d={D} eta={ETA} beta={BETA} "
        f"sigma={SIGMA} delta={DELTA} probe={PROBE_RADIUS}"
    )
    print()
    print(
        f"{'N':>5} {'steps':>7} {'scale*':>7} {'time*':>7} {'space*':>7} "
        f"{'scale':>12} {'timeLBFGS':>12} {'spaceLBFGS':>12} "
        f"{'space-scale':>14} {'space-time':>14}"
    )
    for n_workers in N_WORKERS:
        pilot = [
            run_event(170_000 + 1000 * n_workers + i, n_workers, blocks)
            for i in range(PILOT_EVENTS)
        ]
        heldout = [
            run_event(171_000 + 1000 * n_workers + i, n_workers, blocks)
            for i in range(HELDOUT_EVENTS)
        ]
        p_scale = select(pilot, blocks, "scale")
        p_time = select(pilot, blocks, "time_lbfgs")
        p_space = select(pilot, blocks, "space_lbfgs")
        chunks = split_batches(heldout, 8)
        scale_vals = [score(c, blocks, "scale", p_scale) for c in chunks]
        time_vals = [score(c, blocks, "time_lbfgs", p_time) for c in chunks]
        space_vals = [score(c, blocks, "space_lbfgs", p_space) for c in chunks]
        diff_scale = [a - b for a, b in zip(space_vals, scale_vals)]
        diff_time = [a - b for a, b in zip(space_vals, time_vals)]
        steps = sum(ev["steps"] for ev in heldout) / len(heldout)
        cells = []
        for vals in (scale_vals, time_vals, space_vals, diff_scale, diff_time):
            m, hw = mean_ci(vals)
            cells.append(f"{m:6.3f}±{hw:5.3f}")
        print(
            f"{n_workers:5d} {steps:7.2f} {p_scale:7.2f} {p_time:7.2f} {p_space:7.2f} "
            f"{cells[0]:>12} {cells[1]:>12} {cells[2]:>12} "
            f"{cells[3]:>14} {cells[4]:>14}"
        )


if __name__ == "__main__":
    main()
