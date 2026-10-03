"""E014: dynamics-aware local-path correction.

E013 used spatial secants inside a plain quasi-Newton consolidation. That is
the wrong operator for K-step local SGD: the workers have already applied part
of the Newton correction. On a quadratic with no momentum,

    x_K - x_0 = -H^{-1} [I - (I - eta H)^K] g_0.

The correction that remains is therefore

    [I - (I - eta H)^K]^{-1} (x_K - x_0),

not another H^{-1} gradient step.

This experiment asks whether estimating H from local/path/probe geometry and
then correcting the local-update map beats a tuned scalar FedAvg scale.
"""

from __future__ import annotations

import math
import random


D = 40
K = 8
ETA = 0.018
SIGMA = 0.04
N_WORKERS = (8, 32, 128)
PILOT_EVENTS = 80
HELDOUT_EVENTS = 160
PROBE_REPS = 4
PROBE_RADIUS = 0.45
RIDGE = 1e-4
LAMBDA_MIN = 0.03
LAMBDA_MAX = 20.0
C_MAX = 30.0
SCALES = [i / 20 for i in range(0, 241)]
GAMMAS = [i / 20 for i in range(0, 41)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


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
    eigs = []
    vecs = []
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
        eigs.append((l1, l2))
        vecs.append(((c, s), (-s, c)))
    return blocks, eigs, vecs


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


def avg_noisy_grad(x, blocks, rng, reps):
    acc = [0.0] * D
    for _ in range(reps):
        g = noisy_grad(x, blocks, rng)
        for j, val in enumerate(g):
            acc[j] += val
    return [val / reps for val in acc]


def mat2_inv(m):
    a, b, c, d = m
    det = a * d - b * c
    if abs(det) < 1e-12:
        det = 1e-12 if det >= 0 else -1e-12
    return (d / det, -b / det, -c / det, a / det)


def mat2_mul(a, b):
    return (
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    )


def mat2_vec(m, v):
    return (m[0] * v[0] + m[1] * v[1], m[2] * v[0] + m[3] * v[1])


def mat2_pow(m, k):
    out = (1.0, 0.0, 0.0, 1.0)
    base = m
    n = k
    while n:
        if n & 1:
            out = mat2_mul(out, base)
        base = mat2_mul(base, base)
        n >>= 1
    return out


def sym_clip_block(h00, h01, h11):
    tr = h00 + h11
    diff = h00 - h11
    rad = math.sqrt(max(0.0, diff * diff + 4.0 * h01 * h01))
    l1 = max(LAMBDA_MIN, min(LAMBDA_MAX, 0.5 * (tr + rad)))
    l2 = max(LAMBDA_MIN, min(LAMBDA_MAX, 0.5 * (tr - rad)))
    theta = 0.5 * math.atan2(2.0 * h01, diff)
    c, s = math.cos(theta), math.sin(theta)
    return (
        c * c * l1 + s * s * l2,
        c * s * (l1 - l2),
        s * s * l1 + c * c * l2,
    )


def block_correction_from_h(h00, h01, h11):
    a = (1.0 - ETA * h00, -ETA * h01, -ETA * h01, 1.0 - ETA * h11)
    ak = mat2_pow(a, K)
    m = (1.0 - ak[0], -ak[1], -ak[2], 1.0 - ak[3])
    c = mat2_inv(m)
    # Hard cap the eigenvalues of the correction operator. This is not a method
    # claim; it prevents one near-zero estimated curvature from dominating the
    # diagnostic.
    h00c, h01c, h11c = sym_clip_block(c[0], 0.5 * (c[1] + c[2]), c[3])
    tr = h00c + h11c
    diff = h00c - h11c
    rad = math.sqrt(max(0.0, diff * diff + 4.0 * h01c * h01c))
    l1 = min(C_MAX, max(0.0, 0.5 * (tr + rad)))
    l2 = min(C_MAX, max(0.0, 0.5 * (tr - rad)))
    theta = 0.5 * math.atan2(2.0 * h01c, diff)
    cc, ss = math.cos(theta), math.sin(theta)
    c00 = cc * cc * l1 + ss * ss * l2
    c01 = cc * ss * (l1 - l2)
    c11 = ss * ss * l1 + cc * cc * l2
    return (c00, c01, c01, c11)


def estimate_blocks(pairs):
    # Per-2D-block least squares H s ~= y.
    acc = [
        {"ss00": 0.0, "ss01": 0.0, "ss11": 0.0, "ys00": 0.0, "ys01": 0.0, "ys10": 0.0, "ys11": 0.0}
        for _ in range(D // 2)
    ]
    for s, y in pairs:
        for b in range(D // 2):
            j = 2 * b
            s0, s1 = s[j], s[j + 1]
            y0, y1 = y[j], y[j + 1]
            cell = acc[b]
            cell["ss00"] += s0 * s0
            cell["ss01"] += s0 * s1
            cell["ss11"] += s1 * s1
            cell["ys00"] += y0 * s0
            cell["ys01"] += y0 * s1
            cell["ys10"] += y1 * s0
            cell["ys11"] += y1 * s1

    blocks = []
    for cell in acc:
        ss = (cell["ss00"] + RIDGE, cell["ss01"], cell["ss01"], cell["ss11"] + RIDGE)
        ys = (cell["ys00"], cell["ys01"], cell["ys10"], cell["ys11"])
        h = mat2_mul(ys, mat2_inv(ss))
        h01 = 0.5 * (h[1] + h[2])
        blocks.append(sym_clip_block(h[0], h01, h[3]))
    return blocks


def dynamics_correct(dbar, h_blocks):
    out = [0.0] * D
    for b, h in enumerate(h_blocks):
        c = block_correction_from_h(*h)
        j = 2 * b
        z0, z1 = mat2_vec(c, (dbar[j], dbar[j + 1]))
        out[j], out[j + 1] = z0, z1
    return out


def random_offsets(rng, count):
    rows = []
    for _ in range(count):
        row = [rng.choice((-1.0, 1.0)) for _ in range(D)]
        scale = PROBE_RADIUS / math.sqrt(D)
        rows.append([scale * z for z in row])
    mean = mean_rows(rows)
    return [sub(row, mean) for row in rows]


def directed_offsets(rng, count, eigs, vecs):
    # Oracle-directed probe design for the diagnostic: cycle through the block
    # eigenvectors, emphasizing the flat blocks where the dynamics correction
    # has the most headroom.
    dirs = []
    for b, ((l1, l2), (v1, v2)) in enumerate(zip(eigs, vecs)):
        order = [(l2, v2), (l1, v1)] if b >= 4 else [(l1, v1), (l2, v2)]
        for _lam, v in order:
            row = [0.0] * D
            row[2 * b] = v[0]
            row[2 * b + 1] = v[1]
            dirs.append(row)
    rows = []
    for i in range(count):
        sign = 1.0 if i % 2 == 0 else -1.0
        base = dirs[(i // 2) % len(dirs)]
        rows.append(mul(sign * PROBE_RADIUS, base))
    mean = mean_rows(rows)
    return [sub(row, mean) for row in rows]


def probe_pairs(x0, blocks, offsets, rng):
    xs = [add(x0, off) for off in offsets]
    grads = [avg_noisy_grad(x, blocks, rng, PROBE_REPS) for x in xs]
    xbar = mean_rows(xs)
    gbar = mean_rows(grads)
    return [(sub(x, xbar), sub(g, gbar)) for x, g in zip(xs, grads)]


def run_event(seed, n_workers, true_blocks, eigs, vecs):
    rng = random.Random(seed)
    x0 = [rng.gauss(0.0, 0.5) for _ in range(D)]
    xs = [list(x0) for _ in range(n_workers)]
    path_by_step = []
    for _step in range(K):
        grads = [noisy_grad(x, true_blocks, rng) for x in xs]
        xbar = mean_rows(xs)
        gbar = mean_rows(grads)
        path_by_step.append((xs, grads, xbar, gbar))
        xs = [sub(x, mul(ETA, g)) for x, g in zip(xs, grads)]
    xbar_final = mean_rows(xs)
    dbar = sub(xbar_final, x0)

    centered_path_pairs = []
    for xs_step, grads_step, xbar_step, gbar_step in path_by_step[1:]:
        for x, g in zip(xs_step, grads_step):
            centered_path_pairs.append((sub(x, xbar_step), sub(g, gbar_step)))

    budget = n_workers * K
    random_pairs = probe_pairs(x0, true_blocks, random_offsets(rng, budget), rng)
    directed_pairs = probe_pairs(x0, true_blocks, directed_offsets(rng, budget, eigs, vecs), rng)

    return {
        "x0": x0,
        "dbar": dbar,
        "base_improvement": loss(x0, true_blocks) - loss(add(x0, dbar), true_blocks),
        "oracle_corr": dynamics_correct(dbar, true_blocks),
        "path_corr": dynamics_correct(dbar, estimate_blocks(centered_path_pairs)),
        "random_corr": dynamics_correct(dbar, estimate_blocks(random_pairs)),
        "directed_corr": dynamics_correct(dbar, estimate_blocks(directed_pairs)),
    }


def candidate_delta(ev, arm, param):
    if arm == "scale":
        return mul(param, ev["dbar"])
    key = {
        "oracle": "oracle_corr",
        "path": "path_corr",
        "random_probe": "random_corr",
        "directed_probe": "directed_corr",
    }[arm]
    return add(ev["dbar"], mul(param, sub(ev[key], ev["dbar"])))


def score(events, blocks, arm, param):
    base = 0.0
    total = 0.0
    for ev in events:
        base += ev["base_improvement"]
        x = add(ev["x0"], candidate_delta(ev, arm, param))
        total += loss(ev["x0"], blocks) - loss(x, blocks)
    return total / base if base > 0 else 1.0


def select(events, blocks, arm):
    grid = SCALES if arm == "scale" else GAMMAS
    return max(grid, key=lambda p: score(events, blocks, arm, p))


def mean_ci(vals):
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
    return m, 1.96 * math.sqrt(var / len(vals))


def split_batches(events, parts):
    size = len(events) // parts
    return [events[i * size : (i + 1) * size] for i in range(parts)]


def main():
    blocks, eigs, vecs = make_blocks()
    print(
        "dynamics-aware correction: "
        f"d={D} k={K} eta={ETA} sigma={SIGMA} probe_reps={PROBE_REPS} "
        f"probe_radius={PROBE_RADIUS} cmax={C_MAX}"
    )
    print()
    arms = ("scale", "oracle", "path", "random_probe", "directed_probe")
    print(
        f"{'N':>5} {'scale*':>7} {'oracle*':>8} {'path*':>7} {'rand*':>7} {'dir*':>7} "
        f"{'scale':>12} {'oracle':>12} {'path':>12} {'random':>12} {'directed':>12} "
        f"{'dir-scale':>13} {'path-scale':>13}"
    )
    for n_workers in N_WORKERS:
        pilot = [
            run_event(210_000 + 1000 * n_workers + i, n_workers, blocks, eigs, vecs)
            for i in range(PILOT_EVENTS)
        ]
        heldout = [
            run_event(211_000 + 1000 * n_workers + i, n_workers, blocks, eigs, vecs)
            for i in range(HELDOUT_EVENTS)
        ]
        params = {arm: select(pilot, blocks, arm) for arm in arms}
        chunks = split_batches(heldout, 8)
        vals = {arm: [score(c, blocks, arm, params[arm]) for c in chunks] for arm in arms}
        diff_dir = [a - b for a, b in zip(vals["directed_probe"], vals["scale"])]
        diff_path = [a - b for a, b in zip(vals["path"], vals["scale"])]
        cells = []
        for arm in arms:
            m, hw = mean_ci(vals[arm])
            cells.append(f"{m:6.3f}±{hw:5.3f}")
        dirm, dirhw = mean_ci(diff_dir)
        pathm, pathhw = mean_ci(diff_path)
        print(
            f"{n_workers:5d} {params['scale']:7.2f} {params['oracle']:8.2f} "
            f"{params['path']:7.2f} {params['random_probe']:7.2f} {params['directed_probe']:7.2f} "
            f"{cells[0]:>12} {cells[1]:>12} {cells[2]:>12} {cells[3]:>12} {cells[4]:>12} "
            f"{dirm:6.3f}±{dirhw:5.3f} {pathm:6.3f}±{pathhw:5.3f}"
        )


if __name__ == "__main__":
    main()
