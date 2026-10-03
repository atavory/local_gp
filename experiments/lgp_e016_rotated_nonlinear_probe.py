"""E016: dynamics-aware probes on a rotated nonlinear block bed.

E015 showed that probe-fed dynamics-aware correction beats a tuned scalar
FedAvg scale on a diagonal nonlinear/heterogeneous synthetic bed. This gate
removes privileged coordinates by using rotated 2D blocks with quartic terms in
the block eigen-directions:

    f_b(z) = 0.5 z^T H_b z + 0.25 q_1 (v_1^T z)^4 + 0.25 q_2 (v_2^T z)^4.

All shape arms use the K-step local-map correction

    [I - (I - eta Hhat)^K]^{-1} Delta_K

applied blockwise to the FedAvg displacement, then pilot-select one shrinkage
gamma against the tuned scalar FedAvg-scale null.
"""

from __future__ import annotations

import math
import random


D = 40
N_SHARP_BLOCKS = 4
K = 8
ETA = 0.025
N_WORKERS = (8, 32, 128)
ROUNDS = 40
PILOT_REPS = 4
HELDOUT_REPS = 4
SIGMA = 0.35
HET = 0.35
PROBE_REPS = 4
PROBE_RADIUS = 0.25
RIDGE = 1e-4
LAMBDA_MIN = 0.03
LAMBDA_MAX = (2.0 - 1e-9) / ETA
C_MAX = 25.0
SECANT_WINDOW = 8
SCALES = [i / 20 for i in range(0, 81)]
GAMMAS = [i / 20 for i in range(0, 21)]


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
    quartics = []
    for b in range(D // 2):
        if b < N_SHARP_BLOCKS:
            l1, l2 = rng.uniform(8.0, 12.0), rng.uniform(1.2, 3.5)
            q1, q2 = rng.uniform(4.0, 10.0), rng.uniform(1.0, 4.0)
        else:
            l1, l2 = rng.uniform(0.2, 1.0), rng.uniform(0.05, 0.5)
            q1, q2 = rng.uniform(0.2, 1.2), rng.uniform(0.05, 0.5)

        theta = rng.uniform(0.25, 1.32)
        c, s = math.cos(theta), math.sin(theta)
        h00 = c * c * l1 + s * s * l2
        h11 = s * s * l1 + c * c * l2
        h01 = c * s * (l1 - l2)
        blocks.append((h00, h01, h11))
        eigs.append((l1, l2))
        vecs.append(((c, s), (-s, c)))
        quartics.append((q1, q2))
    return blocks, eigs, vecs, quartics


def hmul(x, blocks):
    out = [0.0] * D
    for b, (h00, h01, h11) in enumerate(blocks):
        j = 2 * b
        x0, x1 = x[j], x[j + 1]
        out[j] = h00 * x0 + h01 * x1
        out[j + 1] = h01 * x0 + h11 * x1
    return out


def loss(x, blocks, vecs, quartics):
    quad = 0.5 * dot(x, hmul(x, blocks))
    nonlin = 0.0
    for b, ((v1, v2), (q1, q2)) in enumerate(zip(vecs, quartics)):
        j = 2 * b
        z = (x[j], x[j + 1])
        t1 = v1[0] * z[0] + v1[1] * z[1]
        t2 = v2[0] * z[0] + v2[1] * z[1]
        nonlin += 0.25 * q1 * t1**4 + 0.25 * q2 * t2**4
    return quad + nonlin


def grad_global(x, blocks, vecs, quartics):
    out = hmul(x, blocks)
    for b, ((v1, v2), (q1, q2)) in enumerate(zip(vecs, quartics)):
        j = 2 * b
        z = (x[j], x[j + 1])
        t1 = v1[0] * z[0] + v1[1] * z[1]
        t2 = v2[0] * z[0] + v2[1] * z[1]
        out[j] += q1 * t1**3 * v1[0] + q2 * t2**3 * v2[0]
        out[j + 1] += q1 * t1**3 * v1[1] + q2 * t2**3 * v2[1]
    return out


def true_hessian_blocks(x, blocks, vecs, quartics):
    out = []
    for b, ((h00, h01, h11), (v1, v2), (q1, q2)) in enumerate(
        zip(blocks, vecs, quartics)
    ):
        j = 2 * b
        z = (x[j], x[j + 1])
        t1 = v1[0] * z[0] + v1[1] * z[1]
        t2 = v2[0] * z[0] + v2[1] * z[1]
        a1 = 3.0 * q1 * t1 * t1
        a2 = 3.0 * q2 * t2 * t2
        out.append(
            (
                h00 + a1 * v1[0] * v1[0] + a2 * v2[0] * v2[0],
                h01 + a1 * v1[0] * v1[1] + a2 * v2[0] * v2[1],
                h11 + a1 * v1[1] * v1[1] + a2 * v2[1] * v2[1],
            )
        )
    return out


def noisy_global_grad(x, blocks, vecs, quartics, rng):
    return [g + rng.gauss(0.0, SIGMA) for g in grad_global(x, blocks, vecs, quartics)]


def avg_noisy_global_grad(x, blocks, vecs, quartics, rng):
    acc = [0.0] * D
    for _ in range(PROBE_REPS):
        g = noisy_global_grad(x, blocks, vecs, quartics, rng)
        for j, val in enumerate(g):
            acc[j] += val
    return [val / PROBE_REPS for val in acc]


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


def sym_clip_block(h00, h01, h11, lo=LAMBDA_MIN, hi=LAMBDA_MAX):
    tr = h00 + h11
    diff = h00 - h11
    rad = math.sqrt(max(0.0, diff * diff + 4.0 * h01 * h01))
    l1 = max(lo, min(hi, 0.5 * (tr + rad)))
    l2 = max(lo, min(hi, 0.5 * (tr - rad)))
    theta = 0.5 * math.atan2(2.0 * h01, diff)
    c, s = math.cos(theta), math.sin(theta)
    return (
        c * c * l1 + s * s * l2,
        c * s * (l1 - l2),
        s * s * l1 + c * c * l2,
    )


def block_correction_from_h(h00, h01, h11):
    h00, h01, h11 = sym_clip_block(h00, h01, h11)
    a = (1.0 - ETA * h00, -ETA * h01, -ETA * h01, 1.0 - ETA * h11)
    ak = mat2_pow(a, K)
    local_map = (1.0 - ak[0], -ak[1], -ak[2], 1.0 - ak[3])
    corr = mat2_inv(local_map)

    # Cap correction eigenvalues so a nearly-zero fitted curvature cannot
    # dominate this synthetic diagnostic.
    c00, c01, c11 = sym_clip_block(corr[0], 0.5 * (corr[1] + corr[2]), corr[3], 0.0, C_MAX)
    return (c00, c01, c01, c11)


def estimate_blocks(pairs):
    acc = [
        {
            "ss00": 0.0,
            "ss01": 0.0,
            "ss11": 0.0,
            "ys00": 0.0,
            "ys01": 0.0,
            "ys10": 0.0,
            "ys11": 0.0,
        }
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

    out = []
    for cell in acc:
        ss = (cell["ss00"] + RIDGE, cell["ss01"], cell["ss01"], cell["ss11"] + RIDGE)
        ys = (cell["ys00"], cell["ys01"], cell["ys10"], cell["ys11"])
        h = mat2_mul(ys, mat2_inv(ss))
        out.append(sym_clip_block(h[0], 0.5 * (h[1] + h[2]), h[3]))
    return out


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
    scale = PROBE_RADIUS / math.sqrt(D)
    for _ in range(count):
        rows.append([scale * rng.choice((-1.0, 1.0)) for _ in range(D)])
    mean = mean_rows(rows)
    return [sub(row, mean) for row in rows]


def coordinate_offsets(count):
    rows = []
    for i in range(count):
        row = [0.0] * D
        row[(i // 2) % D] = PROBE_RADIUS if i % 2 == 0 else -PROBE_RADIUS
        rows.append(row)
    mean = mean_rows(rows)
    return [sub(row, mean) for row in rows]


def directed_offsets(count, vecs):
    dirs = []
    for b, (v1, v2) in enumerate(vecs):
        for v in (v1, v2):
            row = [0.0] * D
            row[2 * b] = v[0]
            row[2 * b + 1] = v[1]
            dirs.append(row)
    rows = []
    for i in range(count):
        sign = 1.0 if i % 2 == 0 else -1.0
        rows.append(mul(sign * PROBE_RADIUS, dirs[(i // 2) % len(dirs)]))
    mean = mean_rows(rows)
    return [sub(row, mean) for row in rows]


def probe_pairs(x0, offsets, blocks, vecs, quartics, rng):
    xs = [add(x0, off) for off in offsets]
    grads = [avg_noisy_global_grad(x, blocks, vecs, quartics, rng) for x in xs]
    xbar = mean_rows(xs)
    gbar = mean_rows(grads)
    return [(sub(x, xbar), sub(g, gbar)) for x, g in zip(xs, grads)]


def path_pairs(steps, grads):
    pairs = []
    for k in range(1, K):
        disps = [steps[i][k] for i in range(len(steps))]
        gs = [grads[i][k] for i in range(len(grads))]
        dbar = mean_rows(disps)
        gbar = mean_rows(gs)
        for d, g in zip(disps, gs):
            pairs.append((sub(d, dbar), sub(g, gbar)))
    return pairs


def make_rounds(seed, n_workers, blocks, vecs, quartics):
    rng = random.Random(seed)
    x = [rng.gauss(0.0, 0.35) for _ in range(D)]
    rounds = []
    prev_x = None
    prev_g_obs = None
    secant_history = []
    for _round in range(ROUNDS):
        start_x = list(x)
        start_loss = loss(x, blocks, vecs, quartics)
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
                z = add(x, disp)
                base_grad = grad_global(z, blocks, vecs, quartics)
                grad = [
                    g + worker_bias[i][j] + rng.gauss(0.0, SIGMA)
                    for j, g in enumerate(base_grad)
                ]
                if k == 0:
                    first_step_grads.append(grad)
                step_displacements[i][k] = disp
                step_gradients[i][k] = grad
                dev[i] = sub(disp, mul(ETA, grad))

        endpoints = [list(v) for v in dev]
        fedavg = mean_rows(endpoints)
        g_obs = mean_rows(first_step_grads)
        if prev_x is None:
            secant = list(fedavg)
        else:
            secant_history.append((sub(start_x, prev_x), sub(g_obs, prev_g_obs)))
            secant = dynamics_correct(fedavg, estimate_blocks(secant_history[-SECANT_WINDOW:]))

        rounds.append(
            (
                start_loss,
                start_x,
                step_displacements,
                step_gradients,
                endpoints,
                fedavg,
                secant,
            )
        )
        x = add(x, fedavg)
        prev_x = start_x
        prev_g_obs = g_obs
    return rounds


def augment_rounds(seed, n_workers, blocks, vecs, quartics):
    rows = []
    raw = make_rounds(seed, n_workers, blocks, vecs, quartics)
    for idx, (start_loss, start_x, steps, grads, endpoints, fedavg, secant) in enumerate(raw):
        rng = random.Random(seed * 1009 + idx)
        budget = n_workers * K
        path_blocks = estimate_blocks(path_pairs(steps, grads))
        random_blocks = estimate_blocks(
            probe_pairs(start_x, random_offsets(rng, budget), blocks, vecs, quartics, rng)
        )
        coord_blocks = estimate_blocks(
            probe_pairs(start_x, coordinate_offsets(budget), blocks, vecs, quartics, rng)
        )
        directed_blocks = estimate_blocks(
            probe_pairs(start_x, directed_offsets(budget, vecs), blocks, vecs, quartics, rng)
        )
        best_worker = min(
            endpoints,
            key=lambda update: loss(add(start_x, update), blocks, vecs, quartics),
        )
        rows.append(
            {
                "start_loss": start_loss,
                "start_x": start_x,
                "fedavg": fedavg,
                "secant": secant,
                "best_worker": best_worker,
                "path": dynamics_correct(fedavg, path_blocks),
                "random_probe": dynamics_correct(fedavg, random_blocks),
                "coord_probe": dynamics_correct(fedavg, coord_blocks),
                "directed_probe": dynamics_correct(fedavg, directed_blocks),
                "oracle": dynamics_correct(fedavg, true_hessian_blocks(start_x, blocks, vecs, quartics)),
            }
        )
    return rows


def candidate(row, arm, param):
    if arm == "scale":
        return mul(param, row["fedavg"])
    if arm in ("secant", "best_worker"):
        return row[arm]
    return add(row["fedavg"], mul(param, sub(row[arm], row["fedavg"])))


def score(rows, arm, param, blocks, vecs, quartics):
    base = 0.0
    total = 0.0
    for row in rows:
        start_x = row["start_x"]
        fedavg = row["fedavg"]
        base += row["start_loss"] - loss(add(start_x, fedavg), blocks, vecs, quartics)
        update = candidate(row, arm, param)
        total += row["start_loss"] - loss(add(start_x, update), blocks, vecs, quartics)
    return total / base if base > 0 else 1.0


def select(rows, arm, blocks, vecs, quartics):
    if arm in ("secant", "best_worker"):
        return 1.0
    grid = SCALES if arm == "scale" else GAMMAS
    return max(grid, key=lambda p: score(rows, arm, p, blocks, vecs, quartics))


def mean_ci(vals):
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
    return m, 1.96 * math.sqrt(var / len(vals))


def main():
    blocks, _eigs, vecs, quartics = make_blocks()
    print(
        "rotated nonlinear probe: "
        f"d={D} blocks={D // 2} sharp_blocks={N_SHARP_BLOCKS} K={K} eta={ETA} "
        f"sigma={SIGMA} het={HET} rounds={ROUNDS} probe_reps={PROBE_REPS} "
        f"probe_radius={PROBE_RADIUS} cmax={C_MAX} secant_window={SECANT_WINDOW}"
    )
    print()
    arms = (
        "scale",
        "oracle",
        "path",
        "random_probe",
        "coord_probe",
        "directed_probe",
        "secant",
        "best_worker",
    )
    print(
        f"{'N':>5} {'scale*':>7} {'oracle*':>8} {'path*':>7} {'rand*':>7} "
        f"{'coord*':>7} {'dir*':>7} {'scalar':>12} {'oracle':>12} {'path':>12} "
        f"{'random':>12} {'coord':>12} {'directed':>12} {'secant':>12} "
        f"{'best':>12} {'rand-scale':>13} {'dir-scale':>13} {'best-rand':>13}"
    )
    for n_workers in N_WORKERS:
        pilot = []
        heldout = []
        for r in range(PILOT_REPS):
            pilot.extend(augment_rounds(230_000 + 1000 * n_workers + r, n_workers, blocks, vecs, quartics))
        for r in range(HELDOUT_REPS):
            heldout.append(
                augment_rounds(231_000 + 1000 * n_workers + r, n_workers, blocks, vecs, quartics)
            )

        params = {arm: select(pilot, arm, blocks, vecs, quartics) for arm in arms}
        vals = {
            arm: [score(rows, arm, params[arm], blocks, vecs, quartics) for rows in heldout]
            for arm in arms
        }
        rand_diff = [a - b for a, b in zip(vals["random_probe"], vals["scale"])]
        dir_diff = [a - b for a, b in zip(vals["directed_probe"], vals["scale"])]
        best_diff = [a - b for a, b in zip(vals["best_worker"], vals["random_probe"])]
        cells = []
        for arm in arms:
            m, hw = mean_ci(vals[arm])
            cells.append(f"{m:6.3f}+/-{hw:5.3f}")
        rm, rhw = mean_ci(rand_diff)
        dm, dhw = mean_ci(dir_diff)
        bm, bhw = mean_ci(best_diff)
        print(
            f"{n_workers:5d} {params['scale']:7.2f} {params['oracle']:8.2f} "
            f"{params['path']:7.2f} {params['random_probe']:7.2f} "
            f"{params['coord_probe']:7.2f} {params['directed_probe']:7.2f} "
            f"{cells[0]:>12} {cells[1]:>12} {cells[2]:>12} {cells[3]:>12} "
            f"{cells[4]:>12} {cells[5]:>12} {cells[6]:>12} {cells[7]:>12} "
            f"{rm:6.3f}+/-{rhw:5.3f} {dm:6.3f}+/-{dhw:5.3f} {bm:6.3f}+/-{bhw:5.3f}"
        )


if __name__ == "__main__":
    main()
