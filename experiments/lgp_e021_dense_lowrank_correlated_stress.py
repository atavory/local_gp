"""E021: dense low-rank geometry with correlated noise.

E016 showed that dynamics-aware probe corrections survive rotated 2D nonlinear
blocks. This gate removes the tiny-block assumption. The bed has diagonal
background curvature plus a dense low-rank spike subspace, quartics in that
same subspace, centered worker heterogeneity, and correlated low-rank gradient
noise.

All correction arms use the K-step local-map operator

    [I - (I - eta Hhat)^K]^{-1} Delta_K,

then shrink estimated corrections with the split-probe residual rule from E017.
"""

from __future__ import annotations

import math
import random


D = 96
RANK = 8
SKETCH_DIM = 8
K = 8
ETA = 0.025
N_WORKERS = (8, 32, 128)
ROUNDS = 24
PILOT_REPS = 3
HELDOUT_REPS = 4
PROBE_REPS = 3
PROBE_RADIUS = 0.32
PROBE_BUDGET_CAP = 128
SUBSPACE_PROBE_BUDGET = 64
SIGMA_DIAG = 0.20
SIGMA_LR = 0.55
HET_DIAG = 0.18
HET_LR = 0.25
RIDGE = 1e-4
EPS = 1e-12
LAMBDA_MIN = 0.03
LAMBDA_MAX = (2.0 - 1e-9) / ETA
C_MAX = 25.0
CG_TOL = 1e-8
CG_MAX_ITER = 160
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


def sqnorm(a):
    return dot(a, a)


def norm(a):
    return math.sqrt(max(0.0, sqnorm(a)))


def mean_rows(rows):
    return [sum(row[j] for row in rows) / len(rows) for j in range(len(rows[0]))]


def clip(x, lo, hi):
    return min(hi, max(lo, x))


def cap_update(base, update):
    base_norm = max(norm(base), EPS)
    update_norm = norm(update)
    if update_norm <= C_MAX * base_norm:
        return update
    return mul(C_MAX * base_norm / update_norm, update)


def orthonormalize(rows, dim, target, rng):
    basis = []
    for row in rows:
        v = list(row)
        for q in basis:
            v = sub(v, mul(dot(v, q), q))
        nv = norm(v)
        if nv > 1e-8:
            basis.append(mul(1.0 / nv, v))
        if len(basis) == target:
            return basis
    while len(basis) < target:
        v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        for q in basis:
            v = sub(v, mul(dot(v, q), q))
        nv = norm(v)
        if nv > 1e-8:
            basis.append(mul(1.0 / nv, v))
    return basis


def qtv(q_basis, x):
    return [dot(q, x) for q in q_basis]


def qmul(q_basis, coeffs):
    out = [0.0] * D
    for q, c in zip(q_basis, coeffs):
        for j, qj in enumerate(q):
            out[j] += c * qj
    return out


def q_project(q_basis, x):
    return qmul(q_basis, qtv(q_basis, x))


def make_bed():
    rng = random.Random(0)
    basis = orthonormalize(
        ([rng.gauss(0.0, 1.0) for _ in range(D)] for _ in range(RANK)),
        D,
        RANK,
        rng,
    )
    base = [rng.uniform(0.06, 0.55) for _ in range(D)]
    spikes = [rng.uniform(7.0, 13.0) for _ in range(RANK)]
    q_diag = [rng.uniform(0.02, 0.16) for _ in range(D)]
    q_lr = [rng.uniform(3.0, 8.0) for _ in range(RANK)]
    random_q = orthonormalize(
        ([rng.gauss(0.0, 1.0) for _ in range(D)] for _ in range(SKETCH_DIM)),
        D,
        SKETCH_DIM,
        rng,
    )
    return {
        "u": basis,
        "base": base,
        "spikes": spikes,
        "q_diag": q_diag,
        "q_lr": q_lr,
        "random_q": random_q,
    }


def loss(x, bed):
    z = qtv(bed["u"], x)
    quad = 0.5 * sum(b * xj * xj for b, xj in zip(bed["base"], x))
    quad += 0.5 * sum(s * zj * zj for s, zj in zip(bed["spikes"], z))
    quartic = 0.25 * sum(q * xj**4 for q, xj in zip(bed["q_diag"], x))
    quartic += 0.25 * sum(q * zj**4 for q, zj in zip(bed["q_lr"], z))
    return quad + quartic


def grad_global(x, bed):
    z = qtv(bed["u"], x)
    out = [b * xj + q * xj**3 for b, q, xj in zip(bed["base"], bed["q_diag"], x)]
    lr_coeff = [s * zj + q * zj**3 for s, q, zj in zip(bed["spikes"], bed["q_lr"], z)]
    return add(out, qmul(bed["u"], lr_coeff))


def true_hessian_matvec(x, bed):
    z = qtv(bed["u"], x)
    diag = [b + 3.0 * q * xj * xj for b, q, xj in zip(bed["base"], bed["q_diag"], x)]
    lr_curv = [s + 3.0 * q * zj * zj for s, q, zj in zip(bed["spikes"], bed["q_lr"], z)]

    def hmul(v):
        out = [dj * vj for dj, vj in zip(diag, v)]
        coeff = [cj * aj for cj, aj in zip(lr_curv, qtv(bed["u"], v))]
        return add(out, qmul(bed["u"], coeff))

    return hmul


def sample_correlated_noise(rng, bed):
    diag = [rng.gauss(0.0, SIGMA_DIAG) for _ in range(D)]
    lowrank = [rng.gauss(0.0, SIGMA_LR) for _ in range(RANK)]
    return add(diag, qmul(bed["u"], lowrank))


def sample_worker_biases(rng, n_workers, bed):
    rows = []
    for _ in range(n_workers):
        diag = [rng.gauss(0.0, HET_DIAG) for _ in range(D)]
        lowrank = [rng.gauss(0.0, HET_LR) for _ in range(RANK)]
        rows.append(add(diag, qmul(bed["u"], lowrank)))
    avg = mean_rows(rows)
    return [sub(row, avg) for row in rows]


def noisy_grad(x, bed, rng):
    return add(grad_global(x, bed), sample_correlated_noise(rng, bed))


def avg_noisy_grad(x, bed, rng):
    acc = [0.0] * D
    for _ in range(PROBE_REPS):
        g = noisy_grad(x, bed, rng)
        for j, val in enumerate(g):
            acc[j] += val
    return [val / PROBE_REPS for val in acc]


def mat_eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def mat_vec(a, x):
    return [dot(row, x) for row in a]


def mat_mul(a, b):
    bt = list(zip(*b))
    return [[dot(row, col) for col in bt] for row in a]


def mat_sub(a, b):
    return [[x - y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def mat_scale(c, a):
    return [[c * x for x in row] for row in a]


def mat_pow(a, k):
    out = mat_eye(len(a))
    base = a
    n = k
    while n:
        if n & 1:
            out = mat_mul(out, base)
        base = mat_mul(base, base)
        n >>= 1
    return out


def mat_inverse(a):
    n = len(a)
    aug = [list(row) + eye for row, eye in zip(a, mat_eye(n))]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            aug[pivot][col] = 1e-12
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        aug[col] = [x / div for x in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor:
                aug[row] = [x - factor * y for x, y in zip(aug[row], aug[col])]
    return [row[n:] for row in aug]


def jacobi_eigh_sym(a):
    n = len(a)
    b = [list(row) for row in a]
    v = mat_eye(n)
    for _ in range(8 * n * n):
        p, q = 0, 1
        best = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                val = abs(b[i][j])
                if val > best:
                    best = val
                    p, q = i, j
        if best < 1e-10:
            break
        if abs(b[p][p] - b[q][q]) < 1e-14:
            angle = math.pi / 4.0
        else:
            angle = 0.5 * math.atan2(2.0 * b[p][q], b[p][p] - b[q][q])
        c, s = math.cos(angle), math.sin(angle)
        for r in range(n):
            brp, brq = b[r][p], b[r][q]
            b[r][p] = c * brp + s * brq
            b[r][q] = -s * brp + c * brq
        for r in range(n):
            bpr, bqr = b[p][r], b[q][r]
            b[p][r] = c * bpr + s * bqr
            b[q][r] = -s * bpr + c * bqr
        for r in range(n):
            vrp, vrq = v[r][p], v[r][q]
            v[r][p] = c * vrp + s * vrq
            v[r][q] = -s * vrp + c * vrq
    vals = [b[i][i] for i in range(n)]
    vecs = [[v[i][j] for i in range(n)] for j in range(n)]
    return vals, vecs


def sym_clip_matrix(a, lo=LAMBDA_MIN, hi=LAMBDA_MAX):
    n = len(a)
    sym = [[0.5 * (a[i][j] + a[j][i]) for j in range(n)] for i in range(n)]
    vals, vecs = jacobi_eigh_sym(sym)
    out = [[0.0 for _ in range(n)] for _ in range(n)]
    for lam, vec in zip(vals, vecs):
        lam = clip(lam, lo, hi)
        for i in range(n):
            for j in range(n):
                out[i][j] += lam * vec[i] * vec[j]
    return out


def coeff_from_diag(lhat):
    out = []
    for lam in lhat:
        if lam <= LAMBDA_MIN or lam >= LAMBDA_MAX:
            out.append(1.0)
            continue
        denom = 1.0 - (1.0 - ETA * lam) ** K
        out.append(1.0 if denom <= 1e-12 else min(C_MAX, max(1.0, 1.0 / denom)))
    return out


def diag_correct(delta, diag_lam):
    return [c * u for c, u in zip(coeff_from_diag(diag_lam), delta)]


def dynamics_correct_small(delta, hmat):
    hmat = sym_clip_matrix(hmat)
    n = len(hmat)
    step = mat_sub(mat_eye(n), mat_scale(ETA, hmat))
    local_map = mat_sub(mat_eye(n), mat_pow(step, K))
    corr = mat_vec(mat_inverse(local_map), delta)
    return cap_update(delta, corr)


def cg_solve(apply_a, b):
    x = [0.0] * len(b)
    r = list(b)
    p = list(r)
    rs = sqnorm(r)
    target = max(1.0, sqnorm(b))
    for _ in range(CG_MAX_ITER):
        if rs <= CG_TOL * CG_TOL * target:
            break
        ap = apply_a(p)
        denom = dot(p, ap)
        if abs(denom) < EPS:
            break
        alpha = rs / denom
        x = add(x, mul(alpha, p))
        r = sub(r, mul(alpha, ap))
        new_rs = sqnorm(r)
        if new_rs <= CG_TOL * CG_TOL * target:
            break
        beta = new_rs / max(rs, EPS)
        p = add(r, mul(beta, p))
        rs = new_rs
    return x


def dynamics_correct_matvec(delta, hmatvec):
    def local_map(v):
        w = list(v)
        for _ in range(K):
            w = sub(w, mul(ETA, hmatvec(w)))
        return sub(v, w)

    return cap_update(delta, cg_solve(local_map, delta))


def lowrank_correct(delta, q_basis, h_sub):
    z = qtv(q_basis, delta)
    corrected_z = dynamics_correct_small(z, h_sub)
    raw_sub = qmul(q_basis, z)
    corrected_sub = qmul(q_basis, corrected_z)
    return add(sub(delta, raw_sub), corrected_sub)


def combined_correct(delta, diag_lam, q_basis, residual_h):
    residual_h = sym_clip_matrix(residual_h, 0.0, LAMBDA_MAX)

    def hmatvec(v):
        out = [dj * vj for dj, vj in zip(diag_lam, v)]
        coeff = mat_vec(residual_h, qtv(q_basis, v))
        return add(out, qmul(q_basis, coeff))

    return dynamics_correct_matvec(delta, hmatvec)


def estimate_diag(pairs):
    num = [0.0] * D
    den = [RIDGE] * D
    for s, y in pairs:
        for j, (sj, yj) in enumerate(zip(s, y)):
            num[j] += sj * yj
            den[j] += sj * sj
    return [clip(n / d, LAMBDA_MIN, LAMBDA_MAX) for n, d in zip(num, den)]


def estimate_subspace_matrix(pairs, q_basis, residual_diag=None):
    k = len(q_basis)
    ss = [[RIDGE if i == j else 0.0 for j in range(k)] for i in range(k)]
    ys = [[0.0 for _ in range(k)] for _ in range(k)]
    for s, y in pairs:
        if residual_diag is not None:
            y = [yj - dj * sj for yj, dj, sj in zip(y, residual_diag, s)]
        a = qtv(q_basis, s)
        b = qtv(q_basis, y)
        for i in range(k):
            for j in range(k):
                ss[i][j] += a[i] * a[j]
                ys[i][j] += b[i] * a[j]
    return sym_clip_matrix(mat_mul(ys, mat_inverse(ss)), 0.0 if residual_diag is not None else LAMBDA_MIN, LAMBDA_MAX)


def model_residual_ratio(pairs, hmatvec):
    residual = 0.0
    target = 0.0
    for s, y in pairs:
        pred = hmatvec(s)
        residual += sqnorm(sub(y, pred))
        target += sqnorm(y)
    return residual / max(target, EPS)


def subspace_residual_ratio(pairs, q_basis, h_sub):
    residual = 0.0
    target = 0.0
    for s, y in pairs:
        a = qtv(q_basis, s)
        b = qtv(q_basis, y)
        pred = mat_vec(h_sub, a)
        residual += sqnorm(sub(b, pred))
        target += sqnorm(b)
    return residual / max(target, EPS)


def risk_gamma(rho):
    return math.sqrt(clip(1.0 - rho, 0.0, 1.0))


def random_offsets(rng, count):
    scale = PROBE_RADIUS / math.sqrt(D)
    rows = [[scale * rng.choice((-1.0, 1.0)) for _ in range(D)] for _ in range(count)]
    avg = mean_rows(rows)
    return [sub(row, avg) for row in rows]


def subspace_offsets(rng, q_basis, count):
    scale = PROBE_RADIUS / math.sqrt(len(q_basis))
    rows = []
    for _ in range(count):
        coeff = [scale * rng.choice((-1.0, 1.0)) for _ in q_basis]
        rows.append(qmul(q_basis, coeff))
    avg = mean_rows(rows)
    return [sub(row, avg) for row in rows]


def probe_pairs(x0, offsets, bed, rng):
    xs = [add(x0, off) for off in offsets]
    grads = [avg_noisy_grad(x, bed, rng) for x in xs]
    xbar = mean_rows(xs)
    gbar = mean_rows(grads)
    return [(sub(x, xbar), sub(g, gbar)) for x, g in zip(xs, grads)]


def split_pairs(pairs):
    mid = max(2, len(pairs) // 2)
    return pairs[:mid], pairs[mid:]


def learned_subspace(pairs, diag_lam, rng):
    residuals = []
    for s, y in pairs:
        residual = [yj - dj * sj for yj, dj, sj in zip(y, diag_lam, s)]
        residuals.append((sqnorm(residual), residual))
    residuals.sort(reverse=True, key=lambda item: item[0])
    return orthonormalize((row for _score, row in residuals), D, SKETCH_DIM, rng)


def make_rounds(seed, n_workers, bed):
    rng = random.Random(seed)
    x = [rng.gauss(0.0, 0.35) for _ in range(D)]
    rows = []
    prev_x = None
    prev_g_obs = None
    secant_history = []
    for _round in range(ROUNDS):
        start_x = list(x)
        start_loss = loss(start_x, bed)
        dev = [[0.0] * D for _ in range(n_workers)]
        biases = sample_worker_biases(rng, n_workers, bed)
        first_step_grads = []

        for k in range(K):
            for i in range(n_workers):
                z = add(start_x, dev[i])
                grad = add(add(grad_global(z, bed), biases[i]), sample_correlated_noise(rng, bed))
                if k == 0:
                    first_step_grads.append(grad)
                dev[i] = sub(dev[i], mul(ETA, grad))

        endpoints = [list(v) for v in dev]
        fedavg = mean_rows(endpoints)
        g_obs = mean_rows(first_step_grads)
        if prev_x is None:
            secant = list(fedavg)
        else:
            secant_history.append((sub(start_x, prev_x), sub(g_obs, prev_g_obs)))
            secant_diag = estimate_diag(secant_history[-SECANT_WINDOW:])
            secant = diag_correct(fedavg, secant_diag)

        rows.append((start_loss, start_x, endpoints, fedavg, secant))
        x = add(x, fedavg)
        prev_x = start_x
        prev_g_obs = g_obs
    return rows


def augment_rounds(seed, n_workers, bed):
    out = []
    raw = make_rounds(seed, n_workers, bed)
    for idx, (start_loss, start_x, endpoints, fedavg, secant) in enumerate(raw):
        rng = random.Random(seed * 1009 + idx)
        full_budget = min(PROBE_BUDGET_CAP, max(48, n_workers * K))
        sub_budget = min(SUBSPACE_PROBE_BUDGET, max(32, n_workers * K))

        full_pairs = probe_pairs(start_x, random_offsets(rng, full_budget), bed, rng)
        full_fit, full_val = split_pairs(full_pairs)
        diag_lam = estimate_diag(full_fit)
        learned_q = learned_subspace(full_fit, diag_lam, rng)

        diag_full = diag_correct(fedavg, diag_lam)
        diag_rho = model_residual_ratio(
            full_val,
            lambda v: [dj * vj for dj, vj in zip(diag_lam, v)],
        )

        lowrank_arms = {}
        for name, q_basis in (
            ("lr_random", bed["random_q"]),
            ("lr_true", bed["u"]),
            ("lr_learned", learned_q),
        ):
            pairs = probe_pairs(start_x, subspace_offsets(rng, q_basis, sub_budget), bed, rng)
            fit, val = split_pairs(pairs)
            h_sub = estimate_subspace_matrix(fit, q_basis)
            rho = subspace_residual_ratio(val, q_basis, h_sub)
            lowrank_arms[name] = {
                "full": lowrank_correct(fedavg, q_basis, h_sub),
                "gamma": risk_gamma(rho),
            }

        combo = {}
        for name, q_basis in (("combo_learned", learned_q), ("combo_true", bed["u"])):
            h_res = estimate_subspace_matrix(full_fit, q_basis, residual_diag=diag_lam)

            def hmv(v, q_basis=q_basis, h_res=h_res):
                out = [dj * vj for dj, vj in zip(diag_lam, v)]
                return add(out, qmul(q_basis, mat_vec(h_res, qtv(q_basis, v))))

            rho = model_residual_ratio(full_val, hmv)
            combo[name] = {
                "full": combined_correct(fedavg, diag_lam, q_basis, h_res),
                "gamma": risk_gamma(rho),
            }

        oracle_full = dynamics_correct_matvec(fedavg, true_hessian_matvec(start_x, bed))
        best_worker = min(endpoints, key=lambda update: loss(add(start_x, update), bed))

        row = {
            "start_loss": start_loss,
            "start_x": start_x,
            "fedavg": fedavg,
            "secant": secant,
            "best_worker": best_worker,
            "diag_full": diag_full,
            "diag_gamma": risk_gamma(diag_rho),
            "oracle_full": oracle_full,
        }
        for name, payload in lowrank_arms.items():
            row[f"{name}_full"] = payload["full"]
            row[f"{name}_gamma"] = payload["gamma"]
        for name, payload in combo.items():
            row[f"{name}_full"] = payload["full"]
            row[f"{name}_gamma"] = payload["gamma"]
        out.append(row)
    return out


def interp(fedavg, full, gamma):
    return add(fedavg, mul(gamma, sub(full, fedavg)))


def candidate(row, arm, param):
    if arm == "fedavg":
        return row["fedavg"]
    if arm == "scale":
        return mul(param, row["fedavg"])
    if arm in ("secant", "best_worker"):
        return row[arm]
    if arm == "oracle":
        return interp(row["fedavg"], row["oracle_full"], param)
    if arm == "combo_tuned":
        return interp(row["fedavg"], row["combo_learned_full"], param)
    if arm in ("diag", "lr_random", "lr_true", "lr_learned", "combo_learned", "combo_true"):
        return interp(row["fedavg"], row[f"{arm}_full"], row[f"{arm}_gamma"])
    raise ValueError(f"unknown arm: {arm}")


def score(rows, arm, param, bed):
    base = 0.0
    total = 0.0
    for row in rows:
        start_x = row["start_x"]
        fedavg = row["fedavg"]
        base += row["start_loss"] - loss(add(start_x, fedavg), bed)
        total += row["start_loss"] - loss(add(start_x, candidate(row, arm, param)), bed)
    return total / base if base > 0 else 1.0


def select(rows, arm, bed):
    if arm == "scale":
        grid = SCALES
    else:
        grid = GAMMAS
    return max(grid, key=lambda p: score(rows, arm, p, bed))


def flatten(batches):
    out = []
    for batch in batches:
        out.extend(batch)
    return out


def mean_ci(vals):
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
    return m, 1.96 * math.sqrt(var / len(vals))


def format_ci(vals):
    m, hw = mean_ci(vals)
    return f"{m:6.3f}+/-{hw:5.3f}"


def mean_gamma(rows, arm):
    return sum(row[f"{arm}_gamma"] for row in rows) / len(rows)


BED = make_bed()


def main():
    print(
        "dense lowrank correlated stress: "
        f"d={D} rank={RANK} sketch_dim={SKETCH_DIM} K={K} eta={ETA} "
        f"sigma_diag={SIGMA_DIAG} sigma_lr={SIGMA_LR} het_diag={HET_DIAG} "
        f"het_lr={HET_LR} rounds={ROUNDS} probe_reps={PROBE_REPS} "
        f"probe_budget_cap={PROBE_BUDGET_CAP} subspace_probe_budget={SUBSPACE_PROBE_BUDGET} "
        "risk_rule=sqrt(clip(1-rho,0,1))"
    )
    print()
    print(
        f"{'N':>5} {'scale*':>7} {'oracle*':>8} {'combo*':>8} "
        f"{'g_diag':>7} {'g_lrL':>7} {'g_combo':>8} "
        f"{'raw':>12} {'scalar':>12} {'diag':>12} {'lr_rand':>12} "
        f"{'lr_learn':>12} {'lr_true':>12} {'combo':>12} {'combo_true':>12} "
        f"{'combo_tuned':>12} {'oracle':>12} {'secant':>12} {'best':>12} "
        f"{'combo-scalar':>13} {'combo-diag':>13} {'best-combo':>13}"
    )
    for n_workers in N_WORKERS:
        pilot = []
        heldout = []
        for r in range(PILOT_REPS):
            pilot.extend(augment_rounds(260_000 + 1000 * n_workers + r, n_workers, BED))
        for r in range(HELDOUT_REPS):
            heldout.append(augment_rounds(261_000 + 1000 * n_workers + r, n_workers, BED))

        pilot_flat = pilot
        heldout_flat = flatten(heldout)
        scale = select(pilot_flat, "scale", BED)
        oracle_gamma = select(pilot_flat, "oracle", BED)
        combo_gamma = select(pilot_flat, "combo_tuned", BED)
        arms = (
            "fedavg",
            "scale",
            "diag",
            "lr_random",
            "lr_learned",
            "lr_true",
            "combo_learned",
            "combo_true",
            "combo_tuned",
            "oracle",
            "secant",
            "best_worker",
        )
        params = {
            "fedavg": None,
            "scale": scale,
            "combo_tuned": combo_gamma,
            "oracle": oracle_gamma,
            "diag": None,
            "lr_random": None,
            "lr_learned": None,
            "lr_true": None,
            "combo_learned": None,
            "combo_true": None,
            "secant": None,
            "best_worker": None,
        }
        vals = {arm: [score(rows, arm, params[arm], BED) for rows in heldout] for arm in arms}
        combo_minus_scalar = [a - b for a, b in zip(vals["combo_learned"], vals["scale"])]
        combo_minus_diag = [a - b for a, b in zip(vals["combo_learned"], vals["diag"])]
        best_minus_combo = [a - b for a, b in zip(vals["best_worker"], vals["combo_learned"])]

        print(
            f"{n_workers:5d} {scale:7.2f} {oracle_gamma:8.2f} {combo_gamma:8.2f} "
            f"{mean_gamma(heldout_flat, 'diag'):7.3f} "
            f"{mean_gamma(heldout_flat, 'lr_learned'):7.3f} "
            f"{mean_gamma(heldout_flat, 'combo_learned'):8.3f} "
            f"{format_ci(vals['fedavg']):>12} {format_ci(vals['scale']):>12} "
            f"{format_ci(vals['diag']):>12} {format_ci(vals['lr_random']):>12} "
            f"{format_ci(vals['lr_learned']):>12} {format_ci(vals['lr_true']):>12} "
            f"{format_ci(vals['combo_learned']):>12} {format_ci(vals['combo_true']):>12} "
            f"{format_ci(vals['combo_tuned']):>12} {format_ci(vals['oracle']):>12} "
            f"{format_ci(vals['secant']):>12} {format_ci(vals['best_worker']):>12} "
            f"{format_ci(combo_minus_scalar):>13} {format_ci(combo_minus_diag):>13} "
            f"{format_ci(best_minus_combo):>13}"
        )


if __name__ == "__main__":
    main()
