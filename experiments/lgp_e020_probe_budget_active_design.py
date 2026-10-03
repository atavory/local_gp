"""E020: probe-budget and active-design gate on the rotated nonlinear bed.

E016 showed that probe-fed dynamics-aware correction survives rotation on a
synthetic CPU bed. E017 showed that residual shrinkage can replace an arbitrary
target-tuned gamma on the diagonal nonlinear bed. This gate asks how much probe
budget is needed and whether simple causal active designs reduce that budget.

All probe arms use the K-step local-map correction

    [I - (I - eta Hhat)^K]^{-1} Delta_K

with residual shrinkage. The only target-tuned scalar is the FedAvg scale null.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.lgp_e016_rotated_nonlinear_probe as e16


N_WORKERS = (8, 32, 128)
ROUNDS = 32
PILOT_REPS = 3
HELDOUT_REPS = 4
POLICIES = ("random", "value_weighted", "uncertainty_value")
RISK_RULES = ("risk_reliability", "risk_sqrt")
SCALES = [i / 20 for i in range(0, 81)]
EPS = 1e-12
PRIOR_RIDGE = 1e-3
PRIOR_BLEND = 0.35
NO_CORRECTION_LAMBDA = 1.0 / e16.ETA


def sqnorm(a):
    return sum(x * x for x in a)


def clip(x, lo, hi):
    return min(hi, max(lo, x))


def gamma_from_rho(rule, rho):
    trust = clip(1.0 - rho, 0.0, 1.0)
    if rule == "risk_reliability":
        return trust
    if rule == "risk_sqrt":
        return math.sqrt(trust)
    raise ValueError(f"unknown risk rule: {rule}")


def no_correction_prior():
    return [(NO_CORRECTION_LAMBDA, 0.0, NO_CORRECTION_LAMBDA) for _ in range(e16.D // 2)]


def eig2_block(block):
    h00, h01, h11 = block
    tr = h00 + h11
    diff = h00 - h11
    rad = math.sqrt(max(0.0, diff * diff + 4.0 * h01 * h01))
    l1 = 0.5 * (tr + rad)
    l2 = 0.5 * (tr - rad)
    theta = 0.5 * math.atan2(2.0 * h01, diff)
    c, s = math.cos(theta), math.sin(theta)
    return ((l1, (c, s)), (l2, (-s, c)))


def rayleigh(block, v):
    h00, h01, h11 = block
    return h00 * v[0] * v[0] + 2.0 * h01 * v[0] * v[1] + h11 * v[1] * v[1]


def correction_coeff(lam):
    if lam <= 1e-9 or lam >= e16.LAMBDA_MAX:
        return 1.0
    denom = 1.0 - (1.0 - e16.ETA * lam) ** e16.K
    if denom <= 1e-12:
        return 1.0
    return min(e16.C_MAX, max(1.0, 1.0 / denom))


def correction_value(lam):
    c = correction_coeff(lam)
    return (1.0 - 1.0 / c) ** 2 / max(c, EPS)


def direction_scores(prior_blocks, uncertainty_blocks=None):
    rows = []
    for b, block in enumerate(prior_blocks):
        for lam, v in eig2_block(block):
            unc = 1.0
            if uncertainty_blocks is not None:
                unc_lam = rayleigh(uncertainty_blocks[b], v)
                prior_lam = max(abs(lam), 1.0)
                unc = 0.05 + min(5.0, abs(unc_lam - lam) / prior_lam)
            rows.append((max(0.0, correction_value(lam) * unc), b, v))
    return rows


def random_offsets(rng, count):
    scale = e16.PROBE_RADIUS / math.sqrt(e16.D)
    return [
        [scale * rng.choice((-1.0, 1.0)) for _ in range(e16.D)]
        for _ in range(count)
    ]


def weighted_direction_offsets(rng, count, scored_dirs):
    total = sum(w for w, _b, _v in scored_dirs)
    if total <= EPS:
        return random_offsets(rng, count)

    cdf = []
    acc = 0.0
    for w, b, v in scored_dirs:
        acc += w
        cdf.append((acc, b, v))

    rows = []
    pair_count = (count + 1) // 2
    for _ in range(pair_count):
        draw = rng.random() * total
        b, v = cdf[-1][1], cdf[-1][2]
        for cut, cb, cv in cdf:
            if draw <= cut:
                b, v = cb, cv
                break
        for sign in (1.0, -1.0):
            if len(rows) >= count:
                break
            row = [0.0] * e16.D
            row[2 * b] = sign * e16.PROBE_RADIUS * v[0]
            row[2 * b + 1] = sign * e16.PROBE_RADIUS * v[1]
            rows.append(row)
    return rows


def top_direction_offsets(count, scored_dirs):
    ranked = sorted(scored_dirs, reverse=True, key=lambda row: row[0])
    if not ranked or ranked[0][0] <= EPS:
        rng = random.Random(0)
        return random_offsets(rng, count)
    rows = []
    idx = 0
    while len(rows) < count:
        _w, b, v = ranked[idx % len(ranked)]
        for sign in (1.0, -1.0):
            if len(rows) >= count:
                break
            row = [0.0] * e16.D
            row[2 * b] = sign * e16.PROBE_RADIUS * v[0]
            row[2 * b + 1] = sign * e16.PROBE_RADIUS * v[1]
            rows.append(row)
        idx += 1
    return rows


def collect_samples(start_x, offsets, blocks, vecs, quartics, rng):
    samples = []
    for off in offsets:
        x = e16.add(start_x, off)
        g = e16.avg_noisy_global_grad(x, blocks, vecs, quartics, rng)
        samples.append((x, g))
    return samples


def samples_to_pairs(samples):
    xbar = e16.mean_rows([x for x, _g in samples])
    gbar = e16.mean_rows([g for _x, g in samples])
    return [(e16.sub(x, xbar), e16.sub(g, gbar)) for x, g in samples]


def split_pairs(pairs):
    fit = pairs[::2]
    val = pairs[1::2]
    if not val:
        val = fit
    return fit, val


def estimate_blocks(pairs, prior_blocks):
    prior = prior_blocks if prior_blocks is not None else no_correction_prior()
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
        for _ in range(e16.D // 2)
    ]
    for s, y in pairs:
        for b in range(e16.D // 2):
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
    for b, cell in enumerate(acc):
        p00, p01, p11 = prior[b]
        ss = (
            cell["ss00"] + PRIOR_RIDGE,
            cell["ss01"],
            cell["ss01"],
            cell["ss11"] + PRIOR_RIDGE,
        )
        ys = (
            cell["ys00"] + PRIOR_RIDGE * p00,
            cell["ys01"] + PRIOR_RIDGE * p01,
            cell["ys10"] + PRIOR_RIDGE * p01,
            cell["ys11"] + PRIOR_RIDGE * p11,
        )
        h = e16.mat2_mul(ys, e16.mat2_inv(ss))
        out.append(e16.sym_clip_block(h[0], 0.5 * (h[1] + h[2]), h[3]))
    return out


def residual_ratio(pairs, h_blocks):
    residual = 0.0
    target = 0.0
    for s, y in pairs:
        pred = e16.hmul(s, h_blocks)
        residual += sqnorm(e16.sub(y, pred))
        target += sqnorm(y)
    return residual / max(target, EPS)


def blend_blocks(old_blocks, new_blocks):
    if old_blocks is None:
        return list(new_blocks)
    out = []
    for old, new in zip(old_blocks, new_blocks):
        out.append(tuple((1.0 - PRIOR_BLEND) * a + PRIOR_BLEND * b for a, b in zip(old, new)))
    return out


def make_probe_samples(policy, start_x, budget, prior_blocks, blocks, vecs, quartics, rng):
    if policy == "random" or prior_blocks is None:
        return collect_samples(
            start_x, random_offsets(rng, budget), blocks, vecs, quartics, rng
        )

    if policy == "value_weighted":
        offsets = weighted_direction_offsets(rng, budget, direction_scores(prior_blocks))
        return collect_samples(start_x, offsets, blocks, vecs, quartics, rng)

    if policy == "uncertainty_value":
        explore = max(2, budget // 2)
        exploit = budget - explore
        first_samples = collect_samples(
            start_x, random_offsets(rng, explore), blocks, vecs, quartics, rng
        )
        first_pairs = samples_to_pairs(first_samples)
        provisional = estimate_blocks(first_pairs, prior_blocks)
        scored = direction_scores(prior_blocks, provisional)
        second_samples = []
        if exploit > 0:
            offsets = top_direction_offsets(exploit, scored)
            second_samples = collect_samples(start_x, offsets, blocks, vecs, quartics, rng)
        return first_samples + second_samples

    raise ValueError(f"unknown policy: {policy}")


def probe_arm(policy, start_x, fedavg, budget, prior_blocks, blocks, vecs, quartics, rng):
    samples = make_probe_samples(
        policy, start_x, budget, prior_blocks, blocks, vecs, quartics, rng
    )
    pairs = samples_to_pairs(samples)
    fit_pairs, val_pairs = split_pairs(pairs)
    h_fit = estimate_blocks(fit_pairs, prior_blocks)
    h_all = estimate_blocks(pairs, prior_blocks)
    rho = residual_ratio(val_pairs, h_fit)
    corrected = e16.dynamics_correct(fedavg, h_fit)
    updates = {}
    gammas = {}
    for rule in RISK_RULES:
        gamma = gamma_from_rho(rule, rho)
        gammas[rule] = gamma
        updates[rule] = e16.add(fedavg, e16.mul(gamma, e16.sub(corrected, fedavg)))
    return {
        "rho": rho,
        "gammas": gammas,
        "updates": updates,
        "prior_update": blend_blocks(prior_blocks, h_all),
    }


def budgets_for(n_workers):
    return sorted({max(4, n_workers // 2), n_workers, 2 * n_workers, n_workers * e16.K})


def stable_policy_seed(seed, round_idx, n_workers, budget, policy):
    policy_id = POLICIES.index(policy) + 1
    return seed * 1009 + round_idx * 9173 + n_workers * 97 + budget * 13 + policy_id


def augment_rounds(seed, n_workers, blocks, vecs, quartics, budgets):
    raw = e16.make_rounds(seed, n_workers, blocks, vecs, quartics)
    priors = {(budget, policy): None for budget in budgets for policy in POLICIES}
    rows = []
    for idx, (start_loss, start_x, _steps, _grads, endpoints, fedavg, secant) in enumerate(raw):
        best_worker = min(
            endpoints,
            key=lambda update: e16.loss(e16.add(start_x, update), blocks, vecs, quartics),
        )
        probes = {}
        for budget in budgets:
            for policy in POLICIES:
                key = (budget, policy)
                rng = random.Random(stable_policy_seed(seed, idx, n_workers, budget, policy))
                arm = probe_arm(
                    policy,
                    start_x,
                    fedavg,
                    budget,
                    priors[key],
                    blocks,
                    vecs,
                    quartics,
                    rng,
                )
                priors[key] = arm["prior_update"]
                probes[key] = arm
        rows.append(
            {
                "start_loss": start_loss,
                "start_x": start_x,
                "fedavg": fedavg,
                "secant": secant,
                "best_worker_true": best_worker,
                "oracle": e16.dynamics_correct(
                    fedavg, e16.true_hessian_blocks(start_x, blocks, vecs, quartics)
                ),
                "probes": probes,
            }
        )
    return rows


def flatten(batches):
    out = []
    for batch in batches:
        out.extend(batch)
    return out


def candidate(row, arm, param=None, budget=None, policy=None, rule=None):
    if arm == "fedavg":
        return row["fedavg"]
    if arm == "scale":
        return e16.mul(param, row["fedavg"])
    if arm in ("secant", "best_worker_true", "oracle"):
        return row[arm]
    if arm == "probe":
        return row["probes"][(budget, policy)]["updates"][rule]
    raise ValueError(f"unknown arm: {arm}")


def score(rows, blocks, vecs, quartics, arm, param=None, budget=None, policy=None, rule=None):
    base = 0.0
    total = 0.0
    for row in rows:
        start_x = row["start_x"]
        fedavg = row["fedavg"]
        base += row["start_loss"] - e16.loss(e16.add(start_x, fedavg), blocks, vecs, quartics)
        update = candidate(row, arm, param, budget, policy, rule)
        total += row["start_loss"] - e16.loss(e16.add(start_x, update), blocks, vecs, quartics)
    return total / base if base > 0 else 1.0


def select_scale(rows, blocks, vecs, quartics):
    return max(SCALES, key=lambda s: score(rows, blocks, vecs, quartics, "scale", param=s))


def select_rule(rows, blocks, vecs, quartics, budget, policy):
    return max(
        RISK_RULES,
        key=lambda rule: score(
            rows, blocks, vecs, quartics, "probe", budget=budget, policy=policy, rule=rule
        ),
    )


def mean_ci(vals):
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
    return m, 1.96 * math.sqrt(var / len(vals))


def format_ci(vals):
    m, hw = mean_ci(vals)
    return f"{m:6.3f}+/-{hw:5.3f}"


def mean_probe_gamma(rows, budget, policy, rule):
    vals = [row["probes"][(budget, policy)]["gammas"][rule] for row in rows]
    return sum(vals) / len(vals)


def main():
    blocks, _eigs, vecs, quartics = e16.make_blocks()
    print(
        "probe-budget active design: "
        f"d={e16.D} blocks={e16.D // 2} K={e16.K} eta={e16.ETA} "
        f"sigma={e16.SIGMA} het={e16.HET} rounds={ROUNDS} "
        f"pilot_reps={PILOT_REPS} heldout_reps={HELDOUT_REPS} "
        f"probe_reps={e16.PROBE_REPS} radius={e16.PROBE_RADIUS} "
        f"prior_ridge={PRIOR_RIDGE} prior_blend={PRIOR_BLEND}",
        flush=True,
    )
    print()
    print(
        "CONTROL "
        f"{'N':>5} {'scale*':>7} {'scalar':>12} {'full_random':>12} "
        f"{'full_rule':>16} {'full_gP':>8} {'full_gH':>8} {'oracle':>12} "
        f"{'secant':>12} {'best_worker':>12} {'full-scalar':>13} "
        f"{'best-full':>13}",
        flush=True,
    )
    print(
        "BUDGET  "
        f"{'N':>5} {'budget':>7} {'policy':>18} {'rule':>16} "
        f"{'gammaP':>8} {'gammaH':>8} {'ratio':>12} {'policy-scalar':>13} "
        f"{'policy-randB':>13} {'policy-full':>13} {'best-policy':>13}",
        flush=True,
    )

    for n_workers in N_WORKERS:
        budgets = budgets_for(n_workers)
        pilot = []
        heldout = []
        for r in range(PILOT_REPS):
            pilot.extend(
                augment_rounds(
                    240_000 + 1000 * n_workers + r,
                    n_workers,
                    blocks,
                    vecs,
                    quartics,
                    budgets,
                )
            )
        for r in range(HELDOUT_REPS):
            heldout.append(
                augment_rounds(
                    241_000 + 1000 * n_workers + r,
                    n_workers,
                    blocks,
                    vecs,
                    quartics,
                    budgets,
                )
            )

        heldout_flat = flatten(heldout)
        best_scale = select_scale(pilot, blocks, vecs, quartics)
        rules = {
            (budget, policy): select_rule(pilot, blocks, vecs, quartics, budget, policy)
            for budget in budgets
            for policy in POLICIES
        }
        full_budget = n_workers * e16.K
        full_rule = rules[(full_budget, "random")]

        control_vals = {
            "scale": [
                score(rows, blocks, vecs, quartics, "scale", param=best_scale)
                for rows in heldout
            ],
            "full_random": [
                score(
                    rows,
                    blocks,
                    vecs,
                    quartics,
                    "probe",
                    budget=full_budget,
                    policy="random",
                    rule=full_rule,
                )
                for rows in heldout
            ],
            "oracle": [score(rows, blocks, vecs, quartics, "oracle") for rows in heldout],
            "secant": [score(rows, blocks, vecs, quartics, "secant") for rows in heldout],
            "best_worker_true": [
                score(rows, blocks, vecs, quartics, "best_worker_true") for rows in heldout
            ],
        }
        full_minus_scalar = [
            a - b for a, b in zip(control_vals["full_random"], control_vals["scale"])
        ]
        best_minus_full = [
            a - b
            for a, b in zip(control_vals["best_worker_true"], control_vals["full_random"])
        ]
        print(
            "CONTROL "
            f"{n_workers:5d} {best_scale:7.2f} {format_ci(control_vals['scale']):>12} "
            f"{format_ci(control_vals['full_random']):>12} {full_rule:>16} "
            f"{mean_probe_gamma(pilot, full_budget, 'random', full_rule):8.3f} "
            f"{mean_probe_gamma(heldout_flat, full_budget, 'random', full_rule):8.3f} "
            f"{format_ci(control_vals['oracle']):>12} {format_ci(control_vals['secant']):>12} "
            f"{format_ci(control_vals['best_worker_true']):>12} "
            f"{format_ci(full_minus_scalar):>13} {format_ci(best_minus_full):>13}",
            flush=True,
        )

        random_by_budget = {}
        for budget in budgets:
            rule = rules[(budget, "random")]
            random_by_budget[budget] = [
                score(
                    rows,
                    blocks,
                    vecs,
                    quartics,
                    "probe",
                    budget=budget,
                    policy="random",
                    rule=rule,
                )
                for rows in heldout
            ]

        for budget in budgets:
            for policy in POLICIES:
                rule = rules[(budget, policy)]
                vals = [
                    score(
                        rows,
                        blocks,
                        vecs,
                        quartics,
                        "probe",
                        budget=budget,
                        policy=policy,
                        rule=rule,
                    )
                    for rows in heldout
                ]
                policy_minus_scalar = [
                    a - b for a, b in zip(vals, control_vals["scale"])
                ]
                policy_minus_random_budget = [
                    a - b for a, b in zip(vals, random_by_budget[budget])
                ]
                policy_minus_full = [
                    a - b for a, b in zip(vals, control_vals["full_random"])
                ]
                best_minus_policy = [
                    a - b for a, b in zip(control_vals["best_worker_true"], vals)
                ]
                print(
                    "BUDGET  "
                    f"{n_workers:5d} {budget:7d} {policy:>18} {rule:>16} "
                    f"{mean_probe_gamma(pilot, budget, policy, rule):8.3f} "
                    f"{mean_probe_gamma(heldout_flat, budget, policy, rule):8.3f} "
                    f"{format_ci(vals):>12} {format_ci(policy_minus_scalar):>13} "
                    f"{format_ci(policy_minus_random_budget):>13} "
                    f"{format_ci(policy_minus_full):>13} {format_ci(best_minus_policy):>13}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
