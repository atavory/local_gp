"""E017: risk-aware shrinkage and best-worker fallback on E015's bed.

E015 used a pilot-selected gamma for probe-fed dynamics-aware correction. This
gate asks whether a target-free residual rule can replace that tuning, and
whether an oracle best-worker endpoint explains the gain.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.lgp_e009_nonlinear_gamma_sweep as e9


PROBE_REPS = 4
PROBE_RADIUS = 0.25
RIDGE = 1e-4
EPS = 1e-12
LAMBDA_MAX = (2.0 - 1e-9) / e9.ETA
SCALES = [i / 20 for i in range(0, 81)]
GAMMAS = [i / 20 for i in range(0, 21)]
RISK_RULES = ("risk_reliability", "risk_sqrt")


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


def sqnorm(a):
    return sum(x * x for x in a)


def clip(x, lo, hi):
    return min(hi, max(lo, x))


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


def residual_ratio(pairs, lhat):
    residual = 0.0
    target = 0.0
    for s, y in pairs:
        pred = [lj * sj for lj, sj in zip(lhat, s)]
        residual += sqnorm(sub(y, pred))
        target += sqnorm(y)
    return residual / max(target, EPS)


def gamma_from_rho(rule, rho):
    trust = clip(1.0 - rho, 0.0, 1.0)
    if rule == "risk_reliability":
        return trust
    if rule == "risk_sqrt":
        return math.sqrt(trust)
    raise ValueError(f"unknown risk rule: {rule}")


def random_offsets(rng, count):
    rows = []
    scale = PROBE_RADIUS / (e9.D**0.5)
    for _ in range(count):
        rows.append([scale * rng.choice((-1.0, 1.0)) for _ in range(e9.D)])
    mean = mean_rows(rows)
    return [sub(row, mean) for row in rows]


def probe_pairs(start_x, offsets, lam, quartic, rng):
    xs = [add(start_x, off) for off in offsets]
    grads = [avg_noisy_global_grad(x, lam, quartic, rng) for x in xs]
    xbar = mean_rows(xs)
    gbar = mean_rows(grads)
    return [(sub(x, xbar), sub(g, gbar)) for x, g in zip(xs, grads)]


def split_random_probe_pairs(start_x, budget, lam, quartic, rng):
    offsets = random_offsets(rng, budget)
    split = max(2, budget // 2)
    fit_pairs = probe_pairs(start_x, offsets[:split], lam, quartic, rng)
    val_pairs = probe_pairs(start_x, offsets[split:], lam, quartic, rng)
    return fit_pairs, val_pairs


def corrected_update(fedavg, lhat):
    c = coeff_from_lam(lhat)
    return [cj * uj for cj, uj in zip(c, fedavg)]


def best_worker_endpoint(start_x, endpoints, lam, quartic):
    return max(
        endpoints,
        key=lambda update: e9.loss(start_x, lam, quartic)
        - e9.loss(add(start_x, update), lam, quartic),
    )


def augment_rounds(seed, n_workers, lam, quartic):
    rows = []
    raw = e9.make_rounds(seed, n_workers, lam, quartic)
    for idx, (start_loss, start_x, _steps, _grads, endpoints, fedavg, secant) in enumerate(raw):
        rng = random.Random(seed * 1009 + idx)
        budget = n_workers * e9.K

        full_pairs = probe_pairs(start_x, random_offsets(rng, budget), lam, quartic, rng)
        full_lam = estimate_diag_from_pairs(full_pairs)

        fit_pairs, val_pairs = split_random_probe_pairs(start_x, budget, lam, quartic, rng)
        risk_lam = estimate_diag_from_pairs(fit_pairs)
        rho = residual_ratio(val_pairs, risk_lam)
        risk_base = corrected_update(fedavg, risk_lam)

        rows.append(
            {
                "start_loss": start_loss,
                "start_x": start_x,
                "fedavg": fedavg,
                "random_probe": corrected_update(fedavg, full_lam),
                "risk_probe": risk_base,
                "risk_reliability_gamma": gamma_from_rho("risk_reliability", rho),
                "risk_sqrt_gamma": gamma_from_rho("risk_sqrt", rho),
                "rho": rho,
                "secant": secant,
                "best_worker_true": best_worker_endpoint(start_x, endpoints, lam, quartic),
            }
        )
    return rows


def candidate(row, arm, param):
    if arm == "fedavg":
        return row["fedavg"]
    if arm == "scale":
        return mul(param, row["fedavg"])
    if arm == "random_probe":
        return add(row["fedavg"], mul(param, sub(row["random_probe"], row["fedavg"])))
    if arm == "full_probe":
        return row["random_probe"]
    if arm in RISK_RULES:
        gamma = row[f"{arm}_gamma"]
        return add(row["fedavg"], mul(gamma, sub(row["risk_probe"], row["fedavg"])))
    if arm in ("secant", "best_worker_true"):
        return row[arm]
    raise ValueError(f"unknown arm: {arm}")


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


def select_param(rows, arm, lam, quartic):
    grid = SCALES if arm == "scale" else GAMMAS
    return max(grid, key=lambda p: score(rows, arm, p, lam, quartic))


def select_risk_rule(rows, lam, quartic):
    return max(RISK_RULES, key=lambda arm: score(rows, arm, None, lam, quartic))


def mean_ci(vals):
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
    return m, 1.96 * (var / len(vals)) ** 0.5


def format_ci(vals):
    m, hw = mean_ci(vals)
    return f"{m:6.3f}+/-{hw:5.3f}"


def mean_gamma(rows, rule):
    return sum(row[f"{rule}_gamma"] for row in rows) / len(rows)


def flatten(batches):
    out = []
    for batch in batches:
        out.extend(batch)
    return out


def main():
    rng = random.Random(0)
    lam = [rng.uniform(8, 12) for _ in range(e9.N_SHARP)] + [
        rng.uniform(0.1, 1.0) for _ in range(e9.D - e9.N_SHARP)
    ]
    quartic = [rng.uniform(4.0, 10.0) for _ in range(e9.N_SHARP)] + [
        rng.uniform(0.1, 1.0) for _ in range(e9.D - e9.N_SHARP)
    ]
    print(
        f"risk-shrinkage fallback: d={e9.D} K={e9.K} eta={e9.ETA} "
        f"sigma={e9.SIGMA} het={e9.HET} probe_reps={PROBE_REPS} radius={PROBE_RADIUS}"
    )
    print()
    print(
        f"{'N':>5} {'scale*':>7} {'gamma*':>7} {'risk_rule':>16} "
        f"{'risk_gP':>8} {'risk_gH':>8} {'raw':>12} {'scalar':>12} "
        f"{'probe':>12} {'full':>12} {'risk':>12} {'secant':>12} "
        f"{'best_worker':>12} {'probe-scalar':>13} {'risk-tuned':>13} "
        f"{'best-probe':>13}"
    )
    for n_workers in e9.N_WORKERS:
        pilot = []
        heldout = []
        for r in range(e9.PILOT_REPS):
            pilot.extend(augment_rounds(230_000 + 1000 * n_workers + r, n_workers, lam, quartic))
        for r in range(e9.HELDOUT_REPS):
            heldout.append(augment_rounds(231_000 + 1000 * n_workers + r, n_workers, lam, quartic))

        pilot_flat = pilot
        heldout_flat = flatten(heldout)
        best_scale = select_param(pilot_flat, "scale", lam, quartic)
        best_gamma = select_param(pilot_flat, "random_probe", lam, quartic)
        risk_rule = select_risk_rule(pilot_flat, lam, quartic)

        vals = {
            "fedavg": [score(rows, "fedavg", None, lam, quartic) for rows in heldout],
            "scale": [score(rows, "scale", best_scale, lam, quartic) for rows in heldout],
            "random_probe": [
                score(rows, "random_probe", best_gamma, lam, quartic) for rows in heldout
            ],
            "full_probe": [score(rows, "full_probe", None, lam, quartic) for rows in heldout],
            "risk": [score(rows, risk_rule, None, lam, quartic) for rows in heldout],
            "secant": [score(rows, "secant", None, lam, quartic) for rows in heldout],
            "best_worker_true": [
                score(rows, "best_worker_true", None, lam, quartic) for rows in heldout
            ],
        }
        probe_minus_scalar = [a - b for a, b in zip(vals["random_probe"], vals["scale"])]
        risk_minus_tuned = [a - b for a, b in zip(vals["risk"], vals["random_probe"])]
        best_minus_probe = [a - b for a, b in zip(vals["best_worker_true"], vals["random_probe"])]

        print(
            f"{n_workers:5d} {best_scale:7.2f} {best_gamma:7.2f} {risk_rule:>16} "
            f"{mean_gamma(pilot_flat, risk_rule):8.3f} {mean_gamma(heldout_flat, risk_rule):8.3f} "
            f"{format_ci(vals['fedavg']):>12} {format_ci(vals['scale']):>12} "
            f"{format_ci(vals['random_probe']):>12} {format_ci(vals['full_probe']):>12} "
            f"{format_ci(vals['risk']):>12} {format_ci(vals['secant']):>12} "
            f"{format_ci(vals['best_worker_true']):>12} {format_ci(probe_minus_scalar):>13} "
            f"{format_ci(risk_minus_tuned):>13} {format_ci(best_minus_probe):>13}"
        )


if __name__ == "__main__":
    main()
