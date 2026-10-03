"""E023: random-probe budget confirmation.

E020 found that simple active designs did not robustly reduce the probe budget,
but random probes stayed strong. This gate drops active policies, spends the
CPU budget on more held-out replicates, and checks whether probe repeats change
the cost story.

All probe arms use the K-step local-map correction

    [I - (I - eta Hhat)^K]^{-1} Delta_K

with residual shrinkage selected between the two E017 rules on pilot rounds.
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
HELDOUT_REPS = 8
PROBE_REP_OPTIONS = (1, 4)
RISK_RULES = ("risk_reliability", "risk_sqrt")
SCALES = [i / 20 for i in range(0, 81)]
GAMMAS = [i / 20 for i in range(0, 21)]
EPS = 1e-12
PRIOR_RIDGE = 1e-3
PRIOR_BLEND = 0.35
NO_CORRECTION_LAMBDA = 1.0 / e16.ETA


# Use the requested local round count even though the E016 helper has its own
# module-level default.
e16.ROUNDS = ROUNDS


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


def random_offsets(rng, count):
    scale = e16.PROBE_RADIUS / math.sqrt(e16.D)
    rows = [
        [scale * rng.choice((-1.0, 1.0)) for _ in range(e16.D)]
        for _ in range(count)
    ]
    mean = e16.mean_rows(rows)
    return [e16.sub(row, mean) for row in rows]


def avg_noisy_global_grad(x, blocks, vecs, quartics, rng, probe_reps):
    acc = [0.0] * e16.D
    for _ in range(probe_reps):
        g = e16.noisy_global_grad(x, blocks, vecs, quartics, rng)
        for j, val in enumerate(g):
            acc[j] += val
    return [val / probe_reps for val in acc]


def collect_samples(start_x, offsets, blocks, vecs, quartics, rng, probe_reps):
    samples = []
    for off in offsets:
        x = e16.add(start_x, off)
        g = avg_noisy_global_grad(x, blocks, vecs, quartics, rng, probe_reps)
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


def probe_arm(start_x, fedavg, budget, probe_reps, prior_blocks, blocks, vecs, quartics, rng):
    samples = collect_samples(
        start_x,
        random_offsets(rng, budget),
        blocks,
        vecs,
        quartics,
        rng,
        probe_reps,
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
        "corrected": corrected,
        "prior_update": blend_blocks(prior_blocks, h_all),
    }


def budgets_for(n_workers):
    budgets = {n_workers, 2 * n_workers, 4 * n_workers, n_workers * e16.K}
    if n_workers == 8:
        budgets.add(max(4, n_workers // 2))
    return sorted(budgets)


def stable_seed(seed, round_idx, n_workers, budget, probe_reps):
    return seed * 1009 + round_idx * 9173 + n_workers * 97 + budget * 13 + probe_reps * 313


def augment_rounds(seed, n_workers, blocks, vecs, quartics, budgets):
    raw = e16.make_rounds(seed, n_workers, blocks, vecs, quartics)
    priors = {(budget, reps): None for budget in budgets for reps in PROBE_REP_OPTIONS}
    rows = []
    for idx, (start_loss, start_x, _steps, _grads, endpoints, fedavg, secant) in enumerate(raw):
        best_worker = min(
            endpoints,
            key=lambda update: e16.loss(e16.add(start_x, update), blocks, vecs, quartics),
        )
        probes = {}
        for budget in budgets:
            for probe_reps in PROBE_REP_OPTIONS:
                key = (budget, probe_reps)
                rng = random.Random(stable_seed(seed, idx, n_workers, budget, probe_reps))
                arm = probe_arm(
                    start_x,
                    fedavg,
                    budget,
                    probe_reps,
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


def candidate(row, arm, param=None, budget=None, probe_reps=None, rule=None):
    if arm == "fedavg":
        return row["fedavg"]
    if arm == "scale":
        return e16.mul(param, row["fedavg"])
    if arm in ("secant", "best_worker_true", "oracle"):
        return row[arm]
    if arm == "probe":
        return row["probes"][(budget, probe_reps)]["updates"][rule]
    if arm == "probe_tuned":
        full = row["probes"][(budget, probe_reps)]["corrected"]
        return e16.add(row["fedavg"], e16.mul(param, e16.sub(full, row["fedavg"])))
    raise ValueError(f"unknown arm: {arm}")


def score(rows, blocks, vecs, quartics, arm, param=None, budget=None, probe_reps=None, rule=None):
    base = 0.0
    total = 0.0
    for row in rows:
        start_x = row["start_x"]
        fedavg = row["fedavg"]
        base += row["start_loss"] - e16.loss(e16.add(start_x, fedavg), blocks, vecs, quartics)
        update = candidate(row, arm, param, budget, probe_reps, rule)
        total += row["start_loss"] - e16.loss(e16.add(start_x, update), blocks, vecs, quartics)
    return total / base if base > 0 else 1.0


def select_scale(rows, blocks, vecs, quartics):
    return max(SCALES, key=lambda s: score(rows, blocks, vecs, quartics, "scale", param=s))


def select_rule(rows, blocks, vecs, quartics, budget, probe_reps):
    return max(
        RISK_RULES,
        key=lambda rule: score(
            rows,
            blocks,
            vecs,
            quartics,
            "probe",
            budget=budget,
            probe_reps=probe_reps,
            rule=rule,
        ),
    )


def select_tuned_gamma(rows, blocks, vecs, quartics, budget, probe_reps):
    return max(
        GAMMAS,
        key=lambda gamma: score(
            rows,
            blocks,
            vecs,
            quartics,
            "probe_tuned",
            param=gamma,
            budget=budget,
            probe_reps=probe_reps,
        ),
    )


def mean_ci(vals):
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
    return m, 1.96 * math.sqrt(var / len(vals))


def format_ci(vals):
    m, hw = mean_ci(vals)
    return f"{m:6.3f}+/-{hw:5.3f}"


def mean_probe_gamma(rows, budget, probe_reps, rule):
    vals = [row["probes"][(budget, probe_reps)]["gammas"][rule] for row in rows]
    return sum(vals) / len(vals)


def main():
    blocks, _eigs, vecs, quartics = e16.make_blocks()
    print(
        "random-probe budget confirm: "
        f"d={e16.D} blocks={e16.D // 2} K={e16.K} eta={e16.ETA} "
        f"sigma={e16.SIGMA} het={e16.HET} rounds={ROUNDS} "
        f"pilot_reps={PILOT_REPS} heldout_reps={HELDOUT_REPS} "
        f"probe_rep_options={PROBE_REP_OPTIONS} radius={e16.PROBE_RADIUS} "
        f"prior_ridge={PRIOR_RIDGE} prior_blend={PRIOR_BLEND}",
        flush=True,
    )
    print()
    print(
        "CONTROL "
        f"{'N':>5} {'scale*':>7} {'scalar':>12} {'full_r1':>12} "
        f"{'full_r4':>12} {'oracle':>12} {'secant':>12} {'best_worker':>12} "
        f"{'full4-scalar':>13} {'best-full4':>13}",
        flush=True,
    )
    print(
        "BUDGET  "
        f"{'N':>5} {'budget':>7} {'reps':>4} {'rule':>16} {'gammaP':>8} "
        f"{'gammaH':>8} {'ratio':>12} {'budget-scalar':>13} "
        f"{'budget-full':>13} {'tuned*':>7} {'tuned':>12} {'risk-tuned':>13}",
        flush=True,
    )
    print(
        "REPS    "
        f"{'N':>5} {'budget':>7} {'rep1-rep4':>13}",
        flush=True,
    )

    for n_workers in N_WORKERS:
        budgets = budgets_for(n_workers)
        pilot = []
        heldout = []
        for r in range(PILOT_REPS):
            pilot.extend(
                augment_rounds(
                    270_000 + 1000 * n_workers + r,
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
                    271_000 + 1000 * n_workers + r,
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
            (budget, reps): select_rule(pilot, blocks, vecs, quartics, budget, reps)
            for budget in budgets
            for reps in PROBE_REP_OPTIONS
        }
        tuned_gammas = {
            (budget, reps): select_tuned_gamma(pilot, blocks, vecs, quartics, budget, reps)
            for budget in budgets
            for reps in PROBE_REP_OPTIONS
        }
        full_budget = n_workers * e16.K
        full_by_reps = {}
        for reps in PROBE_REP_OPTIONS:
            full_rule = rules[(full_budget, reps)]
            full_by_reps[reps] = [
                score(
                    rows,
                    blocks,
                    vecs,
                    quartics,
                    "probe",
                    budget=full_budget,
                    probe_reps=reps,
                    rule=full_rule,
                )
                for rows in heldout
            ]

        control_vals = {
            "scale": [
                score(rows, blocks, vecs, quartics, "scale", param=best_scale)
                for rows in heldout
            ],
            "oracle": [score(rows, blocks, vecs, quartics, "oracle") for rows in heldout],
            "secant": [score(rows, blocks, vecs, quartics, "secant") for rows in heldout],
            "best_worker_true": [
                score(rows, blocks, vecs, quartics, "best_worker_true") for rows in heldout
            ],
        }
        full4_minus_scalar = [a - b for a, b in zip(full_by_reps[4], control_vals["scale"])]
        best_minus_full4 = [
            a - b for a, b in zip(control_vals["best_worker_true"], full_by_reps[4])
        ]
        print(
            "CONTROL "
            f"{n_workers:5d} {best_scale:7.2f} {format_ci(control_vals['scale']):>12} "
            f"{format_ci(full_by_reps[1]):>12} {format_ci(full_by_reps[4]):>12} "
            f"{format_ci(control_vals['oracle']):>12} {format_ci(control_vals['secant']):>12} "
            f"{format_ci(control_vals['best_worker_true']):>12} "
            f"{format_ci(full4_minus_scalar):>13} {format_ci(best_minus_full4):>13}",
            flush=True,
        )

        values = {}
        for budget in budgets:
            for reps in PROBE_REP_OPTIONS:
                rule = rules[(budget, reps)]
                vals = [
                    score(
                        rows,
                        blocks,
                        vecs,
                        quartics,
                        "probe",
                        budget=budget,
                        probe_reps=reps,
                        rule=rule,
                    )
                    for rows in heldout
                ]
                tuned_gamma = tuned_gammas[(budget, reps)]
                tuned_vals = [
                    score(
                        rows,
                        blocks,
                        vecs,
                        quartics,
                        "probe_tuned",
                        param=tuned_gamma,
                        budget=budget,
                        probe_reps=reps,
                    )
                    for rows in heldout
                ]
                values[(budget, reps)] = vals
                budget_minus_scalar = [a - b for a, b in zip(vals, control_vals["scale"])]
                budget_minus_full = [a - b for a, b in zip(vals, full_by_reps[reps])]
                risk_minus_tuned = [a - b for a, b in zip(vals, tuned_vals)]
                print(
                    "BUDGET  "
                    f"{n_workers:5d} {budget:7d} {reps:4d} {rule:>16} "
                    f"{mean_probe_gamma(pilot, budget, reps, rule):8.3f} "
                    f"{mean_probe_gamma(heldout_flat, budget, reps, rule):8.3f} "
                    f"{format_ci(vals):>12} {format_ci(budget_minus_scalar):>13} "
                    f"{format_ci(budget_minus_full):>13} {tuned_gamma:7.2f} "
                    f"{format_ci(tuned_vals):>12} {format_ci(risk_minus_tuned):>13}",
                    flush=True,
                )

        for budget in budgets:
            rep_diff = [a - b for a, b in zip(values[(budget, 1)], values[(budget, 4)])]
            print(
                "REPS    "
                f"{n_workers:5d} {budget:7d} {format_ci(rep_diff):>13}",
                flush=True,
            )


if __name__ == "__main__":
    main()
