"""E025: validation-cost pricing for the E022 dense trust gate.

E022 showed that a same-round held-out mini-loss proxy can choose a small
rowwise gamma for the learned low-rank-plus-diagonal dense correction and avoid
the E021 crash. E024 weakened that claim with a delayed-gamma control. E025
prices the E022 validation grid and compares it to simple matched-compute
rivals.

All dense correction arms keep the K-step local-map operator

    [I - (I - eta Hhat)^K]^-1 Delta_K

from E021/E022.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.lgp_e021_dense_lowrank_correlated_stress as e21
import experiments.lgp_e022_dense_trust_validation as e22


N_WORKERS = (8, 32, 128)
ROUNDS = 24
PILOT_REPS = 3
HELDOUT_REPS = 4
VAL_REPS = (1, 2, 4, 8)

e21.ROUNDS = ROUNDS
e22.ROUNDS = ROUNDS


def full_budget_for(n_workers):
    return min(e21.PROBE_BUDGET_CAP, max(48, n_workers * e21.K))


def validation_forward_equiv(reps):
    component_terms = reps * (e22.LOSS_PROXY_COORD_BATCH + e22.LOSS_PROXY_LR_BATCH)
    full_forward_terms = e21.D + e21.RANK
    return len(e22.VAL_GAMMAS) * component_terms / full_forward_terms


def validation_grad_equiv(reps, backward_to_forward):
    return validation_forward_equiv(reps) / (1.0 + backward_to_forward)


def extra_fit_offsets(reps=e22.LOSS_PROXY_REPS):
    target_grad_calls = validation_grad_equiv(reps, backward_to_forward=2.0)
    return max(1, int(round(target_grad_calls / e21.PROBE_REPS)))


EXTRA_FIT_OFFSETS = extra_fit_offsets()
EXTRA_WORKERS = 1


def make_loss_proxy_batches(rng, reps):
    batches = []
    for _ in range(reps):
        coords = [rng.randrange(e21.D) for _ in range(e22.LOSS_PROXY_COORD_BATCH)]
        lowrank = [rng.randrange(e21.RANK) for _ in range(e22.LOSS_PROXY_LR_BATCH)]
        batches.append((coords, lowrank))
    return batches


def mini_loss_proxy(x, bed, batches):
    z = e21.qtv(bed["u"], x)
    total = 0.0
    for coords, lowrank in batches:
        diag = 0.0
        for j in coords:
            xj = x[j]
            diag += 0.5 * bed["base"][j] * xj * xj + 0.25 * bed["q_diag"][j] * xj**4
        diag *= e21.D / len(coords)

        lr = 0.0
        for k in lowrank:
            zk = z[k]
            lr += 0.5 * bed["spikes"][k] * zk * zk + 0.25 * bed["q_lr"][k] * zk**4
        lr *= e21.RANK / len(lowrank)
        total += diag + lr
    return total / len(batches)


def validation_objective(start_x, update, fedavg, bed, batches):
    candidate = e21.add(start_x, update)
    proxy = mini_loss_proxy(candidate, bed, batches)
    base_norm = max(e21.norm(fedavg), e21.EPS)
    ratio = e21.norm(update) / base_norm
    if ratio > e22.RADIUS_RATIO:
        proxy += e22.RADIUS_PENALTY * (ratio - e22.RADIUS_RATIO) ** 2 * max(1.0, abs(proxy))
    return proxy


def choose_proxy_gamma(start_x, fedavg, combo_full, bed, rng, reps):
    batches = make_loss_proxy_batches(rng, reps)
    proxy_by_gamma = {}
    for gamma in e22.VAL_GAMMAS:
        update = e21.interp(fedavg, combo_full, gamma)
        proxy_by_gamma[gamma] = validation_objective(start_x, update, fedavg, bed, batches)
    gamma = min(e22.VAL_GAMMAS, key=lambda g: proxy_by_gamma[g])
    return gamma, e21.interp(fedavg, combo_full, gamma)


def choose_exact_gamma(start_x, fedavg, combo_full, bed):
    gamma = min(
        e22.VAL_GAMMAS,
        key=lambda g: e21.loss(e21.add(start_x, e21.interp(fedavg, combo_full, g)), bed),
    )
    return gamma, e21.interp(fedavg, combo_full, gamma)


def estimate_combo_payload(fedavg, fit_pairs, val_pairs, bed, rng):
    diag_lam = e21.estimate_diag(fit_pairs)
    learned_q = e21.learned_subspace(fit_pairs, diag_lam, rng)
    diag_full = e21.diag_correct(fedavg, diag_lam)
    diag_rho = e21.model_residual_ratio(
        val_pairs,
        lambda v: [dj * vj for dj, vj in zip(diag_lam, v)],
    )
    h_res = e21.estimate_subspace_matrix(fit_pairs, learned_q, residual_diag=diag_lam)

    def hmv(v):
        out = [dj * vj for dj, vj in zip(diag_lam, v)]
        return e21.add(out, e21.qmul(learned_q, e21.mat_vec(h_res, e21.qtv(learned_q, v))))

    combo_rho = e21.model_residual_ratio(val_pairs, hmv)
    combo_full = e21.combined_correct(fedavg, diag_lam, learned_q, h_res)
    return {
        "diag_full": diag_full,
        "diag_gamma": e21.risk_gamma(diag_rho),
        "combo_full": combo_full,
        "combo_gamma": e21.risk_gamma(combo_rho),
    }


def augment_rounds(seed, n_workers, bed):
    out = []
    raw = e21.make_rounds(seed, n_workers, bed)
    for idx, (start_loss, start_x, endpoints, fedavg, secant) in enumerate(raw):
        rng = random.Random(seed * 1009 + idx)
        budget = full_budget_for(n_workers)

        base_pairs = e21.probe_pairs(start_x, e21.random_offsets(rng, budget), bed, rng)
        base_fit, base_val = e21.split_pairs(base_pairs)
        payload = estimate_combo_payload(fedavg, base_fit, base_val, bed, rng)

        extra_pairs = e21.probe_pairs(
            start_x,
            e21.random_offsets(rng, budget + EXTRA_FIT_OFFSETS),
            bed,
            rng,
        )
        extra_fit, extra_val = e21.split_pairs(extra_pairs)
        extra_payload = estimate_combo_payload(fedavg, extra_fit, extra_val, bed, rng)
        extra_update = e21.interp(
            fedavg, extra_payload["combo_full"], extra_payload["combo_gamma"]
        )

        val_updates = {}
        val_gammas = {}
        for reps in VAL_REPS:
            gamma, update = choose_proxy_gamma(
                start_x,
                fedavg,
                payload["combo_full"],
                bed,
                random.Random(seed * 2003 + idx * 17 + reps),
                reps,
            )
            val_gammas[reps] = gamma
            val_updates[reps] = update

        exact_gamma, exact_update = choose_exact_gamma(
            start_x, fedavg, payload["combo_full"], bed
        )
        best_worker = min(endpoints, key=lambda update: e21.loss(e21.add(start_x, update), bed))

        row = {
            "start_loss": start_loss,
            "start_x": start_x,
            "fedavg": fedavg,
            "secant": secant,
            "best_worker": best_worker,
            "diag_full": payload["diag_full"],
            "diag_gamma": payload["diag_gamma"],
            "combo_learned_full": payload["combo_full"],
            "combo_learned_gamma": payload["combo_gamma"],
            "extra_fit_budget": extra_update,
            "extra_fit_gamma": extra_payload["combo_gamma"],
            "oracle_full": e21.dynamics_correct_matvec(
                fedavg, e21.true_hessian_matvec(start_x, bed)
            ),
            "combo_exact_gamma": exact_gamma,
            "combo_exact_gamma_update": exact_update,
        }
        for reps in VAL_REPS:
            row[f"combo_val_gamma_reps_{reps}"] = val_gammas[reps]
            row[f"combo_val_gamma_reps_{reps}_update"] = val_updates[reps]
        out.append(row)
    return out


def make_scalar_rounds(seed, n_workers, bed):
    rows = []
    for start_loss, start_x, _endpoints, fedavg, _secant in e21.make_rounds(seed, n_workers, bed):
        rows.append({"start_loss": start_loss, "start_x": start_x, "fedavg": fedavg})
    return rows


def candidate(row, arm, params):
    if arm == "fedavg":
        return row["fedavg"]
    if arm == "scale":
        return e21.mul(params["scale"], row["fedavg"])
    if arm == "diag":
        return e21.interp(row["fedavg"], row["diag_full"], row["diag_gamma"])
    if arm == "combo_learned":
        return e21.interp(row["fedavg"], row["combo_learned_full"], row["combo_learned_gamma"])
    if arm == "extra_fit_budget":
        return row["extra_fit_budget"]
    if arm == "combo_exact_gamma":
        return row["combo_exact_gamma_update"]
    if arm == "oracle":
        return e21.interp(row["fedavg"], row["oracle_full"], params["oracle"])
    if arm in ("secant", "best_worker"):
        return row[arm]
    if arm.startswith("combo_val_gamma_reps_"):
        return row[f"{arm}_update"]
    raise ValueError(f"unknown arm: {arm}")


def score(rows, arm, params, bed):
    base = 0.0
    total = 0.0
    for row in rows:
        start_x = row["start_x"]
        fedavg = row["fedavg"]
        base += row["start_loss"] - e21.loss(e21.add(start_x, fedavg), bed)
        total += row["start_loss"] - e21.loss(e21.add(start_x, candidate(row, arm, params)), bed)
    return total / base if base > 0 else 1.0


def score_scalar_rows(rows, scale, bed):
    base = 0.0
    total = 0.0
    for row in rows:
        start_x = row["start_x"]
        fedavg = row["fedavg"]
        base += row["start_loss"] - e21.loss(e21.add(start_x, fedavg), bed)
        total += row["start_loss"] - e21.loss(e21.add(start_x, e21.mul(scale, fedavg)), bed)
    return total / base if base > 0 else 1.0


def select_grid(rows, arm, grid, params, bed):
    best = None
    best_score = -float("inf")
    for value in grid:
        trial = dict(params)
        trial[arm] = value
        val = score(rows, arm, trial, bed)
        if val > best_score:
            best = value
            best_score = val
    return best


def select_scalar_scale(rows, bed):
    return max(e21.SCALES, key=lambda scale: score_scalar_rows(rows, scale, bed))


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


def mean_value(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def raw_gradient_calls(n_workers):
    return n_workers * e21.K


def print_cost_ledger():
    print("COST_LEDGER")
    print(
        f"{'arm':>26} {'N':>5} {'fit_grad':>9} {'val_fwd':>9} {'val_grad_b2':>11} "
        f"{'val_grad_b1':>11} {'total_rel_b2':>12} {'total_rel_b1':>12} "
        f"{'match_note':>18}",
        flush=True,
    )
    for n_workers in N_WORKERS:
        raw = raw_gradient_calls(n_workers)
        budget = full_budget_for(n_workers)
        base_fit = budget * e21.PROBE_REPS
        extra_fit = (budget + EXTRA_FIT_OFFSETS) * e21.PROBE_REPS
        arms = [
            ("scalar", 0.0, 0.0, "base"),
            ("diag_or_E021_combo", base_fit, 0.0, "base_fit"),
            ("extra_fit_budget", extra_fit, 0.0, f"+{EXTRA_FIT_OFFSETS} offsets"),
            ("extra_worker_scalar", 0.0, 0.0, f"N+{EXTRA_WORKERS}"),
        ]
        for reps in VAL_REPS:
            arms.append(
                (
                    f"combo_val_gamma_reps_{reps}",
                    base_fit,
                    validation_forward_equiv(reps),
                    f"{reps} proxy reps",
                )
            )
        for arm, fit_grad, val_fwd, note in arms:
            val_b2 = val_fwd / 3.0
            val_b1 = val_fwd / 2.0
            if arm == "extra_worker_scalar":
                total_b2 = ((n_workers + EXTRA_WORKERS) * e21.K) / raw
                total_b1 = total_b2
            else:
                total_b2 = (raw + fit_grad + val_b2) / raw
                total_b1 = (raw + fit_grad + val_b1) / raw
            print(
                f"COST_LEDGER {arm:>26} {n_workers:5d} {fit_grad:9.3f} "
                f"{val_fwd:9.3f} {val_b2:11.3f} {val_b1:11.3f} "
                f"{total_b2:12.3f} {total_b1:12.3f} {note:>18}",
                flush=True,
            )
    full_val_b2 = validation_grad_equiv(e22.LOSS_PROXY_REPS, 2.0)
    full_val_b1 = validation_grad_equiv(e22.LOSS_PROXY_REPS, 1.0)
    extra_fit_grad = EXTRA_FIT_OFFSETS * e21.PROBE_REPS
    worker_grad = EXTRA_WORKERS * e21.K
    print(
        "MATCHING "
        f"full8_val_grad_b2={full_val_b2:.3f} extra_fit_grad={extra_fit_grad:.3f} "
        f"extra_fit_delta_b2={extra_fit_grad - full_val_b2:.3f} "
        f"extra_worker_grad={worker_grad:.3f} "
        f"extra_worker_delta_b2={worker_grad - full_val_b2:.3f}",
        flush=True,
    )
    print(
        "MATCHING "
        f"full8_val_grad_b1={full_val_b1:.3f} extra_fit_grad={extra_fit_grad:.3f} "
        f"extra_fit_delta_b1={extra_fit_grad - full_val_b1:.3f} "
        f"extra_worker_grad={worker_grad:.3f} "
        f"extra_worker_delta_b1={worker_grad - full_val_b1:.3f}",
        flush=True,
    )


def main():
    bed = e21.make_bed()
    print(
        "validation cost pricing: "
        f"d={e21.D} rank={e21.RANK} K={e21.K} eta={e21.ETA} rounds={ROUNDS} "
        f"pilot_reps={PILOT_REPS} heldout_reps={HELDOUT_REPS} "
        f"probe_reps={e21.PROBE_REPS} probe_budget_cap={e21.PROBE_BUDGET_CAP} "
        f"val_gammas={e22.VAL_GAMMAS} val_reps={VAL_REPS} "
        f"coord_batch={e22.LOSS_PROXY_COORD_BATCH} lr_batch={e22.LOSS_PROXY_LR_BATCH} "
        f"extra_fit_offsets={EXTRA_FIT_OFFSETS} extra_workers={EXTRA_WORKERS}",
        flush=True,
    )
    print_cost_ledger()
    print()
    print(
        f"{'N':>5} {'scale*':>7} {'scaleW*':>8} {'oracle*':>8} "
        f"{'g_r1':>6} {'g_r2':>6} {'g_r4':>6} {'g_r8':>6} {'g_extra':>8} "
        f"{'scalar':>12} {'extra_worker':>12} {'val_r1':>12} {'val_r2':>12} "
        f"{'val_r4':>12} {'val_r8':>12} {'extra_fit':>12} {'combo':>12} "
        f"{'diag':>12} {'exact':>12} {'oracle':>12} {'secant':>12} {'best':>12} "
        f"{'val8-scalar':>13} {'val8-extraF':>13} {'val8-worker':>13} "
        f"{'val8-combo':>13} {'val8-diag':>13} {'best-val8':>13}",
        flush=True,
    )

    for n_workers in N_WORKERS:
        pilot = []
        heldout = []
        pilot_worker = []
        heldout_worker = []
        for r in range(PILOT_REPS):
            seed = 280_000 + 1000 * n_workers + r
            pilot.extend(augment_rounds(seed, n_workers, bed))
            pilot_worker.extend(make_scalar_rounds(seed, n_workers + EXTRA_WORKERS, bed))
        for r in range(HELDOUT_REPS):
            seed = 281_000 + 1000 * n_workers + r
            heldout.append(augment_rounds(seed, n_workers, bed))
            heldout_worker.append(make_scalar_rounds(seed, n_workers + EXTRA_WORKERS, bed))

        heldout_flat = flatten(heldout)
        params = {"scale": 1.0, "oracle": 1.0}
        params["scale"] = select_grid(pilot, "scale", e21.SCALES, params, bed)
        params["oracle"] = select_grid(pilot, "oracle", e21.GAMMAS, params, bed)
        worker_scale = select_scalar_scale(pilot_worker, bed)

        arms = (
            "scale",
            "combo_val_gamma_reps_1",
            "combo_val_gamma_reps_2",
            "combo_val_gamma_reps_4",
            "combo_val_gamma_reps_8",
            "extra_fit_budget",
            "combo_learned",
            "diag",
            "combo_exact_gamma",
            "oracle",
            "secant",
            "best_worker",
        )
        vals = {arm: [score(rows, arm, params, bed) for rows in heldout] for arm in arms}
        extra_worker_vals = [
            score_scalar_rows(rows, worker_scale, bed) for rows in heldout_worker
        ]
        val8 = vals["combo_val_gamma_reps_8"]
        val8_minus_scalar = [a - b for a, b in zip(val8, vals["scale"])]
        val8_minus_extra_fit = [a - b for a, b in zip(val8, vals["extra_fit_budget"])]
        val8_minus_worker = [a - b for a, b in zip(val8, extra_worker_vals)]
        val8_minus_combo = [a - b for a, b in zip(val8, vals["combo_learned"])]
        val8_minus_diag = [a - b for a, b in zip(val8, vals["diag"])]
        best_minus_val8 = [a - b for a, b in zip(vals["best_worker"], val8)]

        print(
            f"{n_workers:5d} {params['scale']:7.2f} {worker_scale:8.2f} "
            f"{params['oracle']:8.2f} "
            f"{mean_value(heldout_flat, 'combo_val_gamma_reps_1'):6.3f} "
            f"{mean_value(heldout_flat, 'combo_val_gamma_reps_2'):6.3f} "
            f"{mean_value(heldout_flat, 'combo_val_gamma_reps_4'):6.3f} "
            f"{mean_value(heldout_flat, 'combo_val_gamma_reps_8'):6.3f} "
            f"{mean_value(heldout_flat, 'extra_fit_gamma'):8.3f} "
            f"{format_ci(vals['scale']):>12} {format_ci(extra_worker_vals):>12} "
            f"{format_ci(vals['combo_val_gamma_reps_1']):>12} "
            f"{format_ci(vals['combo_val_gamma_reps_2']):>12} "
            f"{format_ci(vals['combo_val_gamma_reps_4']):>12} "
            f"{format_ci(vals['combo_val_gamma_reps_8']):>12} "
            f"{format_ci(vals['extra_fit_budget']):>12} "
            f"{format_ci(vals['combo_learned']):>12} {format_ci(vals['diag']):>12} "
            f"{format_ci(vals['combo_exact_gamma']):>12} {format_ci(vals['oracle']):>12} "
            f"{format_ci(vals['secant']):>12} {format_ci(vals['best_worker']):>12} "
            f"{format_ci(val8_minus_scalar):>13} {format_ci(val8_minus_extra_fit):>13} "
            f"{format_ci(val8_minus_worker):>13} {format_ci(val8_minus_combo):>13} "
            f"{format_ci(val8_minus_diag):>13} {format_ci(best_minus_val8):>13}",
            flush=True,
        )


if __name__ == "__main__":
    main()
