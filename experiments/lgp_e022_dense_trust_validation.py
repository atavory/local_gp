"""E022: trust validation for dense low-rank-plus-diagonal correction.

E021 showed a split-probe residual problem on the dense low-rank correlated
noise bed: oracle dense-H correction had large headroom, but learned
low-rank-plus-diagonal correction was destructive. This gate keeps the same bed
and tests whether a stricter same-round validation layer can reject or shrink
that combined update.

All correction arms use the K-step local-map operator

    [I - (I - eta Hhat)^K]^-1 Delta_K

not a plain Newton, BB, or L-BFGS consolidation step.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.lgp_e021_dense_lowrank_correlated_stress as e21


N_WORKERS = (8, 32, 128)
ROUNDS = 24
PILOT_REPS = 3
HELDOUT_REPS = 4
VAL_GAMMAS = (0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0)
LOSS_PROXY_REPS = 8
LOSS_PROXY_COORD_BATCH = 24
LOSS_PROXY_LR_BATCH = 4
VAL_MARGIN_FRAC = 0.001
RADIUS_RATIO = 4.0
RADIUS_PENALTY = 0.02


e21.ROUNDS = ROUNDS


def make_loss_proxy_batches(rng):
    batches = []
    for _ in range(LOSS_PROXY_REPS):
        coords = [rng.randrange(e21.D) for _ in range(LOSS_PROXY_COORD_BATCH)]
        lowrank = [rng.randrange(e21.RANK) for _ in range(LOSS_PROXY_LR_BATCH)]
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
    if ratio > RADIUS_RATIO:
        proxy += RADIUS_PENALTY * (ratio - RADIUS_RATIO) ** 2 * max(1.0, abs(proxy))
    return proxy


def choose_proxy_gamma(start_x, fedavg, combo_full, bed, rng):
    batches = make_loss_proxy_batches(rng)
    proxy_by_gamma = {}
    for gamma in VAL_GAMMAS:
        update = e21.interp(fedavg, combo_full, gamma)
        proxy_by_gamma[gamma] = validation_objective(start_x, update, fedavg, bed, batches)
    gamma = min(VAL_GAMMAS, key=lambda g: proxy_by_gamma[g])
    return gamma, e21.interp(fedavg, combo_full, gamma), proxy_by_gamma


def choose_exact_gamma(start_x, fedavg, combo_full, bed):
    gamma = min(
        VAL_GAMMAS,
        key=lambda g: e21.loss(e21.add(start_x, e21.interp(fedavg, combo_full, g)), bed),
    )
    return gamma, e21.interp(fedavg, combo_full, gamma)


def should_accept_proxy(start_x, fedavg, candidate_update, bed, rng):
    batches = make_loss_proxy_batches(rng)
    fed_proxy = validation_objective(start_x, fedavg, fedavg, bed, batches)
    cand_proxy = validation_objective(start_x, candidate_update, fedavg, bed, batches)
    margin = VAL_MARGIN_FRAC * max(1.0, abs(fed_proxy))
    return cand_proxy + margin < fed_proxy, cand_proxy - fed_proxy


def augment_rounds(seed, n_workers, bed):
    rows = []
    raw = e21.make_rounds(seed, n_workers, bed)
    for idx, (start_loss, start_x, endpoints, fedavg, secant) in enumerate(raw):
        rng = random.Random(seed * 1009 + idx)
        full_budget = min(e21.PROBE_BUDGET_CAP, max(48, n_workers * e21.K))
        sub_budget = min(e21.SUBSPACE_PROBE_BUDGET, max(32, n_workers * e21.K))

        full_pairs = e21.probe_pairs(start_x, e21.random_offsets(rng, full_budget), bed, rng)
        full_fit, full_val = e21.split_pairs(full_pairs)
        diag_lam = e21.estimate_diag(full_fit)
        learned_q = e21.learned_subspace(full_fit, diag_lam, rng)

        diag_full = e21.diag_correct(fedavg, diag_lam)
        diag_rho = e21.model_residual_ratio(
            full_val,
            lambda v: [dj * vj for dj, vj in zip(diag_lam, v)],
        )

        lowrank_arms = {}
        for name, q_basis in (
            ("lr_random", bed["random_q"]),
            ("lr_true", bed["u"]),
            ("lr_learned", learned_q),
        ):
            pairs = e21.probe_pairs(
                start_x, e21.subspace_offsets(rng, q_basis, sub_budget), bed, rng
            )
            fit, val = e21.split_pairs(pairs)
            h_sub = e21.estimate_subspace_matrix(fit, q_basis)
            rho = e21.subspace_residual_ratio(val, q_basis, h_sub)
            lowrank_arms[name] = {
                "full": e21.lowrank_correct(fedavg, q_basis, h_sub),
                "gamma": e21.risk_gamma(rho),
            }

        combo = {}
        for name, q_basis in (("combo_learned", learned_q), ("combo_true", bed["u"])):
            h_res = e21.estimate_subspace_matrix(full_fit, q_basis, residual_diag=diag_lam)

            def hmv(v, q_basis=q_basis, h_res=h_res):
                out = [dj * vj for dj, vj in zip(diag_lam, v)]
                return e21.add(out, e21.qmul(q_basis, e21.mat_vec(h_res, e21.qtv(q_basis, v))))

            rho = e21.model_residual_ratio(full_val, hmv)
            combo[name] = {
                "full": e21.combined_correct(fedavg, diag_lam, q_basis, h_res),
                "gamma": e21.risk_gamma(rho),
            }

        combo_risk = e21.interp(
            fedavg, combo["combo_learned"]["full"], combo["combo_learned"]["gamma"]
        )
        val_gamma, val_update, proxy_by_gamma = choose_proxy_gamma(
            start_x,
            fedavg,
            combo["combo_learned"]["full"],
            bed,
            random.Random(seed * 2003 + idx),
        )
        reject_accept, reject_proxy_delta = should_accept_proxy(
            start_x,
            fedavg,
            combo_risk,
            bed,
            random.Random(seed * 3001 + idx),
        )
        exact_gamma, exact_update = choose_exact_gamma(
            start_x, fedavg, combo["combo_learned"]["full"], bed
        )

        best_worker = min(endpoints, key=lambda update: e21.loss(e21.add(start_x, update), bed))
        row = {
            "start_loss": start_loss,
            "start_x": start_x,
            "fedavg": fedavg,
            "secant": secant,
            "best_worker": best_worker,
            "diag_full": diag_full,
            "diag_gamma": e21.risk_gamma(diag_rho),
            "oracle_full": e21.dynamics_correct_matvec(
                fedavg, e21.true_hessian_matvec(start_x, bed)
            ),
            "combo_val_gamma": val_gamma,
            "combo_val_gamma_update": val_update,
            "combo_val_reject_accept": reject_accept,
            "combo_val_reject_proxy_delta": reject_proxy_delta,
            "combo_exact_gamma": exact_gamma,
            "combo_exact_gamma_update": exact_update,
            "combo_proxy_best_delta": proxy_by_gamma[val_gamma] - proxy_by_gamma[0.0],
        }
        for name, payload in lowrank_arms.items():
            row[f"{name}_full"] = payload["full"]
            row[f"{name}_gamma"] = payload["gamma"]
        for name, payload in combo.items():
            row[f"{name}_full"] = payload["full"]
            row[f"{name}_gamma"] = payload["gamma"]
        rows.append(row)
    return rows


def candidate(row, arm, params):
    if arm == "fedavg":
        return row["fedavg"]
    if arm == "scale":
        return e21.mul(params["scale"], row["fedavg"])
    if arm in ("secant", "best_worker"):
        return row[arm]
    if arm == "oracle":
        return e21.interp(row["fedavg"], row["oracle_full"], params["oracle"])
    if arm == "combo_tuned":
        return e21.interp(row["fedavg"], row["combo_learned_full"], params["combo_tuned"])
    if arm == "combo_val_gamma":
        return row["combo_val_gamma_update"]
    if arm == "combo_val_reject":
        if row["combo_val_reject_accept"]:
            return e21.interp(
                row["fedavg"], row["combo_learned_full"], row["combo_learned_gamma"]
            )
        return e21.mul(params["scale"], row["fedavg"])
    if arm == "combo_exact_gamma":
        return row["combo_exact_gamma_update"]
    if arm in (
        "diag",
        "lr_random",
        "lr_true",
        "lr_learned",
        "combo_learned",
        "combo_true",
    ):
        return e21.interp(row["fedavg"], row[f"{arm}_full"], row[f"{arm}_gamma"])
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


def mean_value(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def accept_rate(rows):
    return sum(1.0 for row in rows if row["combo_val_reject_accept"]) / len(rows)


def main():
    bed = e21.make_bed()
    print(
        "dense trust validation: "
        f"d={e21.D} rank={e21.RANK} sketch_dim={e21.SKETCH_DIM} K={e21.K} eta={e21.ETA} "
        f"sigma_diag={e21.SIGMA_DIAG} sigma_lr={e21.SIGMA_LR} "
        f"het_diag={e21.HET_DIAG} het_lr={e21.HET_LR} rounds={ROUNDS} "
        f"pilot_reps={PILOT_REPS} heldout_reps={HELDOUT_REPS} "
        f"probe_reps={e21.PROBE_REPS} probe_budget_cap={e21.PROBE_BUDGET_CAP} "
        f"subspace_probe_budget={e21.SUBSPACE_PROBE_BUDGET} "
        f"val_gammas={VAL_GAMMAS} loss_proxy_reps={LOSS_PROXY_REPS} "
        f"coord_batch={LOSS_PROXY_COORD_BATCH} lr_batch={LOSS_PROXY_LR_BATCH} "
        f"val_margin_frac={VAL_MARGIN_FRAC} radius_ratio={RADIUS_RATIO}",
        flush=True,
    )
    print()
    print(
        f"{'N':>5} {'scale*':>7} {'oracle*':>8} {'combo*':>8} "
        f"{'g_diag':>7} {'g_lrL':>7} {'g_combo':>8} {'g_val':>7} "
        f"{'g_exact':>8} {'rej%':>7} {'raw':>12} {'scalar':>12} "
        f"{'diag':>12} {'lr_learn':>12} {'combo':>12} {'val_gamma':>12} "
        f"{'val_reject':>12} {'combo_true':>12} {'combo_tuned':>12} "
        f"{'exact_gate':>12} {'oracle':>12} {'secant':>12} {'best':>12} "
        f"{'valG-scalar':>13} {'valG-combo':>13} {'valG-diag':>13} "
        f"{'best-valG':>13} {'rej-scalar':>13} {'rej-combo':>13} "
        f"{'rej-diag':>13} {'best-rej':>13}",
        flush=True,
    )

    for n_workers in N_WORKERS:
        pilot = []
        heldout = []
        for r in range(PILOT_REPS):
            pilot.extend(augment_rounds(270_000 + 1000 * n_workers + r, n_workers, bed))
        for r in range(HELDOUT_REPS):
            heldout.append(augment_rounds(271_000 + 1000 * n_workers + r, n_workers, bed))

        pilot_flat = pilot
        heldout_flat = flatten(heldout)
        params = {"scale": 1.0, "oracle": 1.0, "combo_tuned": 0.0}
        params["scale"] = select_grid(pilot_flat, "scale", e21.SCALES, params, bed)
        params["oracle"] = select_grid(pilot_flat, "oracle", e21.GAMMAS, params, bed)
        params["combo_tuned"] = select_grid(
            pilot_flat, "combo_tuned", e21.GAMMAS, params, bed
        )

        arms = (
            "fedavg",
            "scale",
            "diag",
            "lr_learned",
            "combo_learned",
            "combo_val_gamma",
            "combo_val_reject",
            "combo_true",
            "combo_tuned",
            "combo_exact_gamma",
            "oracle",
            "secant",
            "best_worker",
        )
        vals = {arm: [score(rows, arm, params, bed) for rows in heldout] for arm in arms}

        val_gamma_minus_scalar = [
            a - b for a, b in zip(vals["combo_val_gamma"], vals["scale"])
        ]
        val_gamma_minus_combo = [
            a - b for a, b in zip(vals["combo_val_gamma"], vals["combo_learned"])
        ]
        val_gamma_minus_diag = [
            a - b for a, b in zip(vals["combo_val_gamma"], vals["diag"])
        ]
        best_minus_val_gamma = [
            a - b for a, b in zip(vals["best_worker"], vals["combo_val_gamma"])
        ]
        reject_minus_scalar = [
            a - b for a, b in zip(vals["combo_val_reject"], vals["scale"])
        ]
        reject_minus_combo = [
            a - b for a, b in zip(vals["combo_val_reject"], vals["combo_learned"])
        ]
        reject_minus_diag = [
            a - b for a, b in zip(vals["combo_val_reject"], vals["diag"])
        ]
        best_minus_reject = [
            a - b for a, b in zip(vals["best_worker"], vals["combo_val_reject"])
        ]

        print(
            f"{n_workers:5d} {params['scale']:7.2f} {params['oracle']:8.2f} "
            f"{params['combo_tuned']:8.2f} {mean_gamma(heldout_flat, 'diag'):7.3f} "
            f"{mean_gamma(heldout_flat, 'lr_learned'):7.3f} "
            f"{mean_gamma(heldout_flat, 'combo_learned'):8.3f} "
            f"{mean_value(heldout_flat, 'combo_val_gamma'):7.3f} "
            f"{mean_value(heldout_flat, 'combo_exact_gamma'):8.3f} "
            f"{accept_rate(heldout_flat):7.3f} {format_ci(vals['fedavg']):>12} "
            f"{format_ci(vals['scale']):>12} {format_ci(vals['diag']):>12} "
            f"{format_ci(vals['lr_learned']):>12} {format_ci(vals['combo_learned']):>12} "
            f"{format_ci(vals['combo_val_gamma']):>12} "
            f"{format_ci(vals['combo_val_reject']):>12} "
            f"{format_ci(vals['combo_true']):>12} {format_ci(vals['combo_tuned']):>12} "
            f"{format_ci(vals['combo_exact_gamma']):>12} {format_ci(vals['oracle']):>12} "
            f"{format_ci(vals['secant']):>12} {format_ci(vals['best_worker']):>12} "
            f"{format_ci(val_gamma_minus_scalar):>13} "
            f"{format_ci(val_gamma_minus_combo):>13} "
            f"{format_ci(val_gamma_minus_diag):>13} {format_ci(best_minus_val_gamma):>13} "
            f"{format_ci(reject_minus_scalar):>13} {format_ci(reject_minus_combo):>13} "
            f"{format_ci(reject_minus_diag):>13} {format_ci(best_minus_reject):>13}",
            flush=True,
        )


if __name__ == "__main__":
    main()
