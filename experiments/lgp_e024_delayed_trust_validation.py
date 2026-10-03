"""E024: delayed-gamma control for E022 dense trust validation.

E022 selected a rowwise `combo_val_gamma` from a same-round held-out mini-loss
proxy. Wanger's E022 verdict raised a winner's-curse concern: per-row proxy
selection can create upward bias that does not average away over rows. This
gate repeats the E022 dense bed and applies the gamma selected on round t to the
combined correction on round t+1.

If the validation signal is a real local trust signal rather than same-round
selection noise, the delayed arm should retain a positive edge over tuned
scalar on the E021 dense low-rank correlated-noise bed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.lgp_e021_dense_lowrank_correlated_stress as e21
import experiments.lgp_e022_dense_trust_validation as e22


N_WORKERS = e22.N_WORKERS
PILOT_REPS = e22.PILOT_REPS
HELDOUT_REPS = e22.HELDOUT_REPS
INITIAL_DELAYED_GAMMA = 0.0
FORWARD_TO_BACKWARD_RATIO = 0.5


def attach_delayed_gamma(rows):
    """Apply the previous round's proxy-selected gamma to the current row."""
    prev_gamma = INITIAL_DELAYED_GAMMA
    out = []
    for row in rows:
        delayed = dict(row)
        delayed["combo_val_gamma_delayed"] = prev_gamma
        delayed["combo_val_gamma_delayed_update"] = e21.interp(
            row["fedavg"],
            row["combo_learned_full"],
            prev_gamma,
        )
        out.append(delayed)
        prev_gamma = row["combo_val_gamma"]
    return out


def augment_rounds(seed, n_workers, bed):
    return attach_delayed_gamma(e22.augment_rounds(seed, n_workers, bed))


def candidate(row, arm, params):
    if arm == "combo_val_gamma_delayed":
        return row["combo_val_gamma_delayed_update"]
    return e22.candidate(row, arm, params)


def score(rows, arm, params, bed, skip_first=False):
    base = 0.0
    total = 0.0
    selected = rows[1:] if skip_first else rows
    for row in selected:
        start_x = row["start_x"]
        fedavg = row["fedavg"]
        base += row["start_loss"] - e21.loss(e21.add(start_x, fedavg), bed)
        total += row["start_loss"] - e21.loss(e21.add(start_x, candidate(row, arm, params)), bed)
    return total / base if base > 0 else 1.0


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


def mean_abs_gamma_delta(rows):
    return (
        sum(abs(row["combo_val_gamma"] - row["combo_val_gamma_delayed"]) for row in rows)
        / len(rows)
    )


def validation_cost_report():
    per_candidate_terms = e22.LOSS_PROXY_REPS * (
        e22.LOSS_PROXY_COORD_BATCH + e22.LOSS_PROXY_LR_BATCH
    )
    full_forward_terms = e21.D + e21.RANK
    candidate_forward_eq = per_candidate_terms / full_forward_terms
    grid_forward_eq = len(e22.VAL_GAMMAS) * candidate_forward_eq
    # Assume one backward costs two forwards, so one full gradient call is
    # roughly forward + backward = three forward equivalents.
    gradient_call_forward_eq = 1.0 + 1.0 / FORWARD_TO_BACKWARD_RATIO
    grid_grad_eq = grid_forward_eq / gradient_call_forward_eq
    return candidate_forward_eq, grid_forward_eq, grid_grad_eq


def main():
    bed = e21.make_bed()
    candidate_forward_eq, grid_forward_eq, grid_grad_eq = validation_cost_report()
    print(
        "delayed dense trust validation: "
        f"d={e21.D} rank={e21.RANK} K={e21.K} eta={e21.ETA} "
        f"rounds={e22.ROUNDS} pilot_reps={PILOT_REPS} heldout_reps={HELDOUT_REPS} "
        f"val_gammas={e22.VAL_GAMMAS} loss_proxy_reps={e22.LOSS_PROXY_REPS} "
        f"coord_batch={e22.LOSS_PROXY_COORD_BATCH} lr_batch={e22.LOSS_PROXY_LR_BATCH}",
        flush=True,
    )
    print(
        "validation cost: "
        f"per gamma={candidate_forward_eq:.3f} full-forward equivalents, "
        f"full grid={grid_forward_eq:.3f} full-forward equivalents, "
        f"full grid={grid_grad_eq:.3f} full-gradient-call equivalents "
        f"(assuming backward costs {1.0 / FORWARD_TO_BACKWARD_RATIO:.1f} forwards)",
        flush=True,
    )
    print()
    print(
        f"{'N':>5} {'scale*':>7} {'combo*':>8} {'g_val':>7} {'g_delay':>8} "
        f"{'|dg|':>7} {'scalar':>12} {'same_val':>12} {'delayed':>12} "
        f"{'delayed_d1':>12} {'exact_gate':>12} {'oracle':>12} {'combo':>12} "
        f"{'diag':>12} {'best':>12} {'delay-scalar':>14} {'d1-scalar':>13} "
        f"{'delay-same':>13} "
        f"{'delay-combo':>13} {'delay-diag':>12} {'best-delay':>12}",
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
        params["scale"] = e22.select_grid(pilot_flat, "scale", e21.SCALES, params, bed)
        params["oracle"] = e22.select_grid(pilot_flat, "oracle", e21.GAMMAS, params, bed)
        params["combo_tuned"] = e22.select_grid(
            pilot_flat,
            "combo_tuned",
            e21.GAMMAS,
            params,
            bed,
        )

        arms = (
            "scale",
            "combo_val_gamma",
            "combo_val_gamma_delayed",
            "combo_exact_gamma",
            "oracle",
            "combo_learned",
            "diag",
            "best_worker",
        )
        vals = {arm: [score(rows, arm, params, bed) for rows in heldout] for arm in arms}
        delayed_drop1 = [
            score(rows, "combo_val_gamma_delayed", params, bed, skip_first=True)
            for rows in heldout
        ]
        scalar_drop1 = [
            score(rows, "scale", params, bed, skip_first=True) for rows in heldout
        ]

        delay_minus_scalar = [
            a - b for a, b in zip(vals["combo_val_gamma_delayed"], vals["scale"])
        ]
        drop1_minus_scalar = [a - b for a, b in zip(delayed_drop1, scalar_drop1)]
        delay_minus_same = [
            a - b
            for a, b in zip(vals["combo_val_gamma_delayed"], vals["combo_val_gamma"])
        ]
        delay_minus_combo = [
            a - b
            for a, b in zip(vals["combo_val_gamma_delayed"], vals["combo_learned"])
        ]
        delay_minus_diag = [
            a - b for a, b in zip(vals["combo_val_gamma_delayed"], vals["diag"])
        ]
        best_minus_delay = [
            a - b for a, b in zip(vals["best_worker"], vals["combo_val_gamma_delayed"])
        ]

        print(
            f"{n_workers:5d} {params['scale']:7.2f} {params['combo_tuned']:8.2f} "
            f"{mean_value(heldout_flat, 'combo_val_gamma'):7.3f} "
            f"{mean_value(heldout_flat, 'combo_val_gamma_delayed'):8.3f} "
            f"{mean_abs_gamma_delta(heldout_flat):7.3f} "
            f"{format_ci(vals['scale']):>12} "
            f"{format_ci(vals['combo_val_gamma']):>12} "
            f"{format_ci(vals['combo_val_gamma_delayed']):>12} "
            f"{format_ci(delayed_drop1):>12} "
            f"{format_ci(vals['combo_exact_gamma']):>12} "
            f"{format_ci(vals['oracle']):>12} "
            f"{format_ci(vals['combo_learned']):>12} "
            f"{format_ci(vals['diag']):>12} "
            f"{format_ci(vals['best_worker']):>12} "
            f"{format_ci(delay_minus_scalar):>14} "
            f"{format_ci(drop1_minus_scalar):>13} "
            f"{format_ci(delay_minus_same):>13} "
            f"{format_ci(delay_minus_combo):>13} "
            f"{format_ci(delay_minus_diag):>12} "
            f"{format_ci(best_minus_delay):>12}",
            flush=True,
        )


if __name__ == "__main__":
    main()
