"""DFW-E027: global information query policy on a CPU synthetic bed.

The test is deliberately small and diagonal.  Workers are query processes into
one shared information state.  A global controller chooses exploit/probe/
replicate actions after each worker minibatch and is compared to policies that
collapse workers into a fixed averaged trajectory or fixed schedule.

The non-oracle controllers never read the true curvature or true noise.  They
use a diagonal posterior over curvature plus empirical per-coordinate noise
estimates from current-state repeats.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


D = 160
WORKERS = 24
ROUNDS = 36
QUERY_BUDGET = WORKERS * ROUNDS
PILOT_REPS = 5
HELDOUT_REPS = 24

LAMBDA_MIN = 0.04
LAMBDA_MAX = 18.0
PROBE_RADIUS = 0.22
H_PRIOR_MEAN = 1.0
H_PRIOR_PREC = 0.25
H_MIN = 0.02
H_MAX_EST = 30.0
NOISE_PRIOR_VAR = 0.35**2
NOISE_BLEND = 0.25
TRUST_FRACTION = 0.65
COORD_TRUST_FRACTION = 0.85
EPS = 1e-12

FEDAVG_ETA_GRID = (0.015, 0.025, 0.035, 0.045, 0.06, 0.08)
PRIVATE_ETA_GRID = (0.01, 0.018, 0.028, 0.04, 0.055)
PRIVATE_BETA_GRID = (0.0, 0.5, 0.85)
ALPHA_GRID = (0.35, 0.50, 0.65, 0.80, 0.95)
POLICY_SEED_OFFSET = {
    "fedavg_fixed": 101,
    "round_robin_probe": 202,
    "global_greedy_voI": 303,
    "oracle_policy": 404,
}


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def mul(c, a):
    return [c * x for x in a]


def sqnorm(a):
    return sum(x * x for x in a)


def norm(a):
    return math.sqrt(sqnorm(a))


def clip(x, lo, hi):
    return min(hi, max(lo, x))


def mean_ci(vals):
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
    return m, 1.96 * math.sqrt(var / len(vals))


def format_ci(vals):
    m, hw = mean_ci(vals)
    return f"{m:9.4f}+/-{hw:8.4f}"


def make_bed():
    log_min = math.log(LAMBDA_MIN)
    log_max = math.log(LAMBDA_MAX)
    lam = [
        math.exp(log_min + (log_max - log_min) * j / (D - 1))
        for j in range(D)
    ]

    rng = random.Random(271828)
    perm = list(range(D))
    rng.shuffle(perm)
    lam = [lam[p] for p in perm]

    noise = []
    for j, h in enumerate(lam):
        base = 0.045 + 0.12 * (h / LAMBDA_MAX) ** 0.30
        band = 1.0
        if j % 11 in (0, 3):
            band += 2.4
        if j % 17 == 5:
            band += 1.2
        noise.append(base * band)
    return {"lambda": lam, "noise": noise}


def make_x0(seed):
    rng = random.Random(seed * 7919 + 17)
    return [rng.gauss(0.0, 0.85) for _ in range(D)]


def loss(x, bed):
    return 0.5 * sum(h * z * z for h, z in zip(bed["lambda"], x))


def true_grad(x, bed):
    return [h * z for h, z in zip(bed["lambda"], x)]


def noisy_grad(x, bed, rng):
    return [
        h * z + rng.gauss(0.0, sig)
        for h, z, sig in zip(bed["lambda"], x, bed["noise"])
    ]


def rel_rmse(est, true):
    num = sum((a - b) ** 2 for a, b in zip(est, true)) / len(est)
    den = sum(b * b for b in true) / len(true)
    return math.sqrt(num / max(den, EPS))


def trust_clip(update, x):
    total_cap = TRUST_FRACTION * max(norm(x), 1.0)
    nrm = norm(update)
    if nrm > total_cap:
        update = mul(total_cap / max(nrm, EPS), update)
    out = []
    for uj, xj in zip(update, x):
        cap = COORD_TRUST_FRACTION * max(abs(xj), 0.05)
        out.append(clip(uj, -cap, cap))
    return out


@dataclass
class RunResult:
    policy: str
    initial_loss: float
    final_loss: float
    improvement: float
    exploit: int
    probe: int
    replicate: int
    h_error: float
    noise_error: float


class SharedInfo:
    def __init__(self):
        self.h_mean = [H_PRIOR_MEAN] * D
        self.h_prec = [H_PRIOR_PREC] * D
        self.noise_var = [NOISE_PRIOR_VAR] * D
        self.probe_cursor = 0
        self.reset_round()

    def reset_round(self):
        self.grad_sum = [0.0] * D
        self.grad_sumsq = [0.0] * D
        self.grad_count = 0
        self.round_probe_count = 0
        self.round_replicate_count = 0
        self.round_exploit_count = 0

    def observe_base(self, g):
        self.grad_count += 1
        for j, gj in enumerate(g):
            self.grad_sum[j] += gj
            self.grad_sumsq[j] += gj * gj

    def grad_mean(self, x):
        if self.grad_count == 0:
            return [hj * xj for hj, xj in zip(self.h_mean, x)]
        inv = 1.0 / self.grad_count
        return [v * inv for v in self.grad_sum]

    def observe_probe(self, j, s, g_probe, x):
        if self.grad_count == 0:
            return
        base = self.grad_sum[j] / self.grad_count
        h_obs = (g_probe[j] - base) / s
        h_obs = clip(h_obs, H_MIN, H_MAX_EST)
        obs_var = (self.noise_var[j] * (1.0 + 1.0 / self.grad_count)) / max(s * s, EPS)
        obs_prec = 1.0 / max(obs_var, 1e-5)
        old_prec = self.h_prec[j]
        new_prec = old_prec + obs_prec
        self.h_mean[j] = clip(
            (self.h_mean[j] * old_prec + h_obs * obs_prec) / new_prec,
            H_MIN,
            H_MAX_EST,
        )
        self.h_prec[j] = new_prec

    def finish_round(self):
        if self.grad_count < 2:
            return
        n = self.grad_count
        for j in range(D):
            mean = self.grad_sum[j] / n
            var = max(0.0, (self.grad_sumsq[j] - n * mean * mean) / (n - 1))
            self.noise_var[j] = (
                (1.0 - NOISE_BLEND) * self.noise_var[j] + NOISE_BLEND * max(var, 1e-6)
            )

    def fixed_probe_dim(self):
        j = self.probe_cursor % D
        self.probe_cursor += 1
        return j

    def greedy_probe_dim(self, x):
        g = self.grad_mean(x)
        best = (float("-inf"), 0)
        for j in range(D):
            h = max(self.h_mean[j], H_MIN)
            h_var = 1.0 / max(self.h_prec[j], EPS)
            obs_var = (self.noise_var[j] * (1.0 + 1.0 / max(self.grad_count, 1))) / (
                PROBE_RADIUS * PROBE_RADIUS
            )
            obs_prec = 1.0 / max(obs_var, 1e-5)
            var_after = 1.0 / (self.h_prec[j] + obs_prec)
            reduction = max(0.0, h_var - var_after)
            score = (g[j] * g[j] + 0.05 * x[j] * x[j]) * reduction / (h**4 + 1e-4)
            if score > best[0]:
                best = (score, j)
        return best

    def grad_value(self):
        n = max(self.grad_count, 1)
        out = 0.0
        for h, var in zip(self.h_mean, self.noise_var):
            out += var * (1.0 / n - 1.0 / (n + 1.0)) / max(h * h, 1e-4)
        return out

    def update_from_posterior(self, x, alpha):
        g = self.grad_mean(x)
        update = [
            -alpha * gj / max(hj, H_MIN)
            for gj, hj in zip(g, self.h_mean)
        ]
        return trust_clip(update, x)

    def update_from_true_h(self, x, alpha, bed):
        g = self.grad_mean(x)
        update = [
            -alpha * gj / max(hj, H_MIN)
            for gj, hj in zip(g, bed["lambda"])
        ]
        return trust_clip(update, x)


def choose_round_robin_action(q, info):
    pattern = (
        "exploit",
        "replicate",
        "probe",
        "exploit",
        "exploit",
        "probe",
        "replicate",
        "exploit",
    )
    action = pattern[q % len(pattern)]
    if info.grad_count == 0 and action == "probe":
        action = "exploit"
    if action == "probe":
        return action, info.fixed_probe_dim()
    return action, None


def choose_global_action(info, x, round_idx):
    if info.grad_count == 0:
        return "exploit", None

    probe_score, j = info.greedy_probe_dim(x)
    grad_value = info.grad_value()
    h_rel_sd = math.sqrt(1.0 / max(info.h_prec[j], EPS)) / max(info.h_mean[j], H_MIN)

    # Early high-value probes are useful, but the controller should stop
    # spending a fixed tax once the shared curvature map is sharp enough.
    if (
        h_rel_sd > 0.11
        and probe_score > 0.11 * grad_value
        and info.round_probe_count < WORKERS // 2
    ):
        return "probe", j

    if info.grad_count < 3:
        return "replicate", None

    noisy_dims = 0
    g = info.grad_mean(x)
    for gj, nv in zip(g, info.noise_var):
        if nv / max(info.grad_count, 1) > 0.20 * (gj * gj + 1e-4):
            noisy_dims += 1
    if noisy_dims > D // 3 and info.round_replicate_count < WORKERS // 3:
        return "replicate", None

    return "exploit", None


def choose_oracle_action(info, x, bed):
    if info.grad_count == 0:
        return "exploit", None
    g = info.grad_mean(x)
    noisy = 0
    for gj, sig in zip(g, bed["noise"]):
        if sig * sig / max(info.grad_count, 1) > 0.12 * (gj * gj + 1e-4):
            noisy += 1
    if info.grad_count < 4 or (noisy > D // 4 and info.round_replicate_count < WORKERS // 2):
        return "replicate", None
    return "exploit", None


def run_shared_policy(seed, policy, bed, params):
    rng = random.Random(seed * 1000003 + POLICY_SEED_OFFSET[policy])
    x = make_x0(seed)
    initial = loss(x, bed)
    info = SharedInfo()
    counts = {"exploit": 0, "probe": 0, "replicate": 0}

    for round_idx in range(ROUNDS):
        info.reset_round()
        for q in range(WORKERS):
            if policy == "fedavg_fixed":
                action, dim = "exploit", None
            elif policy == "round_robin_probe":
                action, dim = choose_round_robin_action(q, info)
            elif policy == "global_greedy_voI":
                action, dim = choose_global_action(info, x, round_idx)
            elif policy == "oracle_policy":
                action, dim = choose_oracle_action(info, x, bed)
            else:
                raise ValueError(f"unknown shared policy: {policy}")

            counts[action] += 1
            if action == "exploit":
                info.round_exploit_count += 1
                info.observe_base(noisy_grad(x, bed, rng))
            elif action == "replicate":
                info.round_replicate_count += 1
                info.observe_base(noisy_grad(x, bed, rng))
            elif action == "probe":
                info.round_probe_count += 1
                sign = -1.0 if (round_idx + q + dim) % 2 else 1.0
                s = sign * PROBE_RADIUS
                z = list(x)
                z[dim] += s
                info.observe_probe(dim, s, noisy_grad(z, bed, rng), x)
            else:
                raise ValueError(action)

        info.finish_round()
        if policy == "fedavg_fixed":
            g = info.grad_mean(x)
            update = trust_clip(mul(-params["eta"], g), x)
        elif policy == "oracle_policy":
            update = info.update_from_true_h(x, params["alpha"], bed)
        else:
            update = info.update_from_posterior(x, params["alpha"])
        x = add(x, update)

    final = loss(x, bed)
    h_error = rel_rmse(info.h_mean, bed["lambda"])
    noise_error = rel_rmse([math.sqrt(v) for v in info.noise_var], bed["noise"])
    return RunResult(
        policy=policy,
        initial_loss=initial,
        final_loss=final,
        improvement=initial - final,
        exploit=counts["exploit"],
        probe=counts["probe"],
        replicate=counts["replicate"],
        h_error=h_error,
        noise_error=noise_error,
    )


def run_private_momentum(seed, bed, params):
    rng = random.Random(seed * 1000003 + 404)
    x0 = make_x0(seed)
    workers = [list(x0) for _ in range(WORKERS)]
    vel = [[0.0] * D for _ in range(WORKERS)]
    initial = loss(x0, bed)
    eta = params["eta"]
    beta = params["beta"]

    for q in range(QUERY_BUDGET):
        i = q % WORKERS
        g = noisy_grad(workers[i], bed, rng)
        for j in range(D):
            vel[i][j] = beta * vel[i][j] + g[j]
            workers[i][j] -= eta * vel[i][j]
            workers[i][j] = clip(workers[i][j], -8.0, 8.0)

    xbar = [sum(row[j] for row in workers) / WORKERS for j in range(D)]
    final = loss(xbar, bed)
    prior_h = [H_PRIOR_MEAN] * D
    prior_noise = [math.sqrt(NOISE_PRIOR_VAR)] * D
    return RunResult(
        policy="private_momentum",
        initial_loss=initial,
        final_loss=final,
        improvement=initial - final,
        exploit=QUERY_BUDGET,
        probe=0,
        replicate=0,
        h_error=rel_rmse(prior_h, bed["lambda"]),
        noise_error=rel_rmse(prior_noise, bed["noise"]),
    )


def run_policy(seed, policy, bed, params):
    if policy == "private_momentum":
        return run_private_momentum(seed, bed, params)
    return run_shared_policy(seed, policy, bed, params)


def mean_final_loss(policy, seeds, bed, params):
    vals = [run_policy(seed, policy, bed, params).final_loss for seed in seeds]
    return sum(vals) / len(vals)


def select_params(bed):
    pilot = [310_000 + i for i in range(PILOT_REPS)]
    selected = {}

    selected["fedavg_fixed"] = min(
        ({"eta": eta} for eta in FEDAVG_ETA_GRID),
        key=lambda p: mean_final_loss("fedavg_fixed", pilot, bed, p),
    )
    selected["private_momentum"] = min(
        ({"eta": eta, "beta": beta} for eta in PRIVATE_ETA_GRID for beta in PRIVATE_BETA_GRID),
        key=lambda p: mean_final_loss("private_momentum", pilot, bed, p),
    )
    for policy in ("round_robin_probe", "global_greedy_voI", "oracle_policy"):
        selected[policy] = min(
            ({"alpha": alpha} for alpha in ALPHA_GRID),
            key=lambda p, pol=policy: mean_final_loss(pol, pilot, bed, p),
        )
    return selected


def summarize(policy, rows, fedavg_rows):
    total = QUERY_BUDGET
    ratios = [
        row.improvement / max(base.improvement, EPS)
        for row, base in zip(rows, fedavg_rows)
    ]
    return {
        "final": [row.final_loss for row in rows],
        "ratio": ratios,
        "exploit": sum(row.exploit for row in rows) / (len(rows) * total),
        "probe": sum(row.probe for row in rows) / (len(rows) * total),
        "replicate": sum(row.replicate for row in rows) / (len(rows) * total),
        "h_error": [row.h_error for row in rows],
        "noise_error": [row.noise_error for row in rows],
    }


def decision_label(summaries):
    g = summaries["global_greedy_voI"]["final"]
    rr = summaries["round_robin_probe"]["final"]
    fed = summaries["fedavg_fixed"]["final"]
    oracle = summaries["oracle_policy"]["final"]

    # Positive means the first policy has lower final objective.
    g_beats_rr = [b - a for a, b in zip(g, rr)]
    g_beats_fed = [b - a for a, b in zip(g, fed)]
    oracle_beats_fed = [b - a for a, b in zip(oracle, fed)]

    m1, hw1 = mean_ci(g_beats_rr)
    m2, hw2 = mean_ci(g_beats_fed)
    m_oracle, hw_oracle = mean_ci(oracle_beats_fed)
    if m_oracle - hw_oracle <= 0.0:
        return "NO_HEADROOM"
    if m1 - hw1 > 0.0 and m2 - hw2 > 0.0:
        return "GLOBAL_INFO_POLICY_LIVE"
    return "FIXED_SCHEDULE_EXPLAINS"


def main():
    bed = make_bed()
    params = select_params(bed)
    policies = (
        "private_momentum",
        "fedavg_fixed",
        "round_robin_probe",
        "global_greedy_voI",
        "oracle_policy",
    )
    heldout = [320_000 + i for i in range(HELDOUT_REPS)]
    rows = {
        policy: [run_policy(seed, policy, bed, params[policy]) for seed in heldout]
        for policy in policies
    }
    summaries = {
        policy: summarize(policy, rows[policy], rows["fedavg_fixed"])
        for policy in policies
    }
    label = decision_label(summaries)

    print(
        "dfw-e027 global information query policy: "
        f"d={D} workers={WORKERS} rounds={ROUNDS} budget={QUERY_BUDGET} "
        f"pilot_reps={PILOT_REPS} heldout_reps={HELDOUT_REPS} "
        f"lambda_range=({LAMBDA_MIN},{LAMBDA_MAX}) probe_radius={PROBE_RADIUS} "
        f"h_prior=({H_PRIOR_MEAN}, prec={H_PRIOR_PREC}) noise_prior_var={NOISE_PRIOR_VAR}",
        flush=True,
    )
    print("SELECTED_PARAMS", params, flush=True)
    print()
    print(
        f"{'policy':>22} {'final_loss':>22} {'progress/fed':>22} "
        f"{'exploit':>8} {'probe':>8} {'replicate':>10} "
        f"{'h_err':>16} {'noise_err':>16}",
        flush=True,
    )
    for policy in policies:
        s = summaries[policy]
        print(
            f"{policy:>22} {format_ci(s['final']):>22} {format_ci(s['ratio']):>22} "
            f"{s['exploit']:8.3f} {s['probe']:8.3f} {s['replicate']:10.3f} "
            f"{format_ci(s['h_error']):>16} {format_ci(s['noise_error']):>16}",
            flush=True,
        )

    print()
    contrast_pairs = (
        ("global_greedy_voI", "fedavg_fixed"),
        ("global_greedy_voI", "round_robin_probe"),
        ("oracle_policy", "fedavg_fixed"),
        ("oracle_policy", "global_greedy_voI"),
        ("round_robin_probe", "fedavg_fixed"),
    )
    print(f"{'contrast':>42} {'delta_progress_ratio':>22}", flush=True)
    for a, b in contrast_pairs:
        vals = [
            x - y
            for x, y in zip(summaries[a]["ratio"], summaries[b]["ratio"])
        ]
        print(f"{a + ' - ' + b:>42} {format_ci(vals):>22}", flush=True)

    print()
    print(f"{'final-objective contrast':>42} {'loss_saved':>22}", flush=True)
    for a, b in contrast_pairs:
        vals = [
            y - x
            for x, y in zip(summaries[a]["final"], summaries[b]["final"])
        ]
        print(f"{a + ' beats ' + b:>42} {format_ci(vals):>22}", flush=True)

    print()
    print("DECISION", label, flush=True)
    print()
    print("SEED_PAIRS", flush=True)
    print(
        f"{'seed':>8} {'private':>12} {'fedavg':>12} {'round_robin':>12} "
        f"{'global':>12} {'oracle':>12}",
        flush=True,
    )
    for idx, seed in enumerate(heldout):
        vals = {
            policy: summaries[policy]["ratio"][idx]
            for policy in policies
        }
        print(
            f"{seed:8d} {vals['private_momentum']:12.4f} "
            f"{vals['fedavg_fixed']:12.4f} {vals['round_robin_probe']:12.4f} "
            f"{vals['global_greedy_voI']:12.4f} {vals['oracle_policy']:12.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
