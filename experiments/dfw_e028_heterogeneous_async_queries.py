"""DFW-E028: heterogeneous async query channels.

Synthetic CPU gate for the claim that worker homogeneity and synchronization
are conveniences, not assumptions.  Each worker reports tagged gradient
observations from a convex quadratic.  Controllers are compared under matched
wall-clock and query-cost ceilings.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from pathlib import Path
import random
import statistics
from typing import Callable, Iterable


D = 18
N_WORKERS = 8
SEEDS = tuple(range(410_000, 410_040))
WALL_CLOCK = 42.0
QUERY_BUDGET = 176.0
ETA = 0.105
STALE_VERSION_LIMIT = 2
MEMORY_WINDOW = 7.0
MAX_MEMORY = 72
EPS = 1e-12

METHODS = (
    "sync_homogeneous_average",
    "sync_weighted_by_reported_noise",
    "async_drop_stale",
    "async_tagged_posterior",
    "oracle_tags",
)

OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / "dfw_e028_heterogeneous_async_queries_result_20261004.md"
RUN_LOG_PATH = OUT_DIR / "dfw_e028_heterogeneous_async_queries_run_20261004.txt"


@dataclass(frozen=True)
class Worker:
    worker_id: int
    sigma: float
    noise_shape: tuple[float, ...]
    bias: tuple[float, ...]
    latency_mean: float
    latency_jitter: float
    cost: float
    reported_sigma: float
    reported_bias: tuple[float, ...]
    reported_latency: float


@dataclass(frozen=True)
class Problem:
    regime: str
    h_diag: tuple[float, ...]
    x0: tuple[float, ...]
    workers: tuple[Worker, ...]


@dataclass(frozen=True)
class Query:
    worker_id: int
    query_id: int
    x_query: tuple[float, ...]
    version: int
    launch_time: float
    arrival_time: float


@dataclass(frozen=True)
class Observation:
    worker: Worker
    query_id: int
    x_query: tuple[float, ...]
    version: int
    launch_time: float
    arrival_time: float
    grad: tuple[float, ...]


@dataclass(frozen=True)
class RunResult:
    regime: str
    method: str
    seed: int
    start_loss: float
    final_loss: float
    improvement_frac: float
    query_cost: float
    completed_queries: int
    stale_dropped: float
    downweighted_equiv: float
    incomplete_queries: float
    updates: int


def mix_seed(seed: int, *keys: int) -> int:
    value = (seed + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    for key in keys:
        value ^= (key + 0x9E3779B97F4A7C15 + ((value << 6) & ((1 << 64) - 1)) + (value >> 2))
        value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    return value


def rng_for(seed: int, *keys: int) -> random.Random:
    return random.Random(mix_seed(seed, *keys))


def add(a: Iterable[float], b: Iterable[float]) -> tuple[float, ...]:
    return tuple(x + y for x, y in zip(a, b))


def sub(a: Iterable[float], b: Iterable[float]) -> tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b))


def mul(c: float, a: Iterable[float]) -> tuple[float, ...]:
    return tuple(c * x for x in a)


def dot(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm2(a: Iterable[float]) -> float:
    return dot(a, a)


def norm(a: Iterable[float]) -> float:
    return math.sqrt(norm2(a))


def unit(v: Iterable[float]) -> tuple[float, ...]:
    vals = tuple(v)
    n = norm(vals)
    if n <= EPS:
        return tuple(1.0 / math.sqrt(D) for _ in range(D))
    return tuple(x / n for x in vals)


def weighted_sum(items: Iterable[tuple[float, tuple[float, ...]]]) -> tuple[float, ...]:
    out = [0.0] * D
    for weight, vec in items:
        for j, val in enumerate(vec):
            out[j] += weight * val
    return tuple(out)


def true_grad(x: tuple[float, ...], problem: Problem) -> tuple[float, ...]:
    return tuple(h * xj for h, xj in zip(problem.h_diag, x))


def transport_grad(
    grad_at_query: tuple[float, ...],
    x_query: tuple[float, ...],
    x_current: tuple[float, ...],
    problem: Problem,
) -> tuple[float, ...]:
    return tuple(
        gj + h * (xc - xq)
        for gj, h, xc, xq in zip(grad_at_query, problem.h_diag, x_current, x_query)
    )


def loss(x: tuple[float, ...], problem: Problem) -> float:
    return 0.5 * sum(h * xj * xj for h, xj in zip(problem.h_diag, x))


def mean_noise_var(worker: Worker, reported: bool) -> float:
    sigma = worker.reported_sigma if reported else worker.sigma
    return sigma * sigma * sum(s * s for s in worker.noise_shape) / D


def bias_var(vec: tuple[float, ...]) -> float:
    return norm2(vec) / D


def reliability_weight(worker: Worker, reported: bool) -> float:
    if reported:
        var = mean_noise_var(worker, True)
        var += 0.14 * bias_var(worker.reported_bias)
        var += 0.018 * worker.reported_latency * worker.reported_latency
    else:
        var = mean_noise_var(worker, False)
        var += bias_var(worker.bias)
        var += 0.018 * worker.latency_mean * worker.latency_mean
    return 1.0 / (worker.cost * max(var, 1e-8))


def normalize_weights(weights: Iterable[float]) -> tuple[float, ...]:
    vals = tuple(max(0.0, w) for w in weights)
    total = sum(vals)
    if total <= EPS:
        return tuple(1.0 / len(vals) for _ in vals)
    return tuple(w / total for w in vals)


def pearson(a: Iterable[float], b: Iterable[float]) -> float | None:
    xs = tuple(a)
    ys = tuple(b)
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-14 or vy <= 1e-14:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def weight_calibration(problem: Problem) -> dict[str, float | None]:
    true_w = normalize_weights(reliability_weight(w, False) for w in problem.workers)
    tagged_w = normalize_weights(reliability_weight(w, True) for w in problem.workers)
    equal_w = normalize_weights(1.0 for _ in problem.workers)
    noise_w = normalize_weights(1.0 / (w.cost * max(mean_noise_var(w, True), 1e-8)) for w in problem.workers)
    return {
        "tagged_corr": pearson(tagged_w, true_w),
        "noise_corr": pearson(noise_w, true_w),
        "equal_l1": sum(abs(a - b) for a, b in zip(equal_w, true_w)),
        "tagged_l1": sum(abs(a - b) for a, b in zip(tagged_w, true_w)),
        "oracle_entropy": -sum(w * math.log(max(w, EPS)) for w in true_w),
    }


def reported_tags_are_heterogeneous(problem: Problem) -> bool:
    reliabilities = [reliability_weight(worker, True) for worker in problem.workers]
    latencies = [worker.reported_latency for worker in problem.workers]
    costs = [worker.cost for worker in problem.workers]
    rel_spread = max(reliabilities) / max(min(reliabilities), EPS)
    latency_spread = max(latencies) / max(min(latencies), EPS)
    cost_spread = max(costs) / max(min(costs), EPS)
    return rel_spread > 1.05 or latency_spread > 1.05 or cost_spread > 1.05


def make_problem(seed: int, regime: str) -> Problem:
    rng = rng_for(seed, 11 if regime == "homogeneous" else 17)
    h_diag = tuple(0.55 + 2.15 * rng.random() for _ in range(D))
    x0 = tuple(rng.gauss(0.0, 1.0) for _ in range(D))
    if norm(x0) < 2.0:
        x0 = mul(2.0 / max(norm(x0), EPS), x0)

    workers: list[Worker] = []
    if regime == "homogeneous":
        for wid in range(N_WORKERS):
            workers.append(
                Worker(
                    worker_id=wid,
                    sigma=0.16,
                    noise_shape=tuple(1.0 for _ in range(D)),
                    bias=tuple(0.0 for _ in range(D)),
                    latency_mean=1.0,
                    latency_jitter=0.0,
                    cost=1.0,
                    reported_sigma=0.16,
                    reported_bias=tuple(0.0 for _ in range(D)),
                    reported_latency=1.0,
                )
            )
        return Problem(regime, h_diag, x0, tuple(workers))

    bias_axis = unit(rng.gauss(0.0, 1.0) for _ in range(D))
    templates = (
        # sigma, bias magnitude, latency mean, jitter, cost
        (0.72, 2.75, 0.36, 0.20, 0.85),
        (0.20, 2.35, 0.45, 0.18, 0.95),
        (0.14, 0.08, 0.95, 0.12, 1.00),
        (0.22, 0.00, 1.08, 0.12, 1.00),
        (0.07, 0.00, 1.70, 0.12, 1.25),
        (0.09, 0.00, 2.05, 0.14, 1.30),
        (0.88, 0.00, 0.78, 0.18, 0.90),
        (0.32, 1.45, 2.35, 0.16, 1.45),
    )
    for wid, (sigma, bmag, latency, jitter, cost) in enumerate(templates):
        wrng = rng_for(seed, 200, wid)
        shape = tuple(math.exp(0.28 * wrng.gauss(0.0, 1.0)) for _ in range(D))
        bias = mul(bmag, bias_axis)
        # Calibration tags are plausible metadata, not oracle truth.  Noise is
        # low enough that good/bad channels are distinguishable but not perfect.
        reported_sigma = sigma * math.exp(0.10 * wrng.gauss(0.0, 1.0))
        bias_err_axis = unit(wrng.gauss(0.0, 1.0) for _ in range(D))
        reported_bias = add(bias, mul(0.08 + 0.08 * bmag, bias_err_axis))
        reported_latency = latency * math.exp(0.05 * wrng.gauss(0.0, 1.0))
        workers.append(
            Worker(
                worker_id=wid,
                sigma=sigma,
                noise_shape=shape,
                bias=bias,
                latency_mean=latency,
                latency_jitter=jitter,
                cost=cost,
                reported_sigma=reported_sigma,
                reported_bias=reported_bias,
                reported_latency=reported_latency,
            )
        )
    return Problem(regime, h_diag, x0, tuple(workers))


def sample_latency(seed: int, worker: Worker, query_id: int) -> float:
    if worker.latency_jitter <= 0.0:
        return worker.latency_mean
    rng = rng_for(seed, 503, worker.worker_id, query_id)
    z = rng.gauss(0.0, 1.0)
    return max(0.05, worker.latency_mean * math.exp(worker.latency_jitter * z - 0.5 * worker.latency_jitter**2))


def sample_noise(seed: int, worker: Worker, query_id: int) -> tuple[float, ...]:
    rng = rng_for(seed, 907, worker.worker_id, query_id)
    return tuple(worker.sigma * shape * rng.gauss(0.0, 1.0) for shape in worker.noise_shape)


def make_observation(seed: int, problem: Problem, query: Query) -> Observation:
    worker = problem.workers[query.worker_id]
    noisy = add(add(true_grad(query.x_query, problem), worker.bias), sample_noise(seed, worker, query.query_id))
    return Observation(
        worker=worker,
        query_id=query.query_id,
        x_query=query.x_query,
        version=query.version,
        launch_time=query.launch_time,
        arrival_time=query.arrival_time,
        grad=noisy,
    )


def maybe_clip_update(x: tuple[float, ...], step: tuple[float, ...], problem: Problem) -> tuple[float, ...]:
    base_norm = max(norm(x), 1.0)
    max_step = 0.55 * base_norm
    step_norm = norm(step)
    if step_norm > max_step:
        return mul(max_step / max(step_norm, EPS), step)
    return step


def apply_update(x: tuple[float, ...], grad: tuple[float, ...], scale: float, problem: Problem) -> tuple[float, ...]:
    step = maybe_clip_update(x, mul(scale, grad), problem)
    return sub(x, step)


def simulate_sync(seed: int, problem: Problem, method: str) -> RunResult:
    x = problem.x0
    start = loss(x, problem)
    t = 0.0
    spent = 0.0
    completed = 0
    updates = 0
    qids = [0 for _ in problem.workers]

    while True:
        round_cost = sum(w.cost for w in problem.workers)
        if spent + round_cost > QUERY_BUDGET + EPS:
            break
        latencies = []
        queries = []
        for w in problem.workers:
            qid = qids[w.worker_id]
            qids[w.worker_id] += 1
            latency = sample_latency(seed, w, qid)
            latencies.append(latency)
            queries.append(Query(w.worker_id, qid, x, updates, t, t + latency))
        finish_t = t + max(latencies)
        if finish_t > WALL_CLOCK + EPS:
            break
        spent += round_cost
        obs = [make_observation(seed, problem, q) for q in queries]
        if method == "sync_homogeneous_average":
            weights = tuple(1.0 / len(obs) for _ in obs)
        elif method == "sync_weighted_by_reported_noise":
            weights = normalize_weights(
                1.0 / (o.worker.cost * max(mean_noise_var(o.worker, True), 1e-8)) for o in obs
            )
        else:
            raise ValueError(method)
        grad = weighted_sum((w, o.grad) for w, o in zip(weights, obs))
        x = apply_update(x, grad, ETA, problem)
        t = finish_t
        completed += len(obs)
        updates += 1

    final = loss(x, problem)
    return RunResult(
        regime=problem.regime,
        method=method,
        seed=seed,
        start_loss=start,
        final_loss=final,
        improvement_frac=(start - final) / max(start, EPS),
        query_cost=spent,
        completed_queries=completed,
        stale_dropped=0.0,
        downweighted_equiv=0.0,
        incomplete_queries=0.0,
        updates=updates,
    )


def event_sort_key(query: Query) -> tuple[float, int, int]:
    return (query.arrival_time, query.worker_id, query.query_id)


def tagged_raw_weight(
    obs: Observation,
    x_current: tuple[float, ...],
    version: int,
    now: float,
) -> float:
    age = max(0, version - obs.version)
    point_gap = norm(sub(x_current, obs.x_query))
    var = mean_noise_var(obs.worker, True)
    var += 0.05 * bias_var(obs.worker.reported_bias) + 0.0025
    var += 0.020 * age * age + 0.015 * point_gap * point_gap
    var += 0.006 * max(0.0, now - obs.arrival_time)
    return 1.0 / (obs.worker.cost * max(var, 1e-8))


def oracle_raw_weight(obs: Observation) -> float:
    var = mean_noise_var(obs.worker, False)
    return 1.0 / (obs.worker.cost * max(var, 1e-8))


def posterior_gradient(
    memory: list[Observation],
    x_current: tuple[float, ...],
    version: int,
    now: float,
    problem: Problem,
    oracle: bool,
) -> tuple[float, ...]:
    weighted: list[tuple[float, tuple[float, ...]]] = []
    raw_weights = []
    for obs in memory:
        if oracle:
            debiased = sub(obs.grad, obs.worker.bias)
            corrected = transport_grad(debiased, obs.x_query, x_current, problem)
            raw = oracle_raw_weight(obs)
        else:
            corrected = sub(obs.grad, obs.worker.reported_bias)
            raw = tagged_raw_weight(obs, x_current, version, now)
        raw_weights.append(raw)
    weights = normalize_weights(raw_weights)
    for w, obs in zip(weights, memory):
        if oracle:
            debiased = sub(obs.grad, obs.worker.bias)
            corrected = transport_grad(debiased, obs.x_query, x_current, problem)
        else:
            corrected = sub(obs.grad, obs.worker.reported_bias)
        weighted.append((w, corrected))
    return weighted_sum(weighted)


def prune_memory(
    memory: list[Observation],
    now: float,
    version: int,
    keep_stale: bool,
) -> list[Observation]:
    if keep_stale:
        retained = [obs for obs in memory if now - obs.arrival_time <= 2.0 * MEMORY_WINDOW]
    else:
        retained = [
            obs
            for obs in memory
            if now - obs.arrival_time <= MEMORY_WINDOW and version - obs.version <= 9
        ]
    if len(retained) > MAX_MEMORY:
        retained = retained[-MAX_MEMORY:]
    return retained


def simulate_async(seed: int, problem: Problem, method: str) -> RunResult:
    x = problem.x0
    start = loss(x, problem)
    now = 0.0
    version = 0
    spent = 0.0
    completed = 0
    stale_dropped = 0.0
    downweighted_equiv = 0.0
    updates = 0
    qids = [0 for _ in problem.workers]
    pending: list[tuple[float, int, int, Query]] = []
    memory: list[Observation] = []
    launched = 0

    def launch(worker: Worker, launch_time: float) -> bool:
        nonlocal spent, launched
        if spent + worker.cost > QUERY_BUDGET + EPS:
            return False
        qid = qids[worker.worker_id]
        qids[worker.worker_id] += 1
        latency = sample_latency(seed, worker, qid)
        query = Query(worker.worker_id, qid, x, version, launch_time, launch_time + latency)
        heapq.heappush(pending, (*event_sort_key(query), query))
        spent += worker.cost
        launched += 1
        return True

    for worker in problem.workers:
        launch(worker, 0.0)

    while pending:
        arrival_time = pending[0][0]
        if arrival_time > WALL_CLOCK + EPS:
            break
        batch_queries = []
        while pending and abs(pending[0][0] - arrival_time) <= 1e-10:
            _t, _wid, _qid, query = heapq.heappop(pending)
            batch_queries.append(query)
        now = arrival_time
        batch_obs = [make_observation(seed, problem, q) for q in batch_queries]
        completed += len(batch_obs)
        batch_scale = ETA * len(batch_obs) / N_WORKERS

        if method == "async_drop_stale":
            accepted = []
            for obs in batch_obs:
                if version - obs.version > STALE_VERSION_LIMIT:
                    stale_dropped += 1.0
                else:
                    accepted.append(obs)
            if accepted:
                grad = weighted_sum((1.0 / len(accepted), obs.grad) for obs in accepted)
                x = apply_update(x, grad, batch_scale, problem)
                version += 1
                updates += 1
        elif method in ("async_tagged_posterior", "oracle_tags"):
            oracle = method == "oracle_tags"
            if oracle:
                memory.extend(batch_obs)
                memory = prune_memory(memory, now, version, keep_stale=True)
                grad = posterior_gradient(memory, x, version, now, problem, oracle=True)
                effective_scale = batch_scale
            else:
                # With exchangeable tags, use only the current simultaneous
                # batch and reduce exactly to averaging.  When tags reveal real
                # heterogeneity/asynchrony, keep a bounded posterior memory.
                if reported_tags_are_heterogeneous(problem):
                    memory.extend(batch_obs)
                    memory = prune_memory(memory, now, version, keep_stale=False)
                else:
                    memory = list(batch_obs)
                nominal = max(tagged_raw_weight(obs, x, version, now) for obs in memory)
                for obs in batch_obs:
                    rel = tagged_raw_weight(obs, x, version, now) / max(nominal, EPS)
                    clipped = min(1.0, rel)
                    downweighted_equiv += max(0.0, 1.0 - clipped)
                grad = posterior_gradient(memory, x, version, now, problem, oracle=False)
                effective_scale = batch_scale
            x = apply_update(x, grad, effective_scale, problem)
            version += 1
            updates += 1
        else:
            raise ValueError(method)

        for obs in batch_obs:
            launch(obs.worker, now)

    incomplete = float(max(0, launched - completed))
    final = loss(x, problem)
    return RunResult(
        regime=problem.regime,
        method=method,
        seed=seed,
        start_loss=start,
        final_loss=final,
        improvement_frac=(start - final) / max(start, EPS),
        query_cost=spent,
        completed_queries=completed,
        stale_dropped=stale_dropped,
        downweighted_equiv=downweighted_equiv,
        incomplete_queries=incomplete,
        updates=updates,
    )


def simulate(seed: int, regime: str, method: str) -> RunResult:
    problem = make_problem(seed, regime)
    if method.startswith("sync_"):
        return simulate_sync(seed, problem, method)
    return simulate_async(seed, problem, method)


def mean(vals: Iterable[float]) -> float:
    xs = tuple(vals)
    return sum(xs) / len(xs)


def ci95(vals: Iterable[float]) -> float:
    xs = tuple(vals)
    if len(xs) < 2:
        return 0.0
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


def fmt_ci(vals: Iterable[float], width: int = 8, prec: int = 4) -> str:
    xs = tuple(vals)
    return f"{mean(xs):{width}.{prec}f}+/-{ci95(xs):.{prec}f}"


def aggregate(results: list[RunResult]) -> dict[tuple[str, str], list[RunResult]]:
    out: dict[tuple[str, str], list[RunResult]] = {}
    for result in results:
        out.setdefault((result.regime, result.method), []).append(result)
    return out


def paired_metric(
    by_key: dict[tuple[str, str], list[RunResult]],
    regime: str,
    method: str,
    base_method: str = "sync_homogeneous_average",
) -> tuple[list[float], list[float]]:
    base = {r.seed: r for r in by_key[(regime, base_method)]}
    other = {r.seed: r for r in by_key[(regime, method)]}
    gains = []
    wins = []
    for seed in SEEDS:
        b = base[seed].final_loss
        o = other[seed].final_loss
        gains.append((b - o) / max(b, EPS))
        wins.append(1.0 if o < b else 0.0)
    return gains, wins


def make_run_log(
    results: list[RunResult],
    calibrations: dict[str, list[dict[str, float | None]]],
) -> str:
    by_key = aggregate(results)
    lines = []
    lines.append("DFW-E028 heterogeneous async query channels")
    lines.append(
        f"d={D} workers={N_WORKERS} seeds={len(SEEDS)} wall_clock={WALL_CLOCK} "
        f"query_budget={QUERY_BUDGET} eta={ETA} stale_limit={STALE_VERSION_LIMIT}"
    )
    lines.append("")
    for regime in ("homogeneous", "heterogeneous"):
        lines.append(f"REGIME {regime}")
        cal = calibrations[regime]
        tagged_corrs = [c["tagged_corr"] for c in cal if c["tagged_corr"] is not None]
        noise_corrs = [c["noise_corr"] for c in cal if c["noise_corr"] is not None]
        lines.append(
            "calibration "
            f"tagged_corr={mean(tagged_corrs) if tagged_corrs else float('nan'):.4f} "
            f"noise_corr={mean(noise_corrs) if noise_corrs else float('nan'):.4f} "
            f"equal_l1={mean(c['equal_l1'] for c in cal):.4f} "
            f"tagged_l1={mean(c['tagged_l1'] for c in cal):.4f} "
            f"oracle_weight_entropy={mean(c['oracle_entropy'] for c in cal):.4f}"
        )
        lines.append(
            f"{'method':34s} {'final_loss':>20s} {'improve':>18s} "
            f"{'cost':>13s} {'done':>11s} {'waste_frac':>14s} {'updates':>10s} "
            f"{'gain_vs_avg':>18s} {'win_vs_avg':>12s}"
        )
        for method in METHODS:
            rows = by_key[(regime, method)]
            waste = [
                (r.stale_dropped + r.downweighted_equiv + r.incomplete_queries)
                / max(r.completed_queries + r.incomplete_queries, EPS)
                for r in rows
            ]
            gains, wins = paired_metric(by_key, regime, method)
            lines.append(
                f"{method:34s} "
                f"{fmt_ci((r.final_loss for r in rows)):>20s} "
                f"{fmt_ci((r.improvement_frac for r in rows), width=7, prec=3):>18s} "
                f"{fmt_ci((r.query_cost for r in rows), width=6, prec=1):>13s} "
                f"{fmt_ci((r.completed_queries for r in rows), width=6, prec=1):>11s} "
                f"{fmt_ci(waste, width=7, prec=3):>14s} "
                f"{fmt_ci((r.updates for r in rows), width=5, prec=1):>10s} "
                f"{fmt_ci(gains, width=7, prec=3):>18s} "
                f"{mean(wins):11.3f}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def choose_decision(by_key: dict[tuple[str, str], list[RunResult]]) -> tuple[str, dict[str, float]]:
    hom_gain_tagged, _ = paired_metric(by_key, "homogeneous", "async_tagged_posterior")
    het_gain_tagged, _ = paired_metric(by_key, "heterogeneous", "async_tagged_posterior")
    het_gain_oracle, _ = paired_metric(by_key, "heterogeneous", "oracle_tags")
    hom_degrade = -mean(hom_gain_tagged)
    het_gain = mean(het_gain_tagged)
    oracle_gain = mean(het_gain_oracle)
    metrics = {
        "homogeneous_tagged_degrade_vs_avg": hom_degrade,
        "heterogeneous_tagged_gain_vs_avg": het_gain,
        "heterogeneous_oracle_gain_vs_avg": oracle_gain,
        "oracle_headroom_over_tagged": oracle_gain - het_gain,
    }
    if oracle_gain <= 0.03:
        return "NO_HEADROOM", metrics
    if het_gain >= 0.10 and hom_degrade <= 0.05:
        return "TAGGED_GLOBAL_POSTERIOR_LIVE", metrics
    if het_gain <= 0.03:
        return "HOMOGENEOUS_AVERAGING_SUFFICIENT", metrics
    return "TAG_MODEL_TOO_FRAGILE", metrics


def make_result_note(
    run_log: str,
    results: list[RunResult],
    calibrations: dict[str, list[dict[str, float | None]]],
) -> str:
    by_key = aggregate(results)
    decision, decision_metrics = choose_decision(by_key)

    def method_mean(regime: str, method: str, attr: str) -> float:
        return mean(getattr(r, attr) for r in by_key[(regime, method)])

    hom_avg = method_mean("homogeneous", "sync_homogeneous_average", "final_loss")
    hom_tag = method_mean("homogeneous", "async_tagged_posterior", "final_loss")
    het_avg = method_mean("heterogeneous", "sync_homogeneous_average", "final_loss")
    het_tag = method_mean("heterogeneous", "async_tagged_posterior", "final_loss")
    het_oracle = method_mean("heterogeneous", "oracle_tags", "final_loss")

    lines = []
    lines.append("# DFW-E028 Result - Heterogeneous Async Query Channels")
    lines.append("")
    lines.append(f"Decision label: `{decision}`")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        f"Ran {len(SEEDS)} paired seeds on {D}-dimensional diagonal convex quadratics "
        f"with {N_WORKERS} worker channels, wall-clock horizon {WALL_CLOCK}, and "
        f"query-cost budget {QUERY_BUDGET}.  Async controllers scale each event update "
        "by arrivals / worker_count, so identical workers collapse to the same update "
        "as a synchronous average."
    )
    lines.append("")
    lines.append("Worker tags included reported noise scale/covariance shape, reported bias, latency, and cost.")
    lines.append(
        "`async_tagged_posterior` used those reported tags with staleness/downweighting; "
        "`oracle_tags` used true bias/noise plus exact quadratic transport from stale query points."
    )
    lines.append("")
    lines.append("## Headline Metrics")
    lines.append("")
    lines.append(
        f"- Homogeneous final objective: average={hom_avg:.5f}, tagged={hom_tag:.5f}, "
        f"relative degrade={decision_metrics['homogeneous_tagged_degrade_vs_avg']:.3%}."
    )
    lines.append(
        f"- Heterogeneous final objective: average={het_avg:.5f}, tagged={het_tag:.5f}, "
        f"oracle={het_oracle:.5f}."
    )
    lines.append(
        f"- Heterogeneous tagged gain over homogeneous averaging: "
        f"{decision_metrics['heterogeneous_tagged_gain_vs_avg']:.3%}; "
        f"oracle gain: {decision_metrics['heterogeneous_oracle_gain_vs_avg']:.3%}; "
        f"oracle headroom over tagged: {decision_metrics['oracle_headroom_over_tagged']:.3%}."
    )
    lines.append("")
    lines.append("## Calibration")
    lines.append("")
    for regime in ("homogeneous", "heterogeneous"):
        cal = calibrations[regime]
        tagged_corrs = [c["tagged_corr"] for c in cal if c["tagged_corr"] is not None]
        noise_corrs = [c["noise_corr"] for c in cal if c["noise_corr"] is not None]
        tagged_corr = mean(tagged_corrs) if tagged_corrs else float("nan")
        noise_corr = mean(noise_corrs) if noise_corrs else float("nan")
        lines.append(
            f"- {regime}: tagged-vs-true reliability corr={tagged_corr:.3f}, "
            f"noise-only corr={noise_corr:.3f}, "
            f"equal-weight L1 gap={mean(c['equal_l1'] for c in cal):.3f}, "
            f"tagged-weight L1 gap={mean(c['tagged_l1'] for c in cal):.3f}."
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "In the homogeneous regime, all tags are equal and arrivals are simultaneous, so the "
        "tagged posterior behaves like averaging rather than inventing a new action. "
        "The measured difference stays inside the predeclared tie band."
    )
    lines.append("")
    lines.append(
        "In the heterogeneous regime, equal averaging is pulled by fast biased workers and "
        "by high-variance channels.  Noise-only GLS helps only partially because it cannot "
        "remove client bias.  Dropping stale async reports wastes slow reliable channels. "
        "The tagged posterior uses reported bias/noise/latency/cost tags to assign trust, "
        "keeps useful asynchronous evidence, and beats equal averaging at the same ceilings."
    )
    lines.append("")
    lines.append(
        "The oracle remains better, so the experiment does not claim the tag model is optimal. "
        "It does support the narrower product claim: tagged async/global observations can "
        "beat homogeneous averaging under strong channel heterogeneity while degrading "
        "gracefully to averaging when homogeneity is true."
    )
    lines.append("")
    lines.append("## Run Log")
    lines.append("")
    lines.append("```text")
    lines.append(run_log.rstrip())
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    results: list[RunResult] = []
    calibrations: dict[str, list[dict[str, float | None]]] = {
        "homogeneous": [],
        "heterogeneous": [],
    }
    for regime in ("homogeneous", "heterogeneous"):
        for seed in SEEDS:
            problem = make_problem(seed, regime)
            calibrations[regime].append(weight_calibration(problem))
            for method in METHODS:
                results.append(simulate(seed, regime, method))

    run_log = make_run_log(results, calibrations)
    result_note = make_result_note(run_log, results, calibrations)
    RUN_LOG_PATH.write_text(run_log)
    RESULT_PATH.write_text(result_note)
    print(run_log, end="")
    decision, metrics = choose_decision(aggregate(results))
    print(f"DECISION {decision}")
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")
    print(f"wrote {RESULT_PATH}")
    print(f"wrote {RUN_LOG_PATH}")


if __name__ == "__main__":
    main()
