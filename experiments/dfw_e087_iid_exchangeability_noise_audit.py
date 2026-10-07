"""DFW-E087: IID exchangeability / noise-selection audit.

The spread=1 rows in E060/E063 collapse to a single LR group, so they prove
that the grouped-HPO rule does not invent a group when no group exists.  They
do not test the subtler null: if we assign arbitrary labels to truly identical
workers, probe weights can still move because worker updates and probe batches
are noisy.

This audit creates artificial worker groups while all workers use the same
inner optimizer.  Any label advantage is therefore stochastic-update filtering,
not HPO.  The measured questions are:

  * do probe weights move away from the uniform prior?
  * does that movement beat uniform FedAvg on held-out target batches?
  * does it beat shuffled probe labels?
  * do arbitrary arm labels show persistent identity advantage?
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import platform
import statistics
import sys
from typing import Iterable

import dfw_e030_gpu_real_path_local_map as e30
import dfw_e032_gpu_probe_outer_control_real_path as e32
import dfw_e059_gpu_stationary_inner_lr_probe_admission as e59
import dfw_e060_gpu_stationary_inner_lr_group_probe_admission as e60


DATE = "20261007"
OUT_DIR = Path(os.environ.get("DFW_E087_OUT_DIR", Path(__file__).resolve().parent)).resolve()
RESULT_PATH = OUT_DIR / f"dfw_e087_iid_exchangeability_noise_audit_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e087_iid_exchangeability_noise_audit_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e087_iid_exchangeability_noise_audit_LEGITIMACY_CHECK_{DATE}.md"

K_VALUES = e32.env_int_tuple("DFW_E087_K_VALUES", (8, 16, 32))
N_WORKERS = int(os.environ.get("DFW_E087_N_WORKERS", "8"))
N_GROUPS = int(os.environ.get("DFW_E087_N_GROUPS", "4"))
ROUNDS = int(os.environ.get("DFW_E087_ROUNDS", "36"))
SELECT_BATCH = int(os.environ.get("DFW_E087_SELECT_BATCH", "384"))
PILOT_SEEDS = e32.env_seed_tuple("DFW_E087_PILOT_SEEDS", tuple(range(987_000, 987_003)))
HELDOUT_SEEDS = e32.env_seed_tuple("DFW_E087_HELDOUT_SEEDS", tuple(range(988_000, 988_016)))
SPEC = e59.DampingSpec("iid_fake_group_mean2_l40_t2", 0.40, 2.0, 2)
EPS = 1e-12

mean = e59.mean
fmt_ci = e59.fmt_ci
mean_ci = e59.mean_ci
paired = e59.paired
positive_ci = e59.positive_ci
spearman = e59.spearman


@dataclass(frozen=True)
class FakeGroup:
    label: int
    update: object
    prior: float


@dataclass(frozen=True)
class PolicyTrace:
    values: list[float]
    shuffled: list[float]
    weight_l1: list[float]
    entropy: list[float]
    probe_target_spearman: list[float]
    top1_hit: list[float]
    selected_labels: list[int]


@dataclass(frozen=True)
class Cell:
    k_value: int
    fixed_pair: tuple[float, float]
    uniform: list[float]
    oracle_single: list[float]
    policy: PolicyTrace
    cost_pct: float


def compute_status() -> tuple[str, str, str]:
    diag = e30.compute_diagnostic()
    if diag.status == "REAL_GPU":
        return diag.status, diag.detail, diag.device
    if e30.torch is None:
        return "NO_TORCH", diag.detail, "cpu"
    version = getattr(e30.torch, "__version__", "unknown")
    return "TORCH_CPU", f"torch={version} cuda_available=False", "cpu"


def partitions(n_workers: int, n_groups: int) -> tuple[tuple[int, ...], ...]:
    if n_workers % n_groups != 0:
        raise ValueError("DFW-E087 requires N_WORKERS divisible by N_GROUPS")
    width = n_workers // n_groups
    return tuple(tuple(range(group * width, (group + 1) * width)) for group in range(n_groups))


def fake_groups(row: e60.GroupRoundRow) -> tuple[FakeGroup, ...]:
    parts = partitions(len(row.worker_updates), N_GROUPS)
    out: list[FakeGroup] = []
    for label, idxs in enumerate(parts):
        update = e30.mean_tensors([row.worker_updates[idx] for idx in idxs]).detach()
        out.append(FakeGroup(label=label, update=update, prior=len(idxs) / len(row.worker_updates)))
    return tuple(out)


def aggregate(groups: tuple[FakeGroup, ...], weights: list[float]):
    total = e30.torch.zeros_like(groups[0].update)
    for group, weight in zip(groups, weights):
        total = total + float(weight) * group.update
    return total.detach()


def losses_for(row: e60.GroupRoundRow, velocity, pair: tuple[float, float]) -> tuple[list[float], list[float]]:
    alpha, beta = pair
    select: list[float] = []
    target: list[float] = []
    for group in fake_groups(row):
        action = alpha * (group.update + beta * velocity)
        loss1 = e30.loss_float(row.theta + action, row.select_x, row.select_y)
        loss2 = e30.loss_float(row.theta + action, row.select2_x, row.select2_y)
        select.append(0.5 * (loss1 + loss2))
        target.append(e30.loss_float(row.theta + action, row.target_x, row.target_y))
    return select, target


def soft_weights(groups: tuple[FakeGroup, ...], losses: list[float]) -> list[float]:
    priors = [group.prior for group in groups]
    if len(losses) <= 1:
        return priors
    loc = mean(losses)
    spread = statistics.stdev(losses)
    temp = max(SPEC.tau * spread, 1e-5)
    logits = [max(-12.0, min(12.0, (loc - loss) / temp)) for loss in losses]
    offset = max(logits)
    exps = [math.exp(logit - offset) for logit in logits]
    denom = max(sum(exps), EPS)
    soft = [value / denom for value in exps]
    return [(1.0 - SPEC.lam) * prior + SPEC.lam * weight for prior, weight in zip(priors, soft)]


def entropy_fraction(weights: list[float]) -> float:
    if len(weights) <= 1:
        return 1.0
    ent = -sum(weight * math.log(max(weight, EPS)) for weight in weights)
    return ent / math.log(len(weights))


def l1_from_prior(groups: tuple[FakeGroup, ...], weights: list[float]) -> float:
    return sum(abs(weight - group.prior) for group, weight in zip(groups, weights))


def shuffled(losses: list[float], row_idx: int, rep_idx: int) -> list[float]:
    if not losses:
        return losses
    shift = (row_idx + 3 * rep_idx + 1) % len(losses)
    return losses[shift:] + losses[:shift]


def score_uniform(rows: list[e60.GroupRoundRow], pair: tuple[float, float]) -> float:
    return e60.score_fixed_group(rows, pair)


def score_oracle_single(rows: list[e60.GroupRoundRow], pair: tuple[float, float]) -> float:
    alpha, beta = pair
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=rows[0].theta.device)
    raw = 0.0
    total = 0.0
    for row in rows:
        groups = fake_groups(row)
        raw += e60.improvement(row, row.fedavg)
        best_loss = float("inf")
        best_update = groups[0].update
        for group in groups:
            loss = e30.loss_float(
                row.theta + alpha * (group.update + beta * velocity),
                row.target_x,
                row.target_y,
            )
            if loss < best_loss:
                best_loss = loss
                best_update = group.update
        total += row.target_start_loss - best_loss
        velocity = (best_update + beta * velocity).detach()
    return total / raw if raw > EPS else total


def score_policy(rows: list[e60.GroupRoundRow], pair: tuple[float, float], rep_idx: int, use_shuffle: bool) -> tuple[float, list[float], list[float], list[float], list[float], list[int]]:
    alpha, beta = pair
    velocity = e30.torch.zeros(e30.DIM, dtype=e30.torch.float32, device=rows[0].theta.device)
    raw = 0.0
    total = 0.0
    l1s: list[float] = []
    ents: list[float] = []
    corrs: list[float] = []
    hits: list[float] = []
    labels: list[int] = []
    for row_idx, row in enumerate(rows):
        groups = fake_groups(row)
        select_losses, target_losses = losses_for(row, velocity, pair)
        observed_losses = shuffled(select_losses, row_idx, rep_idx) if use_shuffle else select_losses
        weights = soft_weights(groups, observed_losses)
        aggregate_update = aggregate(groups, weights)
        direction = (aggregate_update + beta * velocity).detach()
        raw += e60.improvement(row, row.fedavg)
        total += e60.improvement(row, alpha * direction)
        selected_idx = min(range(len(observed_losses)), key=lambda idx: observed_losses[idx])
        target_idx = min(range(len(target_losses)), key=lambda idx: target_losses[idx])
        l1s.append(l1_from_prior(groups, weights))
        ents.append(entropy_fraction(weights))
        corrs.append(spearman(observed_losses, target_losses))
        hits.append(float(selected_idx == target_idx))
        labels.append(groups[selected_idx].label)
        velocity = direction
    return total / raw if raw > EPS else total, l1s, ents, corrs, hits, labels


def label_max_mass(labels: Iterable[int]) -> float:
    rows = list(labels)
    if not rows:
        return 0.0
    return max(rows.count(label) for label in set(rows)) / len(rows)


def cost_pct(k_value: int) -> float:
    return 100.0 * N_GROUPS * SPEC.slices * SELECT_BATCH / (
        e30.TRAIN_SIZE * 3.0 * N_WORKERS * k_value
    )


def diagnose_cell(k_value: int, device: str) -> Cell:
    e30.ROUNDS = ROUNDS
    e60.SELECT_BATCH = SELECT_BATCH
    pilot_rows = e60.make_groups(k_value, N_WORKERS, 1.0, PILOT_SEEDS, device)
    fixed_pair = e60.tune_fixed_pair(pilot_rows)
    heldout_rows = e60.make_groups(k_value, N_WORKERS, 1.0, HELDOUT_SEEDS, device)
    uniform_vals = [score_uniform(rows, fixed_pair) for rows in heldout_rows]
    oracle_vals = [score_oracle_single(rows, fixed_pair) for rows in heldout_rows]
    values: list[float] = []
    shuffled_values: list[float] = []
    l1s: list[float] = []
    ents: list[float] = []
    corrs: list[float] = []
    hits: list[float] = []
    labels: list[int] = []
    for rep_idx, rows in enumerate(heldout_rows):
        value, value_l1, value_ent, value_corr, value_hit, value_labels = score_policy(
            rows,
            fixed_pair,
            rep_idx,
            False,
        )
        shuf, _shuf_l1, _shuf_ent, _shuf_corr, _shuf_hit, _shuf_labels = score_policy(
            rows,
            fixed_pair,
            rep_idx,
            True,
        )
        values.append(value)
        shuffled_values.append(shuf)
        l1s.extend(value_l1)
        ents.extend(value_ent)
        corrs.extend(value_corr)
        hits.extend(value_hit)
        labels.extend(value_labels)
    return Cell(
        k_value=k_value,
        fixed_pair=fixed_pair,
        uniform=uniform_vals,
        oracle_single=oracle_vals,
        policy=PolicyTrace(values, shuffled_values, l1s, ents, corrs, hits, labels),
        cost_pct=cost_pct(k_value),
    )


def negative_ci(vals: Iterable[float]) -> bool:
    m, hw = mean_ci(vals)
    return m + hw < 0.0


def decide(cells: list[Cell], status: str) -> str:
    if status == "NO_TORCH":
        return "NO_TORCH"
    gains: list[float] = []
    shuffled_gaps: list[float] = []
    l1s: list[float] = []
    for cell in cells:
        gains.extend(paired(cell.policy.values, cell.uniform))
        shuffled_gaps.extend(paired(cell.policy.values, cell.policy.shuffled))
        l1s.extend(cell.policy.weight_l1)
    if positive_ci(gains) and positive_ci(shuffled_gaps):
        return "IID_STOCHASTIC_FILTERING_POSITIVE"
    if negative_ci(gains):
        return "IID_PROBE_NOISE_HURTS"
    if mean(l1s) > 0.05:
        return "IID_WEIGHTS_MOVE_NO_PERSISTENT_HPO_GAIN"
    return "IID_NULL_UNIFORM_WEIGHTS"


def run_experiment() -> tuple[str, str, str]:
    status, detail, device = compute_status()
    if status == "NO_TORCH":
        run_log = "\n".join(
            (
                "DFW-E087 IID exchangeability noise audit",
                f"host={platform.node()} python={sys.executable}",
                f"DECISION NO_TORCH detail={detail}",
            )
        )
        result = "# DFW-E087 Result - IID Exchangeability Noise Audit\n\nDecision label: `NO_TORCH`\n"
        legit = "# DFW-E087 Legitimacy Check\n\n- PyTorch workload: not run.\n"
        return run_log + "\n", result + "\n", legit + "\n"

    e30.torch.set_default_dtype(e30.torch.float32)
    try:
        e30.torch.set_num_threads(int(os.environ.get("DFW_E087_TORCH_THREADS", "1")))
    except Exception:
        pass

    cells: list[Cell] = []
    run_lines = [
        "DFW-E087 IID exchangeability noise audit",
        f"host={platform.node()} python={sys.executable} compute_status={status} {detail}",
        f"N={N_WORKERS} fake_groups={N_GROUPS} K={K_VALUES} rounds={ROUNDS} select_batch={SELECT_BATCH}",
        f"pilot_seeds={len(PILOT_SEEDS)} heldout_seeds={len(HELDOUT_SEEDS)} spec={SPEC.name}",
        "",
    ]
    for k_value in K_VALUES:
        cell = diagnose_cell(k_value, device)
        cells.append(cell)
        label_mass = label_max_mass(cell.policy.selected_labels)
        run_lines.append(
            f"CELL K={k_value} fixed_pair=({cell.fixed_pair[0]:.1f},{cell.fixed_pair[1]:.2f}) "
            f"uniform={fmt_ci(cell.uniform)} oracle_single={fmt_ci(cell.oracle_single)} "
            f"selector={fmt_ci(cell.policy.values)} selector-uniform={fmt_ci(paired(cell.policy.values, cell.uniform))} "
            f"selector-shuffled={fmt_ci(paired(cell.policy.values, cell.policy.shuffled))} "
            f"target_oracle-uniform={fmt_ci(paired(cell.oracle_single, cell.uniform))} "
            f"weight_l1={mean(cell.policy.weight_l1):.3f} entropy={mean(cell.policy.entropy):.3f} "
            f"probe_target_spearman={mean(cell.policy.probe_target_spearman):.3f} top1_hit={mean(cell.policy.top1_hit):.3f} "
            f"max_label_mass={label_mass:.3f} cost={cell.cost_pct:.2f}%"
        )
        print(run_lines[-1], flush=True)

    decision = decide(cells, status)
    pooled_gain: list[float] = []
    pooled_shuffled: list[float] = []
    pooled_oracle: list[float] = []
    pooled_l1: list[float] = []
    pooled_entropy: list[float] = []
    pooled_corr: list[float] = []
    pooled_hit: list[float] = []
    for cell in cells:
        pooled_gain.extend(paired(cell.policy.values, cell.uniform))
        pooled_shuffled.extend(paired(cell.policy.values, cell.policy.shuffled))
        pooled_oracle.extend(paired(cell.oracle_single, cell.uniform))
        pooled_l1.extend(cell.policy.weight_l1)
        pooled_entropy.extend(cell.policy.entropy)
        pooled_corr.extend(cell.policy.probe_target_spearman)
        pooled_hit.extend(cell.policy.top1_hit)
    run_lines.extend(
        [
            "",
            f"POOLED selector-uniform={fmt_ci(pooled_gain)} selector-shuffled={fmt_ci(pooled_shuffled)} "
            f"target_oracle-uniform={fmt_ci(pooled_oracle)} weight_l1={mean(pooled_l1):.3f} "
            f"entropy={mean(pooled_entropy):.3f} probe_target_spearman={mean(pooled_corr):.3f} top1_hit={mean(pooled_hit):.3f}",
            f"DECISION {decision}",
            f"compute_status={status}",
        ]
    )

    lines = [
        "# DFW-E087 Result - IID Exchangeability Noise Audit",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{status}`: `{detail}`.",
        "",
        "## Pooled",
        "",
        f"- selector minus uniform: `{fmt_ci(pooled_gain).strip()}`",
        f"- selector minus shuffled: `{fmt_ci(pooled_shuffled).strip()}`",
        f"- target oracle minus uniform: `{fmt_ci(pooled_oracle).strip()}`",
        f"- mean weight L1 from uniform prior: `{mean(pooled_l1):.3f}`",
        f"- mean weight entropy fraction: `{mean(pooled_entropy):.3f}`",
        f"- mean probe/target Spearman: `{mean(pooled_corr):.3f}`",
        f"- top-1 probe hit rate: `{mean(pooled_hit):.3f}`",
        "",
        "## Cells",
        "",
        "| K | fixed `(alpha,beta)` | uniform | target oracle single | selector | selector minus uniform | selector minus shuffled | oracle minus uniform | weight L1 | entropy | probe/target rho | top1 hit | max label mass | cost |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{cell.k_value}",
                    f"({cell.fixed_pair[0]:.1f},{cell.fixed_pair[1]:.2f})",
                    fmt_ci(cell.uniform).strip(),
                    fmt_ci(cell.oracle_single).strip(),
                    fmt_ci(cell.policy.values).strip(),
                    fmt_ci(paired(cell.policy.values, cell.uniform)).strip(),
                    fmt_ci(paired(cell.policy.values, cell.policy.shuffled)).strip(),
                    fmt_ci(paired(cell.oracle_single, cell.uniform)).strip(),
                    f"{mean(cell.policy.weight_l1):.3f}",
                    f"{mean(cell.policy.entropy):.3f}",
                    f"{mean(cell.policy.probe_target_spearman):.3f}",
                    f"{mean(cell.policy.top1_hit):.3f}",
                    f"{label_max_mass(cell.policy.selected_labels):.3f}",
                    f"{cell.cost_pct:.2f}%",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "Artificial labels create non-uniform realized probe weights without creating "
        "a durable hyperparameter signal.  A positive selector row here would be "
        "stochastic-update filtering and must be separated from the HPO claim."
    )
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])

    legit = "\n".join(
        [
            "# DFW-E087 Legitimacy Check",
            "",
            f"- Compute: `{status}` / `{detail}`.",
            "- All workers use the same inner learning rate; fake arm labels are arbitrary worker partitions.",
            "- Probe batches select weights; target batches are used only for scoring and oracle diagnostics.",
            "- Shuffled labels rotate probe losses across fake groups.",
            "- This is an IID/null audit, not an HPO positive gate.",
            "",
        ]
    )
    return "\n".join(run_lines) + "\n", "\n".join(lines) + "\n", legit + "\n"


def main() -> None:
    run_log, result, legit = run_experiment()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_PATH.write_text(run_log)
    RESULT_PATH.write_text(result)
    LEGITIMACY_PATH.write_text(legit)
    print(result)
    print(f"wrote {RESULT_PATH}")
    print(f"wrote {RUN_LOG_PATH}")
    print(f"wrote {LEGITIMACY_PATH}")


if __name__ == "__main__":
    main()
