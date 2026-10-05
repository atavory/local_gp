"""DFW-E063: focused replication of the grouped LR probe positive.

E062 showed that an online quadratic response tracker ties the raw same-round
group probe rule.  Before adding more modeling machinery, this experiment
tightens the favorable claim: fixed grouped LR arms plus the frozen
`group_mean2_l40_t2` damped probe rule, with more held-out seeds and a shuffled
control.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import platform
import sys

import dfw_e030_gpu_real_path_local_map as e30
import dfw_e032_gpu_probe_outer_control_real_path as e32
import dfw_e059_gpu_stationary_inner_lr_probe_admission as e59
import dfw_e060_gpu_stationary_inner_lr_group_probe_admission as e60


DATE = "20261005"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / f"dfw_e063_gpu_group_probe_replication_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e063_gpu_group_probe_replication_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e063_gpu_group_probe_replication_LEGITIMACY_CHECK_{DATE}.md"

K_VALUES = e32.env_int_tuple("DFW_E063_K_VALUES", (8, 16, 32))
N_WORKERS = e32.env_int_tuple("DFW_E063_N_WORKERS", (8,))
ROUNDS = int(os.environ.get("DFW_E063_ROUNDS", "36"))
SELECT_BATCH = int(os.environ.get("DFW_E063_SELECT_BATCH", "384"))
PILOT_SEEDS = e32.env_seed_tuple("DFW_E063_PILOT_SEEDS", tuple(range(929_000, 929_003)))
HELDOUT_SEEDS = e32.env_seed_tuple("DFW_E063_HELDOUT_SEEDS", tuple(range(930_000, 930_012)))
SPREAD_VALUES = e59.env_float_tuple("DFW_E063_SPREAD_VALUES", (1.0, 16.0))
EPS = 1e-12
RAW_SPEC = next(spec for spec in e60.GROUP_SPECS if spec.name == "group_mean2_l40_t2")


mean = e59.mean
fmt_ci = e59.fmt_ci
paired = e59.paired
positive_ci = e59.positive_ci
spread_text = e59.spread_text


@dataclass(frozen=True)
class Cell:
    spread: float
    n_workers: int
    k_value: int
    group_values: tuple[float, ...]
    fixed_pair: tuple[float, float]
    fixed: list[float]
    oracle_subset: list[float]
    raw: list[float]
    shuffled: list[float]
    cost_pct: float


def cost_pct(group_count: int, n_workers: int, k_value: int) -> float:
    return 100.0 * group_count * RAW_SPEC.slices * SELECT_BATCH / (
        e30.TRAIN_SIZE * 3.0 * n_workers * k_value
    )


def capture(vals: list[float], fixed: list[float], oracle: list[float]) -> list[float]:
    out = []
    for val, fix, ora in zip(vals, fixed, oracle):
        gap = ora - fix
        out.append((val - fix) / gap if abs(gap) > EPS else 0.0)
    return out


def diagnose_cell(spread: float, k_value: int, n_workers: int, device: str) -> Cell:
    e30.ROUNDS = ROUNDS
    e60.SELECT_BATCH = SELECT_BATCH
    pilot_groups = e60.make_groups(k_value, n_workers, spread, PILOT_SEEDS, device)
    fixed_pair = e60.tune_fixed_pair(pilot_groups)
    heldout_groups = e60.make_groups(k_value, n_workers, spread, HELDOUT_SEEDS, device)
    fixed_vals = [e60.score_fixed_group(rows, fixed_pair) for rows in heldout_groups]
    oracle_vals = [e60.score_group_oracle(rows, fixed_pair)[0] for rows in heldout_groups]
    raw_vals = []
    shuffled_vals = []
    for rep_idx, rows in enumerate(heldout_groups):
        raw_vals.append(e60.score_damped_group_policy(rows, fixed_pair, RAW_SPEC, rep_idx, False)[0])
        shuffled_vals.append(e60.score_damped_group_policy(rows, fixed_pair, RAW_SPEC, rep_idx, True)[0])
    first = heldout_groups[0][0]
    groups = e60.group_updates(first)
    return Cell(
        spread=spread,
        n_workers=n_workers,
        k_value=k_value,
        group_values=tuple(value for value, _update, _prior in groups),
        fixed_pair=fixed_pair,
        fixed=fixed_vals,
        oracle_subset=oracle_vals,
        raw=raw_vals,
        shuffled=shuffled_vals,
        cost_pct=cost_pct(len(groups), n_workers, k_value),
    )


def decide(cells: list[Cell], compute_status: str) -> str:
    if compute_status != "REAL_GPU":
        return "NO_GPU_HARNESS"
    spread_cells = [cell for cell in cells if cell.spread > 1.0 + 1e-12]
    raw_gains = []
    raw_vs_shuffled = []
    oracle_gains = []
    for cell in spread_cells:
        raw_gains.extend(paired(cell.raw, cell.fixed))
        raw_vs_shuffled.extend(paired(cell.raw, cell.shuffled))
        oracle_gains.extend(paired(cell.oracle_subset, cell.fixed))
    collapse_cells = [cell for cell in cells if cell.spread <= 1.0 + 1e-12]
    collapse_ok = all(abs(mean(paired(cell.raw, cell.fixed))) < 1e-9 for cell in collapse_cells)
    if collapse_ok and positive_ci(raw_gains) and positive_ci(raw_vs_shuffled):
        return "RAW_GROUP_PROBE_REPLICATES"
    if positive_ci(oracle_gains):
        return "ORACLE_GROUP_HEADROOM_EXTRACTION_WEAK"
    return "GROUP_PROBE_REPLICATION_NEGATIVE"


def no_gpu_artifacts(diag: e30.ComputeDiagnostic) -> tuple[str, str, str]:
    run_log = "\n".join(
        (
            "DFW-E063 grouped LR probe replication",
            f"host={platform.node()} python={sys.executable}",
            f"DECISION NO_GPU_HARNESS compute_status={diag.status} detail={diag.detail}",
        )
    )
    result = "\n".join(
        (
            "# DFW-E063 Result - Group Probe Replication",
            "",
            "Decision label: `NO_GPU_HARNESS`",
            "",
            f"Diagnostic: `{diag.detail}`.",
            "",
        )
    )
    legit = "# DFW-E063 Legitimacy Check\n\n- GPU workload: not run.\n"
    return run_log + "\n", result + "\n", legit


def run_experiment() -> tuple[str, str, str]:
    diag = e30.compute_diagnostic()
    if diag.status != "REAL_GPU":
        return no_gpu_artifacts(diag)
    e30.torch.set_default_dtype(e30.torch.float32)
    try:
        e30.torch.set_num_threads(1)
    except Exception:
        pass
    device = diag.device
    cells: list[Cell] = []
    run_lines = [
        "DFW-E063 grouped LR probe replication",
        f"host={platform.node()} python={sys.executable} compute_status={diag.status} {diag.detail}",
        f"model=torch_two_layer_binary_classifier dim={e30.DIM} N={N_WORKERS} K={K_VALUES} rounds={ROUNDS} eta={e30.ETA}",
        f"pilot_seeds={len(PILOT_SEEDS)} heldout_seeds={len(HELDOUT_SEEDS)} spread_values={SPREAD_VALUES} select_batch={SELECT_BATCH} raw_spec={RAW_SPEC.name}",
        "",
    ]
    for n_workers in N_WORKERS:
        for k_value in K_VALUES:
            for spread in SPREAD_VALUES:
                cell = diagnose_cell(spread, k_value, n_workers, device)
                cells.append(cell)
                run_lines.append(
                    f"CELL spread={spread_text(spread)} N={n_workers} K={k_value} "
                    f"fixed_pair=({cell.fixed_pair[0]:.1f},{cell.fixed_pair[1]:.2f}) "
                    f"groups={','.join(f'{value:.3g}' for value in cell.group_values)} "
                    f"fixed={fmt_ci(cell.fixed)} oracle={fmt_ci(cell.oracle_subset)} "
                    f"raw={fmt_ci(cell.raw)} raw-fixed={fmt_ci(paired(cell.raw, cell.fixed))} "
                    f"raw-shuffled={fmt_ci(paired(cell.raw, cell.shuffled))} "
                    f"capture={fmt_ci(capture(cell.raw, cell.fixed, cell.oracle_subset))} cost={cell.cost_pct:.2f}%"
                )
                print(run_lines[-1], flush=True)
    decision = decide(cells, diag.status)
    run_lines.append("")
    run_lines.append(f"DECISION {decision}")
    run_lines.append(f"compute_status={diag.status}")

    lines = [
        "# DFW-E063 Result - Group Probe Replication",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{diag.status}` via PyTorch CUDA: `{diag.detail}`.",
        "",
        "## Held-Out Scores",
        "",
        "| spread | N | K | groups | fixed `(alpha,beta)` | fixed | oracle | raw | raw minus fixed | raw minus shuffled | capture | cost |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            "| "
            + " | ".join(
                [
                    spread_text(cell.spread),
                    f"{cell.n_workers}",
                    f"{cell.k_value}",
                    ",".join(f"{value:.3g}" for value in cell.group_values),
                    f"({cell.fixed_pair[0]:.1f},{cell.fixed_pair[1]:.2f})",
                    fmt_ci(cell.fixed).strip(),
                    fmt_ci(cell.oracle_subset).strip(),
                    fmt_ci(cell.raw).strip(),
                    fmt_ci(paired(cell.raw, cell.fixed)).strip(),
                    fmt_ci(paired(cell.raw, cell.shuffled)).strip(),
                    fmt_ci(capture(cell.raw, cell.fixed, cell.oracle_subset)).strip(),
                    f"{cell.cost_pct:.2f}%",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Run Log", "", "```text", "\n".join(run_lines), "```"])
    legit = "\n".join(
        [
            "# DFW-E063 Legitimacy Check",
            "",
            f"- Compute: `{diag.status}` / `{diag.detail}`.",
            "- Target batches are used only for scoring and oracle diagnostics.",
            "- Uses frozen `group_mean2_l40_t2`; no pilot policy selection.",
            "- Shuffled control rotates probe values across LR groups.",
            "",
        ]
    )
    return "\n".join(run_lines) + "\n", "\n".join(lines) + "\n", legit + "\n"


def main() -> None:
    run_log, result, legit = run_experiment()
    RUN_LOG_PATH.write_text(run_log)
    RESULT_PATH.write_text(result)
    LEGITIMACY_PATH.write_text(legit)
    print(result)
    print(f"wrote {RESULT_PATH}")
    print(f"wrote {RUN_LOG_PATH}")
    print(f"wrote {LEGITIMACY_PATH}")


if __name__ == "__main__":
    main()
