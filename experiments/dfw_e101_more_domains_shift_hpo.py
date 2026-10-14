"""DFW-E101: additional-domain adaptive HPO audit.

E100 showed the adaptive HPO mechanism across five small synthetic domains.
E101 adds six more domain shapes without changing the controller:

* multiclass tabular classification;
* sparse bag-of-words text-like classification;
* pairwise ranking;
* graph/node neighborhood classification;
* sequence classification;
* matrix-interaction regression.

Each stream is deliberately nonstationary.  The early rounds are off-target or
label-corrupted while probe/select/target batches remain target-domain.  The
internal controller evaluates LR-arm candidate next states from the same base
model each round and probe-selects the next shared state.  External HPO spends
the same candidate-update budget on separate fixed-LR trajectories and selects
one by validation loss.
"""

from __future__ import annotations

from pathlib import Path
import os
import platform
import sys

import dfw_e100_multidomain_shift_hpo as e100


DATE = "20261014"
OUT_DIR = Path(os.environ.get("DFW_E101_OUT_DIR", Path(__file__).resolve().parent)).resolve()
RESULT_PATH = OUT_DIR / f"dfw_e101_more_domains_shift_hpo_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e101_more_domains_shift_hpo_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e101_more_domains_shift_hpo_LEGITIMACY_CHECK_{DATE}.md"
SUMMARY_PATH = OUT_DIR / "E101_MORE_DOMAINS_SHIFT_SUMMARY.md"


def parse_seeds() -> tuple[int, ...]:
    raw = os.environ.get(
        "DFW_E101_HELDOUT_SEEDS",
        ",".join(str(seed) for seed in range(999_400, 999_432)),
    )
    return tuple(int(part) for part in raw.split(",") if part)


def configure_e100_settings() -> None:
    e100.ROUNDS = int(os.environ.get("DFW_E101_ROUNDS", "24"))
    e100.SWITCH_ROUND = int(os.environ.get("DFW_E101_SWITCH_ROUND", "18"))
    e100.BATCH_SIZE = int(os.environ.get("DFW_E101_BATCH_SIZE", "64"))
    e100.PROBE_BATCH_SIZE = int(os.environ.get("DFW_E101_PROBE_BATCH_SIZE", "64"))
    e100.EVAL_BATCHES = int(os.environ.get("DFW_E101_EVAL_BATCHES", "2"))
    e100.BASE_LR = float(os.environ.get("DFW_E101_BASE_LR", "0.02"))
    e100.WEIGHT_DECAY = float(os.environ.get("DFW_E101_WEIGHT_DECAY", "0.0"))
    e100.OPTIMIZER = os.environ.get("DFW_E101_OPTIMIZER", "sgd")
    e100.POLICY = os.environ.get("DFW_E101_POLICY", "argmin")
    e100.TAU_MULT = float(os.environ.get("DFW_E101_TAU_MULT", "2.0"))
    e100.TOPK = int(os.environ.get("DFW_E101_TOPK", "3"))
    e100.HELDOUT_SEEDS = parse_seeds()
    e100.LR_MULTS = tuple(
        float(part)
        for part in os.environ.get("DFW_E101_LR_MULTS", "0.0,0.25,0.5,1.0,2.0,4.0").split(",")
        if part
    )


def multiclass_tabular_batch(seed: int, batch_size: int, target: bool, device: str) -> tuple[object, object]:
    g = e100.gen(seed)
    torch = e100.torch
    x = torch.randn(batch_size, 16, generator=g)
    basis = torch.arange(16, dtype=torch.float32)
    weight = torch.stack(
        [
            torch.sin(0.30 * basis),
            torch.cos(0.45 * basis),
            torch.linspace(-1.0, 1.0, 16),
            torch.linspace(1.0, -0.6, 16),
        ],
        dim=0,
    )
    logits = x @ weight.T
    logits[:, 0] += 0.35 * torch.sin(x[:, 0] * x[:, 1])
    logits[:, 1] += 0.35 * torch.cos(x[:, 2] - x[:, 3])
    logits[:, 2] += 0.25 * x[:, 4].pow(2)
    logits[:, 3] -= 0.30 * x[:, 5] * x[:, 6]
    y = logits.argmax(dim=1)
    if not target:
        x = x + 0.35
        y = (y + 1) % 4
    return x.to(device), y.to(device)


def sparse_text_batch(seed: int, batch_size: int, target: bool, device: str) -> tuple[object, object]:
    g = e100.gen(seed)
    torch = e100.torch
    classes = 3
    vocab = 72
    y = torch.randint(0, classes, (batch_size,), generator=g)
    probs = torch.full((batch_size, vocab), 0.025)
    for cls in range(classes):
        mask = y == cls
        start = cls * 18
        probs[mask, start : start + 18] = 0.38
        probs[mask, 54 + cls * 6 : 54 + (cls + 1) * 6] = 0.24
    x = torch.bernoulli(probs, generator=g)
    x = x + 0.04 * torch.randn(batch_size, vocab, generator=g)
    if not target:
        x = x + 0.05 * torch.randn(batch_size, vocab, generator=g)
        y = (y + 1) % classes
    return x.to(device), y.to(device)


def ranking_pair_batch(seed: int, batch_size: int, target: bool, device: str) -> tuple[object, object]:
    g = e100.gen(seed)
    torch = e100.torch
    dim = 8
    a = torch.randn(batch_size, dim, generator=g)
    b = torch.randn(batch_size, dim, generator=g)
    weight = torch.linspace(-0.9, 0.9, dim)
    score_a = a @ weight + 0.25 * torch.sin(a[:, 0] + a[:, 1])
    score_b = b @ weight + 0.25 * torch.sin(b[:, 0] + b[:, 1])
    y = (score_a > score_b).long()
    x = torch.cat([a, b, a - b, a * b], dim=1)
    if not target:
        x = 1.35 * x + 0.05 * torch.randn(batch_size, 4 * dim, generator=g)
        y = 1 - y
    return x.to(device), y.to(device)


def graph_node_batch(seed: int, batch_size: int, target: bool, device: str) -> tuple[object, object]:
    g = e100.gen(seed)
    torch = e100.torch
    dim = 8
    node = torch.randn(batch_size, dim, generator=g)
    neigh = torch.randn(batch_size, 4, dim, generator=g)
    neigh_mean = neigh.mean(dim=1)
    node_w = torch.linspace(-0.8, 0.9, dim)
    neigh_w = torch.linspace(0.7, -0.6, dim)
    score = node @ node_w + 0.85 * (neigh_mean @ neigh_w) + 0.25 * node[:, 0] * neigh_mean[:, 1]
    y = (score > 0).long()
    x = torch.cat([node, neigh_mean, node * neigh_mean], dim=1)
    if not target:
        x = x + 0.20
        y = 1 - y
    return x.to(device), y.to(device)


def sequence_class_batch(seed: int, batch_size: int, target: bool, device: str) -> tuple[object, object]:
    g = e100.gen(seed)
    torch = e100.torch
    classes = 3
    steps = torch.arange(24, dtype=torch.float32).unsqueeze(0).expand(batch_size, 24)
    y = torch.randint(0, classes, (batch_size,), generator=g)
    freq = torch.tensor([0.18, 0.34, 0.57])[y].unsqueeze(1)
    phase = 2.0 * e100.math.pi * torch.rand(batch_size, 1, generator=g)
    trend = torch.tensor([-0.035, 0.0, 0.035])[y].unsqueeze(1) * steps
    x = torch.sin(freq * steps + phase) + 0.55 * torch.cos(0.5 * freq * steps - phase)
    x = x + trend + 0.12 * torch.randn(batch_size, 24, generator=g)
    if not target:
        x = 1.25 * x + 0.08 * torch.randn(batch_size, 24, generator=g)
        y = (y + 1) % classes
    return x.to(device), y.to(device)


def matrix_interaction_batch(seed: int, batch_size: int, target: bool, device: str) -> tuple[object, object]:
    g = e100.gen(seed)
    torch = e100.torch
    dim = 6
    user = torch.randn(batch_size, dim, generator=g)
    item = torch.randn(batch_size, dim, generator=g)
    interaction = user * item
    y = (
        interaction.sum(dim=1)
        + 0.30 * torch.sin(user[:, 0] + item[:, 1])
        + 0.15 * user[:, 2].pow(2)
        - 0.15 * item[:, 3].pow(2)
    ).unsqueeze(1)
    x = torch.cat([user, item, interaction], dim=1)
    if not target:
        x = 1.20 * x + 0.05 * torch.randn(batch_size, 3 * dim, generator=g)
        y = -2.00 * y + 0.25
    return x.to(device), y.to(device)


def make_domains() -> tuple[e100.DomainSpec, ...]:
    return (
        e100.DomainSpec(
            "multiclass_tabular",
            (16,),
            4,
            "ce",
            lambda _device: e100.make_mlp(16, 4),
            multiclass_tabular_batch,
        ),
        e100.DomainSpec(
            "sparse_text",
            (72,),
            3,
            "ce",
            lambda _device: e100.make_mlp(72, 3),
            sparse_text_batch,
        ),
        e100.DomainSpec(
            "ranking_pair",
            (32,),
            2,
            "ce",
            lambda _device: e100.make_mlp(32, 2),
            ranking_pair_batch,
        ),
        e100.DomainSpec(
            "graph_node",
            (24,),
            2,
            "ce",
            lambda _device: e100.make_mlp(24, 2),
            graph_node_batch,
        ),
        e100.DomainSpec(
            "sequence_class",
            (24,),
            3,
            "ce",
            lambda _device: e100.make_mlp(24, 3),
            sequence_class_batch,
        ),
        e100.DomainSpec(
            "matrix_interaction",
            (18,),
            1,
            "mse",
            lambda _device: e100.make_mlp(18, 1),
            matrix_interaction_batch,
        ),
    )


def domain_decision(score: e100.DomainScores, status: str) -> str:
    if status == "NO_TORCH":
        return "NO_TORCH"
    internal_default = e100.paired(score.internal_probe, score.default)
    internal_shuffled = e100.paired(score.internal_probe, score.internal_shuffled)
    internal_external = e100.paired(score.internal_probe, score.external_valid)
    if (
        e100.positive_ci(internal_default)
        and e100.positive_ci(internal_shuffled)
        and e100.positive_ci(internal_external)
    ):
        return "MORE_DOMAIN_SHIFT_HPO_WIN"
    if e100.positive_ci(internal_default) and e100.positive_ci(internal_shuffled):
        return "MORE_DOMAIN_SHIFT_LABEL_CONTENT"
    return "MORE_DOMAIN_SHIFT_NOT_CLEARED"


def no_torch_artifacts(detail: str) -> tuple[str, str, str, str]:
    run_log = "\n".join(
        (
            "DFW-E101 more-domain shift HPO",
            f"host={platform.node()} python={sys.executable}",
            f"DECISION NO_TORCH detail={detail}",
        )
    )
    result = "# DFW-E101 Result - More-Domain Shift HPO\n\nDecision label: `NO_TORCH`\n"
    legit = "# DFW-E101 Legitimacy Check\n\n- PyTorch workload: not run.\n"
    summary = "# E101 More-Domain Shift HPO Summary\n\nDecision: `NO_TORCH`.\n"
    return run_log + "\n", result + "\n", legit + "\n", summary + "\n"


def result_lines(
    scores: list[e100.DomainScores],
    status: str,
    detail: str,
    decision: str,
    pooled_default: list[float],
    pooled_shuffled: list[float],
    pooled_external: list[float],
    pooled_oracle: list[float],
) -> list[str]:
    lines = [
        "# DFW-E101 Result - More-Domain Shift HPO",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{status}`: `{detail}`.",
        "",
        "## Pooled",
        "",
        f"- internal minus default: `{e100.fmt_ci(pooled_default).strip()}`",
        f"- internal minus shuffled: `{e100.fmt_ci(pooled_shuffled).strip()}`",
        f"- internal minus external validation-selected HPO: `{e100.fmt_ci(pooled_external).strip()}`",
        f"- internal minus external oracle HPO: `{e100.fmt_ci(pooled_oracle).strip()}`",
        "",
        "## Domains",
        "",
        "| domain | decision | internal-default | internal-shuffled | internal-external | cost |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for score in scores:
        lines.append(
            "| "
            + " | ".join(
                [
                    score.name,
                    domain_decision(score, status),
                    e100.fmt_ci(e100.paired(score.internal_probe, score.default)).strip(),
                    e100.fmt_ci(e100.paired(score.internal_probe, score.internal_shuffled)).strip(),
                    e100.fmt_ci(e100.paired(score.internal_probe, score.external_valid)).strip(),
                    f"{score.probe_cost_pct:.2f}%",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    if decision == "MORE_DOMAINS_SHIFT_HPO_WIN":
        lines.append(
            "The added-domain audit clears the pooled fixed-compute comparator: "
            "same-round probe-selected aggregation beats default fixed LR, shuffled probes, "
            "and validation-selected external fixed-LR HPO across six additional domain shapes."
        )
    else:
        lines.append("The added-domain audit is a boundary result; inspect per-domain rows.")
    return lines


def legitimacy_text(status: str, detail: str) -> str:
    return "\n".join(
        [
            "# DFW-E101 Legitimacy Check",
            "",
            f"- Compute: `{status}` / `{detail}`.",
            "- Domains: multiclass tabular, sparse text-like bags, pairwise ranking, graph/node summaries, sequence classification, matrix-interaction regression.",
            f"- Optimizer family: `{e100.OPTIMIZER}` with LR multipliers `{', '.join(f'{mult:g}' for mult in e100.LR_MULTS)}`.",
            "- Training stream: off-target or label-corrupted before the switch, target-domain after the switch.",
            "- Probe/select and target batches are target-domain.",
            "- Same candidate-update budget: internal and external HPO both train one candidate update per LR arm per round.",
            f"- Internal HPO uses `{e100.POLICY}` same-round probe selection from one shared base state each round.",
            "- External HPO keeps separate fixed-LR trajectories and selects by held-out validation loss.",
            "- Shuffled control preserves candidate updates but permutes probe-to-arm labels.",
            "- This is an additional small synthetic domain-suite audit, not a foundation-scale benchmark.",
            "",
        ]
    )


def summary_lines(
    scores: list[e100.DomainScores],
    status: str,
    detail: str,
    decision: str,
    pooled_default: list[float],
    pooled_shuffled: list[float],
    pooled_external: list[float],
    pooled_oracle: list[float],
) -> list[str]:
    domains = ", ".join(score.name for score in scores)
    arms = ", ".join(f"{mult:g}" for mult in e100.LR_MULTS)
    lines = [
        "# E101 More-Domain Shift HPO Summary",
        "",
        f"Decision: `{decision}`.",
        "",
        "Run:",
        "",
        "- script: `experiments/dfw_e101_more_domains_shift_hpo.py`",
        f"- output dir: `{OUT_DIR.relative_to(Path.cwd()) if OUT_DIR.is_relative_to(Path.cwd()) else OUT_DIR}`",
        f"- seeds: `{min(e100.HELDOUT_SEEDS)}..{max(e100.HELDOUT_SEEDS)}` ({len(e100.HELDOUT_SEEDS)} seeds)",
        f"- domains: {domains}",
        f"- optimizer arms: {e100.OPTIMIZER.upper()} with LR multipliers `({arms})`",
        f"- policy: one-hot probe selection (`{e100.POLICY}`)",
        f"- rounds: `{e100.ROUNDS}`, target-domain switch at round `{e100.SWITCH_ROUND}`",
        f"- batch: `{e100.BATCH_SIZE}`, probe batch: `{e100.PROBE_BATCH_SIZE}`, eval batches: `{e100.EVAL_BATCHES}`",
        f"- estimated probe overhead: `{scores[0].probe_cost_pct:.2f}%`",
        f"- compute: Buck/PyTorch `{status}`, `{detail}`",
        "",
        "Pooled readout:",
        "",
        f"- internal minus default: `{e100.fmt_ci(pooled_default).strip()}`",
        f"- internal minus shuffled probes: `{e100.fmt_ci(pooled_shuffled).strip()}`",
        f"- internal minus validation-selected external fixed-LR HPO: `{e100.fmt_ci(pooled_external).strip()}`",
        f"- internal minus external oracle fixed-LR HPO: `{e100.fmt_ci(pooled_oracle).strip()}`",
        "",
        "Per-domain readout:",
        "",
        "| domain | decision | internal-default | internal-shuffled | internal-external |",
        "|---|---|---:|---:|---:|",
    ]
    for score in scores:
        lines.append(
            "| "
            + " | ".join(
                [
                    score.name,
                    f"`{domain_decision(score, status)}`",
                    f"`{e100.fmt_ci(e100.paired(score.internal_probe, score.default)).strip()}`",
                    f"`{e100.fmt_ci(e100.paired(score.internal_probe, score.internal_shuffled)).strip()}`",
                    f"`{e100.fmt_ci(e100.paired(score.internal_probe, score.external_valid)).strip()}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "E101 is a breadth extension of E100.  It keeps the same adaptive-HPO controller and fixed-compute external-HPO comparator, but moves to six new synthetic domain shapes.  A pooled win here strengthens the paper's claim that the mechanism is not tied to one tabular toy, one tiny-LM stream, or one local-SGD outer-control ledger.",
        ]
    )
    return lines


def run_experiment() -> tuple[str, str, str, str]:
    configure_e100_settings()
    status, detail, device = e100.compute_status()
    if status == "NO_TORCH":
        return no_torch_artifacts(detail)
    e100.torch.set_num_threads(int(os.environ.get("DFW_E101_TORCH_THREADS", "1")))
    scores = [e100.diagnose_domain(spec, device) for spec in make_domains()]
    pooled_default = e100.pooled(scores, "internal_probe", "default")
    pooled_shuffled = e100.pooled(scores, "internal_probe", "internal_shuffled")
    pooled_external = e100.pooled(scores, "internal_probe", "external_valid")
    pooled_oracle = e100.pooled(scores, "internal_probe", "external_oracle")
    decision = (
        "MORE_DOMAINS_SHIFT_HPO_WIN"
        if (
            e100.positive_ci(pooled_default)
            and e100.positive_ci(pooled_shuffled)
            and e100.positive_ci(pooled_external)
        )
        else "MORE_DOMAINS_SHIFT_NOT_CLEARED"
    )
    arms = ", ".join(f"{mult:g}" for mult in e100.LR_MULTS)
    run_lines = [
        "DFW-E101 more-domain shift HPO",
        f"host={platform.node()} python={sys.executable} compute_status={status} {detail}",
        (
            f"rounds={e100.ROUNDS} switch_round={e100.SWITCH_ROUND} batch={e100.BATCH_SIZE} "
            f"probe_batch={e100.PROBE_BATCH_SIZE} eval_batches={e100.EVAL_BATCHES} policy={e100.POLICY}"
        ),
        (
            f"optimizer={e100.OPTIMIZER} base_lr={e100.BASE_LR} weight_decay={e100.WEIGHT_DECAY} "
            f"lr_mults=({arms}) seeds={len(e100.HELDOUT_SEEDS)}"
        ),
        f"POOLED internal-default={e100.fmt_ci(pooled_default)} "
        f"internal-shuffled={e100.fmt_ci(pooled_shuffled)} "
        f"internal-external_valid={e100.fmt_ci(pooled_external)} "
        f"internal-external_oracle={e100.fmt_ci(pooled_oracle)}",
        f"DECISION {decision}",
        f"compute_status={status}",
    ]
    result = result_lines(scores, status, detail, decision, pooled_default, pooled_shuffled, pooled_external, pooled_oracle)
    legit = legitimacy_text(status, detail)
    summary = summary_lines(scores, status, detail, decision, pooled_default, pooled_shuffled, pooled_external, pooled_oracle)
    return "\n".join(run_lines) + "\n", "\n".join(result) + "\n", legit, "\n".join(summary) + "\n"


def main() -> None:
    run_log, result, legit, summary = run_experiment()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_PATH.write_text(run_log)
    RESULT_PATH.write_text(result)
    LEGITIMACY_PATH.write_text(legit)
    SUMMARY_PATH.write_text(summary)
    print(result)
    print(f"wrote {RESULT_PATH}")
    print(f"wrote {RUN_LOG_PATH}")
    print(f"wrote {LEGITIMACY_PATH}")
    print(f"wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
