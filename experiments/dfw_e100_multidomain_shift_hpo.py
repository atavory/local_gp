"""DFW-E100: multidomain domain-shift HPO audit.

The local-SGD and tiny-LM results show the central mechanism in two settings.
E100 broadens the stress test across small synthetic domains:

* tabular binary classification;
* tabular regression;
* PFN-style tabular in-context classification;
* time-series next-step forecasting;
* synthetic image classification.

Each domain has a deliberately nonstationary training stream: the first half is
off-target or label-flipped, the second half is target-domain.  Probe/select
and target batches are always target-domain.  The internal controller evaluates
LR-arm candidate next states from the same base model each round and
probe-weights their parameter deltas into one shared trajectory.  External HPO
spends the same candidate-update budget on separate fixed-LR trajectories and
selects one by validation loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import math
import os
import platform
import statistics
import sys
from typing import Callable, Iterable

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None


DATE = "20261014"
OUT_DIR = Path(os.environ.get("DFW_E100_OUT_DIR", Path(__file__).resolve().parent)).resolve()
RESULT_PATH = OUT_DIR / f"dfw_e100_multidomain_shift_hpo_result_{DATE}.md"
RUN_LOG_PATH = OUT_DIR / f"dfw_e100_multidomain_shift_hpo_run_{DATE}.txt"
LEGITIMACY_PATH = OUT_DIR / f"dfw_e100_multidomain_shift_hpo_LEGITIMACY_CHECK_{DATE}.md"

ROUNDS = int(os.environ.get("DFW_E100_ROUNDS", "24"))
SWITCH_ROUND = int(os.environ.get("DFW_E100_SWITCH_ROUND", "18"))
BATCH_SIZE = int(os.environ.get("DFW_E100_BATCH_SIZE", "64"))
PROBE_BATCH_SIZE = int(os.environ.get("DFW_E100_PROBE_BATCH_SIZE", "64"))
EVAL_BATCHES = int(os.environ.get("DFW_E100_EVAL_BATCHES", "2"))
BASE_LR = float(os.environ.get("DFW_E100_BASE_LR", "0.02"))
WEIGHT_DECAY = float(os.environ.get("DFW_E100_WEIGHT_DECAY", "0.0"))
OPTIMIZER = os.environ.get("DFW_E100_OPTIMIZER", "sgd")
POLICY = os.environ.get("DFW_E100_POLICY", "argmin")
TAU_MULT = float(os.environ.get("DFW_E100_TAU_MULT", "2.0"))
TOPK = int(os.environ.get("DFW_E100_TOPK", "3"))
HELDOUT_SEEDS = tuple(
    int(part)
    for part in os.environ.get(
        "DFW_E100_HELDOUT_SEEDS",
        ",".join(str(seed) for seed in range(999_300, 999_332)),
    ).split(",")
    if part
)
LR_MULTS = tuple(
    float(part)
    for part in os.environ.get("DFW_E100_LR_MULTS", "0.0,0.25,0.5,1.0,2.0,4.0").split(",")
    if part
)


@dataclass(frozen=True)
class Arm:
    name: str
    lr_mult: float


@dataclass(frozen=True)
class DomainSpec:
    name: str
    input_shape: tuple[int, ...]
    output_dim: int
    task: str
    make_model: Callable[[str], nn.Module]
    make_batch: Callable[[int, int, bool, str], tuple[object, object]]


@dataclass(frozen=True)
class DomainScores:
    name: str
    initial_loss: list[float]
    default: list[float]
    internal_probe: list[float]
    internal_shuffled: list[float]
    internal_oracle: list[float]
    external_valid: list[float]
    external_oracle: list[float]
    probe_cost_pct: float


def mean(vals: Iterable[float]) -> float:
    rows = tuple(vals)
    return sum(rows) / len(rows) if rows else 0.0


def ci95(vals: Iterable[float]) -> float:
    rows = tuple(vals)
    if len(rows) < 2:
        return 0.0
    return 1.96 * statistics.stdev(rows) / math.sqrt(len(rows))


def fmt_ci(vals: Iterable[float]) -> str:
    rows = tuple(vals)
    return f"{mean(rows):7.3f}+/-{ci95(rows):.3f}"


def paired(lhs: list[float], rhs: list[float]) -> list[float]:
    return [a - b for a, b in zip(lhs, rhs)]


def positive_ci(vals: Iterable[float]) -> bool:
    rows = tuple(vals)
    return bool(rows) and mean(rows) - ci95(rows) > 0.0


def compute_status() -> tuple[str, str, str]:
    if torch is None:
        return "NO_TORCH", "PyTorch import failed", "cpu"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        detail = f"torch={torch.__version__} cuda_available=True device={device_name}"
        return "REAL_GPU", detail, "cuda"
    return "TORCH_CPU", f"torch={torch.__version__} cuda_available=False", "cpu"


def gen(seed: int) -> object:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def clone_state(model: nn.Module) -> dict[str, object]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def load_state(model: nn.Module, state: dict[str, object]) -> None:
    model.load_state_dict(state)


def make_optimizer(model: nn.Module, lr: float) -> object:
    if OPTIMIZER == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    if OPTIMIZER == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    raise ValueError(f"unknown optimizer: {OPTIMIZER}")


def make_mlp(input_dim: int, output_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.GELU(),
        nn.Linear(64, 64),
        nn.GELU(),
        nn.Linear(64, output_dim),
    )


def make_cnn(_device: str) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv2d(8, 8, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Flatten(),
        nn.Linear(8 * 8 * 8, 32),
        nn.GELU(),
        nn.Linear(32, 2),
    )


def tabular_class_batch(seed: int, batch_size: int, target: bool, device: str) -> tuple[object, object]:
    g = gen(seed)
    x = torch.randn(batch_size, 12, generator=g)
    weight = torch.linspace(-1.2, 1.2, 12)
    logits = x @ weight + 0.4 * torch.sin(x[:, 0] * x[:, 1])
    y = (logits > 0).long()
    if not target:
        y = 1 - y
        x = x + 0.4
    return x.to(device), y.to(device)


def tabular_reg_batch(seed: int, batch_size: int, target: bool, device: str) -> tuple[object, object]:
    g = gen(seed)
    x = torch.randn(batch_size, 10, generator=g)
    weight = torch.linspace(0.7, -0.9, 10)
    y = (x @ weight + 0.25 * torch.sin(2.0 * x[:, 0])).unsqueeze(1)
    if not target:
        y = -0.9 * y + 0.5 * torch.cos(x[:, 1]).unsqueeze(1)
        x = x - 0.3
    return x.to(device), y.to(device)


def tabular_context_batch(seed: int, batch_size: int, target: bool, device: str) -> tuple[object, object]:
    g = gen(seed)
    dim = 8
    ctx = 16
    w = torch.randn(batch_size, dim, generator=g)
    x_ctx = torch.randn(batch_size, ctx, dim, generator=g)
    ctx_logits = (x_ctx * w.unsqueeze(1)).sum(dim=2) + 0.2 * torch.sin(x_ctx[:, :, 0] * w[:, 0:1])
    y_ctx = (ctx_logits > 0).long()
    x_q = torch.randn(batch_size, dim, generator=g)
    q_logits = (x_q * w).sum(dim=1) + 0.2 * torch.sin(x_q[:, 0] * w[:, 0])
    y_q = (q_logits > 0).long()

    mask0 = (y_ctx == 0).float().unsqueeze(2)
    mask1 = (y_ctx == 1).float().unsqueeze(2)
    count0 = mask0.sum(dim=1).clamp_min(1.0)
    count1 = mask1.sum(dim=1).clamp_min(1.0)
    mean0 = (x_ctx * mask0).sum(dim=1) / count0
    mean1 = (x_ctx * mask1).sum(dim=1) / count1
    if not target:
        y_q = 1 - y_q
    counts = torch.cat([count0, count1], dim=1) / float(ctx)
    proto = (x_q - 0.5 * (mean0 + mean1)) * (mean1 - mean0)
    proto_score = proto.sum(dim=1, keepdim=True)
    features = torch.cat(
        [x_q, mean0, mean1, x_q - mean0, x_q - mean1, counts, 3.0 * proto_score, proto_score.abs()],
        dim=1,
    )
    return features.to(device), y_q.to(device)


def timeseries_batch(seed: int, batch_size: int, target: bool, device: str) -> tuple[object, object]:
    g = gen(seed)
    steps = torch.arange(16, dtype=torch.float32).unsqueeze(0).expand(batch_size, 16)
    phase = 2.0 * math.pi * torch.rand(batch_size, 1, generator=g)
    freq = 0.38 if target else 0.16
    signal = torch.sin(freq * steps + phase)
    signal = signal + 0.15 * torch.randn(batch_size, 16, generator=g)
    x = signal[:, :15]
    y = signal[:, 15:16]
    if not target:
        y = -y
    return x.to(device), y.to(device)


def vision_batch(seed: int, batch_size: int, target: bool, device: str) -> tuple[object, object]:
    g = gen(seed)
    y = torch.randint(0, 2, (batch_size,), generator=g)
    imgs = 0.15 * torch.randn(batch_size, 1, 8, 8, generator=g)
    for idx, label in enumerate(y.tolist()):
        if label == 0:
            imgs[idx, 0, :, 2:4] += 1.0
        else:
            imgs[idx, 0, 2:4, :] += 1.0
    if not target:
        y = 1 - y
        imgs = torch.flip(imgs, dims=(3,))
    return imgs.to(device), y.to(device)


def make_domains() -> tuple[DomainSpec, ...]:
    return (
        DomainSpec("tabular_class", (12,), 2, "ce", lambda _device: make_mlp(12, 2), tabular_class_batch),
        DomainSpec("tabular_reg", (10,), 1, "mse", lambda _device: make_mlp(10, 1), tabular_reg_batch),
        DomainSpec("tabular_context", (44,), 2, "ce", lambda _device: make_mlp(44, 2), tabular_context_batch),
        DomainSpec("timeseries", (15,), 1, "mse", lambda _device: make_mlp(15, 1), timeseries_batch),
        DomainSpec("vision8x8", (1, 8, 8), 2, "ce", make_cnn, vision_batch),
    )


def loss_on_batch(model: nn.Module, batch: tuple[object, object], task: str) -> object:
    x, y = batch
    pred = model(x)
    if task == "ce":
        return F.cross_entropy(pred, y)
    return F.mse_loss(pred, y)


def eval_loss(model: nn.Module, spec: DomainSpec, seed: int, batch_size: int, device: str) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for idx in range(EVAL_BATCHES):
            batch = spec.make_batch(seed + 10_000 * idx, batch_size, True, device)
            losses.append(float(loss_on_batch(model, batch, spec.task).detach().cpu()))
    return mean(losses)


def train_worker_candidate(
    worker: nn.Module,
    opt: object,
    state: dict[str, object],
    batch: tuple[object, object],
    task: str,
) -> dict[str, object]:
    load_state(worker, state)
    worker.train()
    opt.zero_grad(set_to_none=True)
    loss = loss_on_batch(worker, batch, task)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(worker.parameters(), 1.0)
    opt.step()
    return clone_state(worker)


def weights_from_losses(losses: list[float]) -> list[float]:
    if POLICY == "argmin":
        best_idx = min(range(len(losses)), key=lambda idx: losses[idx])
        return [1.0 if idx == best_idx else 0.0 for idx in range(len(losses))]
    if POLICY == "topk_mean":
        order = sorted(range(len(losses)), key=lambda idx: losses[idx])
        keep = set(order[: min(TOPK, len(order))])
        return [1.0 / len(keep) if idx in keep else 0.0 for idx in range(len(losses))]
    if POLICY in ("soft_std1", "soft_std2"):
        mult = 1.0 if POLICY == "soft_std1" else TAU_MULT
        denom = max(statistics.pstdev(losses) * mult, 0.005)
        low = min(losses)
        raw = [math.exp(-(loss - low) / denom) for loss in losses]
        total = sum(raw)
        return [val / total for val in raw]
    raise ValueError(f"unknown policy: {POLICY}")


def aggregate_states(
    base_state: dict[str, object],
    candidate_states: list[dict[str, object]],
    weights: list[float],
) -> dict[str, object]:
    out = {}
    for key, base_value in base_state.items():
        if not torch.is_floating_point(base_value):
            out[key] = base_value.clone()
            continue
        agg = torch.zeros_like(base_value)
        for weight, state in zip(weights, candidate_states):
            agg = agg + weight * state[key]
        out[key] = agg
    return out


def score_state(model: nn.Module, state: dict[str, object], spec: DomainSpec, seed: int, device: str) -> float:
    load_state(model, state)
    return eval_loss(model, spec, seed, PROBE_BATCH_SIZE, device)


def run_internal(seed: int, spec: DomainSpec, arms: tuple[Arm, ...], device: str, mode: str) -> tuple[float, float]:
    torch.manual_seed(seed)
    model = spec.make_model(device).to(device)
    workers = [copy.deepcopy(model).to(device) for _arm in arms]
    opts = [
        make_optimizer(worker, BASE_LR * arm.lr_mult) for worker, arm in zip(workers, arms)
    ]
    initial = eval_loss(model, spec, seed + 900_000, PROBE_BATCH_SIZE, device)
    for round_idx in range(ROUNDS):
        base_state = clone_state(model)
        target_phase = round_idx >= SWITCH_ROUND
        batch = spec.make_batch(seed + 1000 * round_idx, BATCH_SIZE, target_phase, device)
        candidate_states = [
            train_worker_candidate(worker, opt, base_state, batch, spec.task)
            for worker, opt in zip(workers, opts)
        ]
        losses = [
            score_state(model, state, spec, seed + 200_000 + round_idx, device)
            for state in candidate_states
        ]
        if mode == "shuffled":
            shift = (seed + round_idx) % len(losses)
            losses = losses[shift:] + losses[:shift]
        elif mode == "oracle":
            losses = [
                score_state(model, state, spec, seed + 300_000 + round_idx, device)
                for state in candidate_states
            ]
        load_state(model, aggregate_states(base_state, candidate_states, weights_from_losses(losses)))
    final = eval_loss(model, spec, seed + 901_000, PROBE_BATCH_SIZE, device)
    return initial, initial - final


def run_fixed_arm(seed: int, spec: DomainSpec, arm: Arm, device: str) -> tuple[float, float]:
    torch.manual_seed(seed)
    model = spec.make_model(device).to(device)
    opt = make_optimizer(model, BASE_LR * arm.lr_mult)
    initial = eval_loss(model, spec, seed + 900_000, PROBE_BATCH_SIZE, device)
    for round_idx in range(ROUNDS):
        target_phase = round_idx >= SWITCH_ROUND
        batch = spec.make_batch(seed + 1000 * round_idx, BATCH_SIZE, target_phase, device)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_on_batch(model, batch, spec.task)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    select = eval_loss(model, spec, seed + 902_000, PROBE_BATCH_SIZE, device)
    target = eval_loss(model, spec, seed + 901_000, PROBE_BATCH_SIZE, device)
    return initial - select, initial - target


def run_external(seed: int, spec: DomainSpec, arms: tuple[Arm, ...], device: str) -> tuple[float, float, float]:
    scores = [run_fixed_arm(seed, spec, arm, device) for arm in arms]
    valid_idx = max(range(len(scores)), key=lambda idx: scores[idx][0])
    oracle_idx = max(range(len(scores)), key=lambda idx: scores[idx][1])
    default_idx = min(range(len(arms)), key=lambda idx: abs(arms[idx].lr_mult - 1.0))
    return scores[default_idx][1], scores[valid_idx][1], scores[oracle_idx][1]


def diagnose_domain(spec: DomainSpec, device: str) -> DomainScores:
    arms = tuple(Arm(f"lr_x{mult:g}", mult) for mult in LR_MULTS)
    initial_loss: list[float] = []
    default: list[float] = []
    internal_probe: list[float] = []
    internal_shuffled: list[float] = []
    internal_oracle: list[float] = []
    external_valid: list[float] = []
    external_oracle: list[float] = []
    for idx, seed in enumerate(HELDOUT_SEEDS):
        initial, probe_score = run_internal(seed, spec, arms, device, "probe")
        _initial_s, shuffled_score = run_internal(seed, spec, arms, device, "shuffled")
        _initial_o, oracle_score = run_internal(seed, spec, arms, device, "oracle")
        default_score, valid_score, oracle_ext_score = run_external(seed, spec, arms, device)
        initial_loss.append(initial)
        default.append(default_score)
        internal_probe.append(probe_score)
        internal_shuffled.append(shuffled_score)
        internal_oracle.append(oracle_score)
        external_valid.append(valid_score)
        external_oracle.append(oracle_ext_score)
        print(
            f"DOMAIN {spec.name} REP {idx:02d} seed={seed} "
            f"default={default_score:.4f} internal={probe_score:.4f} "
            f"shuffled={shuffled_score:.4f} external_valid={valid_score:.4f}",
            flush=True,
        )
    probe_cost = 100.0 * PROBE_BATCH_SIZE / (3.0 * BATCH_SIZE)
    return DomainScores(
        name=spec.name,
        initial_loss=initial_loss,
        default=default,
        internal_probe=internal_probe,
        internal_shuffled=internal_shuffled,
        internal_oracle=internal_oracle,
        external_valid=external_valid,
        external_oracle=external_oracle,
        probe_cost_pct=probe_cost,
    )


def decide(scores: DomainScores, status: str) -> str:
    if status == "NO_TORCH":
        return "NO_TORCH"
    internal_default = paired(scores.internal_probe, scores.default)
    internal_shuffled = paired(scores.internal_probe, scores.internal_shuffled)
    internal_external = paired(scores.internal_probe, scores.external_valid)
    if positive_ci(internal_default) and positive_ci(internal_shuffled) and positive_ci(internal_external):
        return "MULTIDOMAIN_SHIFT_HPO_WIN"
    if positive_ci(internal_default) and positive_ci(internal_shuffled) and mean(internal_external) > 0.0:
        return "MULTIDOMAIN_SHIFT_HPO_LIVE"
    if positive_ci(internal_default) and positive_ci(internal_shuffled):
        return "MULTIDOMAIN_SHIFT_LABEL_CONTENT"
    return "MULTIDOMAIN_SHIFT_NOT_CLEARED"


def pooled(scores: list[DomainScores], attr: str, baseline: str) -> list[float]:
    out: list[float] = []
    for score in scores:
        out.extend(paired(getattr(score, attr), getattr(score, baseline)))
    return out


def no_torch_artifacts(detail: str) -> tuple[str, str, str]:
    run_log = "\n".join(
        (
            "DFW-E100 multidomain shift HPO",
            f"host={platform.node()} python={sys.executable}",
            f"DECISION NO_TORCH detail={detail}",
        )
    )
    result = "# DFW-E100 Result - Multidomain Shift HPO\n\nDecision label: `NO_TORCH`\n"
    legit = "# DFW-E100 Legitimacy Check\n\n- PyTorch workload: not run.\n"
    return run_log + "\n", result + "\n", legit + "\n"


def run_experiment() -> tuple[str, str, str]:
    status, detail, device = compute_status()
    if status == "NO_TORCH":
        return no_torch_artifacts(detail)
    torch.set_num_threads(int(os.environ.get("DFW_E100_TORCH_THREADS", "1")))
    scores = [diagnose_domain(spec, device) for spec in make_domains()]
    pooled_default = pooled(scores, "internal_probe", "default")
    pooled_shuffled = pooled(scores, "internal_probe", "internal_shuffled")
    pooled_external = pooled(scores, "internal_probe", "external_valid")
    pooled_oracle = pooled(scores, "internal_probe", "external_oracle")
    decision = "MULTIDOMAIN_SHIFT_HPO_WIN" if (
        positive_ci(pooled_default) and positive_ci(pooled_shuffled) and positive_ci(pooled_external)
    ) else "MULTIDOMAIN_SHIFT_NOT_CLEARED"
    arms = ", ".join(f"{mult:g}" for mult in LR_MULTS)
    run_lines = [
        "DFW-E100 multidomain shift HPO",
        f"host={platform.node()} python={sys.executable} compute_status={status} {detail}",
        (
            f"rounds={ROUNDS} switch_round={SWITCH_ROUND} batch={BATCH_SIZE} "
            f"probe_batch={PROBE_BATCH_SIZE} eval_batches={EVAL_BATCHES} policy={POLICY}"
        ),
        (
            f"optimizer={OPTIMIZER} base_lr={BASE_LR} weight_decay={WEIGHT_DECAY} "
            f"lr_mults=({arms}) seeds={len(HELDOUT_SEEDS)}"
        ),
        f"POOLED internal-default={fmt_ci(pooled_default)} "
        f"internal-shuffled={fmt_ci(pooled_shuffled)} "
        f"internal-external_valid={fmt_ci(pooled_external)} "
        f"internal-external_oracle={fmt_ci(pooled_oracle)}",
        f"DECISION {decision}",
        f"compute_status={status}",
    ]
    lines = result_lines(scores, status, detail, decision, pooled_default, pooled_shuffled, pooled_external, pooled_oracle)
    legit = legitimacy_text(status, detail)
    return "\n".join(run_lines) + "\n", "\n".join(lines) + "\n", legit


def result_lines(
    scores: list[DomainScores],
    status: str,
    detail: str,
    decision: str,
    pooled_default: list[float],
    pooled_shuffled: list[float],
    pooled_external: list[float],
    pooled_oracle: list[float],
) -> list[str]:
    lines = [
        "# DFW-E100 Result - Multidomain Shift HPO",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Compute Status",
        "",
        f"`{status}`: `{detail}`.",
        "",
        "## Pooled",
        "",
        f"- internal minus default: `{fmt_ci(pooled_default).strip()}`",
        f"- internal minus shuffled: `{fmt_ci(pooled_shuffled).strip()}`",
        f"- internal minus external validation-selected HPO: `{fmt_ci(pooled_external).strip()}`",
        f"- internal minus external oracle HPO: `{fmt_ci(pooled_oracle).strip()}`",
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
                    decide(score, status),
                    fmt_ci(paired(score.internal_probe, score.default)).strip(),
                    fmt_ci(paired(score.internal_probe, score.internal_shuffled)).strip(),
                    fmt_ci(paired(score.internal_probe, score.external_valid)).strip(),
                    f"{score.probe_cost_pct:.2f}%",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    if decision == "MULTIDOMAIN_SHIFT_HPO_WIN":
        lines.append(
            "The multidomain shift audit clears the pooled fixed-compute comparator: "
            "same-round probe-weighted aggregation beats default controls, shuffled probes, and "
            "validation-selected fixed-LR external HPO across the domain suite."
        )
    else:
        lines.append("The multidomain shift audit is a boundary result; inspect per-domain rows.")
    return lines


def legitimacy_text(status: str, detail: str) -> str:
    return "\n".join(
        [
            "# DFW-E100 Legitimacy Check",
            "",
            f"- Compute: `{status}` / `{detail}`.",
            "- Domains: tabular classification, tabular regression, PFN-style tabular in-context classification, time-series forecasting, synthetic vision.",
            f"- Optimizer family: `{OPTIMIZER}` with LR multipliers `{', '.join(f'{mult:g}' for mult in LR_MULTS)}`.",
            "- Training stream: off-target or label-flipped before the switch, target-domain after the switch.",
            "- Probe/select and target batches are target-domain.",
            "- Same candidate-update budget: internal and external HPO both train one candidate update per LR arm per round.",
            f"- Internal HPO uses `{POLICY}` probe-weighted aggregation from one shared base state each round.",
            "- External HPO keeps separate fixed-LR trajectories and selects by held-out validation loss.",
            "- Shuffled control preserves candidate updates but permutes probe-to-arm labels.",
            "- This is a small synthetic domain-suite audit, not a replacement for large task-family experiments.",
            "",
        ]
    )


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
