"""Practitioner-facing PyTorch demo for HPO-aware aggregation.

The demo is intentionally a normal PyTorch training loop. The stream contains
clean target batches mixed with off-target/noisy batches. A fixed external HPO
run can choose one learning rate for the whole stream; internal probe-HPO can
choose to move aggressively on useful batches and skip harmful batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import os
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from hpo_aware_aggregation import ArmSpec, ProbeHPOAggregator


@dataclass(frozen=True)
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    kind: str


class TinyClassifier(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def target_labels(x: torch.Tensor) -> torch.Tensor:
    score = x[:, 0] + 0.65 * x[:, 1] + 0.35 * torch.sin(2.5 * x[:, 0])
    return (score > 0.0).long()


def make_batch(seed: int, size: int, kind: str, device: torch.device) -> Batch:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn(size, 2, generator=gen)
    if kind == "clean":
        y = target_labels(x)
    elif kind == "shifted":
        shifted = x + torch.tensor([1.2, -0.8])
        y = target_labels(shifted)
    elif kind == "anti":
        y = 1 - target_labels(x)
    else:
        raise ValueError(f"unknown batch kind: {kind}")
    return Batch(x=x.to(device), y=y.to(device), kind=kind)


def stream_kind(round_idx: int) -> str:
    # Deterministic, nonstationary data quality. The useful action changes by
    # round: move on clean/shifted batches, usually skip anti-target batches.
    if round_idx % 10 in (3, 7):
        return "anti"
    if round_idx % 10 in (4, 8):
        return "shifted"
    return "clean"


def loss_fn(model: nn.Module, batch: Batch) -> torch.Tensor:
    return F.cross_entropy(model(batch.x), batch.y)


@torch.no_grad()
def accuracy(model: nn.Module, batches: Iterable[Batch]) -> float:
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for batch in batches:
        pred = model(batch.x).argmax(dim=-1)
        correct += int((pred == batch.y).sum().item())
        total += int(batch.y.numel())
    model.train(was_training)
    return correct / max(total, 1)


@torch.no_grad()
def mean_loss(model: nn.Module, batches: Iterable[Batch]) -> float:
    was_training = model.training
    model.eval()
    losses = [float(loss_fn(model, batch).detach().cpu()) for batch in batches]
    model.train(was_training)
    return sum(losses) / len(losses)


def train_fixed(
    initial_state: dict[str, torch.Tensor],
    lr: float,
    train_stream: list[Batch],
    probe_stream: list[Batch],
    eval_batches: list[Batch],
    *,
    device: torch.device,
) -> dict[str, float | str]:
    model = TinyClassifier().to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for batch in train_stream:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch)
        loss.backward()
        optimizer.step()
    return {
        "name": f"fixed_lr={lr:g}",
        "lr": lr,
        "probe_loss": mean_loss(model, probe_stream),
        "eval_loss": mean_loss(model, eval_batches),
        "eval_acc": accuracy(model, eval_batches),
    }


def train_internal(
    initial_state: dict[str, torch.Tensor],
    train_stream: list[Batch],
    probe_stream: list[Batch],
    eval_batches: list[Batch],
    arms: tuple[ArmSpec, ...],
    *,
    device: torch.device,
    shuffle_probe_scores: bool,
    seed: int,
) -> dict[str, float | str | dict[str, int]]:
    model = TinyClassifier().to(device)
    model.load_state_dict(initial_state)
    selector = ProbeHPOAggregator(arms=arms, loss_fn=loss_fn, selector="argmin")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    counts: dict[str, int] = {arm.name: 0 for arm in arms}
    for round_idx, batch in enumerate(train_stream):
        result = selector.step(
            model,
            train_batches=[batch],
            probe_batches=[probe_stream[round_idx % len(probe_stream)]],
            shuffle_probe_scores=shuffle_probe_scores,
            generator=gen,
        )
        counts[result.selected_arm] += 1
    return {
        "name": "internal_probe_hpo_shuffled" if shuffle_probe_scores else "internal_probe_hpo",
        "probe_loss": mean_loss(model, probe_stream),
        "eval_loss": mean_loss(model, eval_batches),
        "eval_acc": accuracy(model, eval_batches),
        "selected_counts": counts,
    }


def summarize(rows: list[dict[str, float | str | dict[str, int]]]) -> str:
    lines = [
        "| method | probe loss | eval loss | eval acc | notes |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        notes = ""
        if "selected_counts" in row:
            notes = json.dumps(row["selected_counts"], sort_keys=True)
        elif "lr" in row:
            notes = f"lr={row['lr']}"
        lines.append(
            f"| {row['name']} | {float(row['probe_loss']):.4f} | "
            f"{float(row['eval_loss']):.4f} | {float(row['eval_acc']):.3f} | {notes} |"
        )
    return "\n".join(lines)


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) >= 2 and argv[0] == "--script":
        argv = argv[2:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=int(os.environ.get("PROBE_HPO_DEMO_ROUNDS", "90")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("PROBE_HPO_DEMO_BATCH_SIZE", "128")))
    parser.add_argument("--probe-batches", type=int, default=int(os.environ.get("PROBE_HPO_DEMO_PROBE_BATCHES", "16")))
    parser.add_argument("--eval-batches", type=int, default=int(os.environ.get("PROBE_HPO_DEMO_EVAL_BATCHES", "32")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("PROBE_HPO_DEMO_SEED", "12345")))
    parser.add_argument("--device", default=os.environ.get("PROBE_HPO_DEMO_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    json_out = os.environ.get("PROBE_HPO_DEMO_JSON_OUT")
    parser.add_argument("--json-out", type=Path, default=Path(json_out) if json_out else None)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    initial_model = TinyClassifier().to(device)
    initial_state = {key: value.detach().clone() for key, value in initial_model.state_dict().items()}

    train_stream = [
        make_batch(args.seed + 10_000 + idx, args.batch_size, stream_kind(idx), device)
        for idx in range(args.rounds)
    ]
    probe_stream = [
        make_batch(args.seed + 20_000 + idx, args.batch_size, "clean", device)
        for idx in range(args.probe_batches)
    ]
    eval_batches = [
        make_batch(args.seed + 30_000 + idx, args.batch_size, "clean", device)
        for idx in range(args.eval_batches)
    ]
    lrs = (0.0, 0.005, 0.02, 0.08, 0.25)
    arms = tuple(ArmSpec.sgd(name=f"sgd_lr={lr:g}", lr=lr, momentum=0.9) for lr in lrs)

    default = train_fixed(initial_state, 0.08, train_stream, probe_stream, eval_batches, device=device)
    fixed_rows = [
        train_fixed(initial_state, lr, train_stream, probe_stream, eval_batches, device=device)
        for lr in lrs
    ]
    external = min(fixed_rows, key=lambda row: float(row["probe_loss"]))
    external = dict(external)
    external["name"] = "external_fixed_lr_hpo"
    internal = train_internal(
        initial_state,
        train_stream,
        probe_stream,
        eval_batches,
        arms,
        device=device,
        shuffle_probe_scores=False,
        seed=args.seed + 1,
    )
    shuffled = train_internal(
        initial_state,
        train_stream,
        probe_stream,
        eval_batches,
        arms,
        device=device,
        shuffle_probe_scores=True,
        seed=args.seed + 2,
    )
    rows = [default, external, shuffled, internal]
    result = {
        "description": (
            "Clean target probe batches detect whether each training batch is useful. "
            "Internal HPO can skip anti-target batches with lr=0 while still moving on clean batches."
        ),
        "stream_counts": {kind: sum(1 for batch in train_stream if batch.kind == kind) for kind in ("clean", "shifted", "anti")},
        "rows": rows,
        "margins": {
            "internal_minus_default_eval_loss": float(internal["eval_loss"]) - float(default["eval_loss"]),
            "internal_minus_external_eval_loss": float(internal["eval_loss"]) - float(external["eval_loss"]),
            "internal_minus_shuffled_eval_loss": float(internal["eval_loss"]) - float(shuffled["eval_loss"]),
        },
    }
    print("# PyTorch Probe-HPO Demo")
    print()
    print(summarize(rows))
    print()
    print("Negative margins are better for internal probe-HPO:")
    for key, value in result["margins"].items():
        print(f"- {key}: {value:+.4f}")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
