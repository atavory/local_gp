"""Small PyTorch implementation of probe-scored optimizer-arm aggregation.

The core object is intentionally narrow. It does not own a dataloader, a
distributed backend, or a training framework. It wraps one training round:

1. copy the current model into candidate optimizer arms;
2. train each candidate on the same mini-batch or local batch list;
3. score candidate next states on held-out probe batches;
4. select or weight candidate states and install the aggregate in the model.

In a real distributed job, candidate arms can live on different workers. This
single-process implementation is the reference recipe and a correctness aid.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal

import torch
import torch.nn as nn


Batch = Any
LossFn = Callable[[nn.Module, Batch], torch.Tensor]
OptimizerFactory = Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]
Selector = Literal["argmin", "softmax", "uniform"]


@dataclass(frozen=True)
class ArmSpec:
    """One optimizer/hyperparameter arm to try from the current model state.

    Args:
        name: Human-readable arm name, for diagnostics.
        make_optimizer: Function that builds a fresh optimizer for a candidate
            model's parameters.
        local_steps: Number of passes over the supplied train batches for this
            arm in one aggregation round.
    """

    name: str
    make_optimizer: OptimizerFactory
    local_steps: int = 1

    @staticmethod
    def sgd(name: str, lr: float, **kwargs: Any) -> "ArmSpec":
        """Convenience constructor for SGD arms."""

        def make_optimizer(params: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
            return torch.optim.SGD(params, lr=lr, **kwargs)

        return ArmSpec(name=name, make_optimizer=make_optimizer)

    @staticmethod
    def adamw(name: str, lr: float, **kwargs: Any) -> "ArmSpec":
        """Convenience constructor for AdamW arms."""

        def make_optimizer(params: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
            return torch.optim.AdamW(params, lr=lr, **kwargs)

        return ArmSpec(name=name, make_optimizer=make_optimizer)


@dataclass(frozen=True)
class CandidateResult:
    """Diagnostics for one candidate arm in one aggregation round."""

    arm: str
    train_loss: float
    probe_loss: float
    scored_probe_loss: float
    weight: float
    delta_norm: float


@dataclass(frozen=True)
class ProbeHPOResult:
    """Result returned by `ProbeHPOAggregator.step`."""

    candidates: tuple[CandidateResult, ...]
    selected_arm: str
    selector: str
    shuffled_probe_scores: bool


@dataclass
class ProbeHPOAggregator:
    """Probe-scored optimizer-arm aggregation for PyTorch modules.

    Args:
        arms: Optimizer arms to evaluate from the same base model state.
        loss_fn: Function returning a scalar loss for `(model, batch)`.
        selector: `"argmin"` for one-hot selection, `"softmax"` for a soft
            weighting, or `"uniform"` for a control.
        temperature: Softmax temperature in loss units. If omitted, the
            per-round probe-loss standard deviation is used.
        shrinkage: Mix candidate weights with a prior. `1.0` is the raw
            selector; smaller values conservatively shrink toward the prior.
        prior: Optional prior over arms. Defaults to uniform.
    """

    arms: tuple[ArmSpec, ...]
    loss_fn: LossFn
    selector: Selector = "argmin"
    temperature: float | None = None
    shrinkage: float = 1.0
    prior: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("at least one arm is required")
        if self.selector not in ("argmin", "softmax", "uniform"):
            raise ValueError("selector must be one of: argmin, softmax, uniform")
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        if self.prior is None:
            self.prior = tuple(1.0 / len(self.arms) for _ in self.arms)
        if len(self.prior) != len(self.arms):
            raise ValueError("prior length must match arms")
        total = sum(self.prior)
        if total <= 0.0:
            raise ValueError("prior must have positive mass")
        self.prior = tuple(float(p) / total for p in self.prior)
        for arm in self.arms:
            if arm.local_steps <= 0:
                raise ValueError("local_steps must be positive for every arm")

    def step(
        self,
        model: nn.Module,
        train_batches: Iterable[Batch],
        probe_batches: Iterable[Batch],
        *,
        shuffle_probe_scores: bool = False,
        generator: torch.Generator | None = None,
    ) -> ProbeHPOResult:
        """Run one HPO-aware aggregation round and update `model` in place.

        `shuffle_probe_scores=True` is the label-content control: candidate
        states are real, but probe scores are assigned to the wrong arm labels.
        """

        train_batch_list = list(train_batches)
        probe_batch_list = list(probe_batches)
        if not train_batch_list:
            raise ValueError("train_batches must not be empty")
        if not probe_batch_list:
            raise ValueError("probe_batches must not be empty")

        was_training = model.training
        base_state = _clone_state(model.state_dict())
        candidate_states: list[dict[str, torch.Tensor]] = []
        train_losses: list[float] = []
        probe_losses: list[float] = []
        delta_norms: list[float] = []

        for arm in self.arms:
            candidate = copy.deepcopy(model)
            candidate.load_state_dict(base_state, strict=True)
            optimizer = arm.make_optimizer(candidate.parameters())
            train_loss = _train_candidate(candidate, optimizer, self.loss_fn, train_batch_list, arm.local_steps)
            probe_loss = _evaluate(candidate, self.loss_fn, probe_batch_list)
            candidate_state = _clone_state(candidate.state_dict())
            candidate_states.append(candidate_state)
            train_losses.append(train_loss)
            probe_losses.append(probe_loss)
            delta_norms.append(_delta_norm(base_state, candidate_state))

        scored_losses = list(probe_losses)
        if shuffle_probe_scores and len(scored_losses) > 1:
            order = torch.randperm(len(scored_losses), generator=generator).tolist()
            scored_losses = [scored_losses[i] for i in order]

        weights = _selector_weights(
            scored_losses,
            selector=self.selector,
            temperature=self.temperature,
            shrinkage=self.shrinkage,
            prior=self.prior,
        )
        _load_weighted_state(model, base_state, candidate_states, weights)
        model.train(was_training)

        selected_idx = max(range(len(weights)), key=lambda idx: weights[idx])
        candidates = tuple(
            CandidateResult(
                arm=arm.name,
                train_loss=train_loss,
                probe_loss=probe_loss,
                scored_probe_loss=scored_loss,
                weight=weight,
                delta_norm=delta_norm,
            )
            for arm, train_loss, probe_loss, scored_loss, weight, delta_norm in zip(
                self.arms,
                train_losses,
                probe_losses,
                scored_losses,
                weights,
                delta_norms,
            )
        )
        return ProbeHPOResult(
            candidates=candidates,
            selected_arm=self.arms[selected_idx].name,
            selector=self.selector,
            shuffled_probe_scores=shuffle_probe_scores,
        )


def _train_candidate(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    batches: list[Batch],
    local_steps: int,
) -> float:
    model.train()
    losses: list[float] = []
    for _ in range(local_steps):
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses)


def _evaluate(model: nn.Module, loss_fn: LossFn, batches: list[Batch]) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in batches:
            losses.append(float(loss_fn(model, batch).detach().cpu()))
    model.train(was_training)
    return sum(losses) / len(losses)


def _clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in state.items()}


def _selector_weights(
    losses: list[float],
    *,
    selector: Selector,
    temperature: float | None,
    shrinkage: float,
    prior: tuple[float, ...],
) -> tuple[float, ...]:
    if selector == "uniform":
        q = torch.full((len(losses),), 1.0 / len(losses), dtype=torch.float64)
    elif selector == "argmin":
        q = torch.zeros(len(losses), dtype=torch.float64)
        q[int(torch.tensor(losses).argmin())] = 1.0
    else:
        loss_tensor = torch.tensor(losses, dtype=torch.float64)
        if temperature is None:
            tau = float(loss_tensor.std(unbiased=False).clamp_min(1e-8))
        else:
            tau = max(float(temperature), 1e-8)
        q = torch.softmax(-(loss_tensor - loss_tensor.min()) / tau, dim=0)

    prior_tensor = torch.tensor(prior, dtype=torch.float64)
    weights = (1.0 - shrinkage) * prior_tensor + shrinkage * q
    weights = weights / weights.sum().clamp_min(1e-12)
    return tuple(float(weight) for weight in weights)


def _load_weighted_state(
    model: nn.Module,
    base_state: Mapping[str, torch.Tensor],
    candidate_states: list[Mapping[str, torch.Tensor]],
    weights: tuple[float, ...],
) -> None:
    selected_idx = max(range(len(weights)), key=lambda idx: weights[idx])
    output: dict[str, torch.Tensor] = {}
    for key, base_value in base_state.items():
        if torch.is_floating_point(base_value) or torch.is_complex(base_value):
            value = base_value.detach().clone()
            for weight, candidate_state in zip(weights, candidate_states):
                candidate_value = candidate_state[key].to(device=value.device, dtype=value.dtype)
                value.add_(candidate_value - base_value, alpha=float(weight))
            output[key] = value
        else:
            output[key] = candidate_states[selected_idx][key].detach().clone()
    model.load_state_dict(output, strict=True)


def _delta_norm(base_state: Mapping[str, torch.Tensor], candidate_state: Mapping[str, torch.Tensor]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for key, base_value in base_state.items():
        if torch.is_floating_point(base_value) or torch.is_complex(base_value):
            diff = candidate_state[key].detach() - base_value.detach()
            total = total + diff.double().pow(2).sum().cpu()
    return float(total.sqrt())
