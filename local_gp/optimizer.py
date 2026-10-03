"""Local-GP local-SGD aggregation prototype.

This module implements the first product-shaped object from the local-GP lane:
a server-side aggregator for K-step local-SGD trajectories.

It is not a standard `torch.optim.Optimizer`. A PyTorch optimizer only sees the
already-reduced gradient/update. This object needs the worker-local trajectory:
the displacement and gradient at each local step, plus the final worker endpoint.

The implemented rule is deliberately conservative:

1. Estimate a diagonal local curvature from within-round local trajectories.
2. Convert curvature to the displacement-law correction
      C_j = 1 / (1 - (1 - eta * lambda_j)^K).
3. Shrink the correction toward FedAvg using one scalar gamma:
      coeff_j = 1 + gamma * (C_j - 1).
4. Fall back to coeff_j = 1 for negative, unstable, or unidentifiable
   curvature directions.
5. Cross-fit by default: fit coefficients on half the workers and apply them to
   the other half's endpoint displacement, then symmetrise.

The gamma shrinkage is not cosmetic. LGP-E002 showed that the current measured
posterior readout is explained by this exact tuned-gamma control. Set
`gamma=None` to use the built-in split-half empirical-Bayes shrinkage rule
instead of a tuned scalar.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Iterable, Literal, Sequence


Vector = list[float]
Matrix = list[Vector]
Tensor3 = list[Matrix]
GainNormalization = Literal["none", "mean", "rms"]
Readout = Literal["posterior_ratio", "plug_in"]

_LOG_GRID_SIZE = 48


@dataclass(frozen=True)
class LocalGPConfig:
    """Configuration for `LocalGPAggregator`.

    Attributes:
        eta: Local worker learning rate used inside each K-step trajectory.
        local_steps: Number of local steps K.
        gamma: Scalar shrinkage toward FedAvg. `0` is FedAvg; `1` applies the
            full displacement-law correction. `None` estimates the shrinkage
            from split-half reliability of the correction vector on each round.
        shrinkage_groups: Number of reliability bins when `gamma=None`. `1`
            gives one scalar shrinkage; larger values give a small group-wise
            action layer over coefficient-magnitude bins.
        ridge: Small denominator ridge for the within-round slope fit.
        min_curvature: Curvatures at or below this value are treated as
            untrusted and receive coefficient 1.
        max_correction: Hard cap on the multiplicative correction. This is a
            safety cap, not the main regularizer.
        crossfit: Fit on half the workers and apply to the other half.
        gain_normalization: Optional normalisation of the corrected update's
            mean or RMS coefficient. Use this only when comparing direction
            shape at matched effective step; leave as "none" for deployment.
    """

    eta: float
    local_steps: int
    gamma: float | None = 0.8
    shrinkage_groups: int = 1
    ridge: float = 1e-12
    min_curvature: float = 1e-9
    max_correction: float = 25.0
    crossfit: bool = True
    gain_normalization: GainNormalization = "none"
    readout: Readout = "posterior_ratio"


@dataclass(frozen=True)
class LocalGPResult:
    """Aggregation result and diagnostics."""

    update: Vector
    fedavg_update: Vector
    coefficients: Vector
    mean_coefficient: float
    rms_coefficient: float
    fallback_fraction: float
    negative_or_unstable_fraction: float
    gamma: float


class LocalGPAggregator:
    """Aggregate K-step local-SGD worker trajectories into a server update."""

    def __init__(self, config: LocalGPConfig) -> None:
        if config.eta <= 0:
            raise ValueError("eta must be positive")
        if config.local_steps <= 0:
            raise ValueError("local_steps must be positive")
        if config.gamma is not None and config.gamma < 0:
            raise ValueError("gamma must be nonnegative")
        if config.shrinkage_groups <= 0:
            raise ValueError("shrinkage_groups must be positive")
        if config.gain_normalization not in ("none", "mean", "rms"):
            raise ValueError("gain_normalization must be one of: none, mean, rms")
        if config.readout not in ("posterior_ratio", "plug_in"):
            raise ValueError("readout must be one of: posterior_ratio, plug_in")
        self.config = config

    def aggregate(
        self,
        step_displacements: Tensor3,
        step_gradients: Tensor3,
        endpoint_displacements: Matrix,
    ) -> LocalGPResult:
        """Return a corrected server update.

        Args:
            step_displacements: `N x K x d` local displacements before each local
                step, relative to the synchronized start point.
            step_gradients: `N x K x d` stochastic gradients evaluated at the
                corresponding `step_displacements`.
            endpoint_displacements: `N x d` final worker displacements after K
                local steps.

        Returns:
            `LocalGPResult` with the corrected update and diagnostics.
        """

        n_workers, dim = _validate_shapes(
            step_displacements,
            step_gradients,
            endpoint_displacements,
            self.config.local_steps,
        )
        fedavg = _mean_rows(endpoint_displacements)

        if self.config.crossfit and n_workers >= 2:
            split = n_workers // 2
            idx_a = list(range(split))
            idx_b = list(range(split, n_workers))
            if self.config.gamma is None:
                raw_a, diag_a = self._fit_coefficients(
                    step_displacements,
                    step_gradients,
                    idx_a,
                    dim,
                    gamma_override=1.0,
                )
                raw_b, diag_b = self._fit_coefficients(
                    step_displacements,
                    step_gradients,
                    idx_b,
                    dim,
                    gamma_override=1.0,
                )
                gamma_by_dim = _split_half_group_gammas(
                    raw_a,
                    raw_b,
                    self.config.shrinkage_groups,
                )
                gamma = sum(gamma_by_dim) / len(gamma_by_dim)
                coeff_a = self._shrink_coefficients(raw_a, gamma_by_dim)
                coeff_b = self._shrink_coefficients(raw_b, gamma_by_dim)
            else:
                gamma = self.config.gamma
                coeff_a, diag_a = self._fit_coefficients(
                    step_displacements,
                    step_gradients,
                    idx_a,
                    dim,
                )
                coeff_b, diag_b = self._fit_coefficients(
                    step_displacements,
                    step_gradients,
                    idx_b,
                    dim,
                )
            end_a = _mean_rows([endpoint_displacements[i] for i in idx_a])
            end_b = _mean_rows([endpoint_displacements[i] for i in idx_b])
            update = [0.5 * (ca * db + cb * da) for ca, cb, da, db in zip(coeff_a, coeff_b, end_a, end_b)]
            coeff = [0.5 * (a + b) for a, b in zip(coeff_a, coeff_b)]
            fallback = 0.5 * (diag_a["fallback_fraction"] + diag_b["fallback_fraction"])
            unstable = 0.5 * (
                diag_a["negative_or_unstable_fraction"] + diag_b["negative_or_unstable_fraction"]
            )
        else:
            gamma = self.config.gamma if self.config.gamma is not None else 1.0
            coeff, diag = self._fit_coefficients(
                step_displacements,
                step_gradients,
                list(range(n_workers)),
                dim,
                gamma_override=gamma,
            )
            update = [c * u for c, u in zip(coeff, fedavg)]
            fallback = diag["fallback_fraction"]
            unstable = diag["negative_or_unstable_fraction"]

        update = _normalise_gain(update, fedavg, self.config.gain_normalization)
        mean_c = sum(coeff) / len(coeff)
        rms_c = (sum(c * c for c in coeff) / len(coeff)) ** 0.5
        return LocalGPResult(
            update=update,
            fedavg_update=fedavg,
            coefficients=coeff,
            mean_coefficient=mean_c,
            rms_coefficient=rms_c,
            fallback_fraction=fallback,
            negative_or_unstable_fraction=unstable,
            gamma=gamma,
        )

    def _fit_coefficients(
        self,
        step_displacements: Tensor3,
        step_gradients: Tensor3,
        worker_indices: Sequence[int],
        dim: int,
        gamma_override: float | None = None,
    ) -> tuple[Vector, dict[str, float]]:
        slopes: Vector = []
        slope_vars: Vector = []
        fallback = 0
        unstable = 0
        max_lam = (2.0 - 1e-9) / self.config.eta

        for j in range(dim):
            xs: list[float] = []
            gs: list[float] = []
            for i in worker_indices:
                for k in range(self.config.local_steps):
                    xs.append(step_displacements[i][k][j])
                    gs.append(step_gradients[i][k][j])

            lam, var = _centered_slope_and_var(xs, gs, self.config.ridge)
            slopes.append(lam)
            slope_vars.append(var)
            if lam <= self.config.min_curvature or lam >= max_lam or not isfinite(lam):
                unstable += 1

        if self.config.readout == "posterior_ratio":
            base_correction = self._posterior_ratio_coefficients(slopes, slope_vars, max_lam)
        else:
            base_correction = [self._plug_in_correction(lam, max_lam) for lam in slopes]

        gamma = self.config.gamma if gamma_override is None else gamma_override
        if gamma is None:
            gamma = 1.0
        coeff = self._shrink_coefficients(base_correction, gamma)
        for c in coeff:
            if c == 1.0:
                fallback += 1

        return coeff, {
            "fallback_fraction": fallback / dim,
            "negative_or_unstable_fraction": unstable / dim,
        }

    def _shrink_coefficients(self, base_correction: Sequence[float], gamma: float | Sequence[float]) -> Vector:
        if isinstance(gamma, float) or isinstance(gamma, int):
            gammas: Iterable[float] = [float(gamma)] * len(base_correction)
        else:
            gammas = gamma
        coeff = []
        for c0, gj in zip(base_correction, gammas):
            if (not isfinite(c0)) or c0 <= 0:
                c = 1.0
            else:
                c0 = min(max(c0, 1.0), self.config.max_correction)
                c = 1.0 + gj * (c0 - 1.0)
            coeff.append(c)
        return coeff

    def _plug_in_correction(self, lam: float, max_lam: float) -> float:
        if (not isfinite(lam)) or lam <= self.config.min_curvature or lam >= max_lam:
            return 1.0
        base = 1.0 - self.config.eta * lam
        denom = 1.0 - base ** self.config.local_steps
        if denom <= 1e-12 or not isfinite(denom):
            return 1.0
        return 1.0 / denom

    def _posterior_ratio_coefficients(
        self,
        slopes: Sequence[float],
        slope_vars: Sequence[float],
        max_lam: float,
    ) -> Vector:
        pos = [b for b in slopes if self.config.min_curvature < b < max_lam]
        if not pos:
            return [1.0] * len(slopes)
        mean_log = sum(log(b) for b in pos) / len(pos)
        var_log = max(1e-2, sum((log(b) - mean_log) ** 2 for b in pos) / len(pos))
        z0 = log(self.config.min_curvature)
        z1 = log(max_lam)
        z_grid = [z0 + i * (z1 - z0) / (_LOG_GRID_SIZE - 1) for i in range(_LOG_GRID_SIZE)]
        lam_grid = [exp(z) for z in z_grid]
        corr_grid = [self._plug_in_correction(lam, max_lam) for lam in lam_grid]
        lam_corr_grid = [lam * corr for lam, corr in zip(lam_grid, corr_grid)]
        out = []
        for bhat, var in zip(slopes, slope_vars):
            var = max(var, 1e-12)
            lw = [
                -((bhat - lam) ** 2) / (2.0 * var)
                - ((z - mean_log) ** 2) / (2.0 * var_log)
                for lam, z in zip(lam_grid, z_grid)
            ]
            mx = max(lw)
            w = [exp(a - mx) for a in lw]
            denom = sum(wi * lam for wi, lam in zip(w, lam_grid))
            if denom <= 0:
                out.append(1.0)
            else:
                out.append(sum(wi * lc for wi, lc in zip(w, lam_corr_grid)) / denom)
        return out


def _validate_shapes(
    step_displacements: Tensor3,
    step_gradients: Tensor3,
    endpoint_displacements: Matrix,
    local_steps: int,
) -> tuple[int, int]:
    if not step_displacements or not step_gradients or not endpoint_displacements:
        raise ValueError("inputs must be nonempty")
    n_workers = len(endpoint_displacements)
    if len(step_displacements) != n_workers or len(step_gradients) != n_workers:
        raise ValueError("worker count mismatch")
    dim = len(endpoint_displacements[0])
    if dim == 0:
        raise ValueError("dimension must be nonzero")
    for row in endpoint_displacements:
        if len(row) != dim:
            raise ValueError("endpoint dimension mismatch")
    for worker_steps, worker_grads in zip(step_displacements, step_gradients):
        if len(worker_steps) != local_steps or len(worker_grads) != local_steps:
            raise ValueError("local step count mismatch")
        for disp, grad in zip(worker_steps, worker_grads):
            if len(disp) != dim or len(grad) != dim:
                raise ValueError("trajectory dimension mismatch")
    return n_workers, dim


def _mean_rows(rows: Sequence[Sequence[float]]) -> Vector:
    dim = len(rows[0])
    return [sum(row[j] for row in rows) / len(rows) for j in range(dim)]


def _centered_slope_and_var(
    xs: Sequence[float],
    ys: Sequence[float],
    ridge: float,
) -> tuple[float, float]:
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    xx = sum((x - mx) ** 2 for x in xs)
    xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = xy / (xx + ridge)
    if len(xs) <= 2 or xx <= ridge:
        return slope, 1e12
    rss = sum((y - my - slope * (x - mx)) ** 2 for x, y in zip(xs, ys))
    noise_var = max(rss / max(1, len(xs) - 2), 1e-12)
    return slope, noise_var / (xx + ridge)


def _normalise_gain(update: Vector, reference: Vector, mode: GainNormalization) -> Vector:
    if mode == "none":
        return update
    if mode == "mean":
        num = sum(abs(x) for x in reference)
        den = sum(abs(x) for x in update)
    elif mode == "rms":
        num = sum(x * x for x in reference) ** 0.5
        den = sum(x * x for x in update) ** 0.5
    else:
        raise ValueError(f"unknown gain normalization: {mode}")
    if den <= 0:
        return update
    scale = num / den
    return [scale * x for x in update]


def _split_half_gamma(coeff_a: Sequence[float], coeff_b: Sequence[float]) -> float:
    """Estimate scalar shrinkage toward coefficient 1 from split-half agreement.

    Let x = coeff - 1 be the correction beyond FedAvg. Two independent worker
    halves give x_a = theta + eps_a and x_b = theta + eps_b. The average
    y=(x_a+x_b)/2 has noise energy E[(eps_a+eps_b)^2]/4, estimated by
    E[(x_a-x_b)^2]/4. Shrink by signal/(signal+noise).
    """

    if len(coeff_a) != len(coeff_b):
        raise ValueError("split-half coefficient length mismatch")
    if not coeff_a:
        return 0.0
    ys = [0.5 * ((a - 1.0) + (b - 1.0)) for a, b in zip(coeff_a, coeff_b)]
    noise = sum(((a - 1.0) - (b - 1.0)) ** 2 for a, b in zip(coeff_a, coeff_b))
    noise /= 4.0 * len(coeff_a)
    total = sum(y * y for y in ys) / len(ys)
    signal = max(0.0, total - noise)
    denom = signal + noise
    if denom <= 0 or not isfinite(denom):
        return 0.0
    return min(1.0, max(0.0, signal / denom))


def _split_half_group_gammas(
    coeff_a: Sequence[float],
    coeff_b: Sequence[float],
    groups: int,
) -> Vector:
    if groups <= 1 or len(coeff_a) < 2 * groups:
        g = _split_half_gamma(coeff_a, coeff_b)
        return [g] * len(coeff_a)
    order = sorted(
        range(len(coeff_a)),
        key=lambda j: 0.5 * ((coeff_a[j] - 1.0) + (coeff_b[j] - 1.0)),
    )
    gamma_by_dim = [0.0] * len(coeff_a)
    for group in range(groups):
        lo = group * len(order) // groups
        hi = (group + 1) * len(order) // groups
        idx = order[lo:hi]
        if not idx:
            continue
        g = _split_half_gamma([coeff_a[j] for j in idx], [coeff_b[j] for j in idx])
        for j in idx:
            gamma_by_dim[j] = g
    return gamma_by_dim
