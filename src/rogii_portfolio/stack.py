"""Convex residual stacking and component-bootstrap intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


STACK_ARMS = ("hgrg", "meta_state", "prefix_boundary", "sequential_final")


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Compute row-level root mean squared error."""

    return float(np.sqrt(np.mean(np.square(np.asarray(truth, float) - np.asarray(prediction, float)))))


def component_weights(component: np.ndarray) -> np.ndarray:
    """Return row weights that give every component equal total mass."""

    values, inverse, counts = np.unique(np.asarray(component).astype(str), return_inverse=True, return_counts=True)
    weight = 1.0 / (len(values) * counts[inverse].astype(float))
    return weight / weight.sum()


@dataclass(frozen=True)
class StackPolicy:
    """Frozen nonnegative weights for moves away from a parent prediction."""

    arms: tuple[str, ...]
    weights: np.ndarray
    ridge: float

    @property
    def parent_weight(self) -> float:
        return float(np.clip(1.0 - np.sum(self.weights), 0.0, 1.0))


def fit_convex_stack(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    component: np.ndarray,
    *,
    ridge: float,
    arms: tuple[str, ...] = STACK_ARMS,
) -> StackPolicy:
    """Fit non-negative move weights with sum <=1 on inner-OOF predictions."""
    parent = np.asarray(predictions["incumbent"], float)
    moves = np.column_stack([np.asarray(predictions[name], float) - parent for name in arms])
    residual = np.asarray(truth, float) - parent
    q = component_weights(component)
    gram = (moves * q[:, None]).T @ moves
    cross = (moves * q[:, None]).T @ residual

    def value(weight: np.ndarray) -> float:
        return float(weight @ gram @ weight - 2.0 * cross @ weight + ridge * (weight @ weight))

    def gradient(weight: np.ndarray) -> np.ndarray:
        return 2.0 * (gram @ weight - cross + ridge * weight)

    result = minimize(
        value,
        np.full(len(arms), 1.0 / (len(arms) + 1.0)),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(arms),
        constraints=[
            {
                "type": "ineq",
                "fun": lambda weight: 1.0 - float(np.sum(weight)),
                "jac": lambda weight: -np.ones_like(weight),
            }
        ],
        options={"ftol": 1e-12, "maxiter": 500, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"nested stack optimizer failed: {result.message}")
    weight = np.clip(np.asarray(result.x, float), 0.0, 1.0)
    if weight.sum() > 1.0 + 1e-8:
        raise RuntimeError("stack weight sum exceeds one")
    return StackPolicy(arms=arms, weights=weight, ridge=float(ridge))


def apply_stack(policy: StackPolicy, predictions: dict[str, np.ndarray]) -> np.ndarray:
    """Apply frozen residual weights to aligned candidate predictions."""

    parent = np.asarray(predictions["incumbent"], float)
    output = policy.parent_weight * parent
    for weight, arm in zip(policy.weights, policy.arms):
        output = output + float(weight) * np.asarray(predictions[arm], float)
    return output


def component_bootstrap(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    component: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    values = np.unique(np.asarray(component).astype(str))
    stats = []
    for value in values:
        mask = np.asarray(component).astype(str) == value
        stats.append(
            (
                int(mask.sum()),
                float(np.sum(np.square(np.asarray(truth)[mask] - np.asarray(reference)[mask]))),
                float(np.sum(np.square(np.asarray(truth)[mask] - np.asarray(candidate)[mask]))),
            )
        )
    stats = np.asarray(stats, float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(stats), size=(draws, len(stats)))
    sample = stats[index].sum(axis=1)
    gain = np.sqrt(sample[:, 1] / sample[:, 0]) - np.sqrt(sample[:, 2] / sample[:, 0])
    return {
        "components": int(len(values)),
        "draws": int(draws),
        "gain_mean": float(gain.mean()),
        "ci95_low": float(np.quantile(gain, 0.025)),
        "ci95_high": float(np.quantile(gain, 0.975)),
        "probability_positive": float(np.mean(gain > 0.0)),
    }
