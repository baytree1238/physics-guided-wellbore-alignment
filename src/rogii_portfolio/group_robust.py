"""Component-risk-aware convex blending of precomputed expert paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class RobustBlendPolicy:
    """Simplex-constrained expert weights selected under component risk."""

    arms: tuple[str, ...]
    weights: np.ndarray
    macro_weight: float
    cvar_weight: float
    ridge: float
    cvar_fraction: float

    @property
    def parent_weight(self) -> float:
        return float(max(0.0, 1.0 - np.sum(self.weights)))


@dataclass(frozen=True)
class _QuadraticGroups:
    gram: np.ndarray
    cross: np.ndarray
    constant: np.ndarray
    count: np.ndarray


def _group_quadratics(
    frame: pd.DataFrame,
    arms: tuple[str, ...],
    *,
    parent: str,
) -> _QuadraticGroups:
    required = {"component", "truth", parent, *arms}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"robust blend input is missing {sorted(missing)}")
    groups = []
    for _, local in frame.groupby("component", sort=True):
        base = local[parent].to_numpy(float)
        residual = local["truth"].to_numpy(float) - base
        moves = np.column_stack([local[arm].to_numpy(float) - base for arm in arms])
        n = len(local)
        groups.append(
            (
                moves.T @ moves / n,
                moves.T @ residual / n,
                float(residual @ residual / n),
                n,
            )
        )
    return _QuadraticGroups(
        gram=np.stack([item[0] for item in groups]),
        cross=np.stack([item[1] for item in groups]),
        constant=np.asarray([item[2] for item in groups], float),
        count=np.asarray([item[3] for item in groups], float),
    )


def _loss_and_gradient(
    weight: np.ndarray,
    groups: _QuadraticGroups,
    *,
    macro_weight: float,
    cvar_weight: float,
    ridge: float,
    cvar_fraction: float,
) -> tuple[float, np.ndarray]:
    mse = np.einsum("i,gij,j->g", weight, groups.gram, weight) - 2 * groups.cross @ weight + groups.constant
    gradients = 2 * np.einsum("gij,j->gi", groups.gram, weight) - 2 * groups.cross

    row_weight = groups.count / groups.count.sum()
    value = float(row_weight @ mse)
    gradient = row_weight @ gradients

    if macro_weight:
        value += float(macro_weight * np.mean(mse))
        gradient = gradient + macro_weight * np.mean(gradients, axis=0)

    if cvar_weight:
        worst_count = max(1, int(np.ceil(cvar_fraction * len(mse))))
        worst = np.argsort(mse, kind="mergesort")[-worst_count:]
        value += float(cvar_weight * np.mean(mse[worst]))
        gradient = gradient + cvar_weight * np.mean(gradients[worst], axis=0)

    value += float(ridge * (weight @ weight))
    gradient = gradient + 2 * ridge * weight
    return value, np.asarray(gradient, float)


def fit_group_robust_blend(
    frame: pd.DataFrame,
    arms: tuple[str, ...],
    *,
    parent: str = "incumbent",
    macro_weight: float = 0.5,
    cvar_weight: float = 0.5,
    ridge: float = 0.02,
    cvar_fraction: float = 0.10,
) -> RobustBlendPolicy:
    """Fit non-negative expert weights with sum at most one."""

    if not 0 < cvar_fraction <= 1:
        raise ValueError("cvar_fraction must lie in (0, 1]")
    groups = _group_quadratics(frame, arms, parent=parent)

    def objective(weight: np.ndarray) -> float:
        return _loss_and_gradient(
            weight,
            groups,
            macro_weight=macro_weight,
            cvar_weight=cvar_weight,
            ridge=ridge,
            cvar_fraction=cvar_fraction,
        )[0]

    def gradient(weight: np.ndarray) -> np.ndarray:
        return _loss_and_gradient(
            weight,
            groups,
            macro_weight=macro_weight,
            cvar_weight=cvar_weight,
            ridge=ridge,
            cvar_fraction=cvar_fraction,
        )[1]

    starts = [
        np.zeros(len(arms)),
        np.full(len(arms), 1.0 / (len(arms) + 1.0)),
        *np.eye(len(arms)),
    ]
    candidates: list[tuple[float, np.ndarray]] = []
    for initial in starts:
        result = minimize(
            objective,
            initial,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(arms),
            constraints={
                "type": "ineq",
                "fun": lambda weight: 1.0 - float(np.sum(weight)),
                "jac": lambda weight: -np.ones_like(weight),
            },
            options={"ftol": 1e-10, "maxiter": 1000, "disp": False},
        )
        weight = np.asarray(result.x, float)
        feasible = (
            np.isfinite(weight).all()
            and np.min(weight) >= -1e-7
            and np.sum(weight) <= 1.0 + 1e-7
        )
        if feasible:
            weight = np.clip(weight, 0.0, 1.0)
            candidates.append((objective(weight), weight))
    if not candidates:
        raise RuntimeError("group-robust optimizer returned no feasible solution")
    _, weights = min(candidates, key=lambda item: item[0])
    if weights.sum() > 1.0:
        weights = weights / weights.sum()
    return RobustBlendPolicy(
        arms=arms,
        weights=weights,
        macro_weight=float(macro_weight),
        cvar_weight=float(cvar_weight),
        ridge=float(ridge),
        cvar_fraction=float(cvar_fraction),
    )


def apply_group_robust_blend(
    frame: pd.DataFrame,
    policy: RobustBlendPolicy,
    *,
    parent: str = "incumbent",
) -> np.ndarray:
    base = frame[parent].to_numpy(float)
    prediction = policy.parent_weight * base
    for weight, arm in zip(policy.weights, policy.arms):
        prediction = prediction + float(weight) * frame[arm].to_numpy(float)
    return prediction
