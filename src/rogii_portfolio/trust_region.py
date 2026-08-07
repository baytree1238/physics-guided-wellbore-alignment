"""A capped correction from a stable parent toward a higher-variance expert."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrustRegionPolicy:
    """Selected blend weight and the component-risk terms used to choose it."""

    weight: float
    maximum_weight: float
    macro_weight: float
    cvar_weight: float
    l2: float
    cvar_fraction: float


def _component_mse_coefficients(
    frame: pd.DataFrame,
    *,
    parent: str,
    expert: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {"component", "truth", parent, expert}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"trust-region input is missing {sorted(missing)}")
    rows: list[tuple[float, float, float, int]] = []
    for _, local in frame.groupby("component", sort=True):
        residual = local["truth"].to_numpy(float) - local[parent].to_numpy(float)
        move = local[expert].to_numpy(float) - local[parent].to_numpy(float)
        rows.append(
            (
                float(np.mean(residual * residual)),
                float(np.mean(residual * move)),
                float(np.mean(move * move)),
                len(local),
            )
        )
    values = np.asarray(rows, float)
    return values[:, 0], values[:, 1], values[:, 2], values[:, 3]


def fit_trust_region_blend(
    frame: pd.DataFrame,
    *,
    parent: str = "sequential_final",
    expert: str = "ridge",
    maximum_weight: float = 0.20,
    macro_weight: float = 0.25,
    cvar_weight: float = 0.25,
    l2: float = 0.10,
    cvar_fraction: float = 0.10,
    grid_step: float = 0.005,
) -> TrustRegionPolicy:
    """Select one non-negative expert move using component-equal risk terms."""

    if not 0 <= maximum_weight <= 1:
        raise ValueError("maximum_weight must lie in [0, 1]")
    if not 0 < cvar_fraction <= 1:
        raise ValueError("cvar_fraction must lie in (0, 1]")
    if grid_step <= 0:
        raise ValueError("grid_step must be positive")
    a, b, c, count = _component_mse_coefficients(frame, parent=parent, expert=expert)
    grid = np.unique(np.r_[np.arange(0.0, maximum_weight + 0.5 * grid_step, grid_step), maximum_weight])
    grid = grid[(grid >= 0.0) & (grid <= maximum_weight)]
    mse = a[:, None] - 2.0 * b[:, None] * grid[None, :] + c[:, None] * np.square(grid)[None, :]
    mse = np.maximum(mse, 0.0)
    row = (count[:, None] * mse).sum(axis=0) / count.sum()
    macro = mse.mean(axis=0)
    worst_count = max(1, int(np.ceil(cvar_fraction * len(a))))
    cvar = np.sort(mse, axis=0)[-worst_count:, :].mean(axis=0)
    objective = (
        row / max(row[0], 1e-12)
        + macro_weight * macro / max(macro[0], 1e-12)
        + cvar_weight * cvar / max(cvar[0], 1e-12)
        + l2 * np.square(grid)
    )
    best = int(np.argmin(objective))
    return TrustRegionPolicy(
        weight=float(grid[best]),
        maximum_weight=float(maximum_weight),
        macro_weight=float(macro_weight),
        cvar_weight=float(cvar_weight),
        l2=float(l2),
        cvar_fraction=float(cvar_fraction),
    )


def apply_trust_region_blend(
    frame: pd.DataFrame,
    policy: TrustRegionPolicy,
    *,
    parent: str = "sequential_final",
    expert: str = "ridge",
) -> np.ndarray:
    base = frame[parent].to_numpy(float)
    return base + policy.weight * (frame[expert].to_numpy(float) - base)
