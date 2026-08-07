"""Hierarchical Geology Regret Gate and movement projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rms(values: np.ndarray) -> float:
    """Return the root mean square of a one-dimensional movement vector."""

    return float(np.sqrt(np.mean(np.square(np.asarray(values, float)))))


@dataclass(frozen=True)
class HGRGPolicy:
    """Frozen gates and movement limits for the HGRG correction."""

    beta: float = 0.5
    movement_budget_ft: float = 2.5
    consensus_strength: float = 2.0
    uncertainty_threshold: float = 1.0
    temperature: float = 2.0
    shrink_floor: float = 0.25
    maximum_weight: float = 0.5
    ramp_ft: float = 250.0
    row_cap_ft: float = 10.0


def apply_hgrg(
    *,
    base: np.ndarray,
    ridge: np.ndarray,
    pf: np.ndarray,
    hmm: np.ndarray,
    horizon: np.ndarray,
    policy: HGRGPolicy = HGRGPolicy(),
) -> tuple[np.ndarray, dict[str, float]]:
    """Move the base trajectory toward the PF/HMM bridge under movement caps."""
    base, ridge, pf, hmm, horizon = map(lambda x: np.asarray(x, float), (base, ridge, pf, hmm, horizon))
    if len({len(base), len(ridge), len(pf), len(hmm), len(horizon)}) != 1:
        raise ValueError("HGRG inputs are not aligned")
    q = pf + policy.beta * (hmm - pf)
    direction = q - base
    direction_rms = rms(direction)
    dispersion = rms(hmm - pf)
    relative_dispersion = dispersion / max(direction_rms, 1e-12)
    projection = float(np.dot(direction, ridge - base) / max(np.dot(direction, direction), 1e-12))
    risk = relative_dispersion * np.exp(-policy.consensus_strength * np.clip(projection, -1.0, 1.0))
    raw_gate = min(1.0, (policy.uncertainty_threshold / max(risk, 1e-12)) ** policy.temperature)
    gate = policy.shrink_floor + (1.0 - policy.shrink_floor) * raw_gate
    budget = min(policy.maximum_weight, policy.movement_budget_ft / max(direction_rms, 1e-12))
    coefficient = float(gate * budget)
    ramp = np.clip(horizon / policy.ramp_ft, 0.0, 1.0)
    move = np.clip(ramp * coefficient * direction, -policy.row_cap_ft, policy.row_cap_ft)
    return base + move, {
        "direction_rms_ft": direction_rms,
        "pf_hmm_rms_ft": dispersion,
        "relative_dispersion": relative_dispersion,
        "consensus_projection": projection,
        "risk": float(risk),
        "gate": float(gate),
        "coefficient": coefficient,
        "move_rms_ft": rms(move),
        "move_absmax_ft": float(np.max(np.abs(move))),
    }


def radial_project(base: np.ndarray, candidate: np.ndarray, *, rms_cap: float, row_cap: float) -> tuple[np.ndarray, float]:
    """Radially scale a move into the RMS/row-cap intersection.

    This preserves the direction of the proposed move.  It is intentionally
    retained for historical parity; it is not the Euclidean projection onto
    the intersection of the two constraint sets.
    """
    move = np.asarray(candidate, float) - np.asarray(base, float)
    scale = min(1.0, rms_cap / max(rms(move), 1e-12), row_cap / max(float(np.max(np.abs(move))), 1e-12))
    return np.asarray(base, float) + scale * move, float(scale)


def exact_l2_linf_project(
    base: np.ndarray,
    candidate: np.ndarray,
    *,
    rms_cap: float,
    row_cap: float,
    iterations: int = 80,
) -> np.ndarray:
    """Euclidean projection of a move onto RMS and coordinatewise caps."""

    base = np.asarray(base, float)
    move = np.asarray(candidate, float) - base
    if base.shape != move.shape or move.ndim != 1:
        raise ValueError("base and candidate must be aligned one-dimensional arrays")
    if rms_cap < 0 or row_cap < 0:
        raise ValueError("projection caps must be non-negative")
    clipped = np.clip(move, -row_cap, row_cap)
    radius = float(rms_cap * np.sqrt(len(move)))
    if np.linalg.norm(clipped) <= radius:
        return base + clipped
    if radius == 0:
        return base.copy()

    magnitude = np.abs(move)
    lower, upper = 0.0, 1.0
    while np.linalg.norm(np.minimum(row_cap, magnitude / (1.0 + upper))) > radius:
        upper *= 2.0
    for _ in range(iterations):
        middle = 0.5 * (lower + upper)
        projected = np.minimum(row_cap, magnitude / (1.0 + middle))
        if np.linalg.norm(projected) > radius:
            lower = middle
        else:
            upper = middle
    projected = np.sign(move) * np.minimum(row_cap, magnitude / (1.0 + upper))
    return base + projected
