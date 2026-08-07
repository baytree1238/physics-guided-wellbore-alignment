"""Meta-State: correlated expert fusion followed by constant-acceleration RTS smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .hgrg import exact_l2_linf_project, radial_project, rms


@dataclass(frozen=True)
class MetaStatePolicy:
    """Expert-covariance, smoother, and movement settings for Meta-State."""

    nominal_weights: tuple[float, float, float] = (0.65, 0.25, 0.10)
    pf_hmm_correlation: float = 0.20
    structural_correlation: float = 0.10
    covariance_shrinkage: float = 0.0
    constrained_weights: bool = False
    exact_projection: bool = False
    prefix_cv_scale_ft: float = 12.0
    jerk_variance: float = 0.05
    stride_rows: int = 20
    movement_budget_ft: float = 5.0
    row_cap_ft: float = 10.0


def constrained_gls_weights(covariance: np.ndarray) -> np.ndarray:
    """Solve min w'Σw on the probability simplex for a small expert set."""

    covariance = np.asarray(covariance, float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, atol=1e-10):
        raise ValueError("covariance must be finite and symmetric")
    n = len(covariance)
    best: tuple[float, np.ndarray] | None = None
    for size in range(1, n + 1):
        for active in combinations(range(n), size):
            index = np.asarray(active, int)
            block = covariance[np.ix_(index, index)]
            try:
                raw = np.linalg.solve(block, np.ones(size))
            except np.linalg.LinAlgError:
                continue
            denominator = float(np.sum(raw))
            if denominator <= 0:
                continue
            local = raw / denominator
            if np.min(local) < -1e-10:
                continue
            weight = np.zeros(n, float)
            weight[index] = np.maximum(local, 0.0)
            weight /= weight.sum()
            value = float(weight @ covariance @ weight)
            if best is None or value < best[0] - 1e-12:
                best = (value, weight)
    if best is None:
        raise np.linalg.LinAlgError("simplex GLS has no feasible active set")
    return best[1]


def correlated_gls(prefix_cv: float, policy: MetaStatePolicy) -> tuple[np.ndarray, float]:
    """Build correlated expert weights and return their fused variance."""

    nominal = np.asarray(policy.nominal_weights, float)
    sigma = 1.0 / np.sqrt(nominal)
    sigma[2] *= np.sqrt(1.0 + (prefix_cv / policy.prefix_cv_scale_ft) ** 2)
    corr = np.array(
        [
            [1.0, policy.pf_hmm_correlation, policy.structural_correlation],
            [policy.pf_hmm_correlation, 1.0, policy.structural_correlation],
            [policy.structural_correlation, policy.structural_correlation, 1.0],
        ]
    )
    covariance = sigma[:, None] * corr * sigma[None, :]
    shrinkage = float(np.clip(policy.covariance_shrinkage, 0.0, 1.0))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance))
    raw = np.linalg.solve(covariance, np.ones(3))
    if np.min(raw) >= 0 or not policy.constrained_weights:
        raw = np.maximum(raw, 0.0)
        weight = raw / raw.sum()
    else:
        weight = constrained_gls_weights(covariance)
    return weight, float(weight @ covariance @ weight)


def _transition(dt: float, jerk_variance: float) -> tuple[np.ndarray, np.ndarray]:
    f = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    q = jerk_variance * np.array(
        [
            [dt**5 / 20, dt**4 / 8, dt**3 / 6],
            [dt**4 / 8, dt**3 / 3, dt**2 / 2],
            [dt**3 / 6, dt**2 / 2, dt],
        ]
    )
    return f, q


def kalman_rts(horizon: np.ndarray, observation: np.ndarray, variance: float, policy: MetaStatePolicy) -> np.ndarray:
    """Smooth a structural observation with a constant-acceleration model."""

    horizon = np.asarray(horizon, float)
    observation = np.asarray(observation, float)
    take = np.unique(np.r_[np.arange(0, len(horizon), policy.stride_rows), len(horizon) - 1])
    u, z = horizon[take] / 100.0, observation[take]
    n = len(take)
    use = min(n, 8)
    design = np.c_[np.ones(use), u[:use] - u[0], 0.5 * np.square(u[:use] - u[0])]
    state = np.linalg.lstsq(design, z[:use], rcond=None)[0]
    covariance = np.diag([variance, 4 * variance, 16 * variance])
    filtered = np.zeros((n, 3)); filtered_cov = np.zeros((n, 3, 3))
    predicted = np.zeros((n, 3)); predicted_cov = np.zeros((n, 3, 3)); transitions = np.zeros((n, 3, 3))
    for index in range(n):
        if index:
            f, q = _transition(float(u[index] - u[index - 1]), policy.jerk_variance)
            state = f @ state
            covariance = f @ covariance @ f.T + q
            transitions[index] = f
        else:
            transitions[index] = np.eye(3)
        predicted[index], predicted_cov[index] = state, covariance
        gain = covariance[:, 0] / (covariance[0, 0] + variance)
        state = state + gain * (z[index] - state[0])
        covariance = covariance - np.outer(gain, covariance[0])
        covariance = 0.5 * (covariance + covariance.T)
        filtered[index], filtered_cov[index] = state, covariance
    smooth = filtered.copy()
    for index in range(n - 2, -1, -1):
        f = transitions[index + 1]
        gain = np.linalg.solve(predicted_cov[index + 1].T, (filtered_cov[index] @ f.T).T).T
        smooth[index] += gain @ (smooth[index + 1] - predicted[index + 1])
    return np.interp(horizon, horizon[take], smooth[:, 0])


def apply_meta_state(
    *,
    hgrg: np.ndarray,
    pf: np.ndarray,
    hmm: np.ndarray,
    structural: np.ndarray,
    horizon: np.ndarray,
    prefix_cv: float,
    policy: MetaStatePolicy = MetaStatePolicy(),
) -> tuple[np.ndarray, np.ndarray, dict[str, float | list[float]]]:
    """Fuse correlated experts, smooth them, and apply the frozen reliability map."""
    weight, variance = correlated_gls(prefix_cv, policy)
    experts = np.c_[pf, hmm, structural]
    state = kalman_rts(horizon, experts @ weight, variance, policy)
    state_move = state - hgrg
    ratio = rms(pf - hmm) / max(rms(state_move), 1e-12)
    reliability = 0.25 + 0.75 * min(1.0, max(ratio, 1e-12) ** -2)
    coefficient = reliability * min(0.32, 3.0 / max(rms(state_move), 1e-12))
    overlay = hgrg + coefficient * state_move
    scale = 0.5 + 1.5 / (1.0 + ratio**2)
    raw = hgrg + scale * (overlay - hgrg)
    if policy.exact_projection:
        candidate = exact_l2_linf_project(
            hgrg,
            raw,
            rms_cap=policy.movement_budget_ft,
            row_cap=policy.row_cap_ft,
        )
        raw_move = raw - hgrg
        projection = rms(candidate - hgrg) / max(rms(raw_move), 1e-12)
    else:
        candidate, projection = radial_project(
            hgrg,
            raw,
            rms_cap=policy.movement_budget_ft,
            row_cap=policy.row_cap_ft,
        )
    return candidate, state, {
        "pf_weight": float(weight[0]),
        "hmm_weight": float(weight[1]),
        "structural_weight": float(weight[2]),
        "effective_variance": variance,
        "dispersion_to_state_ratio": ratio,
        "reliability": reliability,
        "coefficient": coefficient,
        "meta_scale": scale,
        "projection": projection,
        "move_rms_ft": rms(candidate - hgrg),
    }
