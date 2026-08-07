"""Fault-aware switching state-space fusion for trajectory experts.

This module is an opt-in research candidate.  It uses only paths already
available at prediction time and does not alter the frozen pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .hgrg import radial_project, rms


REGIME_NAMES = ("smooth", "fault_up", "fault_down", "uncertain")


@dataclass(frozen=True)
class SwitchingStatePolicy:
    """Fixed parameters for the four-regime filter.

    Transition probabilities apply between sampled rows.  Fault jumps are
    added only when a path enters a fault regime, so a persistent fault state
    does not accumulate the same displacement on every row.
    """

    transition_matrix: tuple[tuple[float, ...], ...] = (
        (0.965, 0.012, 0.012, 0.011),
        (0.580, 0.320, 0.005, 0.095),
        (0.580, 0.005, 0.320, 0.095),
        (0.280, 0.025, 0.025, 0.670),
    )
    initial_probabilities: tuple[float, ...] = (0.85, 0.05, 0.05, 0.05)
    process_jerk_variances: tuple[float, ...] = (0.03, 0.12, 0.12, 0.80)
    observation_inflation: tuple[float, ...] = (1.0, 1.35, 1.35, 4.0)
    expert_std_ft: tuple[float, float, float] = (3.0, 5.0, 8.0)
    pf_hmm_correlation: float = 0.20
    structural_correlation: float = 0.10
    structural_prefix_scale: float = 1.0
    fault_jump_ft: float = 6.0
    md_scale_ft: float = 100.0
    initial_position_std_ft: float = 6.0
    initial_rate_std_ft: float = 5.0
    initial_curvature_std_ft: float = 4.0
    stride_rows: int = 6
    boundary_ramp_ft: float = 75.0
    uncertain_shrink: float = 0.75
    expert_residual_cap_ft: float = 100.0
    latent_position_cap_ft: float = 100.0
    latent_rate_cap_ft: float = 40.0
    latent_curvature_cap_ft: float = 40.0
    movement_budget_ft: float = 5.0
    row_cap_ft: float = 10.0

    def validate(self) -> None:
        transition = np.asarray(self.transition_matrix, float)
        initial = np.asarray(self.initial_probabilities, float)
        process = np.asarray(self.process_jerk_variances, float)
        inflation = np.asarray(self.observation_inflation, float)
        if transition.shape != (4, 4) or np.any(transition < 0.0):
            raise ValueError("transition_matrix must be a non-negative 4 by 4 matrix")
        if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("transition_matrix rows must sum to one")
        if initial.shape != (4,) or np.any(initial < 0.0) or not np.isclose(initial.sum(), 1.0):
            raise ValueError("initial_probabilities must be a four-element simplex")
        if process.shape != (4,) or np.any(process <= 0.0):
            raise ValueError("process_jerk_variances must contain four positive values")
        if inflation.shape != (4,) or np.any(inflation <= 0.0):
            raise ValueError("observation_inflation must contain four positive values")
        if len(self.expert_std_ft) != 3 or np.any(np.asarray(self.expert_std_ft) <= 0.0):
            raise ValueError("expert_std_ft must contain three positive values")
        if not -0.49 < self.pf_hmm_correlation < 0.99:
            raise ValueError("pf_hmm_correlation is outside the supported range")
        if not -0.49 < self.structural_correlation < 0.99:
            raise ValueError("structural_correlation is outside the supported range")
        positive = (
            self.fault_jump_ft,
            self.md_scale_ft,
            self.initial_position_std_ft,
            self.initial_rate_std_ft,
            self.initial_curvature_std_ft,
            self.expert_residual_cap_ft,
            self.latent_position_cap_ft,
            self.latent_rate_cap_ft,
            self.latent_curvature_cap_ft,
            self.movement_budget_ft,
            self.row_cap_ft,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("scale and cap parameters must be positive")
        if self.stride_rows < 1:
            raise ValueError("stride_rows must be at least one")
        if self.boundary_ramp_ft < 0.0:
            raise ValueError("boundary_ramp_ft cannot be negative")
        if not 0.0 <= self.uncertain_shrink <= 1.0:
            raise ValueError("uncertain_shrink must lie in [0, 1]")


def _transition(dt: float, jerk_variance: float) -> tuple[np.ndarray, np.ndarray]:
    transition = np.array(
        [
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ]
    )
    process = jerk_variance * np.array(
        [
            [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
            [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
            [dt**3 / 6.0, dt**2 / 2.0, dt],
        ]
    )
    return transition, process


def _observation_covariance(prefix_cv: float, policy: SwitchingStatePolicy) -> np.ndarray:
    sigma = np.asarray(policy.expert_std_ft, float).copy()
    sigma[2] = np.hypot(sigma[2], policy.structural_prefix_scale * prefix_cv)
    correlation = np.array(
        [
            [1.0, policy.pf_hmm_correlation, policy.structural_correlation],
            [policy.pf_hmm_correlation, 1.0, policy.structural_correlation],
            [policy.structural_correlation, policy.structural_correlation, 1.0],
        ]
    )
    covariance = sigma[:, None] * correlation * sigma[None, :]
    if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
        raise ValueError("expert observation covariance is not positive definite")
    return covariance


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def _normal_weights(log_weights: np.ndarray) -> np.ndarray:
    return np.exp(log_weights - _logsumexp(log_weights))


def _measurement_update(
    mean: np.ndarray,
    covariance: np.ndarray,
    observation: np.ndarray,
    observation_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    measurement = np.zeros((3, 3), float)
    measurement[:, 0] = 1.0
    innovation = observation - measurement @ mean
    innovation_covariance = measurement @ covariance @ measurement.T + observation_covariance
    innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.T)
    sign, log_determinant = np.linalg.slogdet(innovation_covariance)
    if sign <= 0.0:
        raise RuntimeError("non-positive innovation covariance")
    solved = np.linalg.solve(innovation_covariance, innovation)
    gain = np.linalg.solve(innovation_covariance, measurement @ covariance).T
    updated_mean = mean + gain @ innovation
    identity_minus_gain = np.eye(3) - gain @ measurement
    updated_covariance = (
        identity_minus_gain @ covariance @ identity_minus_gain.T
        + gain @ observation_covariance @ gain.T
    )
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
    log_likelihood = -0.5 * (
        3.0 * np.log(2.0 * np.pi) + log_determinant + float(innovation @ solved)
    )
    return updated_mean, updated_covariance, float(log_likelihood)


def _clip_state(mean: np.ndarray, policy: SwitchingStatePolicy) -> tuple[np.ndarray, bool]:
    caps = np.array(
        [
            policy.latent_position_cap_ft,
            policy.latent_rate_cap_ft,
            policy.latent_curvature_cap_ft,
        ]
    )
    clipped = np.clip(mean, -caps, caps)
    return clipped, bool(np.any(clipped != mean))


def _validate_inputs(
    hgrg: np.ndarray,
    pf: np.ndarray,
    hmm: np.ndarray,
    structural: np.ndarray,
    horizon: np.ndarray,
    prefix_cv: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(value, float) for value in (hgrg, pf, hmm, structural, horizon))
    if any(value.ndim != 1 for value in arrays):
        raise ValueError("switching-state inputs must be one-dimensional")
    if len(arrays[0]) == 0 or len({len(value) for value in arrays}) != 1:
        raise ValueError("switching-state inputs are empty or not aligned")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("switching-state inputs must be finite")
    if arrays[4][0] < 0.0 or (len(arrays[4]) > 1 and np.any(np.diff(arrays[4]) <= 0.0)):
        raise ValueError("horizon must be non-negative and strictly increasing")
    if not np.isfinite(prefix_cv) or prefix_cv < 0.0:
        raise ValueError("prefix_cv must be finite and non-negative")
    return arrays


def apply_switching_state(
    *,
    hgrg: np.ndarray,
    pf: np.ndarray,
    hmm: np.ndarray,
    structural: np.ndarray,
    horizon: np.ndarray,
    prefix_cv: float,
    policy: SwitchingStatePolicy = SwitchingStatePolicy(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fuse three target-free expert paths with a fault-aware IMM approximation.

    Returns the capped candidate, a row-aligned ``[offset, rate, curvature]``
    state, row-aligned probabilities in ``REGIME_NAMES`` order, and scalar
    diagnostics.  Rate and curvature use 100-ft measured-depth units.
    """

    policy.validate()
    hgrg, pf, hmm, structural, horizon = _validate_inputs(
        hgrg, pf, hmm, structural, horizon, prefix_cv
    )
    take = np.unique(
        np.r_[np.arange(0, len(horizon), policy.stride_rows, dtype=int), len(horizon) - 1]
    )
    sampled_horizon = horizon[take]
    raw_observation = np.c_[pf[take] - hgrg[take], hmm[take] - hgrg[take], structural[take] - hgrg[take]]
    observation = np.clip(
        raw_observation,
        -policy.expert_residual_cap_ft,
        policy.expert_residual_cap_ft,
    )
    clipped_observations = int(np.count_nonzero(observation != raw_observation))

    transition_probability = np.asarray(policy.transition_matrix, float)
    mode_probability = np.asarray(policy.initial_probabilities, float).copy()
    base_observation_covariance = _observation_covariance(prefix_cv, policy)
    means = np.zeros((4, 3), float)
    initial_variance = np.square(
        [
            policy.initial_position_std_ft,
            policy.initial_rate_std_ft,
            policy.initial_curvature_std_ft,
        ]
    )
    covariances = np.repeat(np.diag(initial_variance)[None, :, :], 4, axis=0)
    state_at_sample = np.zeros((len(take), 3), float)
    probability_at_sample = np.zeros((len(take), 4), float)
    state_clip_count = 0
    tiny = np.finfo(float).tiny

    for row in range(len(take)):
        if row == 0:
            updated_means = np.empty_like(means)
            updated_covariances = np.empty_like(covariances)
            mode_logs = np.empty(4, float)
            for mode in range(4):
                covariance_r = base_observation_covariance * policy.observation_inflation[mode]
                updated_means[mode], updated_covariances[mode], likelihood = _measurement_update(
                    means[mode], covariances[mode], observation[row], covariance_r
                )
                updated_means[mode], clipped = _clip_state(updated_means[mode], policy)
                state_clip_count += int(clipped)
                mode_logs[mode] = np.log(max(mode_probability[mode], tiny)) + likelihood
            mode_probability = _normal_weights(mode_logs)
            means, covariances = updated_means, updated_covariances
        else:
            dt = float(sampled_horizon[row] - sampled_horizon[row - 1]) / policy.md_scale_ft
            branch_means = np.empty((4, 4, 3), float)
            branch_covariances = np.empty((4, 4, 3, 3), float)
            branch_logs = np.empty((4, 4), float)
            for previous_mode in range(4):
                for mode in range(4):
                    transition, process = _transition(dt, policy.process_jerk_variances[mode])
                    predicted_mean = transition @ means[previous_mode]
                    if mode == 1 and previous_mode != 1:
                        predicted_mean[0] += policy.fault_jump_ft
                    elif mode == 2 and previous_mode != 2:
                        predicted_mean[0] -= policy.fault_jump_ft
                    predicted_covariance = (
                        transition @ covariances[previous_mode] @ transition.T + process
                    )
                    covariance_r = base_observation_covariance * policy.observation_inflation[mode]
                    updated_mean, updated_covariance, likelihood = _measurement_update(
                        predicted_mean, predicted_covariance, observation[row], covariance_r
                    )
                    updated_mean, clipped = _clip_state(updated_mean, policy)
                    state_clip_count += int(clipped)
                    branch_means[previous_mode, mode] = updated_mean
                    branch_covariances[previous_mode, mode] = updated_covariance
                    branch_logs[previous_mode, mode] = (
                        np.log(max(mode_probability[previous_mode], tiny))
                        + np.log(max(transition_probability[previous_mode, mode], tiny))
                        + likelihood
                    )

            mode_logs = np.array([_logsumexp(branch_logs[:, mode]) for mode in range(4)])
            next_probability = _normal_weights(mode_logs)
            next_means = np.empty_like(means)
            next_covariances = np.empty_like(covariances)
            for mode in range(4):
                conditional = _normal_weights(branch_logs[:, mode])
                next_means[mode] = np.sum(conditional[:, None] * branch_means[:, mode], axis=0)
                covariance = np.zeros((3, 3), float)
                for previous_mode in range(4):
                    difference = branch_means[previous_mode, mode] - next_means[mode]
                    covariance += conditional[previous_mode] * (
                        branch_covariances[previous_mode, mode] + np.outer(difference, difference)
                    )
                next_covariances[mode] = 0.5 * (covariance + covariance.T)
            means, covariances, mode_probability = next_means, next_covariances, next_probability

        state_at_sample[row] = np.sum(mode_probability[:, None] * means, axis=0)
        probability_at_sample[row] = mode_probability

    state = np.column_stack(
        [np.interp(horizon, sampled_horizon, state_at_sample[:, column]) for column in range(3)]
    )
    regime_probability = np.column_stack(
        [
            np.interp(horizon, sampled_horizon, probability_at_sample[:, mode])
            for mode in range(4)
        ]
    )
    regime_probability /= regime_probability.sum(axis=1, keepdims=True)
    ramp = (
        np.ones(len(horizon), float)
        if policy.boundary_ramp_ft == 0.0
        else np.clip(horizon / policy.boundary_ramp_ft, 0.0, 1.0)
    )
    confidence = 1.0 - policy.uncertain_shrink * regime_probability[:, 3]
    raw_candidate = hgrg + ramp * confidence * state[:, 0]
    candidate, projection = radial_project(
        hgrg,
        raw_candidate,
        rms_cap=policy.movement_budget_ft,
        row_cap=policy.row_cap_ft,
    )
    mean_probability = regime_probability.mean(axis=0)
    diagnostics: dict[str, Any] = {
        "regime_names": list(REGIME_NAMES),
        "sampled_rows": int(len(take)),
        "input_rows": int(len(horizon)),
        "mean_regime_probability": {
            name: float(mean_probability[index]) for index, name in enumerate(REGIME_NAMES)
        },
        "peak_fault_up_probability": float(np.max(regime_probability[:, 1])),
        "peak_fault_down_probability": float(np.max(regime_probability[:, 2])),
        "peak_uncertain_probability": float(np.max(regime_probability[:, 3])),
        "prefix_cv_ft": float(prefix_cv),
        "structural_observation_std_ft": float(np.sqrt(base_observation_covariance[2, 2])),
        "clipped_expert_values": clipped_observations,
        "clipped_state_branches": state_clip_count,
        "raw_move_rms_ft": rms(raw_candidate - hgrg),
        "projection": projection,
        "move_rms_ft": rms(candidate - hgrg),
        "move_absmax_ft": float(np.max(np.abs(candidate - hgrg))),
    }
    return candidate, state, regime_probability, diagnostics
