"""No-prior GeoHMM smoother derived from the B=1 prototype.

It uses target-well MD, Z, GR and TVT_input together with type-well TVT and GR.
The transition and backward recursions match the prototype; the reported
spread is not a calibrated posterior interval.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


MODEL_COLUMNS = ("MD", "Z", "GR", "TVT_input")
TYPEWELL_COLUMNS = ("TVT", "GR")


class GeoHMMNoPriorConfig:
    grid_step = 0.5
    grid_margin = 25.0
    slope_bins = 25
    slope_max = 0.15
    slope_random_walk = 0.003
    slope_pull = 0.001
    slope_prior_sigma = 0.02
    value_blur = 0.005
    residual_correlation_cap = 25.0
    calibration_rows = 600
    emission_distance_cap = 64.0
    emission_floor = 0.004
    gr_gap_fill = 40
    gr_gap_weight = 0.5
    sigma_low = 5.0
    sigma_high = 40.0
    prefix_rows = 600
    prefix_clamp_sigma = 0.5
    stride = 2
    checkpoint = 48


def _validate_inputs(horizontal: pd.DataFrame, typewell: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_horizontal = set(MODEL_COLUMNS) - set(horizontal.columns)
    missing_typewell = set(TYPEWELL_COLUMNS) - set(typewell.columns)
    if missing_horizontal:
        raise ValueError(f"horizontal input is missing {sorted(missing_horizontal)}")
    if missing_typewell:
        raise ValueError(f"typewell input is missing {sorted(missing_typewell)}")

    hw = horizontal.loc[:, MODEL_COLUMNS].copy().reset_index(drop=True)
    tw = typewell.loc[:, TYPEWELL_COLUMNS].copy().sort_values("TVT", kind="mergesort").reset_index(drop=True)
    for column in MODEL_COLUMNS:
        hw[column] = pd.to_numeric(hw[column], errors="coerce")
    for column in TYPEWELL_COLUMNS:
        tw[column] = pd.to_numeric(tw[column], errors="coerce")

    if len(hw) == 0 or len(tw) < 4:
        raise ValueError("empty horizontal well or too-short typewell")
    md = hw["MD"].to_numpy(float)
    if not np.isfinite(md).all() or np.any(np.diff(md) <= 0.0):
        raise ValueError("MD must be finite and strictly increasing in source-row order")
    known = hw["TVT_input"].notna().to_numpy()
    if int(known.sum()) < 30 or int((~known).sum()) == 0:
        raise ValueError("GeoHMM requires at least 30 known-prefix rows and one evaluation row")

    tw = tw[np.isfinite(tw["TVT"].to_numpy(float))].copy().reset_index(drop=True)
    if len(tw) < 4 or np.any(np.diff(tw["TVT"].to_numpy(float)) <= 0.0):
        raise ValueError("typewell TVT must be finite and strictly increasing")
    return hw, tw


def _typewell_curve(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tvt = typewell["TVT"].to_numpy(float)
    gr = typewell["GR"].to_numpy(float)
    finite = np.isfinite(gr)
    if int(finite.sum()) < 2:
        gr = np.zeros_like(tvt)
    else:
        gr = np.interp(tvt, tvt[finite], gr[finite])
    return tvt, gr


def _offset_calibration(observed_gr: np.ndarray, reference_gr: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(observed_gr) & np.isfinite(reference_gr)
    if int(valid.sum()) < 20:
        return 1.0, 0.0
    return 1.0, float(np.median(observed_gr[valid] - reference_gr[valid]))


def _reference_pf_single(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    seed: int,
    particles: int,
) -> tuple[np.ndarray, float]:
    """Run one member of the reference comparison PF."""

    hw, tw = _validate_inputs(horizontal, typewell)
    tw_tvt = tw["TVT"].to_numpy(float)
    tw_gr_series = tw["GR"].astype(float)
    tw_gr = tw_gr_series.fillna(float(tw_gr_series.mean())).to_numpy(float)
    if not np.isfinite(tw_gr).all():
        tw_gr = np.nan_to_num(tw_gr, nan=0.0)

    known = hw[hw["TVT_input"].notna()]
    evaluation = hw[hw["TVT_input"].isna()]
    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_z = float(last["Z"])
    previous_md = float(last["MD"])
    typewell_at_known = np.interp(known["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    gr_sigma = float(
        np.clip(
            np.nanstd(known["GR"].fillna(0.0).to_numpy(float) - typewell_at_known),
            10.0,
            60.0,
        )
    )
    if not np.isfinite(gr_sigma) or gr_sigma <= 0.0:
        gr_sigma = 30.0

    tail = known.tail(30)
    delta_tvt = np.diff(tail["TVT_input"].to_numpy(float))
    delta_z = np.diff(tail["Z"].to_numpy(float))
    delta_md = np.diff(tail["MD"].to_numpy(float))
    usable = delta_md > 0.0
    initial_rate = (
        float(np.median((delta_tvt + delta_z)[usable] / delta_md[usable]))
        if int(usable.sum()) >= 3
        else 0.0
    )

    rng = np.random.default_rng(int(seed))
    count = int(particles)
    position = last_tvt + last_z + 4.5 * rng.standard_normal(count)
    rate = initial_rate + 0.01 * rng.standard_normal(count)
    weight = np.full(count, 1.0 / count)
    gr_interpolated = hw["GR"].interpolate(limit_direction="both").fillna(float(np.mean(tw_gr)))
    evaluation_md = evaluation["MD"].to_numpy(float)
    evaluation_z = evaluation["Z"].to_numpy(float)
    evaluation_gr = gr_interpolated.to_numpy(float)[evaluation.index.to_numpy(int)]
    prediction = np.empty(len(evaluation), dtype=float)
    log_likelihood = 0.0

    for index in range(len(evaluation)):
        md_step = max(float(evaluation_md[index]) - previous_md, 1.0)
        rate = 0.998 * rate + 0.002 * rng.standard_normal(count)
        position = position + rate * md_step + 0.005 * rng.standard_normal(count)
        tvt_particles = np.clip(position - evaluation_z[index], tw_tvt[0] - 100.0, tw_tvt[-1] + 100.0)
        position = tvt_particles + evaluation_z[index]
        expected_gr = np.interp(tvt_particles, tw_tvt, tw_gr)
        distance = (evaluation_gr[index] - expected_gr) / gr_sigma
        likelihood = np.maximum(np.exp(-0.5 * np.minimum(np.square(distance), 600.0)), 1e-300)
        average_likelihood = float(np.sum(weight * likelihood))
        log_likelihood += float(np.log(max(average_likelihood, 1e-300)))
        weight *= likelihood
        weight_sum = float(weight.sum())
        weight = weight / weight_sum if weight_sum > 0.0 else np.full(count, 1.0 / count)
        effective_count = 1.0 / float(np.sum(np.square(weight)))
        if effective_count < 0.5 * count:
            cumulative = np.cumsum(weight)
            offset = rng.uniform(0.0, 1.0 / count)
            selected = np.clip(np.searchsorted(cumulative, offset + np.arange(count) / count), 0, count - 1)
            position = position[selected] + 0.1 * rng.standard_normal(count)
            rate = rate[selected] + 0.001 * rng.standard_normal(count)
            weight = np.full(count, 1.0 / count)
        prediction[index] = float(np.dot(weight, position - evaluation_z[index]))
        previous_md = float(evaluation_md[index])
    return prediction, float(log_likelihood)


def geohmm_reference_pf(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    seeds: int = 8,
    likelihood_temperature: float = 8.0,
    particles: int = 500,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the exact PF reference used by the 40-well GeoHMM bridge.

    The returned array is full-row aligned: known rows retain ``TVT_input`` and
    suffix rows contain the likelihood-weighted ensemble prediction.
    """

    hw, tw = _validate_inputs(horizontal, typewell)
    if int(seeds) < 1 or int(particles) < 8 or float(likelihood_temperature) <= 0.0:
        raise ValueError("invalid reference-PF ensemble configuration")
    predictions = []
    log_likelihoods = []
    for seed in range(int(seeds)):
        prediction, log_likelihood = _reference_pf_single(
            hw,
            tw,
            seed=seed,
            particles=int(particles),
        )
        predictions.append(prediction)
        log_likelihoods.append(log_likelihood)
    prediction_matrix = np.stack(predictions, axis=0)
    log_likelihood_array = np.asarray(log_likelihoods, dtype=float)
    centered = log_likelihood_array - float(log_likelihood_array.max())
    ensemble_weight = np.exp(centered / float(likelihood_temperature))
    ensemble_weight /= ensemble_weight.sum()
    suffix_prediction = np.sum(ensemble_weight[:, None] * prediction_matrix, axis=0)
    full_prediction = hw["TVT_input"].to_numpy(float).copy()
    evaluation_mask = hw["TVT_input"].isna().to_numpy()
    full_prediction[evaluation_mask] = suffix_prediction
    if not np.isfinite(full_prediction[evaluation_mask]).all():
        raise RuntimeError("reference PF produced non-finite suffix predictions")
    return full_prediction, {
        "seeds": int(seeds),
        "particles": int(particles),
        "likelihood_temperature": float(likelihood_temperature),
        "weight_min": float(ensemble_weight.min()),
        "weight_max": float(ensemble_weight.max()),
        "spatial_prior_used": False,
        "allowed_horizontal_columns": list(MODEL_COLUMNS),
        "allowed_typewell_columns": list(TYPEWELL_COLUMNS),
    }


def geohmm_no_prior(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    config: type[GeoHMMNoPriorConfig] = GeoHMMNoPriorConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return full-row TVT means, suffix standard deviations, and diagnostics.

    Only ``MODEL_COLUMNS`` and ``TYPEWELL_COLUMNS`` are copied into the model.
    The function has no argument through which a spatial or target-derived prior
    can enter.
    """

    hw, tw = _validate_inputs(horizontal, typewell)
    tw_tvt, tw_gr = _typewell_curve(tw)
    known_mask = hw["TVT_input"].notna().to_numpy()
    known_index = np.flatnonzero(known_mask)
    evaluation_index = np.flatnonzero(~known_mask)
    output = hw["TVT_input"].to_numpy(float).copy()

    grid_step = float(config.grid_step)
    value_grid = np.arange(
        float(tw_tvt[0]) - float(config.grid_margin),
        float(tw_tvt[-1]) + float(config.grid_margin) + grid_step,
        grid_step,
    )
    slopes = np.linspace(-float(config.slope_max), float(config.slope_max), int(config.slope_bins))
    slope_count = len(slopes)
    value_count = len(value_grid)

    prefix_index = known_index[-int(config.prefix_rows) :]
    evaluation_use = np.unique(np.r_[evaluation_index[:: int(config.stride)], evaluation_index[-1]])
    use = np.unique(np.r_[prefix_index[:: int(config.stride)], evaluation_use])
    use.sort()

    md_all = hw["MD"].to_numpy(float)
    z_all = hw["Z"].to_numpy(float)
    md = md_all[use]
    z = z_all[use]
    gr_raw_series = hw["GR"].astype(float)
    gr_filled_series = gr_raw_series.interpolate(limit=int(config.gr_gap_fill), limit_area="inside")
    gr_raw = gr_raw_series.to_numpy(float)[use]
    gr = gr_filled_series.to_numpy(float)[use]
    observation_weight = np.where(
        np.isfinite(gr_raw),
        1.0,
        np.where(np.isfinite(gr), float(config.gr_gap_weight), 0.0),
    )
    tvt_input = hw["TVT_input"].to_numpy(float)[use]
    time_count = len(use)
    delta_md = np.clip(np.diff(md), 0.25, 12.0)
    delta_z = np.diff(z)

    known_frame = hw.loc[known_mask].tail(int(config.calibration_rows))
    known_gr = known_frame["GR"].to_numpy(float)
    reference_known_gr = np.interp(known_frame["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    calibration_gain, calibration_offset = _offset_calibration(known_gr, reference_known_gr)
    residual = known_gr - (calibration_gain * reference_known_gr + calibration_offset)
    residual_mad = float(np.nanmedian(np.abs(residual - np.nanmedian(residual)))) * 1.4826
    residual_fallback = float(np.nanstd(residual))
    residual_scale = residual_mad if np.isfinite(residual_mad) and residual_mad > 0.0 else residual_fallback
    if not np.isfinite(residual_scale) or residual_scale <= 0.0:
        residual_scale = float(config.sigma_high)
    residual_scale = float(np.clip(residual_scale, float(config.sigma_low), float(config.sigma_high)))
    reference_gr_grid = calibration_gain * np.interp(value_grid, tw_tvt, tw_gr) + calibration_offset

    residual_series = pd.Series(residual).interpolate(limit_direction="both").to_numpy(float)
    residual_series = residual_series - np.nanmean(residual_series)
    correlation_length = 1.0
    if len(residual_series) > 80 and float(np.nanstd(residual_series)) > 1e-6:
        variance = float(np.nanmean(residual_series * residual_series))
        for lag in range(1, 61):
            autocorrelation = float(np.nanmean(residual_series[:-lag] * residual_series[lag:])) / variance
            if autocorrelation <= 0.05:
                break
            correlation_length += 2.0 * autocorrelation
    correlation_length = float(np.clip(correlation_length, 1.0, float(config.residual_correlation_cap)))

    prefix_tail = hw.loc[known_mask].tail(200)
    delta_u = np.diff(prefix_tail["TVT_input"].to_numpy(float) + prefix_tail["Z"].to_numpy(float))
    delta_prefix_md = np.diff(prefix_tail["MD"].to_numpy(float))
    usable_slope = delta_prefix_md > 0.0
    initial_slope = (
        float(np.median(delta_u[usable_slope] / delta_prefix_md[usable_slope]))
        if int(usable_slope.sum()) >= 5
        else 0.0
    )
    initial_slope = float(np.clip(initial_slope, -0.8 * float(config.slope_max), 0.8 * float(config.slope_max)))
    slope_prior = np.exp(-0.5 * np.square((slopes - initial_slope) / float(config.slope_prior_sigma)))
    slope_prior = slope_prior / slope_prior.sum()

    def shift_slab(slab: np.ndarray, bins: int) -> np.ndarray:
        shifted = np.zeros_like(slab)
        if bins == 0:
            shifted[:] = slab
        elif bins > 0:
            if bins < value_count:
                shifted[bins:] = slab[:-bins]
        elif -bins < value_count:
            shifted[:bins] = slab[-bins:]
        return shifted

    def shift_rows(matrix: np.ndarray, md_step: float, z_step: float, sign: float) -> np.ndarray:
        shifted = np.empty_like(matrix)
        for slope_index in range(slope_count):
            distance = sign * (slopes[slope_index] * md_step - z_step) / grid_step
            lower_bin = int(math.floor(distance))
            fraction = distance - lower_bin
            shifted[slope_index] = (
                (1.0 - fraction) * shift_slab(matrix[slope_index], lower_bin)
                + fraction * shift_slab(matrix[slope_index], lower_bin + 1)
            )
        return shifted

    def value_blur(matrix: np.ndarray, steps: float) -> np.ndarray:
        weight = min(float(config.value_blur) * steps, 0.24)
        blurred = (1.0 - 2.0 * weight) * matrix
        blurred[:, 1:] += weight * matrix[:, :-1]
        blurred[:, :-1] += weight * matrix[:, 1:]
        return blurred

    def slope_mix(matrix: np.ndarray, steps: float) -> np.ndarray:
        walk = min(float(config.slope_random_walk) * steps, 0.45)
        pull = min(float(config.slope_pull) * steps, 0.1)
        mixed = (1.0 - 2.0 * walk - pull) * matrix
        mixed[1:] += walk * matrix[:-1]
        mixed[:-1] += walk * matrix[1:]
        mixed[0] += walk * matrix[0]
        mixed[-1] += walk * matrix[-1]
        mixed += pull * slope_prior[:, None] * matrix.sum(axis=0, keepdims=True)
        return mixed

    def emission(time_index: int) -> np.ndarray:
        likelihood = np.ones(value_count, dtype=float)
        observed = gr[time_index]
        if np.isfinite(observed) and observation_weight[time_index] > 0.0:
            step_feet = delta_md[time_index - 1] if time_index > 0 else 1.0
            effective_weight = observation_weight[time_index] * min(step_feet / correlation_length, 1.0)
            distance_squared = effective_weight * np.square((observed - reference_gr_grid) / residual_scale)
            likelihood = (
                (1.0 - float(config.emission_floor))
                * np.exp(-0.5 * np.minimum(distance_squared, float(config.emission_distance_cap)))
                + float(config.emission_floor)
            )
        if np.isfinite(tvt_input[time_index]):
            likelihood *= np.exp(
                -0.5 * np.square((value_grid - tvt_input[time_index]) / float(config.prefix_clamp_sigma))
            )
        return likelihood

    def forward_step(matrix: np.ndarray, time_index: int) -> np.ndarray:
        steps = float(delta_md[time_index - 1])
        matrix = shift_rows(matrix, steps, float(delta_z[time_index - 1]), +1.0)
        matrix = value_blur(matrix, steps)
        matrix = slope_mix(matrix, steps)
        matrix *= emission(time_index)[None, :]
        total = float(matrix.sum())
        if total <= 0.0 or not np.isfinite(total):
            return np.full((slope_count, value_count), 1.0 / (slope_count * value_count))
        return matrix / total

    def backward_step(matrix: np.ndarray, time_index: int) -> np.ndarray:
        steps = float(delta_md[time_index - 1])
        matrix = matrix * emission(time_index)[None, :]
        matrix = slope_mix(matrix, steps)
        matrix = value_blur(matrix, steps)
        matrix = shift_rows(matrix, steps, float(delta_z[time_index - 1]), -1.0)
        total = float(matrix.sum())
        if total <= 0.0 or not np.isfinite(total):
            return np.full((slope_count, value_count), 1.0 / (slope_count * value_count))
        return matrix / total

    checkpoint_interval = int(config.checkpoint)
    checkpoints: dict[int, np.ndarray] = {}
    forward = np.full((slope_count, value_count), 1.0 / (slope_count * value_count))
    forward *= emission(0)[None, :]
    initial_normalizer = float(forward.sum())
    log_likelihood = float(np.log(max(initial_normalizer, 1e-300)))
    forward /= initial_normalizer
    checkpoints[0] = forward.copy()
    for time_index in range(1, time_count):
        forward = shift_rows(forward, float(delta_md[time_index - 1]), float(delta_z[time_index - 1]), +1.0)
        forward = value_blur(forward, float(delta_md[time_index - 1]))
        forward = slope_mix(forward, float(delta_md[time_index - 1]))
        forward *= emission(time_index)[None, :]
        normalizer = float(forward.sum())
        log_likelihood += float(np.log(max(normalizer, 1e-300)))
        forward = (
            forward / normalizer
            if normalizer > 0.0 and np.isfinite(normalizer)
            else np.full((slope_count, value_count), 1.0 / (slope_count * value_count))
        )
        if time_index % checkpoint_interval == 0:
            checkpoints[time_index] = forward.copy()

    posterior_mean = np.empty(time_count, dtype=float)
    posterior_std = np.empty(time_count, dtype=float)
    backward = np.full((slope_count, value_count), 1.0 / (slope_count * value_count))
    high = time_count - 1
    while high >= 0:
        low = (high // checkpoint_interval) * checkpoint_interval
        forward_buffer = np.empty((high - low + 1, slope_count, value_count), dtype=float)
        local_forward = checkpoints[low]
        forward_buffer[0] = local_forward
        for time_index in range(low + 1, high + 1):
            local_forward = forward_step(local_forward, time_index)
            forward_buffer[time_index - low] = local_forward
        for time_index in range(high, low - 1, -1):
            posterior = forward_buffer[time_index - low] * backward
            posterior_total = float(posterior.sum())
            if posterior_total <= 0.0 or not np.isfinite(posterior_total):
                posterior = forward_buffer[time_index - low]
                posterior_total = float(posterior.sum())
            posterior /= posterior_total
            value_probability = posterior.sum(axis=0)
            mean = float(value_probability @ value_grid)
            posterior_mean[time_index] = mean
            posterior_std[time_index] = float(
                np.sqrt(max(float(value_probability @ np.square(value_grid - mean)), 0.0))
            )
            if time_index > 0:
                backward = backward_step(backward, time_index)
        high = low - 1

    evaluation_md = md_all[evaluation_index]
    evaluation_in_use = np.isin(use, evaluation_use)
    output[evaluation_index] = np.interp(evaluation_md, md[evaluation_in_use], posterior_mean[evaluation_in_use])
    suffix_std = np.interp(evaluation_md, md[evaluation_in_use], posterior_std[evaluation_in_use])
    if not np.isfinite(output[evaluation_index]).all() or not np.isfinite(suffix_std).all():
        raise RuntimeError("GeoHMM produced non-finite suffix predictions")

    diagnostics: dict[str, Any] = {
        "calibration_gain": float(calibration_gain),
        "calibration_offset": float(calibration_offset),
        "residual_scale": float(residual_scale),
        "correlation_length": float(correlation_length),
        "initial_slope": float(initial_slope),
        "time_steps": int(time_count),
        "value_bins": int(value_count),
        "log_likelihood": float(log_likelihood),
        "spatial_prior_used": False,
        "allowed_horizontal_columns": list(MODEL_COLUMNS),
        "allowed_typewell_columns": list(TYPEWELL_COLUMNS),
    }
    return output, suffix_std, diagnostics
