"""Particle-filter seed paths, predictive evidence and robust aggregation.

The dynamics match the reference PF. The implementation uses target-well logs
and a type well, without coordinates, neighbouring wells or structural tops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


HORIZONTAL_COLUMNS = ("MD", "Z", "GR", "TVT_input")
TYPEWELL_COLUMNS = ("TVT", "GR")


@dataclass(frozen=True)
class PFSeedEnsemble:
    """Suffix-aligned seed trajectories and predictive evidence increments."""

    predictions: np.ndarray  # (seed, suffix row)
    log_likelihood_increments: np.ndarray  # (seed, suffix row)
    seeds: tuple[int, ...]
    particles: int

    @property
    def log_likelihoods(self) -> np.ndarray:
        return np.sum(self.log_likelihood_increments, axis=1)


def _validate_inputs(
    horizontal: pd.DataFrame, typewell: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_h = set(HORIZONTAL_COLUMNS) - set(horizontal.columns)
    missing_t = set(TYPEWELL_COLUMNS) - set(typewell.columns)
    if missing_h:
        raise ValueError(f"horizontal input is missing {sorted(missing_h)}")
    if missing_t:
        raise ValueError(f"typewell input is missing {sorted(missing_t)}")

    hw = horizontal.loc[:, HORIZONTAL_COLUMNS].copy().reset_index(drop=True)
    tw = (
        typewell.loc[:, TYPEWELL_COLUMNS]
        .copy()
        .sort_values("TVT", kind="mergesort")
        .reset_index(drop=True)
    )
    for column in HORIZONTAL_COLUMNS:
        hw[column] = pd.to_numeric(hw[column], errors="coerce")
    for column in TYPEWELL_COLUMNS:
        tw[column] = pd.to_numeric(tw[column], errors="coerce")

    md = hw["MD"].to_numpy(float)
    if len(hw) == 0 or not np.isfinite(md).all() or np.any(np.diff(md) <= 0.0):
        raise ValueError("MD must be finite and strictly increasing")
    known = hw["TVT_input"].notna().to_numpy()
    if int(known.sum()) < 30 or int((~known).sum()) == 0:
        raise ValueError("PF requires at least 30 known rows and one suffix row")

    finite_tvt = np.isfinite(tw["TVT"].to_numpy(float))
    tw = tw.loc[finite_tvt].reset_index(drop=True)
    if len(tw) < 4 or np.any(np.diff(tw["TVT"].to_numpy(float)) <= 0.0):
        raise ValueError("typewell TVT must be finite and strictly increasing")
    return hw, tw


def _particle_filter_single_trace(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    seed: int,
    particles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference PF member plus rowwise predictive log evidence."""

    hw, tw = _validate_inputs(horizontal, typewell)
    tw_tvt = tw["TVT"].to_numpy(float)
    tw_gr_series = tw["GR"].astype(float)
    tw_gr = tw_gr_series.fillna(float(tw_gr_series.mean())).to_numpy(float)
    if not np.isfinite(tw_gr).all():
        tw_gr = np.nan_to_num(tw_gr, nan=0.0)

    known = hw.loc[hw["TVT_input"].notna()]
    evaluation = hw.loc[hw["TVT_input"].isna()]
    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_z = float(last["Z"])
    previous_md = float(last["MD"])
    reference_known = np.interp(
        known["TVT_input"].to_numpy(float), tw_tvt, tw_gr
    )
    gr_sigma = float(
        np.clip(
            np.nanstd(known["GR"].fillna(0.0).to_numpy(float) - reference_known),
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
    if count < 8:
        raise ValueError("particles must be at least 8")
    position = last_tvt + last_z + 4.5 * rng.standard_normal(count)
    rate = initial_rate + 0.01 * rng.standard_normal(count)
    weight = np.full(count, 1.0 / count)
    gr_interpolated = (
        hw["GR"].interpolate(limit_direction="both").fillna(float(np.mean(tw_gr)))
    )
    evaluation_md = evaluation["MD"].to_numpy(float)
    evaluation_z = evaluation["Z"].to_numpy(float)
    evaluation_gr = gr_interpolated.to_numpy(float)[evaluation.index.to_numpy(int)]
    prediction = np.empty(len(evaluation), dtype=float)
    log_increment = np.empty(len(evaluation), dtype=float)

    for index in range(len(evaluation)):
        md_step = max(float(evaluation_md[index]) - previous_md, 1.0)
        rate = 0.998 * rate + 0.002 * rng.standard_normal(count)
        position = position + rate * md_step + 0.005 * rng.standard_normal(count)
        tvt_particles = np.clip(
            position - evaluation_z[index],
            tw_tvt[0] - 100.0,
            tw_tvt[-1] + 100.0,
        )
        position = tvt_particles + evaluation_z[index]
        expected_gr = np.interp(tvt_particles, tw_tvt, tw_gr)
        distance = (evaluation_gr[index] - expected_gr) / gr_sigma
        likelihood = np.maximum(
            np.exp(-0.5 * np.minimum(np.square(distance), 600.0)), 1e-300
        )
        average_likelihood = float(np.sum(weight * likelihood))
        log_increment[index] = float(np.log(max(average_likelihood, 1e-300)))
        weight *= likelihood
        weight_sum = float(weight.sum())
        weight = (
            weight / weight_sum
            if weight_sum > 0.0
            else np.full(count, 1.0 / count)
        )
        effective_count = 1.0 / float(np.sum(np.square(weight)))
        if effective_count < 0.5 * count:
            cumulative = np.cumsum(weight)
            offset = rng.uniform(0.0, 1.0 / count)
            selected = np.clip(
                np.searchsorted(
                    cumulative, offset + np.arange(count) / count
                ),
                0,
                count - 1,
            )
            position = position[selected] + 0.1 * rng.standard_normal(count)
            rate = rate[selected] + 0.001 * rng.standard_normal(count)
            weight = np.full(count, 1.0 / count)
        prediction[index] = float(np.dot(weight, position - evaluation_z[index]))
        previous_md = float(evaluation_md[index])
    return prediction, log_increment


def pf_seed_trajectories(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    seeds: int = 8,
    particles: int = 500,
    seed_offset: int = 0,
) -> PFSeedEnsemble:
    """Generate one suffix path and log-evidence sequence per seed."""

    hw, tw = _validate_inputs(horizontal, typewell)
    if int(seeds) < 2:
        raise ValueError("at least two seeds are required for robust aggregation")
    paths: list[np.ndarray] = []
    increments: list[np.ndarray] = []
    if isinstance(seed_offset, bool) or int(seed_offset) != seed_offset or int(seed_offset) < 0:
        raise ValueError("seed_offset must be a non-negative integer")
    seed_values = tuple(int(seed_offset) + index for index in range(int(seeds)))
    for seed in seed_values:
        path, log_increment = _particle_filter_single_trace(
            hw, tw, seed=seed, particles=int(particles)
        )
        paths.append(path)
        increments.append(log_increment)
    result = PFSeedEnsemble(
        predictions=np.stack(paths, axis=0),
        log_likelihood_increments=np.stack(increments, axis=0),
        seeds=seed_values,
        particles=int(particles),
    )
    if not np.isfinite(result.predictions).all():
        raise RuntimeError("PF produced non-finite predictions")
    if not np.isfinite(result.log_likelihood_increments).all():
        raise RuntimeError("PF produced non-finite predictive evidence")
    return result


def stable_softmax(score: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("score must be a non-empty vector")
    centered = values - float(np.max(values))
    weights = np.exp(np.clip(centered, -700.0, 0.0))
    total = float(np.sum(weights))
    return weights / total if total > 0.0 else np.full(len(values), 1.0 / len(values))


def effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    total = float(values.sum())
    if values.ndim != 1 or len(values) == 0 or total <= 0.0:
        raise ValueError("weights must be a positive vector")
    values = values / total
    return float(1.0 / np.sum(np.square(values)))


def constrain_weight_ess(weights: np.ndarray, minimum_ess: float) -> np.ndarray:
    """Maximally retain weights while mixing with uniform to meet an ESS floor."""

    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or len(values) == 0 or np.any(values < 0.0):
        raise ValueError("weights must be a non-negative vector")
    if not 1.0 <= float(minimum_ess) <= len(values):
        raise ValueError("minimum_ess must lie between one and the seed count")
    total = float(values.sum())
    values = values / total if total > 0.0 else np.full(len(values), 1.0 / len(values))
    if effective_sample_size(values) >= float(minimum_ess) - 1e-12:
        return values
    uniform = np.full(len(values), 1.0 / len(values))
    low, high = 0.0, 1.0
    # rho=0 is uniform; choose the largest rho retaining the raw weights.
    for _ in range(64):
        rho = 0.5 * (low + high)
        proposal = uniform + rho * (values - uniform)
        if effective_sample_size(proposal) >= float(minimum_ess):
            low = rho
        else:
            high = rho
    return uniform + low * (values - uniform)


def global_likelihood_weights(
    log_likelihoods: np.ndarray,
    *,
    temperature: float = 8.0,
    minimum_ess: float | None = None,
) -> np.ndarray:
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    weights = stable_softmax(np.asarray(log_likelihoods, dtype=float) / float(temperature))
    if minimum_ess is not None:
        weights = constrain_weight_ess(weights, float(minimum_ess))
    return weights


def likelihood_aggregate(
    predictions: np.ndarray,
    log_likelihoods: np.ndarray,
    *,
    temperature: float = 8.0,
    minimum_ess: float | None = None,
) -> np.ndarray:
    paths = np.asarray(predictions, dtype=float)
    if paths.ndim != 2 or paths.shape[0] != len(log_likelihoods):
        raise ValueError("seed path and likelihood shapes disagree")
    weights = global_likelihood_weights(
        log_likelihoods, temperature=temperature, minimum_ess=minimum_ess
    )
    return np.sum(weights[:, None] * paths, axis=0)


def prequential_aggregate(
    predictions: np.ndarray,
    log_likelihood_increments: np.ndarray,
    *,
    temperature: float = 8.0,
    minimum_ess: float | None = None,
    decay: float = 1.0,
) -> np.ndarray:
    """Aggregate row t with evidence available strictly before row t.

    A decay below one turns cumulative evidence into a rolling exponentially
    discounted score.  This reduces lock-in to an early Monte Carlo winner.
    """

    paths = np.asarray(predictions, dtype=float)
    increments = np.asarray(log_likelihood_increments, dtype=float)
    if paths.ndim != 2 or increments.shape != paths.shape:
        raise ValueError("predictions and log-likelihood increments must align")
    if not 0.0 < float(decay) <= 1.0:
        raise ValueError("decay must lie in (0, 1]")
    score = np.zeros(paths.shape[0], dtype=float)
    output = np.empty(paths.shape[1], dtype=float)
    for row in range(paths.shape[1]):
        weights = global_likelihood_weights(
            score, temperature=float(temperature), minimum_ess=minimum_ess
        )
        output[row] = float(np.dot(weights, paths[:, row]))
        score = float(decay) * score + increments[:, row]
    return output


def robust_seed_aggregate(predictions: np.ndarray, method: str) -> np.ndarray:
    paths = np.asarray(predictions, dtype=float)
    if paths.ndim != 2 or paths.shape[0] < 2:
        raise ValueError("predictions must have shape (at least two seeds, rows)")
    if method == "uniform":
        return np.mean(paths, axis=0)
    if method == "median":
        return np.median(paths, axis=0)
    if method == "trim1":
        if paths.shape[0] < 4:
            raise ValueError("trim1 requires at least four seeds")
        ordered = np.sort(paths, axis=0)
        return np.mean(ordered[1:-1], axis=0)
    raise ValueError(f"unknown robust aggregation method {method!r}")


def registered_candidates(ensemble: PFSeedEnsemble) -> dict[str, np.ndarray]:
    """Return the fixed candidate grid used in the Dev40 experiment."""

    paths = ensemble.predictions
    log_likelihoods = ensemble.log_likelihoods
    output: dict[str, np.ndarray] = {
        "uniform": robust_seed_aggregate(paths, "uniform"),
        "median": robust_seed_aggregate(paths, "median"),
        "trim1": robust_seed_aggregate(paths, "trim1"),
    }
    for temperature in (2.0, 4.0, 8.0, 16.0, 32.0):
        output[f"lik_t{int(temperature)}"] = likelihood_aggregate(
            paths, log_likelihoods, temperature=temperature
        )
    for temperature, ess in (
        (4.0, 4.0),
        (4.0, 6.0),
        (4.0, 7.0),
        (8.0, 4.0),
        (8.0, 6.0),
        (8.0, 7.0),
        (16.0, 6.0),
    ):
        output[f"ess{int(ess)}_t{int(temperature)}"] = likelihood_aggregate(
            paths,
            log_likelihoods,
            temperature=temperature,
            minimum_ess=ess,
        )
    for temperature, ess, decay, label in (
        (4.0, 6.0, 1.0, "preq_t4_e6_d1"),
        (8.0, 6.0, 1.0, "preq_t8_e6_d1"),
        (4.0, 6.0, 0.995, "preq_t4_e6_d995"),
        (4.0, 6.0, 0.98, "preq_t4_e6_d98"),
    ):
        output[label] = prequential_aggregate(
            paths,
            ensemble.log_likelihood_increments,
            temperature=temperature,
            minimum_ess=ess,
            decay=decay,
        )
    # Progressive Bayesian model averaging (PBMA): unlike the incumbent,
    # evidence accumulated after row t is not applied retroactively to row t.
    # No ESS floor is imposed because the temperature itself controls expert
    # concentration.  The curve is registered before confirmation.
    for temperature in (8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0, 32.0):
        output[f"pbma_t{int(temperature)}"] = prequential_aggregate(
            paths,
            ensemble.log_likelihood_increments,
            temperature=temperature,
            minimum_ess=None,
            decay=1.0,
        )
    incumbent = output["lik_t8"]
    for percent in (10, 20, 30, 40, 50, 60, 80, 100):
        blend = percent / 100.0
        output[f"pbma16_blend{percent:03d}"] = (
            incumbent + blend * (output["pbma_t16"] - incumbent)
        )
    # Robust location shrinkage around the incumbent likelihood aggregate.
    output["uniform50_lik8"] = 0.5 * output["uniform"] + 0.5 * incumbent
    output["median25_lik8"] = 0.25 * output["median"] + 0.75 * incumbent
    output["median50_lik8"] = 0.5 * output["median"] + 0.5 * incumbent
    output["trim50_lik8"] = 0.5 * output["trim1"] + 0.5 * incumbent
    return output


def inference_metadata(ensemble: PFSeedEnsemble) -> dict[str, Any]:
    incumbent_weight = global_likelihood_weights(
        ensemble.log_likelihoods, temperature=8.0
    )
    return {
        "seeds": list(ensemble.seeds),
        "particles": ensemble.particles,
        "suffix_rows": int(ensemble.predictions.shape[1]),
        "incumbent_temperature": 8.0,
        "incumbent_ess": effective_sample_size(incumbent_weight),
        "incumbent_weight_min": float(incumbent_weight.min()),
        "incumbent_weight_max": float(incumbent_weight.max()),
        "spatial_prior_used": False,
        "target_used_by_aggregation": False,
        "allowed_horizontal_columns": list(HORIZONTAL_COLUMNS),
        "allowed_typewell_columns": list(TYPEWELL_COLUMNS),
    }
