"""Preparation, PF/GeoHMM inference, and structural surface estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator

from .contracts import PipelineConfig, PreparedWell, ScoredPreparedWell, WellRecord
from .geohmm import GeoHMMNoPriorConfig, geohmm_no_prior
from .particle import likelihood_aggregate, pf_seed_trajectories


def _prepare_inference_frame(record: WellRecord, *, component: str = "") -> PreparedWell:
    """Validate a real visible-prefix contract without consulting truth."""

    record.validate(require_truth=False)
    inference = record.model_frame()
    existing = pd.to_numeric(inference["TVT_input"], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(existing))
    suffix = np.flatnonzero(~np.isfinite(existing))
    contiguous = (
        len(known) >= 30
        and len(suffix) >= 10
        and np.array_equal(known, np.arange(known[-1] + 1))
        and np.array_equal(suffix, np.arange(known[-1] + 1, len(existing)))
    )
    if not contiguous:
        raise ValueError(
            f"{record.well_id}: TVT_input must be a contiguous visible prefix followed by a non-empty suffix"
        )
    last = int(known[-1])
    md = pd.to_numeric(inference["MD"], errors="coerce").to_numpy(float)
    prepared = PreparedWell(
        well_id=record.well_id,
        inference=inference,
        typewell=record.typewell.loc[:, ["TVT", "GR"]].copy().reset_index(drop=True),
        suffix_mask=np.isin(np.arange(len(inference)), suffix),
        suffix_rows=suffix.astype(int),
        horizon=(md[suffix] - md[last]).astype(float),
        row_ids=np.asarray([f"{record.well_id}_{row}" for row in suffix], dtype=str),
        component=str(component),
    )
    prepared.validate()
    return prepared


def prepare_inference_well(record: WellRecord, *, component: str = "deployment") -> PreparedWell:
    """Prepare a well with an existing visible-prefix mask for inference."""

    return _prepare_inference_frame(record, component=component)


def prepare_scored_well(
    record: WellRecord,
    config: PipelineConfig,
    *,
    component: str = "",
) -> ScoredPreparedWell:
    """Prepare model inputs and the separate suffix targets used for scoring.

    Synthetic smoke wells arrive with TVT_input filled, so smoke mode creates
    their mask. Official-data runs must supply the observed mask.
    """

    record.validate(require_truth=True)
    candidate = record
    existing = pd.to_numeric(record.horizontal["TVT_input"], errors="coerce").to_numpy(float)
    if not np.isnan(existing).any():
        if config.mode != "smoke":
            raise ValueError(
                f"{record.well_id}: official-data mode cannot synthesize TVT_input from truth"
            )
        truth = pd.to_numeric(record.horizontal["TVT"], errors="coerce").to_numpy(float)
        cut = int(np.clip(round(config.prefix_fraction * len(existing)), 30, len(existing) - 10))
        masked = record.horizontal.copy()
        masked.loc[:, "TVT_input"] = truth
        masked.loc[cut:, "TVT_input"] = np.nan
        candidate = WellRecord(record.well_id, masked, record.typewell)
    prepared = _prepare_inference_frame(candidate, component=component)
    truth = pd.to_numeric(record.horizontal["TVT"], errors="coerce").to_numpy(float)
    scored = ScoredPreparedWell(prepared, truth[prepared.suffix_rows].copy())
    scored.validate()
    return scored


# Backward-compatible scoring adapter. Model-facing code never receives its
# ScoredPreparedWell wrapper; use ``prepare_inference_well`` for deployment.
prepare_well = prepare_scored_well


@dataclass(frozen=True)
class PhysicsExperts:
    """Target-free particle and GeoHMM paths with suffix uncertainty."""

    pf_full: np.ndarray
    pf_std_suffix: np.ndarray
    hmm_full: np.ndarray
    hmm_std_suffix: np.ndarray
    diagnostics: dict[str, object]


def run_physics(prepared: PreparedWell, config: PipelineConfig) -> PhysicsExperts:
    """Run the Monte-Carlo PF and forward/backward GeoHMM."""
    ensemble = pf_seed_trajectories(
        prepared.inference,
        prepared.typewell,
        seeds=config.pf_seeds,
        particles=config.pf_particles,
        seed_offset=config.pf_seed_offset,
    )
    suffix_pf = likelihood_aggregate(
        ensemble.predictions,
        ensemble.log_likelihoods,
        temperature=8.0,
        minimum_ess=min(float(config.pf_seeds), max(2.0, 0.5 * config.pf_seeds)),
    )
    # ``to_numpy`` can return a view. Copy before filling the PF suffix so the
    # downstream GeoHMM still sees missing TVT_input there.
    pf_full = (
        pd.to_numeric(prepared.inference["TVT_input"], errors="coerce")
        .to_numpy(float)
        .copy()
    )
    pf_full[prepared.suffix_rows] = suffix_pf
    pf_std = np.std(ensemble.predictions, axis=0)

    class RuntimeGeoHMM(GeoHMMNoPriorConfig):
        stride = int(config.hmm_stride)
        checkpoint = int(config.hmm_checkpoint)

    if prepared.inference.loc[prepared.suffix_rows, "TVT_input"].notna().any():
        raise RuntimeError("PF inference mutated the fail-closed suffix mask")
    hmm_full, hmm_std, hmm_diagnostics = geohmm_no_prior(
        prepared.inference,
        prepared.typewell,
        RuntimeGeoHMM,
    )
    return PhysicsExperts(
        pf_full=np.asarray(pf_full, float),
        pf_std_suffix=np.asarray(pf_std, float),
        hmm_full=np.asarray(hmm_full, float),
        hmm_std_suffix=np.asarray(hmm_std, float),
        diagnostics={
            "pf_seeds": config.pf_seeds,
            "pf_particles": config.pf_particles,
            "pf_seed_offset": config.pf_seed_offset,
            "pf_weight_ess_floor": min(float(config.pf_seeds), max(2.0, 0.5 * config.pf_seeds)),
            "hmm": hmm_diagnostics,
        },
    )


@dataclass
class StructuralSurface:
    """Regional structural level fitted from training-well anchors only."""

    model: RBFInterpolator
    center: np.ndarray
    scale: np.ndarray
    source_wells: tuple[str, ...]

    @classmethod
    def fit(cls, records: list[WellRecord], *, xy_scale_ft: float = 10_000.0) -> "StructuralSurface":
        if len(records) < 4:
            raise ValueError("structural surface needs at least four training wells")
        rows = []
        for record in records:
            frame = record.horizontal
            x = float(pd.to_numeric(frame["X"], errors="coerce").median())
            y = float(pd.to_numeric(frame["Y"], errors="coerce").median())
            if "ASTNU" in frame and np.isfinite(pd.to_numeric(frame["ASTNU"], errors="coerce")).any():
                target = float(pd.to_numeric(frame["ASTNU"], errors="coerce").median())
            else:
                target = float(np.nanmedian(pd.to_numeric(frame["TVT"], errors="coerce") + pd.to_numeric(frame["Z"], errors="coerce")))
            rows.append((record.well_id, x, y, target))
        table = pd.DataFrame(rows, columns=["well", "x", "y", "target"])
        xy = table[["x", "y"]].to_numpy(float)
        center = xy.mean(axis=0)
        scale = np.full(2, float(xy_scale_ft))
        model = RBFInterpolator(
            (xy - center) / scale,
            table["target"].to_numpy(float),
            kernel="linear",
            neighbors=min(80, len(table)),
            smoothing=0.0,
            degree=1,
        )
        return cls(model, center, scale, tuple(table["well"].astype(str)))

    def predict(self, prepared: PreparedWell) -> tuple[np.ndarray, float]:
        """Predict suffix TVT and visible-prefix rolling error without suffix truth."""
        frame = prepared.inference
        xy = frame[["X", "Y"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        sampled = np.asarray(self.model((xy - self.center) / self.scale), float)
        z = pd.to_numeric(frame["Z"], errors="coerce").to_numpy(float)
        tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        offset = float(np.median(tvt_input[known] + z[known] - sampled[known]))
        full = sampled + offset - z
        # Four one-sided rolling-origin checks inside the visible prefix.
        errors: list[np.ndarray] = []
        for fraction in (0.40, 0.55, 0.70, 0.85):
            cut = int(np.clip(round(fraction * len(known)), 20, len(known) - 10))
            train = known[:cut]
            valid = known[cut:min(len(known), cut + max(10, len(known) // 10))]
            local_offset = float(np.median(tvt_input[train] + z[train] - sampled[train]))
            errors.append(sampled[valid] + local_offset - z[valid] - tvt_input[valid])
        joined = np.concatenate(errors) if errors else np.asarray([12.0])
        prefix_rmse = float(np.sqrt(np.mean(np.square(joined))))
        return full[prepared.suffix_rows], prefix_rmse
