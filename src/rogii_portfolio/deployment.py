"""Training, target-well inference and submission assembly."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .artifacts import sha256_file, write_json
from .contracts import PipelineConfig, WellRecord
from .features import build_fast_safe_features
from .physics import StructuralSurface, prepare_inference_well, run_physics
from .pipeline import InferenceCache, _cache_well, _fit_models, _predict_one
from .stack import StackPolicy, apply_stack


@dataclass
class FittedCleanPipeline:
    """A model refitted from this pipeline with a selected stack policy."""

    config: PipelineConfig
    base_models: object
    structural_surface: StructuralSurface
    stack_policy: StackPolicy
    training_wells: tuple[str, ...]
    lineage_mode: str = "retrained_cleanroom"


@dataclass(frozen=True)
class WellPrediction:
    """Submission-ready suffix prediction and its auditable candidate paths."""

    well_id: str
    row_id: np.ndarray
    row_number: np.ndarray
    tvt: np.ndarray
    arms: dict[str, np.ndarray]
    diagnostics: dict[str, object]


def fit_frozen_pipeline(
    training_wells: list[WellRecord],
    config: PipelineConfig,
    stack_policy: StackPolicy,
) -> FittedCleanPipeline:
    """Fit base models on all training wells using a selected stack policy."""

    if len(training_wells) < 4:
        raise ValueError("deployment fit needs at least four training wells")
    caches = [_cache_well(record, "deployment_train", config) for record in training_wells]
    models = _fit_models(caches, config)
    surface = StructuralSurface.fit(training_wells)
    return FittedCleanPipeline(
        config=config,
        base_models=models,
        structural_surface=surface,
        stack_policy=stack_policy,
        training_wells=tuple(record.well_id for record in training_wells),
    )


def predict_well(model: FittedCleanPipeline, well: WellRecord) -> WellPrediction:
    """Predict one well after rejecting inputs that contain TVT."""

    if "TVT" in well.horizontal.columns:
        raise ValueError("deployment input must not contain TVT")
    prepared = prepare_inference_well(well)
    physics = run_physics(prepared, model.config)
    features = build_fast_safe_features(
        prepared,
        pf_full=physics.pf_full,
        pf_std_suffix=physics.pf_std_suffix,
        hmm_full=physics.hmm_full,
    )
    output = _predict_one(
        InferenceCache(prepared=prepared, physics=physics, features=features),
        model.base_models,
        model.structural_surface,
        enable_switching_state=model.config.switching_state_enabled,
        enable_trust_region_ridge=model.config.trust_region_ridge_enabled,
        trust_region_ridge_weight=model.config.trust_region_ridge_weight,
    )
    arms = {name: np.asarray(value, np.float64) for name, value in output["predictions"].items()}
    arms["nested_stack"] = apply_stack(model.stack_policy, arms).astype(np.float64, copy=False)
    return WellPrediction(
        well_id=well.well_id,
        row_id=prepared.row_ids.copy(),
        row_number=prepared.suffix_rows.copy(),
        tvt=arms["nested_stack"].copy(),
        arms=arms,
        diagnostics=output["diagnostics"],
    )


def verify_submission(frame: pd.DataFrame, sample_submission: pd.DataFrame) -> dict[str, object]:
    """Require schema, ID order and cardinality to match the competition template."""

    if list(frame.columns) != ["id", "tvt"] or list(sample_submission.columns) != ["id", "tvt"]:
        raise ValueError("submission and template schema must be exactly ['id', 'tvt']")
    ids = frame["id"].astype(str).to_numpy()
    expected = sample_submission["id"].astype(str).to_numpy()
    if not np.array_equal(ids, expected):
        raise ValueError("submission ID order differs from the template; reordering is not automatic")
    predictions = pd.to_numeric(frame["tvt"], errors="coerce").to_numpy(np.float64)
    if len(np.unique(ids)) != len(ids) or not np.isfinite(predictions).all():
        raise ValueError("submission IDs must be unique and predictions finite")
    import hashlib

    return {
        "rows": int(len(frame)),
        "id_order_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        "prediction_float64_sha256": hashlib.sha256(
            np.ascontiguousarray(predictions.astype("<f8", copy=False)).tobytes()
        ).hexdigest(),
        "status": "PASS",
    }


def predict_submission(
    model: FittedCleanPipeline,
    test_wells: Iterable[WellRecord],
    sample_submission: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Predict all test wells and restore the template's exact row order."""

    predictions = [predict_well(model, well) for well in test_wells]
    generated_ids = np.concatenate([item.row_id for item in predictions]).astype(str)
    generated_tvt = np.concatenate([item.tvt for item in predictions]).astype(np.float64)
    frame = pd.DataFrame({"id": generated_ids, "tvt": generated_tvt})
    audit = verify_submission(frame, sample_submission)
    audit["lineage_mode"] = model.lineage_mode
    audit["training_wells"] = len(model.training_wells)
    audit["config"] = asdict(model.config)
    return frame, audit


def write_submission(
    frame: pd.DataFrame,
    sample_submission: pd.DataFrame,
    path: Path,
    *,
    audit_path: Path | None = None,
) -> dict[str, object]:
    audit = verify_submission(frame, sample_submission)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    audit["csv_sha256"] = sha256_file(path)
    audit["csv_bytes"] = path.stat().st_size
    if audit_path is not None:
        write_json(audit_path, audit)
    return audit
