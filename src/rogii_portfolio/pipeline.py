"""Nested component-CV training and holdout evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .components import assert_component_disjoint, build_components, split_components
from .contracts import PipelineConfig, PredictionBundle, PreparedWell, WellRecord
from .features import SAFE_FEATURES, build_fast_safe_features
from .hgrg import apply_hgrg
from .meta_state import apply_meta_state
from .models import FastSafeModels, incumbent_parent
from .overlays import apply_conditional_shape, apply_prefix_boundary
from .physics import PhysicsExperts, StructuralSurface, prepare_scored_well, run_physics
from .stack import StackPolicy, apply_stack, component_bootstrap, fit_convex_stack, rmse
from .switching_state import apply_switching_state


@dataclass(frozen=True)
class InferenceCache:
    """Cached target-well inputs and expert paths."""

    prepared: PreparedWell
    physics: PhysicsExperts
    features: pd.DataFrame


@dataclass(frozen=True)
class ScoredCache:
    """Model inputs paired with targets for fitting and scoring."""

    source_record: WellRecord
    model_input: InferenceCache
    truth_suffix: np.ndarray


@dataclass
class NestedResult:
    """Outputs from nested component validation and final holdout scoring.

    ``outer_oof`` contains fold-frozen development predictions. ``holdout``
    contains predictions from the final development refit. The two populations
    must remain separate when reporting results.
    """

    summary: dict[str, object]
    outer_oof: PredictionBundle
    holdout: PredictionBundle
    stack_policy: StackPolicy
    components: pd.DataFrame
    fold_records: pd.DataFrame
    component_universe: pd.DataFrame | None = None


def _cache_well(record: WellRecord, component: str, config: PipelineConfig) -> ScoredCache:
    scored = prepare_scored_well(record, config, component=component)
    prepared = scored.model_input
    physics = run_physics(prepared, config)
    features = build_fast_safe_features(
        prepared,
        pf_full=physics.pf_full,
        pf_std_suffix=physics.pf_std_suffix,
        hmm_full=physics.hmm_full,
    )
    return ScoredCache(record, InferenceCache(prepared, physics, features), scored.truth_suffix)


def _concat_training(caches: list[ScoredCache]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    features = pd.concat([cache.model_input.features for cache in caches], ignore_index=True)
    truth = np.concatenate([cache.truth_suffix for cache in caches])
    pf = np.concatenate([
        cache.model_input.physics.pf_full[cache.model_input.prepared.suffix_rows]
        for cache in caches
    ])
    return features, truth, pf


def _fit_models(caches: list[ScoredCache], config: PipelineConfig) -> FastSafeModels:
    features, truth, pf = _concat_training(caches)
    return FastSafeModels.fit(features, truth, pf, config)


def _predict_one(
    cache: InferenceCache,
    models: FastSafeModels,
    surface: StructuralSurface,
    *,
    enable_switching_state: bool = False,
    enable_trust_region_ridge: bool = False,
    trust_region_ridge_weight: float = 0.05,
) -> dict[str, object]:
    """Build every candidate path for one target-free cached well."""

    prepared, physics = cache.prepared, cache.physics
    suffix = prepared.suffix_rows
    pf = physics.pf_full[suffix]
    hmm = physics.hmm_full[suffix]
    ridge, nonlinear = models.predict(cache.features, pf)
    incumbent, parent_diag = incumbent_parent(ridge=ridge, pf=pf, nonlinear=nonlinear)
    hgrg, hgrg_diag = apply_hgrg(
        base=incumbent,
        ridge=ridge,
        pf=pf,
        hmm=hmm,
        horizon=prepared.horizon,
    )
    structural, prefix_cv = surface.predict(prepared)
    meta, state, meta_diag = apply_meta_state(
        hgrg=hgrg,
        pf=pf,
        hmm=hmm,
        structural=structural,
        horizon=prepared.horizon,
        prefix_cv=prefix_cv,
    )
    boundary, boundary_diag = apply_prefix_boundary(prepared, meta)
    final, shape_diag = apply_conditional_shape(base=boundary, pf=pf, hmm_stride6=hmm)
    predictions = {
        "pf": pf,
        "ridge": ridge,
        "nonlinear": nonlinear,
        "incumbent": incumbent,
        "hgrg": hgrg,
        "state": state,
        "meta_state": meta,
        "prefix_boundary": boundary,
        "sequential_final": final,
    }
    diagnostics: dict[str, object] = {
        "well": prepared.well_id,
        "component": prepared.component,
        "rows": len(suffix),
        "parent": parent_diag,
        "hgrg": hgrg_diag,
        "meta_state": meta_diag,
        "prefix_boundary": boundary_diag,
        "conditional_shape": shape_diag,
    }
    if enable_switching_state:
        switching, _, _, switching_diag = apply_switching_state(
            hgrg=hgrg,
            pf=pf,
            hmm=hmm,
            structural=structural,
            horizon=prepared.horizon,
            prefix_cv=prefix_cv,
        )
        predictions["switching_state"] = switching
        diagnostics["switching_state"] = switching_diag
    if enable_trust_region_ridge:
        weight = float(trust_region_ridge_weight)
        if not np.isfinite(weight) or not 0.0 <= weight <= 0.25:
            raise ValueError("trust_region_ridge_weight must lie in [0, 0.25]")
        predictions["trust_region_ridge"] = final + weight * (ridge - final)
        diagnostics["trust_region_ridge"] = {
            "ridge_weight": weight,
            "sequential_final_weight": 1.0 - weight,
        }
    return {
        "predictions": predictions,
        "structural": structural,
        "prefix_cv": prefix_cv,
        "diagnostics": diagnostics,
    }


def _bundle(
    caches: list[ScoredCache],
    models: FastSafeModels,
    surface: StructuralSurface,
    *,
    enable_switching_state: bool = False,
    enable_trust_region_ridge: bool = False,
    trust_region_ridge_weight: float = 0.05,
) -> PredictionBundle:
    outputs = [
        _predict_one(
            cache.model_input,
            models,
            surface,
            enable_switching_state=enable_switching_state,
            enable_trust_region_ridge=enable_trust_region_ridge,
            trust_region_ridge_weight=trust_region_ridge_weight,
        )
        for cache in caches
    ]
    names = tuple(outputs[0]["predictions"])
    predictions = {
        name: np.concatenate([np.asarray(output["predictions"][name], float) for output in outputs])
        for name in names
    }
    bundle = PredictionBundle(
        well=np.concatenate([
            np.repeat(cache.model_input.prepared.well_id, len(cache.model_input.prepared.suffix_rows))
            for cache in caches
        ]),
        row_id=np.concatenate([cache.model_input.prepared.row_ids for cache in caches]),
        row_number=np.concatenate([cache.model_input.prepared.suffix_rows for cache in caches]),
        component=np.concatenate([
            np.repeat(cache.model_input.prepared.component, len(cache.model_input.prepared.suffix_rows))
            for cache in caches
        ]),
        horizon=np.concatenate([cache.model_input.prepared.horizon for cache in caches]),
        truth=np.concatenate([cache.truth_suffix for cache in caches]),
        features=pd.concat([cache.model_input.features for cache in caches], ignore_index=True),
        predictions=predictions,
        diagnostics=[output["diagnostics"] for output in outputs],
    )
    bundle.validate()
    return bundle


def _combine(bundles: Iterable[PredictionBundle]) -> PredictionBundle:
    bundles = list(bundles)
    names = tuple(bundles[0].predictions)
    result = PredictionBundle(
        well=np.concatenate([bundle.well for bundle in bundles]),
        row_id=np.concatenate([bundle.row_id for bundle in bundles]),
        row_number=np.concatenate([bundle.row_number for bundle in bundles]),
        component=np.concatenate([bundle.component for bundle in bundles]),
        horizon=np.concatenate([bundle.horizon for bundle in bundles]),
        truth=np.concatenate([bundle.truth for bundle in bundles]),
        features=pd.concat([bundle.features for bundle in bundles], ignore_index=True),
        predictions={name: np.concatenate([bundle.predictions[name] for bundle in bundles]) for name in names},
        diagnostics=sum((bundle.diagnostics for bundle in bundles), []),
    )
    order = np.argsort(result.row_id.astype(str), kind="mergesort")
    result.well = result.well[order]
    result.row_id = result.row_id[order]
    result.row_number = result.row_number[order]
    result.component = result.component[order]
    result.horizon = result.horizon[order]
    result.truth = result.truth[order]
    result.features = result.features.iloc[order].reset_index(drop=True)
    result.predictions = {name: value[order] for name, value in result.predictions.items()}
    result.validate()
    return result


def _balanced_assignment(components: list[str], row_count: dict[str, int], folds: int) -> dict[str, int]:
    load = np.zeros(folds, float)
    assignment: dict[str, int] = {}
    for component in sorted(components, key=lambda name: (-row_count[name], name)):
        fold = int(np.argmin(load))
        assignment[component] = fold
        load[fold] += row_count[component]
    if set(assignment.values()) != set(range(folds)):
        raise RuntimeError("not enough components for every nested fold")
    return assignment


def _inner_oof(
    caches: list[ScoredCache],
    config: PipelineConfig,
) -> PredictionBundle:
    components = sorted({cache.model_input.prepared.component for cache in caches})
    counts = {
        component: sum(
            len(cache.model_input.prepared.suffix_rows)
            for cache in caches
            if cache.model_input.prepared.component == component
        )
        for component in components
    }
    assignment = _balanced_assignment(components, counts, config.inner_folds)
    bundles = []
    for fold in range(config.inner_folds):
        train = [cache for cache in caches if assignment[cache.model_input.prepared.component] != fold]
        valid = [cache for cache in caches if assignment[cache.model_input.prepared.component] == fold]
        if not train or not valid:
            raise RuntimeError("empty inner split")
        models = _fit_models(train, config)
        surface = StructuralSurface.fit([cache.source_record for cache in train])
        bundles.append(_bundle(valid, models, surface))
    return _combine(bundles)


def _arm_scores(bundle: PredictionBundle) -> dict[str, float]:
    return {name: rmse(bundle.truth, prediction) for name, prediction in bundle.predictions.items()}


def run_nested_experiment(
    wells: list[WellRecord],
    config: PipelineConfig,
    *,
    component_table: pd.DataFrame | None = None,
    component_graph_scope_wells: int | None = None,
    component_universe: pd.DataFrame | None = None,
) -> NestedResult:
    """Run nested component CV, refit on development data and score the holdout."""
    config.validate()
    if component_table is None:
        component_table = build_components(wells, config)
    else:
        component_table = component_table.copy().reset_index(drop=True)
        expected = {record.well_id for record in wells}
        observed = set(component_table["well"].astype(str))
        if observed != expected or component_table["well"].duplicated().any():
            raise ValueError("external component table must cover each evaluated well exactly once")
    development_components, holdout_components, outer_assignment = split_components(component_table, config)
    component_map = component_table.set_index("well")["component"].astype(str).to_dict()
    caches = [_cache_well(record, component_map[record.well_id], config) for record in wells]
    development = [
        cache for cache in caches if cache.model_input.prepared.component in development_components
    ]
    holdout_caches = [
        cache for cache in caches if cache.model_input.prepared.component in holdout_components
    ]
    row_count = component_table.set_index("component").groupby(level=0)["rows"].sum().astype(int).to_dict()
    if len(set(outer_assignment.values())) < config.outer_folds:
        outer_assignment = _balanced_assignment(sorted(development_components), row_count, config.outer_folds)

    outer_bundles: list[PredictionBundle] = []
    fold_records: list[dict[str, object]] = []
    frozen_sequential_policy = StackPolicy(
        arms=("sequential_final",),
        weights=np.asarray([1.0], dtype=float),
        ridge=0.0,
    )
    for outer_fold in range(config.outer_folds):
        train = [
            cache
            for cache in development
            if outer_assignment[cache.model_input.prepared.component] != outer_fold
        ]
        valid = [
            cache
            for cache in development
            if outer_assignment[cache.model_input.prepared.component] == outer_fold
        ]
        assert_component_disjoint(
            {cache.source_record.well_id for cache in train},
            {cache.source_record.well_id for cache in valid},
            component_table,
        )
        if config.skip_nested_stack:
            policy = frozen_sequential_policy
        else:
            inner = _inner_oof(train, config)
            policy = fit_convex_stack(
                inner.truth,
                inner.predictions,
                inner.component,
                ridge=config.stack_ridge,
            )
        models = _fit_models(train, config)
        surface = StructuralSurface.fit([cache.source_record for cache in train])
        validation = _bundle(
            valid,
            models,
            surface,
            enable_switching_state=config.switching_state_enabled,
            enable_trust_region_ridge=config.trust_region_ridge_enabled,
            trust_region_ridge_weight=config.trust_region_ridge_weight,
        )
        validation.predictions["nested_stack"] = apply_stack(policy, validation.predictions)
        validation.validate()
        outer_bundles.append(validation)
        fold_records.append(
            {
                "outer_fold": outer_fold,
                "train_wells": len(train),
                "valid_wells": len(valid),
                "train_components": len({cache.model_input.prepared.component for cache in train}),
                "valid_components": len({cache.model_input.prepared.component for cache in valid}),
                "incumbent_rmse": rmse(validation.truth, validation.predictions["incumbent"]),
                "sequential_final_rmse": rmse(validation.truth, validation.predictions["sequential_final"]),
                "nested_stack_rmse": rmse(validation.truth, validation.predictions["nested_stack"]),
                "stack_parent_weight": policy.parent_weight,
                **{f"stack_{arm}": float(weight) for arm, weight in zip(policy.arms, policy.weights)},
            }
        )

    outer_oof = _combine(outer_bundles)
    # Fit the deployment stack on outer-OOF development predictions.
    final_policy = (
        frozen_sequential_policy
        if config.skip_nested_stack
        else fit_convex_stack(
            outer_oof.truth,
            outer_oof.predictions,
            outer_oof.component,
            ridge=config.stack_ridge,
        )
    )
    # Keep the reported OOF scores from fold-specific policies. Apply this
    # refitted policy only to the holdout.
    final_models = _fit_models(development, config)
    final_surface = StructuralSurface.fit([cache.source_record for cache in development])
    holdout = _bundle(
        holdout_caches,
        final_models,
        final_surface,
        enable_switching_state=config.switching_state_enabled,
        enable_trust_region_ridge=config.trust_region_ridge_enabled,
        trust_region_ridge_weight=config.trust_region_ridge_weight,
    )
    holdout.predictions["nested_stack"] = apply_stack(final_policy, holdout.predictions)
    holdout.validate()

    development_names = {cache.source_record.well_id for cache in development}
    holdout_names = {cache.source_record.well_id for cache in holdout_caches}
    assert_component_disjoint(development_names, holdout_names, component_table)
    oof_scores = _arm_scores(outer_oof)
    holdout_scores = _arm_scores(holdout)
    sequential_bootstrap = component_bootstrap(
        holdout.truth,
        holdout.predictions["incumbent"],
        holdout.predictions["sequential_final"],
        holdout.component,
        draws=config.bootstrap_draws,
        seed=config.split_seed,
    )
    stack_bootstrap = component_bootstrap(
        holdout.truth,
        holdout.predictions["incumbent"],
        holdout.predictions["nested_stack"],
        holdout.component,
        draws=config.bootstrap_draws,
        seed=(config.split_seed + 1) % (2**32),
    )
    summary: dict[str, object] = {
        "mode": config.mode,
        "contract": "fully_nested_geological_component_cv_plus_single_untouched_holdout",
        "feature_count": len(SAFE_FEATURES),
        "wells": len(wells),
        "components": int(component_table["component"].nunique()),
        "component_graph_scope_wells": int(component_graph_scope_wells or len(wells)),
        "development_wells": len(development),
        "holdout_wells": len(holdout_caches),
        "development_components": len(development_components),
        "holdout_components": len(holdout_components),
        "outer_oof_rmse": oof_scores,
        "untouched_holdout_rmse": holdout_scores,
        "untouched_holdout_gain": {
            "sequential_final_vs_incumbent": holdout_scores["incumbent"] - holdout_scores["sequential_final"],
            "nested_stack_vs_incumbent": holdout_scores["incumbent"] - holdout_scores["nested_stack"],
        },
        "component_bootstrap": {
            "sequential_final_vs_incumbent": sequential_bootstrap,
            "nested_stack_vs_incumbent": stack_bootstrap,
        },
        "final_stack": {
            "parent_weight": final_policy.parent_weight,
            "weights": {arm: float(weight) for arm, weight in zip(final_policy.arms, final_policy.weights)},
            "fit_source": (
                "predeclared_sequential_passthrough_for_frozen_arm_validation"
                if config.skip_nested_stack
                else "outer_oof_development_only"
            ),
        },
        "target_isolation": {
            "model_frame_columns": ["MD", "X", "Y", "Z", "GR", "TVT_input"],
            "suffix_truth_in_prediction_api": False,
            "scoring_truth_object_separate_from_model_input": True,
            "official_prefix_fallback_to_truth_count": 0,
            "holdout_used_for_selection": False,
            "component_overlap_count": 0,
        },
        "config": asdict(config),
        "historical_lineage_note": (
            "The scored 9.091 lineage is incumbent→HGRG→Meta-State→Prefix-Boundary→shape. "
            "The 121-feature retrain and nested stack are separately reported research controls."
        ),
    }
    if any(
        name in config.extra
        for name in ("split_seed", "refit_seed", "pf_seed_offset", "skip_nested_stack")
    ):
        summary["seed_contract"] = {
            "split_seed": config.split_seed,
            "refit_seed": config.refit_seed,
            "pf_seed_offset": config.pf_seed_offset,
            "pf_member_seeds": list(
                range(config.pf_seed_offset, config.pf_seed_offset + config.pf_seeds)
            ),
            "split_is_frozen_across_refits": True,
        }
    return NestedResult(
        summary=summary,
        outer_oof=outer_oof,
        holdout=holdout,
        stack_policy=final_policy,
        components=component_table,
        fold_records=pd.DataFrame(fold_records),
        component_universe=None if component_universe is None else component_universe.copy(),
    )
