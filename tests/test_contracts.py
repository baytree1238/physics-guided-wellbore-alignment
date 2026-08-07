from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rogii_portfolio.components import build_components, split_components
from rogii_portfolio.contracts import PipelineConfig, WellRecord
from rogii_portfolio.features import SAFE_FEATURES, build_fast_safe_features
from rogii_portfolio.hgrg import apply_hgrg, exact_l2_linf_project, radial_project, rms
from rogii_portfolio.meta_state import constrained_gls_weights
from rogii_portfolio.particle import pf_seed_trajectories
from rogii_portfolio.physics import prepare_scored_well
from rogii_portfolio.reproduce import select_complete_components
from rogii_portfolio.synthetic import make_synthetic_wells


def test_feature_schema_is_exactly_121_and_target_isolated() -> None:
    record = make_synthetic_wells(n_wells=10, rows=100)[0]
    config = PipelineConfig(outer_folds=2, inner_folds=2, pf_seeds=2, pf_particles=16)
    scored = prepare_scored_well(record, config)
    prepared = scored.model_input
    known = np.flatnonzero(np.isfinite(prepared.inference["TVT_input"]))
    last = known[-1]
    slope = np.median(np.diff(prepared.inference.loc[known, "TVT_input"]) / np.diff(prepared.inference.loc[known, "MD"]))
    pf_full = prepared.inference["TVT_input"].to_numpy(float).copy()
    pf_full[prepared.suffix_rows] = pf_full[last] + slope * prepared.horizon
    hmm_full = pf_full.copy()
    hmm_full[prepared.suffix_rows] += 0.2 * np.sin(prepared.horizon / 12.0)
    features = build_fast_safe_features(
        prepared,
        pf_full=pf_full,
        pf_std_suffix=np.full(len(prepared.suffix_rows), 0.3),
        hmm_full=hmm_full,
    )
    assert tuple(features.columns) == SAFE_FEATURES
    assert features.shape == (len(prepared.suffix_rows), 121)
    assert np.isfinite(features.to_numpy()).all()
    assert "TVT" not in prepared.inference
    assert prepared.inference.loc[prepared.suffix_rows, "TVT_input"].isna().all()

    poisoned_frame = record.horizontal.copy()
    poisoned_frame.loc[prepared.suffix_rows, "TVT"] += 100_000.0
    poisoned = WellRecord(record.well_id, poisoned_frame, record.typewell)
    poisoned_scored = prepare_scored_well(poisoned, config)
    assert poisoned_scored.model_input.inference.equals(prepared.inference)
    assert not np.array_equal(poisoned_scored.truth_suffix, scored.truth_suffix)


def test_hgrg_enforces_direction_and_movement_budgets() -> None:
    n = 200
    base = np.linspace(10_000.0, 10_150.0, n)
    pf = base + 20.0 * np.sin(np.linspace(0, 4, n))
    hmm = pf + 5.0 * np.cos(np.linspace(0, 3, n))
    ridge = base + 0.4 * (0.5 * (pf + hmm) - base)
    candidate, diagnostics = apply_hgrg(
        base=base,
        ridge=ridge,
        pf=pf,
        hmm=hmm,
        horizon=np.arange(n, dtype=float),
    )
    assert rms(candidate - base) <= 2.5 + 1e-10
    assert np.max(np.abs(candidate - base)) <= 10.0 + 1e-10
    assert 0.0 <= diagnostics["gate"] <= 1.0
    assert diagnostics["coefficient"] >= 0.0


def test_geological_pairs_remain_in_one_component_and_holdout_is_disjoint() -> None:
    wells = make_synthetic_wells(n_wells=18, rows=100)
    config = PipelineConfig(outer_folds=3, inner_folds=2, pf_seeds=2, pf_particles=16)
    table = build_components(wells, config)
    lookup = table.set_index("well")["component"]
    for index in range(0, 18, 2):
        assert lookup[f"syn_{index:03d}"] == lookup[f"syn_{index + 1:03d}"]
    development, holdout, folds = split_components(table, config)
    assert development.isdisjoint(holdout)
    assert set(folds) == development


def test_split_and_refit_randomness_are_separate_contracts() -> None:
    wells = make_synthetic_wells(n_wells=18, rows=100)
    common = {
        "split_seed": 1409,
        "pf_seed_offset": 200,
        "skip_nested_stack": True,
    }
    left = PipelineConfig(
        outer_folds=3,
        inner_folds=2,
        pf_seeds=2,
        pf_particles=16,
        extra={**common, "refit_seed": 3101},
    )
    right = replace(left, extra={**common, "refit_seed": 7727})
    table = build_components(wells, left)
    assert split_components(table, left) == split_components(table, right)
    assert left.split_seed == right.split_seed == 1409
    assert left.refit_seed != right.refit_seed
    assert left.pf_seed_offset == right.pf_seed_offset == 200
    assert left.skip_nested_stack is True


def test_particle_seed_offset_is_explicit_and_reproducible() -> None:
    record = make_synthetic_wells(n_wells=10, rows=100)[0]
    config = PipelineConfig(outer_folds=2, inner_folds=2, pf_seeds=2, pf_particles=16)
    prepared = prepare_scored_well(record, config).model_input
    first = pf_seed_trajectories(
        prepared.inference,
        prepared.typewell,
        seeds=2,
        particles=16,
        seed_offset=31,
    )
    repeated = pf_seed_trajectories(
        prepared.inference,
        prepared.typewell,
        seeds=2,
        particles=16,
        seed_offset=31,
    )
    shifted = pf_seed_trajectories(
        prepared.inference,
        prepared.typewell,
        seeds=2,
        particles=16,
        seed_offset=37,
    )
    assert first.seeds == repeated.seeds == (31, 32)
    assert shifted.seeds == (37, 38)
    assert np.array_equal(first.predictions, repeated.predictions)
    assert not np.array_equal(first.predictions, shifted.predictions)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"split_seed": True}, "split_seed"),
        ({"refit_seed": -1}, "refit_seed"),
        ({"pf_seed_offset": 2**32}, "pf_seed_offset"),
        ({"skip_nested_stack": 1}, "skip_nested_stack"),
    ],
)
def test_seed_contract_rejects_ambiguous_values(extra: dict[str, object], message: str) -> None:
    config = PipelineConfig(
        outer_folds=2,
        inner_folds=2,
        pf_seeds=2,
        pf_particles=16,
        extra=extra,
    )
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_official_mode_refuses_truth_derived_prefix_and_global_subset_is_component_complete() -> None:
    wells = make_synthetic_wells(n_wells=18, rows=100)
    full_config = PipelineConfig(
        outer_folds=3,
        inner_folds=2,
        pf_seeds=2,
        pf_particles=16,
        mode="full",
    )
    with pytest.raises(ValueError, match="cannot synthesize TVT_input from truth"):
        prepare_scored_well(wells[0], full_config)

    table = build_components(wells, full_config)
    primary, primary_table = select_complete_components(wells, table, max_wells=10, seed=17)
    primary_names = {well.well_id for well in primary}
    complement_table = table.loc[~table["well"].isin(primary_names)]
    assert len(primary) <= 10
    assert set(primary_table["component"]).isdisjoint(set(complement_table["component"]))


def test_constrained_gls_improves_on_clip_and_renormalize() -> None:
    covariance = np.array(
        [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.8],
            [0.1, 0.8, 1.0],
        ]
    )
    raw = np.linalg.solve(covariance, np.ones(3))
    clipped = np.maximum(raw, 0.0)
    clipped /= clipped.sum()
    constrained = constrained_gls_weights(covariance)
    assert np.all(constrained >= 0)
    assert np.isclose(constrained.sum(), 1.0)
    assert constrained @ covariance @ constrained <= clipped @ covariance @ clipped + 1e-12


def test_exact_l2_linf_projection_is_not_radial_scaling() -> None:
    base = np.zeros(2)
    candidate = np.array([10.0, 1.0])
    radial, _ = radial_project(base, candidate, rms_cap=100.0, row_cap=5.0)
    exact = exact_l2_linf_project(base, candidate, rms_cap=100.0, row_cap=5.0)
    assert np.allclose(radial, [5.0, 0.5])
    assert np.allclose(exact, [5.0, 1.0])
    assert np.linalg.norm(candidate - exact) < np.linalg.norm(candidate - radial)

    exact = exact_l2_linf_project(base, candidate, rms_cap=2.0, row_cap=5.0)
    assert rms(exact - base) <= 2.0 + 1e-10
    assert np.max(np.abs(exact - base)) <= 5.0 + 1e-10
