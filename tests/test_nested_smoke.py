from __future__ import annotations

from dataclasses import replace

import numpy as np

from rogii_portfolio.contracts import PipelineConfig
from rogii_portfolio.pipeline import run_nested_experiment
from rogii_portfolio.synthetic import make_synthetic_wells


def test_full_graph_runs_with_nested_component_contract() -> None:
    config = PipelineConfig(
        outer_folds=3,
        inner_folds=2,
        holdout_fraction=0.20,
        pf_seeds=2,
        pf_particles=16,
        hmm_stride=12,
        hmm_checkpoint=12,
        model_max_iter=5,
        model_max_leaf_nodes=7,
        bootstrap_draws=40,
        mode="smoke",
        extra={"synthetic_wells": 18, "synthetic_rows": 100},
    )
    result = run_nested_experiment(make_synthetic_wells(18, 100), config)
    assert result.summary["target_isolation"]["component_overlap_count"] == 0
    assert result.summary["target_isolation"]["holdout_used_for_selection"] is False
    assert result.summary["feature_count"] == 121
    assert "nested_stack" in result.holdout.predictions
    assert np.isfinite(result.holdout.predictions["sequential_final"]).all()
    assert 0.0 <= result.stack_policy.parent_weight <= 1.0
    # The reported development stack must be the fold-frozen OOF path, not a
    # policy refit and re-scored on those same labels.
    assert result.summary["final_stack"]["fit_source"] == "outer_oof_development_only"
    fold_rows = result.fold_records
    assert len(fold_rows) == config.outer_folds


def test_frozen_trust_region_arm_uses_no_weight_refit() -> None:
    config = PipelineConfig(
        seed=91,
        outer_folds=3,
        inner_folds=2,
        holdout_fraction=0.20,
        pf_seeds=2,
        pf_particles=16,
        hmm_stride=12,
        hmm_checkpoint=12,
        model_max_iter=3,
        model_max_leaf_nodes=7,
        bootstrap_draws=20,
        mode="smoke",
        extra={
            "split_seed": 91,
            "refit_seed": 3101,
            "pf_seed_offset": 3101,
            "skip_nested_stack": True,
            "enable_trust_region_ridge": True,
            "trust_region_ridge_weight": 0.05,
        },
    )
    result = run_nested_experiment(make_synthetic_wells(18, 100), config)
    for bundle in (result.outer_oof, result.holdout):
        expected = bundle.predictions["sequential_final"] + 0.05 * (
            bundle.predictions["ridge"] - bundle.predictions["sequential_final"]
        )
        assert np.allclose(bundle.predictions["trust_region_ridge"], expected, atol=1e-12)
        assert np.array_equal(
            bundle.predictions["nested_stack"], bundle.predictions["sequential_final"]
        )
    assert result.summary["final_stack"]["fit_source"] == (
        "predeclared_sequential_passthrough_for_frozen_arm_validation"
    )
    assert result.summary["seed_contract"]["split_seed"] == 91
    assert result.summary["seed_contract"]["refit_seed"] == 3101
    assert result.summary["seed_contract"]["pf_member_seeds"] == [3101, 3102]
