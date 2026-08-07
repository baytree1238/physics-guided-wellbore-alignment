from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rogii_portfolio.evaluation import (
    ExperimentRegistry,
    _bootstrap_rank_draws,
    _fold_metrics,
    _horizon_metrics,
    _load_predictions,
    _long_group_metrics,
    _model_metrics,
)


MODELS = ("incumbent", "candidate")


def example_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"r{i}" for i in range(6)],
            "well": ["w1", "w1", "w1", "w1", "w2", "w3"],
            "row_number": [1, 2, 3, 4, 1, 1],
            "component": ["c1", "c1", "c1", "c1", "c2", "c3"],
            "horizon_ft": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "truth": np.zeros(6),
            "incumbent": np.full(6, 2.0),
            "candidate": [1.0, 1.0, 1.0, 1.0, 3.0, 4.0],
        }
    )


def test_group_metrics_do_not_hide_small_wells() -> None:
    frame = example_frame()
    wells = _long_group_metrics(frame, MODELS, "incumbent", "well", "demo")
    components = _long_group_metrics(frame, MODELS, "incumbent", "component", "demo")
    metrics = _model_metrics(
        frame,
        MODELS,
        "incumbent",
        "demo",
        wells,
        components,
        0.10,
    ).set_index("model")
    candidate = metrics.loc["candidate"]
    assert candidate["pooled_row_rmse"] == pytest.approx(math.sqrt(29 / 6))
    assert candidate["macro_well_rmse"] == pytest.approx(8 / 3)
    assert candidate["macro_component_rmse"] == pytest.approx(8 / 3)
    assert candidate["harmed_well_rate"] == pytest.approx(2 / 3)
    assert candidate["worst_10pct_component_rmse_cvar"] == pytest.approx(4.0)
    assert candidate["worst_tail_components"] == 1


def test_horizon_bins_include_boundaries_once() -> None:
    result = _horizon_metrics(
        example_frame(), MODELS, "incumbent", "demo", (2.0, 4.0)
    )
    counts = (
        result.loc[result["model"] == "candidate"]
        .set_index("horizon_bin")["rows"]
        .to_dict()
    )
    assert counts == {"0-2": 2, ">2-4": 2, ">4": 2}


def test_fold_and_bootstrap_rank_results_are_reproducible() -> None:
    frame = example_frame()
    frame["outer_fold"] = [0, 0, 0, 0, 1, 1]
    folds = _fold_metrics(frame, MODELS, "incumbent", "demo")
    candidate = folds.loc[folds["model"] == "candidate"].set_index("outer_fold")
    assert candidate.loc[0, "rank"] == 1
    assert candidate.loc[1, "rank"] == 2

    components = _long_group_metrics(
        frame, MODELS, "incumbent", "component", "demo"
    )
    first = _bootstrap_rank_draws(components, MODELS, "incumbent", "demo", 25, 101)
    second = _bootstrap_rank_draws(components, MODELS, "incumbent", "demo", 25, 101)
    pd.testing.assert_frame_equal(first, second)
    assert first["seed"].nunique() == 25


def test_prediction_loader_rejects_duplicate_row_ids(tmp_path) -> None:
    frame = example_frame()
    frame.loc[1, "row_id"] = frame.loc[0, "row_id"]
    path = tmp_path / "predictions.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="row_id_unique"):
        _load_predictions(path, MODELS, "demo", [])


def test_registry_requires_baseline_and_valid_disjoint_panels() -> None:
    contract = {
        "contract_id": "demo_oof",
        "panel": "demo",
        "split": "outer_oof",
        "role": "development",
        "training_seed": 7,
        "predictions": "predictions.csv",
        "components": "components.csv",
        "summary": "summary.json",
        "config": "config.json",
        "manifest": "manifest.json",
        "fold_metrics": "folds.csv",
    }
    payload = {
        "schema": "rogii_robustness_registry_v1",
        "baseline": "incumbent",
        "models": list(MODELS),
        "horizon_upper_bounds_ft": [100, 500],
        "component_bootstrap": {"draws": 10, "seed": 7},
        "worst_component_fraction": 0.1,
        "contracts": [contract],
    }
    registry = ExperimentRegistry.from_dict(payload)
    assert registry.baseline == "incumbent"
    payload["baseline"] = "missing"
    with pytest.raises(ValueError, match="baseline"):
        ExperimentRegistry.from_dict(payload)
