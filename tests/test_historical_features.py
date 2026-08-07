from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rogii_portfolio.contracts import PipelineConfig
from rogii_portfolio.features import SAFE_FEATURES
from rogii_portfolio.historical_features import (
    HISTORICAL_FEATURE_ORDER,
    HISTORICAL_ORIGINAL_BUILDER_SHA256,
    HISTORICAL_SAFE_SOURCE_SHA256,
    build_historical_fast_safe_frame,
    canonical_historical_feature_hash,
    verify_historical_source_identity,
)
from rogii_portfolio.physics import prepare_well
from rogii_portfolio.synthetic import make_synthetic_wells


def _write_historical_inputs(directory: Path) -> tuple[Path, Path, np.ndarray]:
    record = make_synthetic_wells(n_wells=10, rows=100)[0]
    config = PipelineConfig(outer_folds=2, inner_folds=2, pf_seeds=2, pf_particles=16)
    scored = prepare_well(record, config)
    prepared = scored.model_input
    horizontal = prepared.inference.copy()
    horizontal["TVT"] = record.horizontal["TVT"].to_numpy(float)
    horizontal_path = directory / f"{record.well_id}__horizontal_well.csv"
    typewell_path = directory / f"{record.well_id}__typewell.csv"
    horizontal.to_csv(horizontal_path, index=False)
    record.typewell.to_csv(typewell_path, index=False)
    return horizontal_path, typewell_path, prepared.suffix_rows


def test_historical_source_and_schema_are_frozen(tmp_path: Path) -> None:
    assert HISTORICAL_ORIGINAL_BUILDER_SHA256 == (
        "500c9255b3bfd2268c8bb2e9be1a4025f0aa01d3483e67b4e28fa31e1d5d2c2c"
    )
    assert HISTORICAL_SAFE_SOURCE_SHA256 == (
        "0e5f184ceee12abd4a1c4ce560dd544c009a6a5689a8430be7c476b15fcaf6f3"
    )
    assert HISTORICAL_FEATURE_ORDER == SAFE_FEATURES
    assert verify_historical_source_identity()
    horizontal_path, typewell_path, suffix_rows = _write_historical_inputs(tmp_path)
    frame = build_historical_fast_safe_frame(horizontal_path, typewell_path)
    assert tuple(frame.columns) == ("well", "id", *SAFE_FEATURES)
    assert frame.shape == (len(suffix_rows), 123)
    assert np.isfinite(frame.loc[:, SAFE_FEATURES].to_numpy(float)).all()


def test_historical_features_are_bitwise_repeatable_and_ignore_suffix_truth(
    tmp_path: Path,
) -> None:
    horizontal_path, typewell_path, suffix_rows = _write_historical_inputs(tmp_path)
    first = build_historical_fast_safe_frame(horizontal_path, typewell_path)
    first_hash = canonical_historical_feature_hash(first)

    # Poison only the scoring suffix. The builder reads the horizontal CSV
    # with an observable allow-list, so TVT cannot affect one feature byte.
    horizontal = pd.read_csv(horizontal_path)
    horizontal.loc[suffix_rows, "TVT"] += 100_000.0
    horizontal.to_csv(horizontal_path, index=False)
    poisoned = build_historical_fast_safe_frame(horizontal_path, typewell_path)

    assert first["id"].equals(poisoned["id"])
    a = np.ascontiguousarray(first.loc[:, SAFE_FEATURES].to_numpy(dtype="<f4"))
    b = np.ascontiguousarray(poisoned.loc[:, SAFE_FEATURES].to_numpy(dtype="<f4"))
    assert np.array_equal(a.view("<u4"), b.view("<u4"))
    assert canonical_historical_feature_hash(poisoned) == first_hash
