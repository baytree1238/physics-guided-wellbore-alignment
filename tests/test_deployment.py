from __future__ import annotations

from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from rogii_portfolio.contracts import PreparedWell, WellRecord
from rogii_portfolio.deployment import verify_submission, write_submission
from rogii_portfolio.physics import prepare_inference_well
from rogii_portfolio.parents import HistoricalArtifactParent
from rogii_portfolio.artifacts import sha256_file
from rogii_portfolio.synthetic import make_synthetic_wells


def test_inference_object_has_no_truth_and_requires_a_real_prefix() -> None:
    source = make_synthetic_wells(n_wells=10, rows=100)[0]
    horizontal = source.horizontal.drop(columns=["TVT"]).copy()
    horizontal.loc[62:, "TVT_input"] = np.nan
    target = WellRecord(source.well_id, horizontal, source.typewell)
    prepared = prepare_inference_well(target)
    assert "truth" not in {field.name for field in fields(PreparedWell)}
    assert prepared.inference.loc[prepared.suffix_rows, "TVT_input"].isna().all()

    broken = horizontal.copy()
    broken.loc[80, "TVT_input"] = 10_000.0
    with pytest.raises(ValueError, match="contiguous visible prefix"):
        prepare_inference_well(WellRecord(source.well_id, broken, source.typewell))


def test_submission_writer_refuses_reordering_and_records_hashes(tmp_path) -> None:
    sample = pd.DataFrame({"id": ["w_2", "w_3"], "tvt": [0.0, 0.0]})
    frame = pd.DataFrame({"id": ["w_2", "w_3"], "tvt": np.asarray([2.5, 3.5], np.float64)})
    assert verify_submission(frame, sample)["status"] == "PASS"
    output = tmp_path / "submission.csv"
    audit = write_submission(frame, sample, output, audit_path=tmp_path / "audit.json")
    assert output.exists() and len(audit["csv_sha256"]) == 64

    reversed_frame = frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="ID order"):
        verify_submission(reversed_frame, sample)


def test_historical_parent_requires_pinned_bytes_and_exact_ids(tmp_path) -> None:
    path = tmp_path / "experts.csv"
    pd.DataFrame(
        {
            "id": ["w_2", "w_3"],
            "incumbent": [2.0, 3.0],
            "ridge": [2.1, 3.1],
            "pf": [1.9, 2.9],
            "hmm": [2.2, 3.2],
        }
    ).to_csv(path, index=False)
    provider = HistoricalArtifactParent.from_csv(path, expected_sha256=sha256_file(path))
    prediction = provider.predict(np.asarray(["w_3", "w_2"]))
    assert prediction["incumbent"].tolist() == [3.0, 2.0]
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        HistoricalArtifactParent.from_csv(path, expected_sha256="0" * 64)
