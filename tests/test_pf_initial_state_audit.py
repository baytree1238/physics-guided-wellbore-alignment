import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "rogii_pf_initial_state_audit.py"
SPEC = importlib.util.spec_from_file_location("rogii_pf_initial_state_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _known() -> pd.DataFrame:
    md = np.arange(140.0)
    return pd.DataFrame(
        {
            "MD": md,
            "Z": 100.0 + 0.25 * md,
            "GR": 50.0,
            "TVT_input": 1000.0 + 0.75 * md,
        }
    )


def test_grid_is_preregistered_and_zero_arms_are_deduplicated() -> None:
    arms = audit.registered_arms()
    assert len(arms) == 36
    assert len({arm.name for arm in arms}) == 36
    assert len(audit.compute_arms()) == 25
    assert audit.REFERENCE == "w030_median_difference_s1.0"


def test_all_estimators_recover_linear_structural_rate() -> None:
    known = _known()
    for window in audit.WINDOWS:
        for estimator in audit.ESTIMATORS:
            arm = audit.Arm(window, estimator, 1.0)
            assert np.isclose(audit.initial_rate(known, arm), 1.0, atol=1e-14)


def test_shrink_is_applied_only_to_initial_rate() -> None:
    known = _known()
    raw = audit.initial_rate(known, audit.Arm(60, "ols", 1.0))
    assert audit.initial_rate(known, audit.Arm(60, "ols", 0.5)) == 0.5 * raw
    for window in audit.WINDOWS:
        for estimator in audit.ESTIMATORS:
            assert audit.initial_rate(known, audit.Arm(window, estimator, 0.0)) == 0.0


def test_prediction_contract_excludes_suffix_truth_and_spatial_inputs() -> None:
    assert audit.MODEL_COLUMNS == ("MD", "Z", "GR", "TVT_input")
    assert audit.TYPEWELL_COLUMNS == ("TVT", "GR")
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert 'horizontal.loc[:, MODEL_COLUMNS]' in source
    assert 'typewell.loc[:, TYPEWELL_COLUMNS]' in source
    assert 'raw_horizontal.loc[evaluation, "TVT"]' in source


def test_rmse_and_fold_scoring() -> None:
    truth = np.array([0.0, 2.0, 0.0, 4.0])
    prediction = np.array([0.0, 0.0, 0.0, 0.0])
    folds = np.array([0, 0, 1, 1])
    assert np.isclose(audit.rmse(truth, prediction), np.sqrt(5.0))
    assert np.allclose(audit.fold_scores(truth, prediction, folds), [np.sqrt(2.0), np.sqrt(8.0)])
