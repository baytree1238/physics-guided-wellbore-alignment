from __future__ import annotations

import numpy as np
import pandas as pd

from rogii_portfolio.trust_region import apply_trust_region_blend, fit_trust_region_blend


def test_trust_region_respects_cap_and_improves_parent() -> None:
    frame = pd.DataFrame(
        {
            "component": np.repeat(["a", "b", "c"], 20),
            "truth": np.tile(np.linspace(0.0, 1.0, 20), 3),
        }
    )
    frame["sequential_final"] = frame["truth"] + 1.0
    frame["ridge"] = frame["truth"] - 1.0
    policy = fit_trust_region_blend(frame, maximum_weight=0.30, grid_step=0.01)
    prediction = apply_trust_region_blend(frame, policy)
    assert 0.0 <= policy.weight <= 0.30
    parent_mse = np.mean(np.square(frame["truth"] - frame["sequential_final"]))
    candidate_mse = np.mean(np.square(frame["truth"].to_numpy(float) - prediction))
    assert candidate_mse < parent_mse
