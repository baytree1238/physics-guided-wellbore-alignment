from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii_portfolio.refit_stability import hierarchical_component_seed_bootstrap


def _component_table() -> pd.DataFrame:
    rows = []
    for refit_seed, gain in ((3101, 0.30), (7727, 0.20), (19001, 0.10)):
        for component, parent_rmse in (("a", 2.0), ("b", 3.0), ("c", 4.0)):
            count = 100
            candidate_rmse = parent_rmse - gain
            rows.append(
                {
                    "refit_seed": refit_seed,
                    "component": component,
                    "rows": count,
                    "parent_sse": count * parent_rmse**2,
                    "candidate_sse": count * candidate_rmse**2,
                }
            )
    return pd.DataFrame(rows)


def test_hierarchical_bootstrap_is_deterministic_and_resamples_both_axes() -> None:
    table = _component_table()
    first, first_draws = hierarchical_component_seed_bootstrap(table, draws=500, seed=17)
    second, second_draws = hierarchical_component_seed_bootstrap(table, draws=500, seed=17)
    assert first == second
    assert first["refit_seeds"] == 3
    assert first["components"] == 3
    assert first["ci95_low"] > 0.0
    assert np.array_equal(first_draws.to_numpy(), second_draws.to_numpy())


def test_hierarchical_bootstrap_rejects_an_incomplete_seed_component_grid() -> None:
    table = _component_table().iloc[:-1].copy()
    with pytest.raises(ValueError, match="every refit seed"):
        hierarchical_component_seed_bootstrap(table, draws=20, seed=17)
