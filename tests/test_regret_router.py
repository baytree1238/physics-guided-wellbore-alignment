from __future__ import annotations

import numpy as np
import pandas as pd

from rogii_portfolio.regret_router import evaluate_nested_regret_router, well_level_table


def _rows(wells: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for well_index in range(wells):
        n = 12
        truth = 10_000 + np.arange(n) + rng.normal(0, 0.4, n)
        incumbent = truth + rng.normal(0, 1.0, n)
        pf = incumbent + rng.normal(0, 0.4, n)
        hgrg = incumbent + 0.25 * (pf - incumbent)
        meta = hgrg + rng.normal(0, 0.15, n)
        boundary = meta + rng.normal(0, 0.10, n)
        sequential = boundary + rng.normal(0, 0.08, n)
        nested = 0.5 * incumbent + 0.5 * sequential
        for row in range(n):
            records.append(
                {
                    "row_id": f"w{well_index}_{row}",
                    "well": f"w{well_index}",
                    "row_number": row,
                    "component": f"c{well_index // 2}",
                    "horizon_ft": row + 1.0,
                    "truth": truth[row],
                    "pf": pf[row],
                    "incumbent": incumbent[row],
                    "hgrg": hgrg[row],
                    "meta_state": meta[row],
                    "prefix_boundary": boundary[row],
                    "sequential_final": sequential[row],
                    "nested_stack": nested[row],
                }
            )
    return pd.DataFrame(records)


def test_router_features_are_target_free_and_nested_graph_runs() -> None:
    development = _rows(30, seed=1)
    holdout = _rows(10, seed=2)
    base, names = well_level_table(development)
    poisoned = development.copy()
    poisoned["truth"] += 100_000.0
    changed, changed_names = well_level_table(poisoned)
    assert names == changed_names
    assert np.array_equal(base[names].to_numpy(float), changed[names].to_numpy(float))

    summary, rows, decisions = evaluate_nested_regret_router(
        development,
        holdout,
        bootstrap_draws=50,
    )
    assert np.isfinite(rows["regret_router"]).all()
    assert summary["selected_blend"] in (0.25, 0.5, 1.0)
    assert summary["selected_quantile"] in (0.5, 0.75, 0.9)
    assert set(decisions["selected_arm"]) <= {
        "incumbent", "hgrg", "meta_state", "prefix_boundary", "sequential_final", "nested_stack"
    }
