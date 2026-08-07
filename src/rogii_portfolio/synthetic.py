"""Deterministic synthetic wells for testing the validation pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import WellRecord


def make_synthetic_wells(
    n_wells: int = 18,
    rows: int = 180,
    *,
    seed: int = 20260806,
) -> list[WellRecord]:
    """Build paired wells that share local structure and nearby coordinates."""
    if n_wells < 10 or rows < 100:
        raise ValueError("synthetic nested validation needs >=10 wells and >=100 rows")
    rng = np.random.default_rng(seed)
    result: list[WellRecord] = []
    tw_grid = np.arange(9_700.0, 10_500.5, 0.5)
    for index in range(n_wells):
        component = index // 2
        local = np.random.default_rng(seed + 1009 * component)
        md = 10_000.0 + np.arange(rows, dtype=float)
        x0 = 2_900_000.0 + 8_000.0 * component
        y0 = 1_000_000.0 + 5_000.0 * component
        x = x0 + 0.35 * np.arange(rows) + (index % 2) * 70.0
        y = y0 + 0.18 * np.arange(rows) + (index % 2) * 55.0
        z = -8_300.0 - 0.88 * np.arange(rows) + 0.3 * np.sin(np.arange(rows) / 23.0)
        curvature = (component % 3 - 1) * 1.4e-4
        tvt = (
            10_000.0
            + (0.94 + 0.008 * (component % 4)) * np.arange(rows)
            + curvature * np.square(np.arange(rows))
            + 0.35 * np.sin(np.arange(rows) / (17.0 + component % 5))
        )
        phase = 0.5 * component
        tw_gr = (
            95.0
            + 23.0 * np.sin((tw_grid - 9_700.0) / 15.0 + phase)
            + 10.0 * np.sin((tw_grid - 9_700.0) / 4.7)
            + local.normal(0.0, 1.0, len(tw_grid))
        )
        gr = np.interp(tvt, tw_grid, tw_gr) + rng.normal(0.0, 5.0, rows)
        u = tvt + z
        formation_base = float(np.median(u)) + 100.0
        frame = pd.DataFrame(
            {
                "MD": md,
                "X": x,
                "Y": y,
                "Z": z,
                "ANCC": formation_base + 150.0,
                "ASTNU": formation_base,
                "ASTNL": formation_base - 35.0,
                "EGFDU": formation_base - 90.0,
                "EGFDL": formation_base - 125.0,
                "BUDA": formation_base - 230.0,
                "TVT": tvt,
                "GR": gr,
                "TVT_input": tvt,
            }
        )
        typewell = pd.DataFrame({"TVT": tw_grid, "GR": tw_gr, "Geology": ""})
        result.append(WellRecord(f"syn_{index:03d}", frame, typewell))
    return result
