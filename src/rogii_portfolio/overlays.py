"""One-sided prefix-boundary and conditional GeoHMM shape corrections."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import PreparedWell
from .hgrg import radial_project, rms


def _slope(md: np.ndarray, values: np.ndarray, window_ft: float) -> float:
    use = md >= md[-1] - window_ft
    x, y = md[use], values[use]
    if len(x) < 8:
        return 0.0
    centered = x - x[-1]
    denominator = float(centered @ centered)
    return float(centered @ (y - y[-1]) / denominator) if denominator > 1e-12 else 0.0


def apply_prefix_boundary(
    prepared: PreparedWell,
    base: np.ndarray,
    *,
    tau_ft: float = 256.0,
    alpha: float = 1.0,
    rms_budget_ft: float = 2.5,
    row_cap_ft: float = 10.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Blend a decaying visible-prefix tangent into a protected parent path."""

    frame = prepared.inference
    tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(frame["Z"], errors="coerce").to_numpy(float)
    md = pd.to_numeric(frame["MD"], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input))
    u = tvt_input[known] + z[known]
    slope = _slope(md[known], u, 256.0)
    tangent = u[-1] + slope * prepared.horizon
    structural = np.asarray(base, float) + z[prepared.suffix_rows]
    direction = np.clip(np.exp(-prepared.horizon / tau_ft) * (tangent - structural), -row_cap_ft, row_cap_ft)
    raw = np.asarray(base, float) + alpha * direction
    candidate, projection = radial_project(base, raw, rms_cap=rms_budget_ft, row_cap=row_cap_ft)
    return candidate, {
        "prefix_u0": float(u[-1]),
        "prefix_slope": slope,
        "projection": projection,
        "move_rms_ft": rms(candidate - base),
        "move_absmax_ft": float(np.max(np.abs(candidate - base))),
    }


def apply_conditional_shape(
    *,
    base: np.ndarray,
    pf: np.ndarray,
    hmm_stride6: np.ndarray,
    alpha: float = 0.25,
    amplitude_scale_ft: float = 6.41825,
    slope_scale: float = 0.0090056,
    rms_budget_ft: float = 0.75,
    row_cap_ft: float = 2.5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Add centered PF/GeoHMM shape information under fixed movement limits."""

    raw_direction = np.asarray(hmm_stride6, float) - np.asarray(pf, float)
    centered = raw_direction - float(np.mean(raw_direction))
    amplitude = rms(raw_direction)
    slope_information = rms(np.diff(raw_direction)) / max(amplitude, 1e-12) if len(raw_direction) > 1 else 0.0
    gate = (
        1.0 / (1.0 + (amplitude / amplitude_scale_ft) ** 2)
        * slope_information / (slope_information + slope_scale)
    )
    raw = np.asarray(base, float) + alpha * gate * centered
    candidate, projection = radial_project(base, raw, rms_cap=rms_budget_ft, row_cap=row_cap_ft)
    return candidate, {
        "correction_rms_ft": amplitude,
        "normalized_slope_rms": slope_information,
        "gate": float(gate),
        "projection": projection,
        "preprojection_mean_move_ft": float(np.mean(alpha * gate * centered)),
        "move_rms_ft": rms(candidate - base),
    }
