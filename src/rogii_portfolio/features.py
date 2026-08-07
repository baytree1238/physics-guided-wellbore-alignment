"""Reimplementation of the 121-column FAST-SAFE feature schema.

The column names match the historical builder, but the formulas were derived
again for nested validation. The historical formulas are kept in
``historical_features.py``. Suffix TVT is not an input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import PreparedWell


SAFE_EXACT = {
    "last_known_tvt", "pf_ancc", "pf_ancc_std", "pf_ancc_delta", "pf_z",
    "pf_z_delta", "pf_vs_z", "beam_mean_d", "beam_std_d", "beam_med_d",
    "sc8_d", "sc8_sc", "sc15_d", "sc15_sc", "sc25_d", "sc25_sc",
    "sc_cons_d", "sc_ens_d", "sc_trust", "hyb_d", "sc_vs_beam", "cal_a",
    "cal_b", "pfx_rmse", "known_len", "eval_len", "slp_all", "slp_50",
    "slp_z", "slp_b_d_all", "slp_b_d_50", "ktvt_range", "ktvt_std",
    "md_since", "frac", "frac2", "sqrt_frac", "z", "dx", "dy", "dz",
    "dxy", "dzdmd", "dxdmd", "dydmd", "gr", "gr_d1", "gr_d2", "gr_env",
    "gr_nrg", "gr_vs_tw_anc", "gr_vs_slp_all", "tw_range", "tw_gr_mean",
}
SAFE_FAMILY_EXACT = {
    *(f"beam_{tag}_d" for tag in ("cons", "loose", "vcons", "sm5", "vloose", "mid", "stiff")),
    *(f"tda{offset}" for offset in (-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80)),
    *(f"tdbc{offset}" for offset in (-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40)),
    *(f"tdsc{offset}" for offset in (-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30)),
    *(f"tdpf{offset}" for offset in (-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30)),
    *(f"grm{window}" for window in (5, 21, 51, 101)),
    *(f"grs{window}" for window in (5, 21, 51, 101)),
    *(f"glag{lag}" for lag in (1, 5, 15, 30)),
    *(f"glead{lead}" for lead in (1, 5, 15, 30)),
}
SAFE_FEATURES = tuple(sorted(SAFE_EXACT | SAFE_FAMILY_EXACT))
if len(SAFE_FEATURES) != 121:
    raise AssertionError(f"FAST-SAFE contract changed: {len(SAFE_FEATURES)} != 121")


def _finite(values: np.ndarray, fill: float = 0.0) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, float), nan=fill, posinf=fill, neginf=fill)


def _fill_series(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, float))
    finite = np.isfinite(series.to_numpy(float))
    fill = float(np.nanmedian(series.to_numpy(float))) if finite.any() else 0.0
    return series.interpolate(limit_direction="both").fillna(fill).to_numpy(float)


def _robust_slope(md: np.ndarray, value: np.ndarray) -> float:
    delta_md = np.diff(md)
    delta = np.diff(value)
    valid = np.isfinite(delta_md) & np.isfinite(delta) & (delta_md > 0)
    return float(np.median(delta[valid] / delta_md[valid])) if valid.sum() >= 3 else 0.0


def _calibration(observed: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(observed) & np.isfinite(reference)
    if valid.sum() < 20:
        return 1.0, 0.0, 30.0
    x = reference[valid]
    y = observed[valid]
    design = np.c_[x, np.ones(len(x))]
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    a = float(np.clip(coefficient[0], 0.25, 4.0))
    b = float(np.clip(np.median(y - a * x), -500.0, 500.0))
    error = y - (a * x + b)
    return a, b, float(np.sqrt(np.mean(np.square(error))))


def build_fast_safe_features(
    prepared: PreparedWell,
    *,
    pf_full: np.ndarray,
    pf_std_suffix: np.ndarray,
    hmm_full: np.ndarray,
) -> pd.DataFrame:
    """Build the feature matrix in canonical ``SAFE_FEATURES`` order."""
    prepared.validate()
    frame = prepared.inference
    if "TVT" in frame.columns:
        raise RuntimeError("model frame must not expose suffix TVT")
    n = len(frame)
    if len(pf_full) != n or len(hmm_full) != n:
        raise ValueError("expert paths are not full-row aligned")
    suffix = prepared.suffix_rows
    if len(pf_std_suffix) != len(suffix):
        raise ValueError("PF uncertainty is not suffix aligned")
    tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if not len(known) or known[-1] >= suffix[0]:
        raise ValueError("features require a contiguous visible prefix")

    md = pd.to_numeric(frame["MD"], errors="coerce").to_numpy(float)
    x = pd.to_numeric(frame["X"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(frame["Y"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(frame["Z"], errors="coerce").to_numpy(float)
    gr = _fill_series(pd.to_numeric(frame["GR"], errors="coerce").to_numpy(float))
    tw_tvt = pd.to_numeric(prepared.typewell["TVT"], errors="coerce").to_numpy(float)
    tw_gr = _fill_series(pd.to_numeric(prepared.typewell["GR"], errors="coerce").to_numpy(float))
    finite_tw = np.isfinite(tw_tvt)
    tw_tvt, tw_gr = tw_tvt[finite_tw], tw_gr[finite_tw]

    last = int(known[-1])
    last_tvt = float(tvt_input[last])
    horizon = md[suffix] - md[last]
    slp_all = _robust_slope(md[known], tvt_input[known])
    tail = known[-min(50, len(known)):]
    slp_50 = _robust_slope(md[tail], tvt_input[tail])
    slp_z = _robust_slope(md[known], tvt_input[known] + z[known])
    straight_all = last_tvt + slp_all * horizon
    straight_50 = last_tvt + slp_50 * horizon
    pf = np.asarray(pf_full, float)[suffix]
    hmm = np.asarray(hmm_full, float)[suffix]
    pf_z = (last_tvt + z[last]) - z[suffix]

    slopes = {
        "cons": 0.75 * slp_50 + 0.25 * slp_all,
        "loose": 1.15 * slp_50,
        "vcons": 0.50 * slp_50 + 0.50 * slp_all,
        "sm5": 0.80 * slp_50 + 0.20 * slp_z,
        "vloose": 1.30 * slp_50,
        "mid": 0.50 * (slp_50 + slp_all),
        "stiff": slp_all,
    }
    beams = {name: last_tvt + slope * horizon for name, slope in slopes.items()}
    beam_matrix = np.column_stack(list(beams.values()))
    beam_mean = beam_matrix.mean(axis=1)
    beam_std = beam_matrix.std(axis=1)
    beam_med = np.median(beam_matrix, axis=1)

    known_tw = np.interp(tvt_input[known], tw_tvt, tw_gr)
    cal_a, cal_b, pfx_rmse = _calibration(gr[known], known_tw)

    def typewell_residual(candidate: np.ndarray, offset: float) -> np.ndarray:
        reference = cal_a * np.interp(candidate + offset, tw_tvt, tw_gr) + cal_b
        return gr[suffix] - reference

    sc8 = 0.80 * pf + 0.20 * hmm
    sc15 = 0.85 * pf + 0.15 * hmm
    sc25 = 0.75 * pf + 0.25 * hmm
    sc_cons = 0.50 * pf + 0.25 * hmm + 0.25 * beam_med
    sc_ens = np.mean(np.c_[pf, hmm, beam_mean, pf_z], axis=1)
    disagreement = np.std(np.c_[pf, hmm, beam_mean, pf_z], axis=1)
    sc_trust = 1.0 / (1.0 + np.square(disagreement / 6.0))
    hybrid = sc_trust * sc_cons + (1.0 - sc_trust) * beam_med

    output: dict[str, np.ndarray] = {}
    count = len(suffix)
    constant = lambda value: np.full(count, float(value), dtype=float)
    output.update(
        {
            "last_known_tvt": constant(last_tvt),
            "pf_ancc": pf,
            "pf_ancc_std": np.asarray(pf_std_suffix, float),
            "pf_ancc_delta": pf - last_tvt,
            "pf_z": pf_z,
            "pf_z_delta": pf_z - last_tvt,
            "pf_vs_z": pf - pf_z,
            "beam_mean_d": beam_mean - last_tvt,
            "beam_std_d": beam_std,
            "beam_med_d": beam_med - last_tvt,
            "sc8_d": sc8 - last_tvt,
            "sc8_sc": (sc8 - beam_mean) / np.maximum(beam_std, 1e-3),
            "sc15_d": sc15 - last_tvt,
            "sc15_sc": (sc15 - beam_mean) / np.maximum(beam_std, 1e-3),
            "sc25_d": sc25 - last_tvt,
            "sc25_sc": (sc25 - beam_mean) / np.maximum(beam_std, 1e-3),
            "sc_cons_d": sc_cons - last_tvt,
            "sc_ens_d": sc_ens - last_tvt,
            "sc_trust": sc_trust,
            "hyb_d": hybrid - last_tvt,
            "sc_vs_beam": sc_cons - beam_med,
            "cal_a": constant(cal_a),
            "cal_b": constant(cal_b),
            "pfx_rmse": constant(pfx_rmse),
            "known_len": constant(len(known)),
            "eval_len": constant(count),
            "slp_all": constant(slp_all),
            "slp_50": constant(slp_50),
            "slp_z": constant(slp_z),
            "slp_b_d_all": straight_all - beam_med,
            "slp_b_d_50": straight_50 - beam_med,
            "ktvt_range": constant(np.ptp(tvt_input[known])),
            "ktvt_std": constant(np.std(tvt_input[known])),
            "md_since": horizon,
            "frac": np.arange(1, count + 1, dtype=float) / count,
            "frac2": np.square(np.arange(1, count + 1, dtype=float) / count),
            "sqrt_frac": np.sqrt(np.arange(1, count + 1, dtype=float) / count),
            "z": z[suffix],
            "dx": x[suffix] - x[last],
            "dy": y[suffix] - y[last],
            "dz": z[suffix] - z[last],
            "dxy": np.hypot(x[suffix] - x[last], y[suffix] - y[last]),
            "dzdmd": (z[suffix] - z[last]) / np.maximum(horizon, 1e-6),
            "dxdmd": (x[suffix] - x[last]) / np.maximum(horizon, 1e-6),
            "dydmd": (y[suffix] - y[last]) / np.maximum(horizon, 1e-6),
            "gr": gr[suffix],
            "gr_d1": np.gradient(gr)[suffix],
            "gr_d2": np.gradient(np.gradient(gr))[suffix],
            "gr_env": np.abs(gr[suffix] - pd.Series(gr).rolling(51, center=True, min_periods=1).mean().to_numpy()[suffix]),
            "gr_nrg": np.square(gr[suffix] - float(np.mean(gr[known]))),
            "gr_vs_tw_anc": typewell_residual(pf_z, 0.0),
            "gr_vs_slp_all": typewell_residual(straight_all, 0.0),
            "tw_range": constant(np.ptp(tw_tvt)),
            "tw_gr_mean": constant(np.mean(tw_gr)),
        }
    )
    for tag, path in beams.items():
        output[f"beam_{tag}_d"] = path - last_tvt
    for offset in (-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80):
        output[f"tda{offset}"] = typewell_residual(hmm, float(offset))
    for offset in (-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40):
        output[f"tdbc{offset}"] = typewell_residual(beams["cons"], float(offset))
    for offset in (-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30):
        output[f"tdsc{offset}"] = typewell_residual(sc_cons, float(offset))
        output[f"tdpf{offset}"] = typewell_residual(pf, float(offset))
    gr_series = pd.Series(gr)
    for window in (5, 21, 51, 101):
        output[f"grm{window}"] = gr_series.rolling(window, center=True, min_periods=1).mean().to_numpy()[suffix]
        output[f"grs{window}"] = gr_series.rolling(window, center=True, min_periods=2).std().fillna(0.0).to_numpy()[suffix]
    for lag in (1, 5, 15, 30):
        output[f"glag{lag}"] = gr_series.shift(lag).bfill().to_numpy()[suffix]
        output[f"glead{lag}"] = gr_series.shift(-lag).ffill().to_numpy()[suffix]

    missing = sorted(set(SAFE_FEATURES) - set(output))
    extra = sorted(set(output) - set(SAFE_FEATURES))
    if missing or extra:
        raise AssertionError(f"121-feature construction mismatch: missing={missing}, extra={extra}")
    result = pd.DataFrame({name: _finite(output[name]) for name in SAFE_FEATURES})
    if result.shape != (count, 121) or not np.isfinite(result.to_numpy(float)).all():
        raise RuntimeError("FAST-SAFE feature matrix failed its finite/shape contract")
    return result
