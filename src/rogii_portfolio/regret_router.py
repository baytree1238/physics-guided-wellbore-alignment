"""Well-level pairwise-regret routing with group-cross-fitted abstention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .stack import component_bootstrap, rmse


FALLBACK = "incumbent"
CANDIDATES = ("hgrg", "meta_state", "prefix_boundary", "sequential_final", "nested_stack")


@dataclass
class _RegretModel:
    scaler: StandardScaler
    ridge: Ridge

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.ridge.predict(self.scaler.transform(x)), float)


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    return float(np.sqrt(np.mean(np.square(values))))


def well_level_table(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate routing features and per-arm gain labels at well level."""

    required = {"well", "component", "horizon_ft", "truth", "pf", FALLBACK, *CANDIDATES}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"router input is missing {sorted(missing)}")
    records: list[dict[str, object]] = []
    for well, group in rows.groupby("well", sort=True):
        group = group.sort_values("row_number", kind="mergesort")
        base = group[FALLBACK].to_numpy(float)
        truth = group["truth"].to_numpy(float)
        horizon = group["horizon_ft"].to_numpy(float)
        record: dict[str, object] = {
            "well": str(well),
            "component": str(group["component"].iloc[0]),
            "rows": len(group),
            "horizon_max": float(np.max(horizon)),
            "horizon_mean": float(np.mean(horizon)),
            "pf_base_rms": _rms(group["pf"].to_numpy(float) - base),
            "pf_base_absmax": float(np.max(np.abs(group["pf"].to_numpy(float) - base))),
            "pf_hgrg_rms": _rms(group["pf"].to_numpy(float) - group["hgrg"].to_numpy(float)),
        }
        previous = base
        for arm in CANDIDATES:
            values = group[arm].to_numpy(float)
            move = values - base
            local = values - previous
            record[f"{arm}__move_rms"] = _rms(move)
            record[f"{arm}__move_absmax"] = float(np.max(np.abs(move)))
            record[f"{arm}__local_rms"] = _rms(local)
            record[f"{arm}__roughness"] = _rms(np.diff(move)) if len(move) > 1 else 0.0
            record[f"gain__{arm}"] = float(
                np.mean(np.square(truth - base)) - np.mean(np.square(truth - values))
            )
            previous = values
        record["fallback_mse"] = float(np.mean(np.square(truth - base)))
        records.append(record)
    table = pd.DataFrame(records)
    feature_names = [
        column
        for column in table.columns
        if column not in {"well", "component", "fallback_mse"}
        and not column.startswith("gain__")
    ]
    if not np.isfinite(table[feature_names].to_numpy(float)).all():
        raise ValueError("router features are non-finite")
    return table, feature_names


def _component_folds(table: pd.DataFrame, folds: int, seed: int) -> np.ndarray:
    components = table.groupby("component", sort=True).size().sort_values(ascending=False)
    if len(components) < folds:
        raise ValueError("not enough components for regret-router folds")
    rng = np.random.default_rng(seed)
    tie = {component: float(rng.random()) for component in components.index}
    ordered = sorted(components.index, key=lambda c: (-int(components[c]), tie[c], str(c)))
    load = np.zeros(folds, int)
    mapping: dict[str, int] = {}
    for component in ordered:
        fold = int(np.argmin(load))
        mapping[str(component)] = fold
        load[fold] += int(components[component])
    return table["component"].astype(str).map(mapping).to_numpy(int)


def _weights(components: np.ndarray) -> np.ndarray:
    values, counts = np.unique(components.astype(str), return_counts=True)
    lookup = {value: count for value, count in zip(values, counts)}
    result = np.asarray([1.0 / lookup[value] for value in components.astype(str)], float)
    return result / np.mean(result)


def _fit(x: np.ndarray, y: np.ndarray, components: np.ndarray) -> _RegretModel:
    weight = _weights(components)
    scaler = StandardScaler().fit(x, sample_weight=weight)
    ridge = Ridge(alpha=10.0).fit(scaler.transform(x), y, sample_weight=weight)
    return _RegretModel(scaler, ridge)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="mergesort")
    values, weights = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return float(values[min(int(np.searchsorted(cdf, q, side="left")), len(values) - 1)])


def _crossfit_gain_predictions(
    table: pd.DataFrame,
    feature_names: list[str],
    *,
    folds: int,
    seed: int,
) -> dict[str, np.ndarray]:
    assignment = _component_folds(table, folds, seed)
    x = table[feature_names].to_numpy(float)
    components = table["component"].to_numpy(str)
    output = {arm: np.full(len(table), np.nan) for arm in CANDIDATES}
    for fold in range(folds):
        train, valid = assignment != fold, assignment == fold
        for arm in CANDIDATES:
            model = _fit(x[train], table.loc[train, f"gain__{arm}"].to_numpy(float), components[train])
            output[arm][valid] = model.predict(x[valid])
    if any(not np.isfinite(values).all() for values in output.values()):
        raise RuntimeError("regret cross-fitting left non-finite predictions")
    return output


def _fit_with_margins(
    table: pd.DataFrame,
    feature_names: list[str],
    *,
    quantiles: tuple[float, ...],
    inner_folds: int,
    seed: int,
) -> tuple[dict[str, _RegretModel], dict[float, dict[str, float]]]:
    inner = _crossfit_gain_predictions(table, feature_names, folds=inner_folds, seed=seed)
    x = table[feature_names].to_numpy(float)
    components = table["component"].to_numpy(str)
    weight = _weights(components)
    models: dict[str, _RegretModel] = {}
    margins = {q: {} for q in quantiles}
    for arm in CANDIDATES:
        target = table[f"gain__{arm}"].to_numpy(float)
        residual = np.abs(target - inner[arm])
        for q in quantiles:
            margins[q][arm] = _weighted_quantile(residual, weight, q)
        models[arm] = _fit(x, target, components)
    return models, margins


def _route(
    rows: pd.DataFrame,
    wells: pd.DataFrame,
    predicted_gain: dict[str, np.ndarray],
    margins: dict[str, float],
    *,
    blend: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    lcb = np.column_stack([predicted_gain[arm] - margins[arm] for arm in CANDIDATES])
    best_index = np.argmax(lcb, axis=1)
    best_lcb = lcb[np.arange(len(wells)), best_index]
    selected = np.asarray([CANDIDATES[index] for index in best_index], object)
    selected[best_lcb <= 0.0] = FALLBACK
    choice = dict(zip(wells["well"].astype(str), selected))
    confidence = dict(zip(wells["well"].astype(str), np.maximum(best_lcb, 0.0)))
    output = rows[FALLBACK].to_numpy(float).copy()
    for well, index in rows.groupby("well", sort=False).groups.items():
        arm = choice[str(well)]
        if arm != FALLBACK:
            positions = np.asarray(index, int)
            output[positions] += blend * (
                rows.loc[positions, arm].to_numpy(float) - rows.loc[positions, FALLBACK].to_numpy(float)
            )
    decisions = wells.loc[:, ["well", "component"]].copy()
    decisions["selected_arm"] = selected
    decisions["positive_lcb"] = np.maximum(best_lcb, 0.0)
    decisions["abstained"] = selected == FALLBACK
    decisions["blend"] = blend
    decisions["confidence"] = decisions["well"].astype(str).map(confidence)
    return output, decisions


def evaluate_nested_regret_router(
    development_rows: pd.DataFrame,
    holdout_rows: pd.DataFrame,
    *,
    outer_folds: int = 5,
    inner_folds: int = 3,
    quantiles: tuple[float, ...] = (0.50, 0.75, 0.90),
    blends: tuple[float, ...] = (0.25, 0.50, 1.00),
    seed: int = 20260806,
    bootstrap_draws: int = 2000,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Select the router by group CV, then evaluate it on the holdout."""

    dev_wells, feature_names = well_level_table(development_rows)
    hold_wells, hold_features = well_level_table(holdout_rows)
    if feature_names != hold_features:
        raise ValueError("router feature order changed between development and holdout")
    outer = _component_folds(dev_wells, outer_folds, seed)
    grid_predictions = {
        (q, blend): np.full(len(development_rows), np.nan)
        for q in quantiles
        for blend in blends
    }
    decision_parts: list[pd.DataFrame] = []
    for fold in range(outer_folds):
        train_well = dev_wells.loc[outer != fold].reset_index(drop=True)
        valid_well = dev_wells.loc[outer == fold].reset_index(drop=True)
        models, margins = _fit_with_margins(
            train_well,
            feature_names,
            quantiles=quantiles,
            inner_folds=inner_folds,
            seed=seed + 1009 * fold,
        )
        x_valid = valid_well[feature_names].to_numpy(float)
        predicted = {arm: models[arm].predict(x_valid) for arm in CANDIDATES}
        valid_names = set(valid_well["well"].astype(str))
        row_mask = development_rows["well"].astype(str).isin(valid_names).to_numpy()
        valid_rows = development_rows.loc[row_mask].reset_index(drop=True)
        original_positions = np.flatnonzero(row_mask)
        for q in quantiles:
            for blend in blends:
                routed, decisions = _route(valid_rows, valid_well, predicted, margins[q], blend=blend)
                grid_predictions[(q, blend)][original_positions] = routed
                decisions["outer_fold"] = fold
                decisions["quantile"] = q
                decision_parts.append(decisions)
    if any(not np.isfinite(value).all() for value in grid_predictions.values()):
        raise RuntimeError("router outer cross-fit is incomplete")
    truth_dev = development_rows["truth"].to_numpy(float)
    grid_scores = {
        f"q{q:.2f}_b{blend:.2f}": rmse(truth_dev, prediction)
        for (q, blend), prediction in grid_predictions.items()
    }
    selected_key = min(grid_scores, key=lambda key: (grid_scores[key], key))
    selected_q = float(selected_key.split("_")[0][1:])
    selected_blend = float(selected_key.split("_")[1][1:])

    final_models, final_margins = _fit_with_margins(
        dev_wells,
        feature_names,
        quantiles=quantiles,
        inner_folds=inner_folds,
        seed=seed + 99991,
    )
    predicted_holdout = {
        arm: final_models[arm].predict(hold_wells[feature_names].to_numpy(float))
        for arm in CANDIDATES
    }
    holdout_prediction, holdout_decisions = _route(
        holdout_rows.reset_index(drop=True),
        hold_wells,
        predicted_holdout,
        final_margins[selected_q],
        blend=selected_blend,
    )
    selected_dev = grid_predictions[(selected_q, selected_blend)]
    truth_hold = holdout_rows["truth"].to_numpy(float)
    fallback_hold = holdout_rows[FALLBACK].to_numpy(float)
    bootstrap = component_bootstrap(
        truth_hold,
        fallback_hold,
        holdout_prediction,
        holdout_rows["component"].to_numpy(str),
        draws=bootstrap_draws,
        seed=seed,
    )
    summary: dict[str, object] = {
        "contract": "well_level_pairwise_regret_group_crossfit_plus_conformal_abstention",
        "feature_names": feature_names,
        "candidate_arms": list(CANDIDATES),
        "selection_grid_rmse": grid_scores,
        "selected_quantile": selected_q,
        "selected_blend": selected_blend,
        "development_fallback_rmse": rmse(truth_dev, development_rows[FALLBACK].to_numpy(float)),
        "development_selected_rmse": rmse(truth_dev, selected_dev),
        "holdout_fallback_rmse": rmse(truth_hold, fallback_hold),
        "holdout_selected_rmse": rmse(truth_hold, holdout_prediction),
        "holdout_gain_ft": rmse(truth_hold, fallback_hold) - rmse(truth_hold, holdout_prediction),
        "holdout_component_bootstrap": bootstrap,
        "holdout_abstention_rate": float(holdout_decisions["abstained"].mean()),
        "holdout_selected_arm_counts": holdout_decisions["selected_arm"].value_counts().to_dict(),
        "claim_note": (
            "The policy was defined before the 160-well run completed. Grid selection uses only "
            "component-cross-fitted development predictions; the component holdout is transfer evidence."
        ),
    }
    row_output = holdout_rows.loc[:, ["row_id", "well", "component", "truth", FALLBACK]].copy()
    row_output["regret_router"] = holdout_prediction
    decisions = pd.concat(decision_parts, ignore_index=True)
    decisions = pd.concat(
        [decisions, holdout_decisions.assign(outer_fold="holdout", quantile=selected_q)],
        ignore_index=True,
    )
    return summary, row_output, decisions
