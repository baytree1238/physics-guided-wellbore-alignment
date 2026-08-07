#!/usr/bin/env python3
"""Cross-fit component-risk-aware blends and transfer them between 160-well panels."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_portfolio.group_robust import apply_group_robust_blend, fit_group_robust_blend  # noqa: E402
from rogii_portfolio.stack import component_bootstrap  # noqa: E402


ARMS = ("ridge", "nonlinear", "hgrg", "meta_state", "prefix_boundary", "sequential_final")
GRID = tuple(
    (macro, cvar, ridge)
    for macro in (0.0, 0.5, 1.0)
    for cvar in (0.0, 0.5, 2.0)
    for ridge in (0.01, 0.10)
)
SEEDS = (3101, 7727, 19001)
FOLDS = 5


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(truth, float) - np.asarray(prediction, float)))))


def metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    local = frame.loc[:, ["well", "component", "truth"]].copy()
    local["squared_error"] = np.square(local["truth"].to_numpy(float) - prediction)
    well_rmse = np.sqrt(local.groupby("well", sort=True)["squared_error"].mean())
    component_rmse = np.sqrt(local.groupby("component", sort=True)["squared_error"].mean())
    worst_count = max(1, int(np.ceil(0.10 * len(component_rmse))))
    worst = component_rmse.sort_values().iloc[-worst_count:]
    return {
        "row_rmse": float(np.sqrt(local["squared_error"].mean())),
        "macro_well_rmse": float(well_rmse.mean()),
        "macro_component_rmse": float(component_rmse.mean()),
        "cvar10_component_rmse": float(worst.mean()),
    }


def balanced_folds(frame: pd.DataFrame, seed: int) -> dict[str, int]:
    counts = frame.groupby("component", sort=True).size()
    rng = np.random.default_rng(seed)
    jitter = {name: value for name, value in zip(counts.index, rng.random(len(counts)))}
    order = sorted(counts.index.astype(str), key=lambda name: (-int(counts.loc[name]), jitter[name], name))
    load = np.zeros(FOLDS, int)
    assignment = {}
    for component in order:
        fold = int(np.argmin(load))
        assignment[component] = fold
        load[fold] += int(counts.loc[component])
    return assignment


def crossfit(frame: pd.DataFrame, setting: tuple[float, float, float], seed: int) -> np.ndarray:
    macro, cvar, ridge = setting
    assignment = balanced_folds(frame, seed)
    output = np.full(len(frame), np.nan)
    components = frame["component"].astype(str)
    for fold in range(FOLDS):
        valid = components.map(assignment).to_numpy(int) == fold
        policy = fit_group_robust_blend(
            frame.loc[~valid],
            ARMS,
            macro_weight=macro,
            cvar_weight=cvar,
            ridge=ridge,
        )
        output[valid] = apply_group_robust_blend(frame.loc[valid], policy)
    if not np.isfinite(output).all():
        raise RuntimeError("cross-fitted robust predictions are incomplete")
    return output


def normalized_selection_score(candidate: dict[str, float], base: dict[str, float]) -> float:
    return float(
        0.50 * candidate["row_rmse"] / base["row_rmse"]
        + 0.25 * candidate["macro_component_rmse"] / base["macro_component_rmse"]
        + 0.25 * candidate["cvar10_component_rmse"] / base["cvar10_component_rmse"]
    )


def fit_and_transfer(source: pd.DataFrame, target: pd.DataFrame, name: str) -> dict[str, object]:
    base_metrics = metrics(source, source["incumbent"].to_numpy(float))
    grid_rows = []
    cached: dict[tuple[tuple[float, float, float], int], np.ndarray] = {}
    for setting in GRID:
        seed_scores = []
        for seed in SEEDS:
            prediction = crossfit(source, setting, seed)
            cached[(setting, seed)] = prediction
            result = metrics(source, prediction)
            seed_scores.append(normalized_selection_score(result, base_metrics))
        grid_rows.append(
            {
                "macro_weight": setting[0],
                "cvar_weight": setting[1],
                "ridge": setting[2],
                "mean_selection_score": float(np.mean(seed_scores)),
                "std_selection_score": float(np.std(seed_scores)),
            }
        )
    grid = pd.DataFrame(grid_rows).sort_values(
        ["mean_selection_score", "std_selection_score", "cvar_weight", "macro_weight", "ridge"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    )
    best = grid.iloc[0]
    setting = (float(best.macro_weight), float(best.cvar_weight), float(best.ridge))
    crossfit_prediction = np.mean([cached[(setting, seed)] for seed in SEEDS], axis=0)
    policy = fit_group_robust_blend(
        source,
        ARMS,
        macro_weight=setting[0],
        cvar_weight=setting[1],
        ridge=setting[2],
    )
    target_prediction = apply_group_robust_blend(target, policy)
    target_truth = target["truth"].to_numpy(float)
    target_component = target["component"].astype(str).to_numpy()
    return {
        "direction": name,
        "selection": {
            "macro_weight": setting[0],
            "cvar_weight": setting[1],
            "ridge": setting[2],
            "mean_score": float(best.mean_selection_score),
            "std_score": float(best.std_selection_score),
            "seeds": list(SEEDS),
            "folds": FOLDS,
        },
        "policy": {
            "parent_weight": policy.parent_weight,
            "weights": {arm: float(value) for arm, value in zip(policy.arms, policy.weights)},
        },
        "source_crossfit": {
            "incumbent": metrics(source, source["incumbent"].to_numpy(float)),
            "sequential": metrics(source, source["sequential_final"].to_numpy(float)),
            "robust": metrics(source, crossfit_prediction),
        },
        "target_frozen": {
            "incumbent": metrics(target, target["incumbent"].to_numpy(float)),
            "sequential": metrics(target, target["sequential_final"].to_numpy(float)),
            "robust": metrics(target, target_prediction),
            "bootstrap_vs_incumbent": component_bootstrap(
                target_truth,
                target["incumbent"].to_numpy(float),
                target_prediction,
                target_component,
                draws=5000,
                seed=91827,
            ),
            "bootstrap_vs_sequential": component_bootstrap(
                target_truth,
                target["sequential_final"].to_numpy(float),
                target_prediction,
                target_component,
                draws=5000,
                seed=91828,
            ),
        },
        "grid": grid.to_dict(orient="records"),
    }


def main() -> None:
    primary = pd.read_csv(ROOT / "artifacts/realdata_nested_160/evidence/outer_oof_predictions.csv.gz")
    complement = pd.read_csv(
        ROOT / "artifacts/realdata_nested_160_confirmation/evidence/outer_oof_predictions.csv.gz"
    )
    result = {
        "schema": "rogii_group_robust_transfer_v1",
        "status": "retrospective_reused_panels",
        "arms": list(ARMS),
        "primary_to_complement": fit_and_transfer(primary, complement, "primary_to_complement"),
        "complement_to_primary": fit_and_transfer(complement, primary, "complement_to_primary"),
    }
    output = ROOT / "artifacts/group_robust_transfer"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
