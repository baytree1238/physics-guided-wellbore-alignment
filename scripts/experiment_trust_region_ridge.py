#!/usr/bin/env python3
"""Repeated component-CV and cross-panel transfer for a capped Ridge correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_portfolio.stack import component_bootstrap  # noqa: E402
from rogii_portfolio.trust_region import apply_trust_region_blend, fit_trust_region_blend  # noqa: E402


SEEDS = (3101, 7727, 19001)
FOLDS = 5
SETTINGS = tuple(
    (cap, 0.25, 0.25, 0.10)
    for cap in (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25)
)


def _folds(frame: pd.DataFrame, seed: int) -> np.ndarray:
    counts = frame.groupby("component", sort=True).size()
    rng = np.random.default_rng(seed)
    order = sorted(counts.index.astype(str), key=lambda x: (-int(counts.loc[x]), float(rng.random()), x))
    load = np.zeros(FOLDS, int)
    assignment: dict[str, int] = {}
    for component in order:
        fold = int(np.argmin(load))
        assignment[component] = fold
        load[fold] += int(counts.loc[component])
    return frame["component"].astype(str).map(assignment).to_numpy(int)


def _metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    local = frame[["well", "component", "truth"]].copy()
    local["se"] = np.square(local["truth"].to_numpy(float) - prediction)
    component_mse = local.groupby("component", sort=True)["se"].mean().to_numpy(float)
    well_rmse = np.sqrt(local.groupby("well", sort=True)["se"].mean().to_numpy(float))
    worst = np.sort(component_mse)[-max(1, int(np.ceil(0.10 * len(component_mse)))) :]
    return {
        "row_rmse": float(np.sqrt(local["se"].mean())),
        "macro_well_rmse": float(well_rmse.mean()),
        "macro_component_rmse": float(np.sqrt(component_mse).mean()),
        "cvar10_component_rmse": float(np.sqrt(worst.mean())),
    }


def _score(candidate: dict[str, float], parent: dict[str, float]) -> float:
    return float(
        0.60 * candidate["row_rmse"] / parent["row_rmse"]
        + 0.20 * candidate["macro_component_rmse"] / parent["macro_component_rmse"]
        + 0.20 * candidate["cvar10_component_rmse"] / parent["cvar10_component_rmse"]
    )


def _crossfit(frame: pd.DataFrame, setting: tuple[float, float, float, float], seed: int) -> tuple[np.ndarray, list[float]]:
    cap, macro, cvar, l2 = setting
    fold = _folds(frame, seed)
    prediction = np.full(len(frame), np.nan)
    weights = []
    for value in range(FOLDS):
        valid = fold == value
        policy = fit_trust_region_blend(
            frame.loc[~valid],
            maximum_weight=cap,
            macro_weight=macro,
            cvar_weight=cvar,
            l2=l2,
        )
        prediction[valid] = apply_trust_region_blend(frame.loc[valid], policy)
        weights.append(policy.weight)
    return prediction, weights


def _direction(source: pd.DataFrame, target: pd.DataFrame, name: str) -> dict[str, object]:
    parent_source = _metrics(source, source["sequential_final"].to_numpy(float))
    rows = []
    cache: dict[tuple[tuple[float, float, float, float], int], tuple[np.ndarray, list[float]]] = {}
    for setting in SETTINGS:
        scores = []
        for seed in SEEDS:
            prediction, weights = _crossfit(source, setting, seed)
            cache[(setting, seed)] = (prediction, weights)
            scores.append(_score(_metrics(source, prediction), parent_source))
        rows.append({
            "maximum_weight": setting[0], "macro_weight": setting[1],
            "cvar_weight": setting[2], "l2": setting[3],
            "mean_score": float(np.mean(scores)), "std_score": float(np.std(scores)),
        })
    grid = pd.DataFrame(rows).sort_values(
        ["mean_score", "std_score", "maximum_weight", "cvar_weight"], kind="mergesort"
    )
    best = grid.iloc[0]
    setting = tuple(float(best[name]) for name in ("maximum_weight", "macro_weight", "cvar_weight", "l2"))
    crossfit = np.mean([cache[(setting, seed)][0] for seed in SEEDS], axis=0)
    fold_weights = [weight for seed in SEEDS for weight in cache[(setting, seed)][1]]
    policy = fit_trust_region_blend(
        source,
        maximum_weight=setting[0], macro_weight=setting[1],
        cvar_weight=setting[2], l2=setting[3],
    )
    frozen = apply_trust_region_blend(target, policy)
    truth = target["truth"].to_numpy(float)
    component = target["component"].astype(str).to_numpy()
    return {
        "direction": name,
        "selection": dict(zip(("maximum_weight", "macro_weight", "cvar_weight", "l2"), setting)),
        "fold_weight_range": [float(min(fold_weights)), float(max(fold_weights))],
        "frozen_weight": policy.weight,
        "source_crossfit": {
            "sequential": parent_source,
            "trust_region": _metrics(source, crossfit),
        },
        "target_frozen": {
            "sequential": _metrics(target, target["sequential_final"].to_numpy(float)),
            "trust_region": _metrics(target, frozen),
            "bootstrap": component_bootstrap(
                truth, target["sequential_final"].to_numpy(float), frozen, component,
                draws=5000, seed=61591,
            ),
        },
        "grid": grid.to_dict(orient="records"),
    }


def _fixed_audit(frame: pd.DataFrame, weight: float, seed: int) -> dict[str, object]:
    parent = frame["sequential_final"].to_numpy(float)
    candidate = parent + weight * (frame["ridge"].to_numpy(float) - parent)
    truth = frame["truth"].to_numpy(float)
    well_error = frame[["well"]].copy()
    well_error["parent_se"] = np.square(truth - parent)
    well_error["candidate_se"] = np.square(truth - candidate)
    by_well = well_error.groupby("well", sort=True)[["parent_se", "candidate_se"]].mean()
    parent_metrics = _metrics(frame, parent)
    candidate_metrics = _metrics(frame, candidate)
    return {
        "weight": weight,
        "sequential": parent_metrics,
        "candidate": candidate_metrics,
        "row_rmse_gain_ft": parent_metrics["row_rmse"] - candidate_metrics["row_rmse"],
        "cvar10_gain_ft": (
            parent_metrics["cvar10_component_rmse"]
            - candidate_metrics["cvar10_component_rmse"]
        ),
        "harmed_well_rate": float((by_well["candidate_se"] > by_well["parent_se"]).mean()),
        "bootstrap": component_bootstrap(
            truth,
            parent,
            candidate,
            frame["component"].astype(str).to_numpy(),
            draws=5000,
            seed=seed,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed-only",
        action="store_true",
        help="Reuse saved directional results and refresh only the fixed-weight audit.",
    )
    args = parser.parse_args()
    primary = pd.read_csv(ROOT / "artifacts/realdata_nested_160/evidence/outer_oof_predictions.csv.gz")
    complement = pd.read_csv(ROOT / "artifacts/realdata_nested_160_confirmation/evidence/outer_oof_predictions.csv.gz")
    primary_holdout = pd.read_csv(
        ROOT / "artifacts/realdata_nested_160/evidence/untouched_holdout_predictions.csv.gz"
    )
    complement_holdout = pd.read_csv(
        ROOT / "artifacts/realdata_nested_160_confirmation/evidence/untouched_holdout_predictions.csv.gz"
    )
    output = ROOT / "artifacts/trust_region_ridge"
    path = output / "summary.json"
    if args.fixed_only:
        if not path.exists():
            raise FileNotFoundError("run the full trust-region experiment before --fixed-only")
        result = json.loads(path.read_text(encoding="utf-8"))
    else:
        result = {
            "schema": "rogii_trust_region_ridge_transfer_v1",
            "status": "retrospective_reused_panels",
            "primary_to_complement": _direction(primary, complement, "primary_to_complement"),
            "complement_to_primary": _direction(complement, primary, "complement_to_primary"),
        }
    result["fixed_weight_0p05_audit"] = {
            "claim": "post-hoc sensitivity audit; not a fresh confirmation or promoted default",
            "primary_outer_oof": _fixed_audit(primary, 0.05, 80101),
            "primary_holdout": _fixed_audit(primary_holdout, 0.05, 80102),
            "complement_outer_oof": _fixed_audit(complement, 0.05, 80103),
            "complement_holdout": _fixed_audit(complement_holdout, 0.05, 80104),
    }
    output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
