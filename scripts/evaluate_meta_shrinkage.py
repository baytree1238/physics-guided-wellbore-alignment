#!/usr/bin/env python3
"""Post-hoc cross-panel ablation of Meta-State and downstream move strength."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rogii_portfolio.artifacts import write_json  # noqa: E402
from rogii_portfolio.stack import component_bootstrap, rmse  # noqa: E402


ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)
DOWNSTREAM = (0.50, 0.75, 1.0)


def candidate(frame: pd.DataFrame, alpha: float, downstream: float) -> np.ndarray:
    hgrg = frame["hgrg"].to_numpy(float)
    meta_move = frame["meta_state"].to_numpy(float) - hgrg
    later_move = frame["sequential_final"].to_numpy(float) - frame["meta_state"].to_numpy(float)
    return hgrg + alpha * meta_move + downstream * later_move


def score(frame: pd.DataFrame, alpha: float, downstream: float) -> float:
    return rmse(frame["truth"].to_numpy(float), candidate(frame, alpha, downstream))


def evaluate(frame: pd.DataFrame, alpha: float, downstream: float, seed: int) -> dict[str, object]:
    truth = frame["truth"].to_numpy(float)
    sequential = frame["sequential_final"].to_numpy(float)
    prediction = candidate(frame, alpha, downstream)
    return {
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "components": int(frame["component"].nunique()),
        "sequential_rmse": rmse(truth, sequential),
        "candidate_rmse": rmse(truth, prediction),
        "gain_vs_sequential_ft": rmse(truth, sequential) - rmse(truth, prediction),
        "component_bootstrap": component_bootstrap(
            truth,
            sequential,
            prediction,
            frame["component"].to_numpy(str),
            draws=5000,
            seed=seed,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    args = parser.parse_args()
    primary_evidence = args.primary_root.resolve() / "evidence"
    confirmation_evidence = args.confirmation_root.resolve() / "evidence"
    primary_oof = pd.read_csv(primary_evidence / "outer_oof_predictions.csv.gz")
    primary_holdout = pd.read_csv(primary_evidence / "untouched_holdout_predictions.csv.gz")
    confirmation_oof = pd.read_csv(confirmation_evidence / "outer_oof_predictions.csv.gz")
    confirmation_holdout = pd.read_csv(confirmation_evidence / "untouched_holdout_predictions.csv.gz")
    grid = [
        {"meta_alpha": alpha, "downstream_alpha": downstream, "primary_oof_rmse": score(primary_oof, alpha, downstream)}
        for alpha in ALPHAS
        for downstream in DOWNSTREAM
    ]
    selected = min(grid, key=lambda row: (row["primary_oof_rmse"], row["meta_alpha"], row["downstream_alpha"]))
    alpha, downstream = selected["meta_alpha"], selected["downstream_alpha"]
    payload = {
        "schema": "rogii_posthoc_meta_shrinkage_cross_panel_v1",
        "status": "POSTHOC_HOLD",
        "selection_source": "primary_outer_oof_only",
        "hypothesis_timing": "conceived_after_inspecting_confirmation stage metrics",
        "selected_meta_alpha": alpha,
        "selected_downstream_alpha": downstream,
        "primary_grid": grid,
        "primary_outer_oof": evaluate(primary_oof, alpha, downstream, 20260806),
        "primary_run_holdout": evaluate(primary_holdout, alpha, downstream, 20260807),
        "confirmation_outer_oof": evaluate(confirmation_oof, alpha, downstream, 20260808),
        "confirmation_run_holdout": evaluate(confirmation_holdout, alpha, downstream, 20260809),
        "claim_note": (
            "The transfer numbers are informative but not a fresh confirmation because the ablation "
            "was motivated after the confirmation stage metrics were visible. A new component panel "
            "would be required for promotion."
        ),
    }
    output = args.confirmation_root.resolve() / "posthoc_meta_shrinkage.json"
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
