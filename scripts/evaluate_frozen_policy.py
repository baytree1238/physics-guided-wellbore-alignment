#!/usr/bin/env python3
"""Evaluate a primary-panel policy on a disjoint complementary panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rogii_portfolio.artifacts import sha256_file, write_json  # noqa: E402
from rogii_portfolio.stack import StackPolicy, apply_stack, component_bootstrap, rmse  # noqa: E402


def evaluate(frame: pd.DataFrame, policy: StackPolicy, *, seed: int) -> dict[str, object]:
    predictions = {name: frame[name].to_numpy(float) for name in ("incumbent", *policy.arms)}
    expanded = apply_stack(policy, predictions)
    truth = frame["truth"].to_numpy(float)
    incumbent = frame["incumbent"].to_numpy(float)
    ridge = frame["ridge"].to_numpy(float)
    sequential = frame["sequential_final"].to_numpy(float)
    return {
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "components": int(frame["component"].nunique()),
        "rmse": {
            "incumbent": rmse(truth, incumbent),
            "sequential_final": rmse(truth, sequential),
            "ridge_only": rmse(truth, ridge),
            "expanded_frozen": rmse(truth, expanded),
        },
        "gain_vs_incumbent": {
            "ridge_only": rmse(truth, incumbent) - rmse(truth, ridge),
            "expanded_frozen": rmse(truth, incumbent) - rmse(truth, expanded),
        },
        "expanded_component_bootstrap": component_bootstrap(
            truth,
            incumbent,
            expanded,
            frame["component"].to_numpy(str),
            draws=5000,
            seed=seed,
        ),
        "ridge_component_bootstrap": component_bootstrap(
            truth,
            incumbent,
            ridge,
            frame["component"].to_numpy(str),
            draws=5000,
            seed=seed + 1,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.policy.read_text())
    policy = StackPolicy(
        arms=tuple(payload["arms"]),
        weights=np.asarray(payload["weights"], float),
        ridge=float(payload["ridge_penalty"]),
    )
    evidence = args.confirmation_root.resolve() / "evidence"
    oof_path = evidence / "outer_oof_predictions.csv.gz"
    hold_path = evidence / "untouched_holdout_predictions.csv.gz"
    oof = pd.read_csv(oof_path)
    holdout = pd.read_csv(hold_path)
    primary_components = pd.read_csv(args.policy.resolve().parent / "evidence" / "geological_components.csv")
    confirmation_components = pd.read_csv(evidence / "geological_components.csv")
    overlap = set(primary_components["component"].astype(str)) & set(
        confirmation_components["component"].astype(str)
    )
    if overlap:
        raise RuntimeError(f"primary/confirmation geological components overlap: {sorted(overlap)}")
    result = {
        "schema": "rogii_disjoint_confirmation_of_primary_policy_v1",
        "primary_policy_sha256": sha256_file(args.policy),
        "confirmation_oof_sha256": sha256_file(oof_path),
        "confirmation_holdout_sha256": sha256_file(hold_path),
        "primary_confirmation_component_overlap": len(overlap),
        "confirmation_outer_oof": evaluate(oof, policy, seed=20260806),
        "confirmation_run_holdout": evaluate(holdout, policy, seed=20260807),
        "claim_note": (
            "Weights and the ridge-only comparator were frozen from the primary outer OOF before "
            "this panel completed. Confirmation OOF is the primary promotion endpoint; the run-level "
            "holdout is reported separately because its training fraction differs."
        ),
    }
    output = args.confirmation_root.resolve() / "frozen_policy_evaluation.json"
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
