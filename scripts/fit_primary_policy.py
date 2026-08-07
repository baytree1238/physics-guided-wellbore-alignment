#!/usr/bin/env python3
"""Fit stack weights from primary outer-OOF predictions and save the policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from rogii_portfolio.artifacts import sha256_file, write_json  # noqa: E402
from rogii_portfolio.stack import apply_stack, fit_convex_stack, rmse  # noqa: E402


ARMS = ("ridge", "nonlinear", "hgrg", "meta_state", "prefix_boundary", "sequential_final")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.artifact_root.resolve() / "evidence"
    oof_path = evidence / "outer_oof_predictions.csv.gz"
    oof = pd.read_csv(oof_path)
    policy = fit_convex_stack(
        oof["truth"].to_numpy(float),
        {name: oof[name].to_numpy(float) for name in ("incumbent", *ARMS)},
        oof["component"].to_numpy(str),
        ridge=0.02,
        arms=ARMS,
    )
    prediction = apply_stack(policy, {name: oof[name].to_numpy(float) for name in ("incumbent", *ARMS)})
    payload = {
        "schema": "rogii_primary_oof_frozen_expanded_policy_v1",
        "fit_source": "primary_160_outer_oof_only",
        "confirmation_labels_read": False,
        "source_outer_oof_sha256": sha256_file(oof_path),
        "arms": list(policy.arms),
        "weights": [float(value) for value in policy.weights],
        "parent_weight": policy.parent_weight,
        "ridge_penalty": policy.ridge,
        "primary_oof_resubstitution_rmse": rmse(oof["truth"], prediction),
        "primary_oof_comparators": {
            name: rmse(oof["truth"], oof[name])
            for name in ("incumbent", "ridge", "nonlinear", "sequential_final")
        },
        "claim_note": (
            "The fitted-policy RMSE is a selection statistic, not honest OOF for the stack. "
            "Promotion depends on the disjoint confirmation component panel."
        ),
    }
    output = args.artifact_root.resolve() / "frozen_primary_policy.json"
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
