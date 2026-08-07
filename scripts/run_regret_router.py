#!/usr/bin/env python3
"""Evaluate the fixed regret-router grid on a nested experiment artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from rogii_portfolio.artifacts import write_json  # noqa: E402
from rogii_portfolio.regret_router import evaluate_nested_regret_router  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.artifact_root.resolve() / "evidence"
    output = args.artifact_root.resolve() / "regret_router"
    output.mkdir(parents=True, exist_ok=True)
    development = pd.read_csv(source / "outer_oof_predictions.csv.gz")
    holdout = pd.read_csv(source / "untouched_holdout_predictions.csv.gz")
    summary, rows, decisions = evaluate_nested_regret_router(development, holdout)
    write_json(output / "summary.json", summary)
    rows.to_csv(output / "holdout_predictions.csv", index=False)
    decisions.to_csv(output / "well_decisions.csv", index=False)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
