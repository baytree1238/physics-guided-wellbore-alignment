#!/usr/bin/env python3
"""Check training, target-only inference and submission order on synthetic data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_portfolio.artifacts import write_json  # noqa: E402
from rogii_portfolio.contracts import WellRecord  # noqa: E402
from rogii_portfolio.deployment import (  # noqa: E402
    fit_frozen_pipeline,
    predict_submission,
    write_submission,
)
from rogii_portfolio.physics import prepare_scored_well  # noqa: E402
from rogii_portfolio.reproduce import load_config  # noqa: E402
from rogii_portfolio.stack import STACK_ARMS, StackPolicy  # noqa: E402
from rogii_portfolio.synthetic import make_synthetic_wells  # noqa: E402


def main() -> int:
    config = load_config(ROOT / "configs" / "smoke.json")
    wells = make_synthetic_wells(
        int(config.extra["synthetic_wells"]),
        int(config.extra["synthetic_rows"]),
        seed=config.seed,
    )
    summary = json.loads((ROOT / "evidence" / "reproduction_summary.json").read_text())
    weight_map = summary["final_stack"]["weights"]
    policy = StackPolicy(
        arms=STACK_ARMS,
        weights=np.asarray([weight_map[arm] for arm in STACK_ARMS], float),
        ridge=config.stack_ridge,
    )
    fitted = fit_frozen_pipeline(wells, config, policy)

    targets: list[WellRecord] = []
    expected_ids: list[str] = []
    for record in wells[:2]:
        scored = prepare_scored_well(record, config)
        inference = scored.model_input
        targets.append(WellRecord(record.well_id, inference.inference.copy(), inference.typewell.copy()))
        expected_ids.extend(inference.row_ids.astype(str))
    sample = pd.DataFrame({"id": expected_ids, "tvt": np.zeros(len(expected_ids), np.float64)})
    frame, prediction_audit = predict_submission(fitted, targets, sample)
    csv_path = ROOT / "evidence" / "deployment_smoke_submission.csv"
    audit_path = ROOT / "evidence" / "deployment_smoke_audit.json"
    csv_audit = write_submission(frame, sample, csv_path)
    audit = {**prediction_audit, **csv_audit, "targets_contained_tvt": False}
    write_json(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
