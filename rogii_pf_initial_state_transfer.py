"""Frozen Exact80 transfer for a Dev-selected PF initial-state arm.

This program refuses to run an arbitrary arm: ``--arm`` must exactly match the
strictly gated ``selected_arm`` in the completed Dev40 summary.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import rogii_pf_initial_state_audit as base


EXACT_PANEL = Path("lightweight_pf_geohmm_evidence/exact80_pf8/predictions.npz")
DEV_SUMMARY = Path("pf_initial_state_dev40_evidence/summary.json")
OUT = Path("pf_initial_state_exact80_evidence")


def find_arm(name: str) -> base.Arm:
    matches = [arm for arm in base.registered_arms() if arm.name == name]
    if len(matches) != 1:
        raise ValueError(f"unknown registered arm: {name}")
    return matches[0]


def _predict(task: tuple[str, str, str, int, int, float]) -> dict[str, object]:
    well, root_text, arm_name, seeds, particles, temperature = task
    root = Path(root_text)
    arm = find_arm(arm_name)
    raw_horizontal = pd.read_csv(root / "train" / f"{well}__horizontal_well.csv").sort_values("MD", kind="mergesort", ignore_index=True)
    raw_typewell = pd.read_csv(root / "train" / f"{well}__typewell.csv").sort_values("TVT", kind="mergesort", ignore_index=True)
    horizontal = raw_horizontal.loc[:, base.MODEL_COLUMNS]
    typewell = raw_typewell.loc[:, base.TYPEWELL_COLUMNS]
    evaluation = raw_horizontal["TVT_input"].isna().to_numpy()
    started = time.perf_counter()
    prediction = base._ensemble(horizontal, typewell, arm, seeds, particles, temperature)
    return {
        "well": well,
        "truth": raw_horizontal.loc[evaluation, "TVT"].to_numpy(np.float32),
        "prediction": prediction,
        "initial_rate": base.initial_rate(horizontal[horizontal["TVT_input"].notna()], arm),
        "seconds": float(time.perf_counter() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--panel", type=Path, default=EXACT_PANEL)
    parser.add_argument("--dev-summary", type=Path, default=DEV_SUMMARY)
    parser.add_argument("--data-root", type=Path, default=base.DATA_ROOT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=8.0)
    args = parser.parse_args()

    arm = find_arm(args.arm)
    dev = json.loads(args.dev_summary.read_text(encoding="utf-8"))
    if dev.get("selected_arm") != arm.name or dev.get("decision") != "TRANSFER_SELECTED_ARM_TO_EXACT80":
        raise RuntimeError("requested arm is not the strictly gated frozen Dev selection")

    with np.load(args.panel, allow_pickle=False) as source:
        panel = {key: source[key].copy() for key in source.files}
    wells = np.unique(panel["well"].astype(str))
    tasks = [(well, str(args.data_root), arm.name, args.seeds, args.particles, args.temperature) for well in wells]
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = [executor.submit(_predict, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            print(f"completed {index}/{len(tasks)}", flush=True)
    wall_seconds = float(time.perf_counter() - started)

    returned = {str(item["well"]): item for item in results}
    panel_well = panel["well"].astype(str)
    candidate = np.empty(len(panel_well), np.float32)
    rates = []
    for well in wells:
        mask = panel_well == well
        result = returned[well]
        if not np.array_equal(np.asarray(result["truth"], np.float32), panel["truth"][mask].astype(np.float32)):
            raise RuntimeError(f"{well}: truth/order canary failed")
        candidate[mask] = result["prediction"]
        rates.append({"well": well, "initial_rate": result["initial_rate"]})

    truth = panel["truth"].astype(float)
    reference = panel["pf"].astype(np.float32)
    folds = panel["fold"].astype(int)
    ref_folds = base.fold_scores(truth, reference, folds)
    cand_folds = base.fold_scores(truth, candidate, folds)
    gains = [left - right for left, right in zip(ref_folds, cand_folds)]
    low, high = base.bootstrap_gain_ci(truth, reference, candidate, panel_well)
    ref_rmse = base.rmse(truth, reference)
    cand_rmse = base.rmse(truth, candidate)
    summary = {
        "method": "pf_initial_structural_rate_frozen_transfer_v1",
        "scope": "Exact80 frozen transfer (historically opened)",
        "frozen_from_dev_summary": str(args.dev_summary),
        "arm": arm.name,
        "reference_rmse": ref_rmse,
        "candidate_rmse": cand_rmse,
        "gain_vs_reference": ref_rmse - cand_rmse,
        "fold_gains": gains,
        "positive_folds": int(sum(value > 0.0 for value in gains)),
        "well_bootstrap_ci95": [low, high],
        "transfer_passes_same_stability_gate": bool(sum(value > 0.0 for value in gains) == 5 and low > 0.0 and ref_rmse - cand_rmse >= 0.05),
        "target_isolation": {"horizontal_prediction_columns": base.MODEL_COLUMNS, "typewell_prediction_columns": base.TYPEWELL_COLUMNS, "suffix_tvt_seen_by_prediction": False},
        "runtime": {"wall_seconds": wall_seconds, "summed_worker_seconds": float(sum(float(item["seconds"]) for item in results)), "workers": int(min(args.workers, len(tasks)))},
        "created_submission_csv": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "predictions.npz", well=panel["well"], fold=panel["fold"], truth=panel["truth"],
        reference_pf=reference, candidate_pf=candidate,
    )
    pd.DataFrame(rates).to_csv(args.output / "initial_rates.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
