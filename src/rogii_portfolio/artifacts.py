"""Deterministic evidence writers and manifest verification."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pipeline import NestedResult


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prediction_frame(result: NestedResult, which: str) -> pd.DataFrame:
    bundle = result.outer_oof if which == "outer_oof" else result.holdout
    frame = pd.DataFrame(
        {
            "row_id": bundle.row_id,
            "well": bundle.well,
            "row_number": bundle.row_number,
            "component": bundle.component,
            "horizon_ft": bundle.horizon,
            "truth": bundle.truth,
        }
    )
    for name, values in bundle.predictions.items():
        frame[name] = values
    return frame


def _plots(result: NestedResult, evidence: Path) -> list[Path]:
    created: list[Path] = []
    score_rows = []
    for contract, scores in (
        ("Outer OOF", result.summary["outer_oof_rmse"]),
        ("Untouched holdout", result.summary["untouched_holdout_rmse"]),
    ):
        for name in ("pf", "incumbent", "hgrg", "meta_state", "prefix_boundary", "sequential_final", "nested_stack"):
            if name in scores:
                score_rows.append({"contract": contract, "stage": name, "rmse": scores[name]})
    score_table = pd.DataFrame(score_rows)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    stages = list(dict.fromkeys(score_table["stage"]))
    x = np.arange(len(stages))
    width = 0.36
    for offset, contract in zip((-0.5, 0.5), ("Outer OOF", "Untouched holdout")):
        local = score_table.set_index(["contract", "stage"])["rmse"]
        values = [local.get((contract, stage), np.nan) for stage in stages]
        ax.bar(x + offset * width, values, width, label=contract)
    ax.set_xticks(x, [name.replace("_", "\n") for name in stages])
    ax.set_ylabel("RMSE (ft; lower is better)")
    ax.set_title("One contract, immediate-parent stages, and an untouched holdout")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = evidence / "rmse_by_stage.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    folds = result.fold_records
    ax.axhline(0.0, color="black", linewidth=1)
    ax.plot(
        folds["outer_fold"],
        folds["incumbent_rmse"] - folds["sequential_final_rmse"],
        marker="o",
        label="Sequential final",
    )
    ax.plot(
        folds["outer_fold"],
        folds["incumbent_rmse"] - folds["nested_stack_rmse"],
        marker="s",
        label="Nested stack",
    )
    ax.set_xlabel("Outer geological-component fold")
    ax.set_ylabel("RMSE gain vs incumbent (ft)")
    ax.set_title("Fold stability matters more than a pooled improvement")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = evidence / "outer_fold_gains.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(path)

    holdout = prediction_frame(result, "holdout")
    example = holdout.loc[holdout["well"] == holdout["well"].iloc[0]].sort_values("row_number")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(example["horizon_ft"], example["truth"], color="black", linewidth=2, label="Truth (scoring only)")
    for name, style in (("incumbent", "--"), ("hgrg", "-."), ("sequential_final", "-"), ("nested_stack", ":")):
        ax.plot(example["horizon_ft"], example[name], linestyle=style, label=name)
    ax.set_xlabel("Distance past visible prefix (ft)")
    ax.set_ylabel("TVT (ft)")
    ax.set_title(f"Trajectory audit: {example['well'].iloc[0]}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = evidence / "example_trajectory.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(path)
    return created


def write_result(result: NestedResult, root: Path, *, config_sha256: str) -> dict[str, Any]:
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    write_json(evidence / "reproduction_summary.json", result.summary)
    result.components.to_csv(evidence / "geological_components.csv", index=False)
    if result.component_universe is not None:
        result.component_universe.to_csv(evidence / "component_graph_universe.csv", index=False)
    result.fold_records.to_csv(evidence / "outer_fold_metrics.csv", index=False)
    deterministic_gzip = {"method": "gzip", "mtime": 0}
    prediction_frame(result, "outer_oof").to_csv(
        evidence / "outer_oof_predictions.csv.gz",
        index=False,
        compression=deterministic_gzip,
    )
    prediction_frame(result, "holdout").to_csv(
        evidence / "untouched_holdout_predictions.csv.gz",
        index=False,
        compression=deterministic_gzip,
    )
    figures = _plots(result, evidence)
    tracked = sorted(
        [
            evidence / "reproduction_summary.json",
            evidence / "geological_components.csv",
            evidence / "outer_fold_metrics.csv",
            evidence / "outer_oof_predictions.csv.gz",
            evidence / "untouched_holdout_predictions.csv.gz",
            *figures,
        ]
    )
    universe_path = evidence / "component_graph_universe.csv"
    if universe_path.exists() and result.component_universe is not None:
        tracked.append(universe_path)
        tracked.sort()
    manifest = {
        "schema": "rogii_portfolio_reproduction_v1",
        "config_sha256": config_sha256,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "artifacts": [
            {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in tracked
        ],
    }
    write_json(evidence / "manifest.json", manifest)
    return manifest


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "evidence" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        path = root / item["path"]
        if not path.exists() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"artifact verification failed: {item['path']}")
    return manifest
