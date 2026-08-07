#!/usr/bin/env python3
"""Build the small companion dataset used by the public Kaggle notebook."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
ARCHIVE = DIST / "rogii_portfolio_companion.zip"
PUBLIC_EXCLUDE = {
    "evidence/portfolio_notebook_rendered.html",
    "evidence/verification_report.json",
}


def selected_files() -> list[Path]:
    files: set[Path] = set()
    for folder in ("src", "evidence", "docs", "configs"):
        files.update(path for path in (ROOT / folder).rglob("*") if path.is_file())

    files.update(
        ROOT / name
        for name in (
            "README.md",
            "LICENSE",
            "requirements.txt",
            "environment.yml",
            "pyproject.toml",
        )
    )

    optional = (
        "artifacts/realdata_smoke/evidence/reproduction_summary.json",
        "artifacts/realdata_nested_160/evidence/reproduction_summary.json",
        "artifacts/realdata_nested_160/frozen_primary_policy.json",
        "artifacts/realdata_nested_160/regret_router/summary.json",
        "artifacts/realdata_nested_160/graph_audit/cross_partition_diagnostics.json",
        "artifacts/realdata_nested_160/graph_audit/component_graph_sensitivity.csv",
        "artifacts/realdata_nested_160_confirmation/evidence/reproduction_summary.json",
        "artifacts/realdata_nested_160_confirmation/frozen_policy_evaluation.json",
        "artifacts/realdata_nested_160_confirmation/posthoc_meta_shrinkage.json",
        "artifacts/robustness_evaluation/summary.json",
        "artifacts/robustness_evaluation/model_metrics.csv",
        "artifacts/robustness_evaluation/rank_stability.csv",
        "artifacts/robustness_evaluation/quality_checks.csv",
        "artifacts/robustness_evaluation/component_metrics.csv",
        "artifacts/robustness_evaluation/well_metrics.csv",
        "artifacts/robustness_evaluation/horizon_metrics.csv",
        "artifacts/robustness_evaluation/fold_metrics.csv",
        "artifacts/robustness_evaluation/bootstrap_rank_draws.csv.gz",
        "artifacts/robustness_evaluation/manifest.json",
        "artifacts/group_robust_transfer/summary.json",
        "artifacts/trust_region_ridge/summary.json",
        "artifacts/realdata_switching_160/evidence/reproduction_summary.json",
    )
    files.update(ROOT / name for name in optional if (ROOT / name).exists())
    return sorted(
        path
        for path in files
        if path.exists() and path.relative_to(ROOT).as_posix() not in PUBLIC_EXCLUDE
    )


def build() -> Path:
    DIST.mkdir(exist_ok=True)
    timestamp = (2026, 8, 7, 0, 0, 0)
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in selected_files():
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            bundle.writestr(info, path.read_bytes())

    with zipfile.ZipFile(ARCHIVE) as bundle:
        names = set(bundle.namelist())
        required = {
            "src/rogii_portfolio/__init__.py",
            "evidence/manifest.json",
            "evidence/reproduction_summary.json",
        }
        if not required.issubset(names):
            raise RuntimeError(f"companion archive is missing {sorted(required - names)}")
        if PUBLIC_EXCLUDE & names:
            raise RuntimeError("private local reports entered the public archive")
        for name in names:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"unsafe archive path: {name}")
        local_markers = (b"/mnt/c/Users/baytr", b"C:\\Users\\baytr")
        for name in names:
            if Path(name).suffix.lower() in {".md", ".json", ".csv", ".py", ".txt", ".yml", ".toml"}:
                payload = bundle.read(name)
                if any(marker in payload for marker in local_markers):
                    raise RuntimeError(f"local path found in public archive: {name}")

    metadata = {
        "title": "ROGII Research Portfolio Companion",
        "id": "baytree1238/rogii-research-portfolio-companion",
        "licenses": [{"name": "other"}],
    }
    (DIST / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{ARCHIVE} ({ARCHIVE.stat().st_size:,} bytes)")
    return ARCHIVE


if __name__ == "__main__":
    build()
