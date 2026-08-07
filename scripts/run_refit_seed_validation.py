#!/usr/bin/env python3
"""Run and evaluate three frozen-split refits of the 5% Ridge trust region."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_portfolio.artifacts import (  # noqa: E402
    sha256_file,
    verify_manifest,
    write_json,
    write_result,
)
from rogii_portfolio.io import CompetitionStore  # noqa: E402
from rogii_portfolio.pipeline import run_nested_experiment  # noqa: E402
from rogii_portfolio.refit_stability import evaluate_refit_runs  # noqa: E402
from rogii_portfolio.reproduce import load_config  # noqa: E402


def _verify_registration(output: Path, data_root: Path, config_path: Path) -> dict[str, object]:
    manifest = json.loads((output / "registration_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        path = output / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"registration artifact failed verification: {item['path']}")
    registration = json.loads((output / "registration.json").read_text(encoding="utf-8"))
    if registration["status"] != "REGISTERED_BEFORE_SCORING":
        raise RuntimeError("panel was not registered before scoring")
    if registration["data_sha256"] != sha256_file(data_root):
        raise RuntimeError("registered data archive differs from --data-root")
    if registration["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("registered config changed after panel registration")
    return registration


def _load_registered_wells(data_root: Path, panel: pd.DataFrame) -> list:
    names = list(panel.sort_values("well", kind="mergesort")["well"].astype(str))
    with CompetitionStore(data_root, split="train") as store:
        available = set(store.wells())
        missing = set(names) - available
        if missing:
            raise RuntimeError(f"registered wells are missing from the archive: {sorted(missing)}")
        return [store.load(name) for name in names]


def _run_one(
    *,
    wells: list,
    panel: pd.DataFrame,
    base_config,
    refit_seed: int,
    pf_seed_offset: int,
    run_root: Path,
    registration_sha256: str,
) -> float:
    manifest_path = run_root / "evidence" / "manifest.json"
    if manifest_path.exists():
        verify_manifest(run_root)
        summary = json.loads(
            (run_root / "evidence" / "reproduction_summary.json").read_text(encoding="utf-8")
        )
        if int(summary["seed_contract"]["refit_seed"]) != int(refit_seed):
            raise RuntimeError(f"cached run has the wrong refit seed: {run_root}")
        return float(summary.get("runtime_seconds", 0.0))

    extra = dict(base_config.extra)
    extra.update(
        {
            "refit_seed": int(refit_seed),
            "pf_seed_offset": int(pf_seed_offset),
            "registered_panel_sha256": registration_sha256,
        }
    )
    config = replace(base_config, extra=extra)
    config.validate()
    run_root.mkdir(parents=True, exist_ok=True)
    run_config_path = run_root / "run_config.json"
    write_json(run_config_path, asdict(config))
    started = perf_counter()
    result = run_nested_experiment(
        wells,
        config,
        component_table=panel,
        component_graph_scope_wells=773,
        component_universe=None,
    )
    elapsed = perf_counter() - started
    result.summary.update(
        {
            "validation_role": "contract_new_retrospective_three_refit_confirmation",
            "frozen_candidate_selected_before_this_panel": True,
            "historically_label_blind": False,
            "registration_sha256": registration_sha256,
            "runtime_seconds": float(elapsed),
        }
    )
    write_result(result, run_root, config_sha256=sha256_file(run_config_path))
    verify_manifest(run_root)
    return float(elapsed)


def _plot(seed_metrics: pd.DataFrame, summary: dict[str, object], path: Path) -> None:
    contracts = ["outer_oof", "holdout"]
    seeds = sorted(seed_metrics["refit_seed"].unique())
    x = np.arange(len(seeds), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for offset, contract in zip((-0.5, 0.5), contracts):
        local = seed_metrics.loc[seed_metrics["contract"] == contract].set_index("refit_seed")
        values = [float(local.loc[seed, "row_rmse_gain_ft"]) for seed in seeds]
        ax.bar(x + offset * width, values, width, label=contract.replace("_", " "))
    ci = summary["contracts"]["outer_oof"]["hierarchical_bootstrap"]
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.set_xlabel("Independent refit seed")
    ax.set_ylabel("RMSE gain from frozen 5% Ridge move (ft)")
    ax.set_title(
        "Frozen 5% trust region across refits\n"
        f"outer-OOF hierarchical 95% CI [{ci['ci95_low']:.3f}, {ci['ci95_high']:.3f}] ft"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_public_registration(
    registration: dict[str, object],
    *,
    registration_sha256: str,
    output: Path,
) -> Path:
    """Export the registered contract without publishing a local data path."""

    public = dict(registration)
    public["schema"] = "rogii_contract_new_panel_public_record_v1"
    public["data_archive"] = Path(str(public.pop("data_path"))).name
    public["private_registration_sha256"] = registration_sha256
    public["path_redaction"] = "Only the local data path was replaced by the archive basename."
    path = output / "public_registration.json"
    write_json(path, public)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "refit_seed_validation_0p05.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "refit_seed_validation_0p05",
    )
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    config_path = args.config.resolve()
    output = args.output_root.resolve()
    registration = _verify_registration(output, data_root, config_path)
    registration_path = output / "registration.json"
    registration_sha256 = sha256_file(registration_path)
    public_registration_path = _write_public_registration(
        registration,
        registration_sha256=registration_sha256,
        output=output,
    )
    panel = pd.read_csv(output / "panel_components.csv")
    if len(panel) != int(registration["registered_panel"]["wells"]):
        raise RuntimeError("registered panel count changed")
    wells = _load_registered_wells(data_root, panel)
    config = load_config(config_path)
    seeds = [int(value) for value in registration["refit_seeds"]]
    offsets = [int(value) for value in registration["pf_seed_offsets"]]
    if len(seeds) != len(offsets) or len(set(seeds)) < 3:
        raise RuntimeError("registration needs at least three independent refit/PF seed pairs")

    run_roots = []
    runtimes = {}
    for refit_seed, pf_seed_offset in zip(seeds, offsets):
        run_root = output / "runs" / f"seed_{refit_seed}"
        print(f"refit seed {refit_seed}: starting", flush=True)
        runtimes[str(refit_seed)] = _run_one(
            wells=wells,
            panel=panel,
            base_config=config,
            refit_seed=refit_seed,
            pf_seed_offset=pf_seed_offset,
            run_root=run_root,
            registration_sha256=registration_sha256,
        )
        run_roots.append(run_root)
        print(f"refit seed {refit_seed}: complete", flush=True)

    summary, seed_metrics, component_metrics, bootstrap = evaluate_refit_runs(
        run_roots,
        refit_seeds=seeds,
        weight=float(registration["frozen_candidate"]["expert_weight"]),
        bootstrap_draws=int(config.bootstrap_draws),
        bootstrap_seed=config.split_seed,
        promotion_rules=dict(registration["promotion_rules"]),
    )
    summary.update(
        {
            "public_registration_path": public_registration_path.relative_to(output).as_posix(),
            "private_registration_sha256": registration_sha256,
            "claim_scope": registration["claim_scope"],
            "panel": registration["registered_panel"],
            "runtime_seconds_by_seed": runtimes,
            "runtime_seconds_total": float(sum(runtimes.values())),
        }
    )
    write_json(output / "summary.json", summary)
    seed_metrics.to_csv(output / "seed_metrics.csv", index=False)
    component_metrics.to_csv(output / "component_seed_metrics.csv", index=False)
    bootstrap.to_csv(
        output / "bootstrap_draws.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    _plot(seed_metrics, summary, output / "refit_seed_gains.png")

    tracked = [
        output / "registration.json",
        output / "registration_manifest.json",
        public_registration_path,
        output / "panel_components.csv",
        output / "full_component_universe.csv",
        output / "summary.json",
        output / "seed_metrics.csv",
        output / "component_seed_metrics.csv",
        output / "bootstrap_draws.csv.gz",
        output / "refit_seed_gains.png",
        *[root / "run_config.json" for root in run_roots],
        *[root / "evidence" / "manifest.json" for root in run_roots],
    ]
    write_json(
        output / "manifest.json",
        {
            "schema": "rogii_refit_seed_stability_manifest_v1",
            "artifacts": [
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(tracked)
            ],
        },
    )
    public_tracked = [
        public_registration_path,
        output / "summary.json",
        output / "seed_metrics.csv",
        output / "component_seed_metrics.csv",
        output / "bootstrap_draws.csv.gz",
        output / "refit_seed_gains.png",
    ]
    write_json(
        output / "public_manifest.json",
        {
            "schema": "rogii_refit_seed_stability_public_manifest_v1",
            "artifacts": [
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(public_tracked)
            ],
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
