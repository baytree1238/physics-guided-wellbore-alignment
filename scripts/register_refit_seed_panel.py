#!/usr/bin/env python3
"""Register a target-free panel that excludes components used to select the 5% arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_portfolio.artifacts import sha256_file, write_json  # noqa: E402
from rogii_portfolio.components import (  # noqa: E402
    build_components_from_metadata,
    well_metadata_record,
)
from rogii_portfolio.io import CompetitionStore  # noqa: E402
from rogii_portfolio.reproduce import load_config  # noqa: E402


def _prior_wells(roots: list[str]) -> tuple[set[str], list[dict[str, object]]]:
    well_sets: list[set[str]] = []
    sources: list[dict[str, object]] = []
    for relative in roots:
        path = ROOT / relative / "evidence" / "component_graph_universe.csv"
        manifest = ROOT / relative / "evidence" / "manifest.json"
        if not path.is_file() or not manifest.is_file():
            raise FileNotFoundError(f"prior panel evidence is incomplete: {relative}")
        table = pd.read_csv(path)
        wells = set(table["well"].astype(str))
        well_sets.append(wells)
        sources.append(
            {
                "root": relative,
                "universe_path": path.relative_to(ROOT).as_posix(),
                "universe_sha256": sha256_file(path),
                "manifest_path": manifest.relative_to(ROOT).as_posix(),
                "manifest_sha256": sha256_file(manifest),
                "wells": len(wells),
            }
        )
    if not well_sets:
        raise ValueError("at least one prior panel root is required")
    if any(wells != well_sets[0] for wells in well_sets[1:]):
        raise RuntimeError("prior panel roots do not share one registered universe")
    return set().union(*well_sets), sources


def _stream_universe(data_root: Path, config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with CompetitionStore(data_root, split="train") as store:
        names = store.wells()
        for index, name in enumerate(names, start=1):
            rows.append(well_metadata_record(store.load(name)))
            if index % 100 == 0 or index == len(names):
                print(f"target-free metadata: {index}/{len(names)} wells", flush=True)
    return build_components_from_metadata(pd.DataFrame(rows), config)


def _select_panel(
    eligible: pd.DataFrame,
    *,
    maximum_wells: int,
    seed: int,
) -> pd.DataFrame:
    groups = {
        str(component): tuple(sorted(group["well"].astype(str)))
        for component, group in eligible.groupby("component", sort=True)
    }
    rng = np.random.default_rng(seed)
    order = np.asarray(sorted(groups), dtype=object)
    rng.shuffle(order)
    selected: set[str] = set()
    for component in order:
        members = groups[str(component)]
        if len(selected) + len(members) <= maximum_wells:
            selected.update(members)
        if len(selected) == maximum_wells:
            break
    panel = eligible.loc[eligible["well"].astype(str).isin(selected)].copy()
    panel = panel.sort_values("well", kind="mergesort", ignore_index=True)
    if len(panel) < max(30, int(0.80 * maximum_wells)):
        raise RuntimeError("contract-new component-complete panel is too small for the registered study")
    if panel["component"].nunique() < 10:
        raise RuntimeError("contract-new panel has too few independent geological components")
    return panel


def _verify_registration(output: Path) -> dict[str, object]:
    manifest_path = output / "registration_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        path = output / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"registration artifact failed verification: {item['path']}")
    return json.loads((output / "registration.json").read_text(encoding="utf-8"))


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
    output.mkdir(parents=True, exist_ok=True)

    if (output / "registration_manifest.json").exists():
        registered = _verify_registration(output)
        if registered["data_sha256"] != sha256_file(data_root):
            raise RuntimeError("registered data archive differs from --data-root")
        print(json.dumps(registered, indent=2, sort_keys=True))
        return 0

    config = load_config(config_path)
    config.validate()
    extra = config.extra
    prior, prior_sources = _prior_wells(list(extra["prior_panel_roots"]))
    universe = _stream_universe(data_root, config)
    universe_wells = set(universe["well"].astype(str))
    missing_prior = prior - universe_wells
    if missing_prior:
        raise RuntimeError(f"current archive is missing {len(missing_prior)} prior-panel wells")

    prior_components = set(
        universe.loc[universe["well"].astype(str).isin(prior), "component"].astype(str)
    )
    eligible = universe.loc[~universe["component"].astype(str).isin(prior_components)].copy()
    panel = _select_panel(
        eligible,
        maximum_wells=int(extra["panel_max_wells"]),
        seed=int(extra["panel_selection_seed"]),
    )
    panel_components = set(panel["component"].astype(str))
    if panel_components & prior_components:
        raise AssertionError("registered panel touches a component containing a prior well")
    if set(panel["well"].astype(str)) & prior:
        raise AssertionError("registered panel contains a well used to select the 5% candidate")

    universe_path = output / "full_component_universe.csv"
    panel_path = output / "panel_components.csv"
    universe.to_csv(universe_path, index=False)
    panel.to_csv(panel_path, index=False)
    registration = {
        "schema": "rogii_contract_new_panel_registration_v1",
        "status": "REGISTERED_BEFORE_SCORING",
        "claim_scope": (
            "New relative to the two 320-well contracts used to select the exact 5% weight; "
            "not historically label-blind because earlier unrelated OOF experiments covered all 773 wells."
        ),
        "genuinely_blind_claim_allowed": False,
        "data_path": str(data_root),
        "data_sha256": sha256_file(data_root),
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "selection_uses_target_columns": False,
        "selection_fields": [
            "well_id",
            "median_X",
            "median_Y",
            "row_count",
            "horizontal_GR_fingerprint",
            "typewell_GR_fingerprint",
        ],
        "prior_sources": prior_sources,
        "full_universe": {
            "wells": int(len(universe)),
            "components": int(universe["component"].nunique()),
        },
        "prior_panel": {
            "wells": int(len(prior)),
            "full_graph_components_touched": int(len(prior_components)),
        },
        "contract_new_eligible_pool": {
            "wells": int(len(eligible)),
            "components": int(eligible["component"].nunique()),
        },
        "registered_panel": {
            "selection_seed": int(extra["panel_selection_seed"]),
            "split_seed": config.split_seed,
            "requested_max_wells": int(extra["panel_max_wells"]),
            "wells": int(len(panel)),
            "components": int(panel["component"].nunique()),
            "rows": int(panel["rows"].sum()),
            "prior_well_overlap": 0,
            "prior_full_graph_component_overlap": 0,
        },
        "frozen_candidate": {
            "parent": "sequential_final",
            "expert": "ridge",
            "expert_weight": config.trust_region_ridge_weight,
        },
        "refit_seeds": [int(value) for value in extra["refit_seeds"]],
        "pf_seed_offsets": [int(value) for value in extra["pf_seed_offsets"]],
        "promotion_rules": extra["promotion_rules"],
    }
    write_json(output / "registration.json", registration)
    tracked = [output / "registration.json", universe_path, panel_path]
    write_json(
        output / "registration_manifest.json",
        {
            "schema": "rogii_contract_new_panel_registration_manifest_v1",
            "artifacts": [
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in tracked
            ],
        },
    )
    print(json.dumps(registration, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
