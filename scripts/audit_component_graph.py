#!/usr/bin/env python3
"""Audit raw-data identity, graph edges, split proximity and threshold sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_portfolio.artifacts import sha256_file, write_json  # noqa: E402
from rogii_portfolio.components import _fingerprint, build_components, split_components  # noqa: E402
from rogii_portfolio.io import CompetitionStore  # noqa: E402
from rogii_portfolio.reproduce import load_config  # noqa: E402


def inventory(root: Path) -> dict[str, object]:
    if root.is_file():
        return {
            "kind": "zip",
            "path": root.name,
            "bytes": root.stat().st_size,
            "sha256": sha256_file(root),
        }
    paths = sorted(
        [*root.glob("*__horizontal_well.csv"), *root.glob("*__typewell.csv")],
        key=lambda path: path.name,
    )
    files = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]
    digest = hashlib.sha256(
        "\n".join(f"{item['path']}:{item['bytes']}:{item['sha256']}" for item in files).encode()
    ).hexdigest()
    return {"kind": "flat_directory", "files": files, "inventory_sha256": digest}


def graph_edges(wells, config) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for record in wells:
        frame = record.horizontal
        rows.append(
            {
                "well": record.well_id,
                "x": float(pd.to_numeric(frame["X"], errors="coerce").median()),
                "y": float(pd.to_numeric(frame["Y"], errors="coerce").median()),
                "gr": _fingerprint(pd.to_numeric(frame["GR"], errors="coerce").to_numpy(float)),
                "tw": _fingerprint(pd.to_numeric(record.typewell["GR"], errors="coerce").to_numpy(float)),
            }
        )
    meta = pd.DataFrame(rows)
    xy = meta[["x", "y"]].to_numpy(float)
    gr = np.stack(meta["gr"]); tw = np.stack(meta["tw"])
    edges = []
    pairs = []
    for left in range(len(meta)):
        for right in range(left + 1, len(meta)):
            distance = float(np.linalg.norm(xy[left] - xy[right]))
            gr_corr = float(gr[left] @ gr[right])
            tw_corr = float(tw[left] @ tw[right])
            spatial = distance <= config.spatial_radius_ft
            similar = distance <= config.similarity_radius_ft and max(gr_corr, tw_corr) >= config.gr_similarity_threshold
            item = {
                "well_left": meta.loc[left, "well"],
                "well_right": meta.loc[right, "well"],
                "distance_ft": distance,
                "gr_correlation": gr_corr,
                "typewell_correlation": tw_corr,
                "edge": bool(spatial or similar),
                "edge_reason": "spatial+similar" if spatial and similar else "spatial" if spatial else "similar" if similar else "none",
            }
            pairs.append(item)
            if item["edge"]:
                edges.append(item)
    return pd.DataFrame(edges), pd.DataFrame(pairs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/realdata_nested_160.json")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    with CompetitionStore(args.data_root.resolve(), split="train") as store:
        wells = store.load_all()
    universe = build_components(wells, config)
    edges, pairs = graph_edges(wells, config)
    selected_path = args.artifact_root.resolve() / "evidence" / "geological_components.csv"
    selected = pd.read_csv(selected_path)
    selected_names = set(selected["well"].astype(str))
    selected_pairs = pairs.loc[
        pairs["well_left"].isin(selected_names) & pairs["well_right"].isin(selected_names)
    ].copy()
    development, holdout, assignment = split_components(selected, config)
    component_of = selected.set_index("well")["component"].astype(str).to_dict()

    def partition(well: str) -> str:
        component = component_of[well]
        return "holdout" if component in holdout else f"outer_{assignment[component]}"

    selected_pairs["partition_left"] = selected_pairs["well_left"].map(partition)
    selected_pairs["partition_right"] = selected_pairs["well_right"].map(partition)
    cross = selected_pairs.loc[selected_pairs["partition_left"] != selected_pairs["partition_right"]]
    diagnostics = {
        "universe_wells": len(wells),
        "universe_components": int(universe["component"].nunique()),
        "selected_wells": len(selected),
        "selected_components": int(selected["component"].nunique()),
        "base_edges": len(edges),
        "cross_partition_pairs": len(cross),
        "cross_partition_min_distance_ft": float(cross["distance_ft"].min()),
        "cross_partition_max_gr_correlation": float(cross["gr_correlation"].max()),
        "cross_partition_max_typewell_correlation": float(cross["typewell_correlation"].max()),
        "thresholds": {
            "spatial_radius_ft": config.spatial_radius_ft,
            "similarity_radius_ft": config.similarity_radius_ft,
            "gr_similarity_threshold": config.gr_similarity_threshold,
        },
        "interpretation": (
            "These are diagnostics conditional on the chosen graph rule. They cannot prove that the "
            "thresholds captured every geologically related pair."
        ),
    }
    sensitivity = []
    for radius in (500.0, 1000.0, 1500.0):
        for correlation in (0.93, 0.95, 0.97):
            candidate = replace(config, spatial_radius_ft=radius, gr_similarity_threshold=correlation)
            table = build_components(wells, candidate)
            sizes = table.groupby("component").size()
            sensitivity.append(
                {
                    "spatial_radius_ft": radius,
                    "similarity_threshold": correlation,
                    "components": int(len(sizes)),
                    "largest_component_wells": int(sizes.max()),
                    "multiwell_components": int((sizes > 1).sum()),
                }
            )
    output = args.artifact_root.resolve() / "graph_audit"
    output.mkdir(parents=True, exist_ok=True)
    edges.to_csv(output / "component_graph_edges.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(output / "component_graph_sensitivity.csv", index=False)
    write_json(output / "cross_partition_diagnostics.json", diagnostics)
    write_json(output / "dataset_inventory.json", inventory(args.data_root.resolve()))
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
