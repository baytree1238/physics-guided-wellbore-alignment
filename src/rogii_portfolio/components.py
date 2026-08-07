"""Geological similarity graph and component-disjoint splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import PipelineConfig, WellRecord


class UnionFind:
    """Small disjoint-set structure used to assemble geological components."""

    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=int)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _fingerprint(values: np.ndarray, points: int = 64) -> np.ndarray:
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    if finite.sum() < 4:
        return np.zeros(points, dtype=float)
    filled = np.interp(np.arange(len(values)), np.flatnonzero(finite), values[finite])
    sample = np.interp(np.linspace(0, len(values) - 1, points), np.arange(len(values)), filled)
    sample -= sample.mean()
    scale = np.linalg.norm(sample)
    return sample / scale if scale > 1e-12 else np.zeros(points, dtype=float)


def well_metadata_record(record: WellRecord) -> dict[str, object]:
    """Build target-free graph metadata for one well.

    Streaming callers use this helper to construct a full-universe graph
    without retaining every raw well in memory.  Only inference-available
    coordinates and GR curves enter the result.
    """

    record.validate(require_truth=False)
    frame = record.horizontal
    return {
        "well": record.well_id,
        "x": float(pd.to_numeric(frame["X"], errors="coerce").median()),
        "y": float(pd.to_numeric(frame["Y"], errors="coerce").median()),
        "rows": len(frame),
        "gr_fingerprint": _fingerprint(
            pd.to_numeric(frame["GR"], errors="coerce").to_numpy(float)
        ),
        "typewell_fingerprint": _fingerprint(
            pd.to_numeric(record.typewell["GR"], errors="coerce").to_numpy(float)
        ),
    }


def well_metadata(wells: list[WellRecord]) -> pd.DataFrame:
    """Collect target-free graph metadata for a sequence of wells."""

    return pd.DataFrame([well_metadata_record(record) for record in wells])


def build_components_from_metadata(
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Connect wells using precomputed target-free metadata."""

    required = {
        "well",
        "x",
        "y",
        "rows",
        "gr_fingerprint",
        "typewell_fingerprint",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"component metadata is missing columns: {sorted(missing)}")
    if metadata.empty or metadata["well"].astype(str).duplicated().any():
        raise ValueError("component metadata must contain unique wells")
    metadata = metadata.copy().reset_index(drop=True)
    metadata["well"] = metadata["well"].astype(str)
    if not np.isfinite(metadata[["x", "y", "rows"]].to_numpy(float)).all():
        raise ValueError("component metadata contains non-finite scalar values")

    xy = metadata[["x", "y"]].to_numpy(float)
    gr = np.stack(metadata["gr_fingerprint"].to_list())
    tw = np.stack(metadata["typewell_fingerprint"].to_list())
    graph = UnionFind(len(metadata))
    edge_count = np.zeros(len(metadata), dtype=int)
    for left in range(len(metadata)):
        for right in range(left + 1, len(metadata)):
            distance = float(np.linalg.norm(xy[left] - xy[right]))
            gr_corr = float(np.dot(gr[left], gr[right]))
            tw_corr = float(np.dot(tw[left], tw[right]))
            spatial = distance <= config.spatial_radius_ft
            similar = (
                distance <= config.similarity_radius_ft
                and max(gr_corr, tw_corr) >= config.gr_similarity_threshold
            )
            if spatial or similar:
                graph.union(left, right)
                edge_count[left] += 1
                edge_count[right] += 1
    roots = [graph.find(i) for i in range(len(metadata))]
    unique = {root: f"component_{index:03d}" for index, root in enumerate(sorted(set(roots)))}
    output = metadata.drop(columns=["gr_fingerprint", "typewell_fingerprint"]).copy()
    output["component"] = [unique[root] for root in roots]
    output["graph_degree"] = edge_count
    return output.sort_values("well", kind="mergesort", ignore_index=True)


def build_components(wells: list[WellRecord], config: PipelineConfig) -> pd.DataFrame:
    """Connect geographically close or strongly similar wells before splitting."""

    return build_components_from_metadata(well_metadata(wells), config)


def split_components(
    component_table: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[set[str], set[str], dict[str, int]]:
    """Assign complete components to a holdout and row-balanced outer folds."""
    rng = np.random.default_rng(config.split_seed)
    sizes = component_table.groupby("component", sort=True)["rows"].sum().sort_values(ascending=False)
    components = sizes.index.to_numpy(str)
    shuffled = components.copy()
    rng.shuffle(shuffled)
    target = config.holdout_fraction * float(sizes.sum())
    holdout: list[str] = []
    running = 0.0
    for component in shuffled:
        if running < target:
            holdout.append(str(component))
            running += float(sizes.loc[component])
    if len(holdout) == 0 or len(holdout) == len(components):
        raise RuntimeError("could not create a non-empty component holdout")
    development = [str(x) for x in components if str(x) not in set(holdout)]

    fold_load = np.zeros(config.outer_folds, dtype=float)
    assignment: dict[str, int] = {}
    for component in sorted(development, key=lambda x: (-float(sizes.loc[x]), x)):
        fold = int(np.argmin(fold_load))
        assignment[component] = fold
        fold_load[fold] += float(sizes.loc[component])
    if set(assignment).intersection(holdout):
        raise AssertionError("holdout component entered outer folds")
    return set(development), set(holdout), assignment


def assert_component_disjoint(train_wells: set[str], valid_wells: set[str], table: pd.DataFrame) -> None:
    """Raise when training and validation wells share a graph component."""

    mapping = table.set_index("well")["component"].astype(str)
    train_components = set(mapping.loc[sorted(train_wells)])
    valid_components = set(mapping.loc[sorted(valid_wells)])
    overlap = train_components & valid_components
    if overlap:
        raise AssertionError(f"geological components crossed a split: {sorted(overlap)}")
