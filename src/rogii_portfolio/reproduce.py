"""Reproduction workflow for synthetic and official-data runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifacts import sha256_file, verify_manifest, write_result
from .components import build_components
from .contracts import PipelineConfig
from .io import CompetitionStore
from .pipeline import NestedResult, run_nested_experiment
from .synthetic import make_synthetic_wells


def load_config(path: Path) -> PipelineConfig:
    """Load and validate a JSON experiment configuration."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return PipelineConfig(**payload)


def select_complete_components(
    wells: list,
    universe: "pd.DataFrame",
    *,
    max_wells: int,
    seed: int,
) -> tuple[list, "pd.DataFrame"]:
    """Sample whole global components without looking at suffix targets."""

    if max_wells >= len(wells):
        return wells, universe.copy()
    rng = np.random.default_rng(seed)
    groups = {
        str(component): tuple(sorted(group["well"].astype(str)))
        for component, group in universe.groupby("component", sort=True)
    }
    order = np.asarray(sorted(groups), dtype=object)
    rng.shuffle(order)
    selected: set[str] = set()
    selected_components: list[str] = []
    for component in order:
        members = groups[str(component)]
        if len(selected) + len(members) <= max_wells:
            selected.update(members)
            selected_components.append(str(component))
        if len(selected) == max_wells:
            break
    if len(selected_components) < 2:
        raise RuntimeError("component-complete subset has too few independent units")
    chosen = [record for record in wells if record.well_id in selected]
    table = universe.loc[universe["well"].isin(selected)].copy().reset_index(drop=True)
    return chosen, table


def reproduce(root: Path, config_path: Path, *, data_root: Path | None = None) -> NestedResult:
    """Run one configured experiment and write its evidence bundle."""

    config = load_config(config_path)
    if config.mode == "smoke":
        wells = make_synthetic_wells(
            n_wells=int(config.extra.get("synthetic_wells", 18)),
            rows=int(config.extra.get("synthetic_rows", 180)),
            seed=config.split_seed,
        )
    elif config.mode == "full":
        if data_root is None:
            raise ValueError("full mode requires --data-root pointing to the official ZIP/directory")
        with CompetitionStore(data_root, split="train") as store:
            limit = config.extra.get("max_wells")
            graph_scope = str(config.extra.get("component_graph_scope", "evaluation_subset"))
            if limit is not None and graph_scope == "full_universe":
                universe_wells = store.load_all(limit=None)
                universe_table = build_components(universe_wells, config)
                primary_wells, primary_table = select_complete_components(
                    universe_wells,
                    universe_table,
                    max_wells=int(limit),
                    seed=config.split_seed,
                )
                subset_role = str(config.extra.get("component_subset", "primary"))
                if subset_role == "primary":
                    wells, component_table = primary_wells, primary_table
                elif subset_role == "complement_of_primary":
                    primary_names = {record.well_id for record in primary_wells}
                    wells = [record for record in universe_wells if record.well_id not in primary_names]
                    component_table = universe_table.loc[
                        ~universe_table["well"].isin(primary_names)
                    ].copy().reset_index(drop=True)
                    if set(primary_table["component"]) & set(component_table["component"]):
                        raise RuntimeError("primary and confirmation component sets overlap")
                else:
                    raise ValueError(f"unknown component_subset: {subset_role}")
            else:
                wells = store.load_all(limit=None if limit is None else int(limit))
                universe_wells = wells
                universe_table = None
                component_table = None
    else:
        raise ValueError(f"unsupported mode: {config.mode}")
    result = run_nested_experiment(
        wells,
        config,
        component_table=component_table if config.mode == "full" else None,
        component_graph_scope_wells=len(universe_wells) if config.mode == "full" else len(wells),
        component_universe=universe_table if config.mode == "full" else None,
    )
    if config.mode == "full":
        result.summary["component_graph_scope"] = (
            str(config.extra.get("component_graph_scope", "evaluation_subset"))
        )
        result.summary["component_graph_universe_components"] = int(
            (universe_table if universe_table is not None else result.components)["component"].nunique()
        )
    write_result(result, root, config_sha256=sha256_file(config_path))
    verify_manifest(root)
    return result
