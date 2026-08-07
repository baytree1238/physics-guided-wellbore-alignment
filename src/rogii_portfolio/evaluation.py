"""Group-aware evaluation for saved prediction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .components import split_components
from .contracts import PipelineConfig


REGISTRY_SCHEMA = "rogii_robustness_registry_v1"
RESULT_SCHEMA = "rogii_robustness_evaluation_v1"
KEY_COLUMNS = ("row_id", "well", "row_number", "component", "horizon_ft", "truth")


@dataclass(frozen=True)
class ContractSpec:
    contract_id: str
    panel: str
    split: str
    role: str
    training_seed: int
    predictions: str
    components: str
    summary: str
    config: str
    manifest: str
    fold_metrics: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ContractSpec":
        required = {
            "contract_id",
            "panel",
            "split",
            "role",
            "training_seed",
            "predictions",
            "components",
            "summary",
            "config",
            "manifest",
        }
        missing = required - set(value)
        if missing:
            raise ValueError(f"contract is missing fields: {sorted(missing)}")
        split = str(value["split"])
        if split not in {"outer_oof", "holdout"}:
            raise ValueError(f"unsupported contract split: {split}")
        return cls(
            contract_id=str(value["contract_id"]),
            panel=str(value["panel"]),
            split=split,
            role=str(value["role"]),
            training_seed=int(value["training_seed"]),
            predictions=str(value["predictions"]),
            components=str(value["components"]),
            summary=str(value["summary"]),
            config=str(value["config"]),
            manifest=str(value["manifest"]),
            fold_metrics=None if value.get("fold_metrics") is None else str(value["fold_metrics"]),
        )


@dataclass(frozen=True)
class ExperimentRegistry:
    baseline: str
    models: tuple[str, ...]
    horizon_upper_bounds_ft: tuple[float, ...]
    bootstrap_draws: int
    bootstrap_seed: int
    worst_component_fraction: float
    contracts: tuple[ContractSpec, ...]
    disjoint_panels: tuple[tuple[str, str], ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentRegistry":
        if value.get("schema") != REGISTRY_SCHEMA:
            raise ValueError(f"registry schema must be {REGISTRY_SCHEMA}")
        bootstrap = value.get("component_bootstrap", {})
        registry = cls(
            baseline=str(value["baseline"]),
            models=tuple(str(x) for x in value["models"]),
            horizon_upper_bounds_ft=tuple(float(x) for x in value["horizon_upper_bounds_ft"]),
            bootstrap_draws=int(bootstrap.get("draws", 2000)),
            bootstrap_seed=int(bootstrap.get("seed", 20260807)),
            worst_component_fraction=float(value.get("worst_component_fraction", 0.10)),
            contracts=tuple(ContractSpec.from_dict(x) for x in value["contracts"]),
            disjoint_panels=tuple(tuple(str(y) for y in x) for x in value.get("disjoint_panels", [])),
        )
        registry.validate()
        return registry

    def validate(self) -> None:
        if not self.models or len(self.models) != len(set(self.models)):
            raise ValueError("registry models must be non-empty and unique")
        if self.baseline not in self.models:
            raise ValueError("baseline must appear in models")
        bounds = np.asarray(self.horizon_upper_bounds_ft, float)
        if not len(bounds) or not np.isfinite(bounds).all() or np.any(bounds <= 0) or np.any(np.diff(bounds) <= 0):
            raise ValueError("horizon bounds must be finite, positive, and strictly increasing")
        if self.bootstrap_draws < 1:
            raise ValueError("component bootstrap needs at least one draw")
        if not 0 < self.worst_component_fraction <= 1:
            raise ValueError("worst_component_fraction must lie in (0, 1]")
        identifiers = [contract.contract_id for contract in self.contracts]
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("contract IDs must be non-empty and unique")
        panels = {contract.panel for contract in self.contracts}
        for pair in self.disjoint_panels:
            if len(pair) != 2 or not set(pair) <= panels:
                raise ValueError(f"invalid disjoint panel pair: {pair}")


@dataclass
class RobustnessResult:
    summary: dict[str, Any]
    quality_checks: pd.DataFrame
    model_metrics: pd.DataFrame
    well_metrics: pd.DataFrame
    component_metrics: pd.DataFrame
    horizon_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    rank_stability: pd.DataFrame
    bootstrap_rank_draws: pd.DataFrame


def load_registry(path: Path) -> ExperimentRegistry:
    return ExperimentRegistry.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"registry path escapes project root: {relative}") from error
    return path


def _record_check(
    checks: list[dict[str, Any]],
    contract_id: str,
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append(
        {
            "contract_id": contract_id,
            "check": name,
            "passed": bool(passed),
            "detail": detail,
        }
    )
    if not passed:
        raise ValueError(f"{contract_id}: {name} failed: {detail}")


def _verify_manifest(manifest_path: Path, checks: list[dict[str, Any]], contract_id: str) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_root = manifest_path.parent.parent
    failures: list[str] = []
    for item in payload.get("artifacts", []):
        path = run_root / str(item["path"])
        if not path.exists():
            failures.append(f"missing {item['path']}")
        elif _sha256(path) != str(item["sha256"]):
            failures.append(f"hash mismatch {item['path']}")
    _record_check(
        checks,
        contract_id,
        "manifest_hashes_match",
        not failures,
        "all tracked artifacts match" if not failures else "; ".join(failures),
    )


def _load_component_table(path: Path, contract_id: str, checks: list[dict[str, Any]]) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"well": str, "component": str})
    required = {"well", "component", "rows"}
    _record_check(
        checks,
        contract_id,
        "component_table_schema",
        required <= set(table),
        f"required={sorted(required)}; observed={sorted(table.columns)}",
    )
    valid = (
        len(table) > 0
        and not table["well"].isna().any()
        and not table["component"].isna().any()
        and not table["well"].duplicated().any()
        and (pd.to_numeric(table["rows"], errors="coerce") > 0).all()
    )
    _record_check(
        checks,
        contract_id,
        "component_table_grain",
        valid,
        f"rows={len(table)}, wells={table['well'].nunique()}, components={table['component'].nunique()}",
    )
    return table


def _load_predictions(
    path: Path,
    models: tuple[str, ...],
    contract_id: str,
    checks: list[dict[str, Any]],
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"row_id": str, "well": str, "component": str},
    )
    required = set(KEY_COLUMNS) | set(models)
    _record_check(
        checks,
        contract_id,
        "prediction_schema",
        required <= set(frame),
        f"required_columns={len(required)}, observed_columns={len(frame.columns)}",
    )
    frame = frame.loc[:, [*KEY_COLUMNS, *models]].copy()
    _record_check(
        checks,
        contract_id,
        "row_id_unique",
        len(frame) > 0 and not frame["row_id"].isna().any() and not frame["row_id"].duplicated().any(),
        f"rows={len(frame)}, unique_row_ids={frame['row_id'].nunique(dropna=False)}",
    )
    numeric = frame.loc[:, ["row_number", "horizon_ft", "truth", *models]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite = np.isfinite(numeric.to_numpy(float)).all()
    _record_check(
        checks,
        contract_id,
        "numeric_values_finite",
        finite,
        f"numeric_cells={numeric.size}",
    )
    frame.loc[:, numeric.columns] = numeric
    _record_check(
        checks,
        contract_id,
        "horizon_nonnegative",
        bool((frame["horizon_ft"] >= 0).all()),
        f"range_ft=[{frame['horizon_ft'].min():.3f}, {frame['horizon_ft'].max():.3f}]",
    )
    well_component_counts = frame.groupby("well", sort=False)["component"].nunique()
    _record_check(
        checks,
        contract_id,
        "one_component_per_well",
        bool((well_component_counts == 1).all()),
        f"wells={len(well_component_counts)}, violations={(well_component_counts != 1).sum()}",
    )
    row_key_duplicates = frame.duplicated(["well", "row_number"]).sum()
    _record_check(
        checks,
        contract_id,
        "well_row_number_unique",
        int(row_key_duplicates) == 0,
        f"duplicate_keys={int(row_key_duplicates)}",
    )
    return frame


def _attach_split(
    frame: pd.DataFrame,
    components: pd.DataFrame,
    config_path: Path,
    spec: ContractSpec,
    checks: list[dict[str, Any]],
) -> tuple[pd.DataFrame, set[str]]:
    mapping = components.set_index("well")["component"].astype(str)
    observed_wells = set(frame["well"])
    coverage = observed_wells <= set(mapping.index)
    _record_check(
        checks,
        spec.contract_id,
        "component_mapping_coverage",
        coverage,
        f"prediction_wells={len(observed_wells)}, mapped_wells={len(observed_wells & set(mapping.index))}",
    )
    expected_mapping = frame["well"].map(mapping)
    mapping_match = bool(expected_mapping.notna().all() and np.array_equal(
        expected_mapping.to_numpy(str), frame["component"].to_numpy(str)
    ))
    _record_check(
        checks,
        spec.contract_id,
        "component_mapping_matches",
        mapping_match,
        f"mismatched_rows={int((expected_mapping.astype(str) != frame['component'].astype(str)).sum())}",
    )

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = PipelineConfig(**config_payload)
    development, holdout, assignment = split_components(components, config)
    expected_components = development if spec.split == "outer_oof" else holdout
    observed_components = set(frame["component"].astype(str))
    _record_check(
        checks,
        spec.contract_id,
        "split_component_set_matches",
        observed_components == expected_components,
        f"expected={len(expected_components)}, observed={len(observed_components)}",
    )
    result = frame.copy()
    if spec.split == "outer_oof":
        result["outer_fold"] = result["component"].map(assignment)
        _record_check(
            checks,
            spec.contract_id,
            "outer_fold_assignment_complete",
            bool(result["outer_fold"].notna().all()),
            f"folds={sorted(result['outer_fold'].dropna().unique().tolist())}",
        )
        result["outer_fold"] = result["outer_fold"].astype(int)
    return result, observed_components


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(truth, float) - np.asarray(prediction, float)))))


def _group_rmse_wide(frame: pd.DataFrame, models: tuple[str, ...], group: str) -> pd.DataFrame:
    errors = pd.DataFrame(
        {
            model: np.square(frame[model].to_numpy(float) - frame["truth"].to_numpy(float))
            for model in models
        },
        index=frame.index,
    )
    errors[group] = frame[group].to_numpy()
    return np.sqrt(errors.groupby(group, sort=True, observed=True)[list(models)].mean())


def _long_group_metrics(
    frame: pd.DataFrame,
    models: tuple[str, ...],
    baseline: str,
    group: str,
    contract_id: str,
) -> pd.DataFrame:
    wide = _group_rmse_wide(frame, models, group)
    counts = frame.groupby(group, sort=True, observed=True).size().rename("rows")
    long = wide.rename_axis(group).reset_index().melt(
        id_vars=group, var_name="model", value_name="rmse"
    )
    long = long.merge(counts.reset_index(), on=group, how="left", validate="many_to_one")
    baseline_scores = wide[baseline].rename("baseline_rmse").reset_index()
    long = long.merge(baseline_scores, on=group, how="left", validate="many_to_one")
    long["gain_vs_baseline"] = long["baseline_rmse"] - long["rmse"]
    if group == "well":
        well_components = frame.groupby("well", sort=True, observed=True)["component"].first().rename("component")
        long = long.merge(well_components.reset_index(), on="well", how="left", validate="many_to_one")
    elif group == "component":
        wells = frame.groupby("component", sort=True, observed=True)["well"].nunique().rename("wells")
        long = long.merge(wells.reset_index(), on="component", how="left", validate="many_to_one")
    long.insert(0, "contract_id", contract_id)
    return long


def _model_metrics(
    frame: pd.DataFrame,
    models: tuple[str, ...],
    baseline: str,
    contract_id: str,
    well_metrics: pd.DataFrame,
    component_metrics: pd.DataFrame,
    tail_fraction: float,
) -> pd.DataFrame:
    local_well = well_metrics.loc[well_metrics["contract_id"] == contract_id]
    local_component = component_metrics.loc[component_metrics["contract_id"] == contract_id]
    baseline_row_rmse = _rmse(frame["truth"].to_numpy(), frame[baseline].to_numpy())
    records: list[dict[str, Any]] = []
    component_count = int(frame["component"].nunique())
    tail_count = max(1, int(math.ceil(tail_fraction * component_count)))
    for model in models:
        wells = local_well.loc[local_well["model"] == model]
        components = local_component.loc[local_component["model"] == model]
        pooled = _rmse(frame["truth"].to_numpy(), frame[model].to_numpy())
        records.append(
            {
                "contract_id": contract_id,
                "model": model,
                "rows": len(frame),
                "wells": int(frame["well"].nunique()),
                "components": component_count,
                "pooled_row_rmse": pooled,
                "macro_well_rmse": float(wells["rmse"].mean()),
                "macro_component_rmse": float(components["rmse"].mean()),
                "harmed_well_rate": float((wells["gain_vs_baseline"] < -1e-12).mean()),
                "harmed_component_rate": float((components["gain_vs_baseline"] < -1e-12).mean()),
                "mean_well_gain_vs_baseline": float(wells["gain_vs_baseline"].mean()),
                "median_well_gain_vs_baseline": float(wells["gain_vs_baseline"].median()),
                "pooled_gain_vs_baseline": baseline_row_rmse - pooled,
                "worst_component_rmse": float(components["rmse"].max()),
                "worst_10pct_component_rmse_cvar": float(
                    components.nlargest(tail_count, "rmse")["rmse"].mean()
                ),
                "worst_tail_components": tail_count,
            }
        )
    metrics = pd.DataFrame(records)
    for column in ("pooled_row_rmse", "macro_well_rmse", "macro_component_rmse", "worst_10pct_component_rmse_cvar"):
        metrics[f"rank_{column}"] = metrics[column].rank(method="average", ascending=True)
    return metrics


def _mark_worst_components(component_metrics: pd.DataFrame, fraction: float) -> pd.DataFrame:
    result = component_metrics.copy()
    result["worst_rank"] = result.groupby(["contract_id", "model"])["rmse"].rank(
        method="first", ascending=False
    )
    counts = result.groupby(["contract_id", "model"])["component"].transform("size")
    result["in_worst_10pct"] = result["worst_rank"] <= np.ceil(fraction * counts)
    return result


def _horizon_labels(bounds: tuple[float, ...]) -> list[str]:
    def display(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else f"{value:g}"

    labels = [f"0-{display(bounds[0])}"]
    labels.extend(f">{display(left)}-{display(right)}" for left, right in zip(bounds[:-1], bounds[1:]))
    labels.append(f">{display(bounds[-1])}")
    return labels


def _horizon_metrics(
    frame: pd.DataFrame,
    models: tuple[str, ...],
    baseline: str,
    contract_id: str,
    bounds: tuple[float, ...],
) -> pd.DataFrame:
    labels = _horizon_labels(bounds)
    index = np.digitize(frame["horizon_ft"].to_numpy(float), np.asarray(bounds), right=True)
    local = frame.copy()
    local["horizon_bin"] = pd.Categorical.from_codes(index, labels, ordered=True)
    records: list[dict[str, Any]] = []
    for label in labels:
        subset = local.loc[local["horizon_bin"] == label]
        if subset.empty:
            continue
        well_wide = _group_rmse_wide(subset, models, "well")
        baseline_rmse = _rmse(subset["truth"].to_numpy(), subset[baseline].to_numpy())
        for model in models:
            pooled = _rmse(subset["truth"].to_numpy(), subset[model].to_numpy())
            records.append(
                {
                    "contract_id": contract_id,
                    "horizon_bin": label,
                    "horizon_bin_order": labels.index(label),
                    "rows": len(subset),
                    "wells": int(subset["well"].nunique()),
                    "components": int(subset["component"].nunique()),
                    "model": model,
                    "pooled_row_rmse": pooled,
                    "macro_well_rmse": float(well_wide[model].mean()),
                    "harmed_well_rate": float((well_wide[model] > well_wide[baseline] + 1e-12).mean()),
                    "gain_vs_baseline": baseline_rmse - pooled,
                }
            )
    return pd.DataFrame(records)


def _fold_metrics(
    frame: pd.DataFrame,
    models: tuple[str, ...],
    baseline: str,
    contract_id: str,
) -> pd.DataFrame:
    if "outer_fold" not in frame:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for fold, subset in frame.groupby("outer_fold", sort=True):
        scores = {model: _rmse(subset["truth"].to_numpy(), subset[model].to_numpy()) for model in models}
        ranks = dict(zip(models, rankdata([scores[model] for model in models], method="average")))
        best = min(scores.values())
        worst = max(scores.values())
        for model in models:
            records.append(
                {
                    "contract_id": contract_id,
                    "outer_fold": int(fold),
                    "rows": len(subset),
                    "wells": int(subset["well"].nunique()),
                    "components": int(subset["component"].nunique()),
                    "model": model,
                    "rmse": scores[model],
                    "rank": float(ranks[model]),
                    "is_best": bool(np.isclose(scores[model], best, rtol=0, atol=1e-12)),
                    "is_worst": bool(np.isclose(scores[model], worst, rtol=0, atol=1e-12)),
                    "gain_vs_baseline": scores[baseline] - scores[model],
                }
            )
    return pd.DataFrame(records)


def _reconcile_fold_metrics(
    calculated: pd.DataFrame,
    stored_path: Path | None,
    contract_id: str,
    checks: list[dict[str, Any]],
) -> None:
    if stored_path is None:
        _record_check(checks, contract_id, "stored_fold_metrics_present", False, "fold metrics path is absent")
        return
    stored = pd.read_csv(stored_path)
    comparable = [
        model for model in calculated["model"].unique() if f"{model}_rmse" in stored.columns
    ]
    failures: list[str] = []
    for model in comparable:
        expected = stored.set_index("outer_fold")[f"{model}_rmse"]
        observed = calculated.loc[calculated["model"] == model].set_index("outer_fold")["rmse"]
        common = expected.index.intersection(observed.index)
        if len(common) != len(expected) or not np.allclose(
            expected.loc[common].to_numpy(float),
            observed.loc[common].to_numpy(float),
            rtol=0,
            atol=1e-9,
        ):
            failures.append(model)
    _record_check(
        checks,
        contract_id,
        "stored_fold_metrics_reconcile",
        bool(comparable) and not failures,
        f"compared={comparable}; mismatches={failures}",
    )


def _reconcile_summary(
    frame: pd.DataFrame,
    summary_path: Path,
    config_path: Path,
    spec: ContractSpec,
    models: tuple[str, ...],
    checks: list[dict[str, Any]],
) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary_seed = summary.get("config", {}).get("seed")
    seeds_match = int(config.get("seed", -1)) == spec.training_seed and int(summary_seed or -1) == spec.training_seed
    _record_check(
        checks,
        spec.contract_id,
        "training_seed_matches_registry",
        seeds_match,
        f"registry={spec.training_seed}, config={config.get('seed')}, summary={summary_seed}",
    )

    count_prefix = "development" if spec.split == "outer_oof" else "holdout"
    expected_wells = int(summary[f"{count_prefix}_wells"])
    expected_components = int(summary[f"{count_prefix}_components"])
    counts_match = (
        frame["well"].nunique() == expected_wells
        and frame["component"].nunique() == expected_components
    )
    _record_check(
        checks,
        spec.contract_id,
        "summary_group_counts_reconcile",
        bool(counts_match),
        (
            f"wells={frame['well'].nunique()}/{expected_wells}, "
            f"components={frame['component'].nunique()}/{expected_components}"
        ),
    )

    score_key = "outer_oof_rmse" if spec.split == "outer_oof" else "untouched_holdout_rmse"
    stored_scores = summary[score_key]
    comparable = [model for model in models if model in stored_scores]
    mismatches = []
    for model in comparable:
        calculated = _rmse(frame["truth"].to_numpy(), frame[model].to_numpy())
        if not np.isclose(calculated, float(stored_scores[model]), rtol=0, atol=1e-9):
            mismatches.append(model)
    _record_check(
        checks,
        spec.contract_id,
        "summary_rmse_reconciles",
        bool(comparable) and not mismatches,
        f"compared={comparable}; mismatches={mismatches}",
    )


def _bootstrap_rank_draws(
    component_metrics: pd.DataFrame,
    models: tuple[str, ...],
    baseline: str,
    contract_id: str,
    draws: int,
    first_seed: int,
) -> pd.DataFrame:
    local = component_metrics.loc[component_metrics["contract_id"] == contract_id]
    wide = local.pivot(index="component", columns="model", values="rmse").loc[:, list(models)]
    values = wide.to_numpy(float)
    model_index = {model: index for index, model in enumerate(models)}
    records: list[dict[str, Any]] = []
    for repeat in range(draws):
        seed = first_seed + repeat
        rng = np.random.default_rng(seed)
        sample = rng.integers(0, len(values), size=len(values))
        scores = values[sample].mean(axis=0)
        ranks = rankdata(scores, method="average")
        best = float(scores.min())
        baseline_score = float(scores[model_index[baseline]])
        for index, model in enumerate(models):
            records.append(
                {
                    "contract_id": contract_id,
                    "repeat": repeat,
                    "seed": seed,
                    "model": model,
                    "macro_component_rmse": float(scores[index]),
                    "rank": float(ranks[index]),
                    "is_best": bool(np.isclose(scores[index], best, rtol=0, atol=1e-12)),
                    "gain_vs_baseline": baseline_score - float(scores[index]),
                }
            )
    return pd.DataFrame(records)


def _rank_summary(
    units: pd.DataFrame,
    scope: str,
    contract_id: str,
    ranking_metric: str,
) -> pd.DataFrame:
    if units.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for model, group in units.groupby("model", sort=False):
        rows.append(
            {
                "scope": scope,
                "contract_id": contract_id,
                "ranking_metric": ranking_metric,
                "model": model,
                "units": int(group["rank"].count()),
                "mean_rank": float(group["rank"].mean()),
                "rank_std": float(group["rank"].std(ddof=0)),
                "best_rate": float(group["is_best"].mean()),
                "beats_baseline_rate": float((group["gain_vs_baseline"] > 1e-12).mean()),
                "mean_gain_vs_baseline": float(group["gain_vs_baseline"].mean()),
                "worst_unit_gain_vs_baseline": float(group["gain_vs_baseline"].min()),
            }
        )
    return pd.DataFrame(rows)


def _contract_rank_units(
    model_metrics: pd.DataFrame,
    contracts: Iterable[str],
    baseline_model: str,
) -> pd.DataFrame:
    local = model_metrics.loc[model_metrics["contract_id"].isin(set(contracts))].copy()
    if local.empty:
        return local
    local["rank"] = local.groupby("contract_id")["macro_component_rmse"].rank(
        method="average", ascending=True
    )
    best = local.groupby("contract_id")["macro_component_rmse"].transform("min")
    local["is_best"] = np.isclose(local["macro_component_rmse"], best, rtol=0, atol=1e-12)
    baseline = local.loc[local["model"] == baseline_model, ["contract_id", "macro_component_rmse"]].rename(
        columns={"macro_component_rmse": "baseline_score"}
    )
    local = local.merge(baseline, on="contract_id", how="left", validate="many_to_one")
    local["gain_vs_baseline"] = local["baseline_score"] - local["macro_component_rmse"]
    return local


def evaluate_registry(
    registry: ExperimentRegistry,
    project_root: Path,
    *,
    verify_manifests: bool = True,
) -> RobustnessResult:
    """Evaluate every registered contract without refitting a model."""

    root = project_root.resolve()
    checks: list[dict[str, Any]] = []
    model_tables: list[pd.DataFrame] = []
    well_tables: list[pd.DataFrame] = []
    component_tables: list[pd.DataFrame] = []
    horizon_tables: list[pd.DataFrame] = []
    fold_tables: list[pd.DataFrame] = []
    bootstrap_tables: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    panel_components: dict[str, set[str]] = {}
    panel_splits: dict[str, dict[str, set[str]]] = {}
    verified_manifests: set[Path] = set()

    for contract_index, spec in enumerate(registry.contracts):
        prediction_path = _resolved(root, spec.predictions)
        component_path = _resolved(root, spec.components)
        summary_path = _resolved(root, spec.summary)
        config_path = _resolved(root, spec.config)
        manifest_path = _resolved(root, spec.manifest)
        fold_path = None if spec.fold_metrics is None else _resolved(root, spec.fold_metrics)
        for name, path in (
            ("predictions", prediction_path),
            ("components", component_path),
            ("summary", summary_path),
            ("config", config_path),
            ("manifest", manifest_path),
        ):
            _record_check(
                checks,
                spec.contract_id,
                f"{name}_exists",
                path.is_file(),
                path.relative_to(root).as_posix(),
            )
        if verify_manifests and manifest_path not in verified_manifests:
            _verify_manifest(manifest_path, checks, spec.contract_id)
            verified_manifests.add(manifest_path)

        components = _load_component_table(component_path, spec.contract_id, checks)
        frame = _load_predictions(prediction_path, registry.models, spec.contract_id, checks)
        frame, observed_components = _attach_split(frame, components, config_path, spec, checks)
        _reconcile_summary(
            frame,
            summary_path,
            config_path,
            spec,
            registry.models,
            checks,
        )
        panel_components.setdefault(spec.panel, set()).update(observed_components)
        panel_splits.setdefault(spec.panel, {})[spec.split] = observed_components
        source_hashes[spec.contract_id] = _sha256(prediction_path)

        well = _long_group_metrics(frame, registry.models, registry.baseline, "well", spec.contract_id)
        component = _long_group_metrics(
            frame, registry.models, registry.baseline, "component", spec.contract_id
        )
        metrics = _model_metrics(
            frame,
            registry.models,
            registry.baseline,
            spec.contract_id,
            well,
            component,
            registry.worst_component_fraction,
        )
        horizon = _horizon_metrics(
            frame,
            registry.models,
            registry.baseline,
            spec.contract_id,
            registry.horizon_upper_bounds_ft,
        )
        folds = _fold_metrics(frame, registry.models, registry.baseline, spec.contract_id)
        if spec.split == "outer_oof":
            _reconcile_fold_metrics(folds, fold_path, spec.contract_id, checks)

        first_seed = registry.bootstrap_seed + contract_index * (registry.bootstrap_draws + 1000)
        bootstrap = _bootstrap_rank_draws(
            component,
            registry.models,
            registry.baseline,
            spec.contract_id,
            registry.bootstrap_draws,
            first_seed,
        )
        model_tables.append(metrics)
        well_tables.append(well)
        component_tables.append(component)
        horizon_tables.append(horizon)
        if not folds.empty:
            fold_tables.append(folds)
        bootstrap_tables.append(bootstrap)

    for panel, splits in panel_splits.items():
        if {"outer_oof", "holdout"} <= set(splits):
            overlap = splits["outer_oof"] & splits["holdout"]
            _record_check(
                checks,
                panel,
                "oof_holdout_component_disjoint",
                not overlap,
                f"overlap_count={len(overlap)}",
            )
    for left, right in registry.disjoint_panels:
        overlap = panel_components[left] & panel_components[right]
        _record_check(
            checks,
            f"{left}__{right}",
            "registered_panels_component_disjoint",
            not overlap,
            f"overlap_count={len(overlap)}",
        )

    model_metrics = pd.concat(model_tables, ignore_index=True)
    well_metrics = pd.concat(well_tables, ignore_index=True)
    component_metrics = _mark_worst_components(
        pd.concat(component_tables, ignore_index=True), registry.worst_component_fraction
    )
    horizon_metrics = pd.concat(horizon_tables, ignore_index=True)
    fold_metrics = pd.concat(fold_tables, ignore_index=True) if fold_tables else pd.DataFrame()
    bootstrap_draws = pd.concat(bootstrap_tables, ignore_index=True)

    stability_tables: list[pd.DataFrame] = []
    for contract_id, local in bootstrap_draws.groupby("contract_id", sort=False):
        stability_tables.append(
            _rank_summary(
                local,
                "component_bootstrap_seed",
                contract_id,
                "macro_component_rmse",
            )
        )
    for contract_id, local in fold_metrics.groupby("contract_id", sort=False):
        stability_tables.append(
            _rank_summary(local, "outer_fold", contract_id, "pooled_row_rmse")
        )
    if not fold_metrics.empty:
        stability_tables.append(
            _rank_summary(
                fold_metrics,
                "outer_fold",
                "all_oof_folds",
                "pooled_row_rmse",
            )
        )
    all_contract_ids = [contract.contract_id for contract in registry.contracts]
    oof_contract_ids = [
        contract.contract_id for contract in registry.contracts if contract.split == "outer_oof"
    ]
    stability_tables.append(
        _rank_summary(
            _contract_rank_units(model_metrics, all_contract_ids, registry.baseline),
            "registered_contract",
            "all_contracts",
            "macro_component_rmse",
        )
    )
    stability_tables.append(
        _rank_summary(
            _contract_rank_units(model_metrics, oof_contract_ids, registry.baseline),
            "registered_contract",
            "outer_oof_only",
            "macro_component_rmse",
        )
    )
    rank_stability = pd.concat(
        [table for table in stability_tables if not table.empty], ignore_index=True
    )

    contract_best: list[dict[str, Any]] = []
    for contract_id, local in model_metrics.groupby("contract_id", sort=False):
        best = local.sort_values(["macro_component_rmse", "model"], kind="mergesort").iloc[0]
        contract_best.append(
            {
                "contract_id": contract_id,
                "model": str(best["model"]),
                "macro_component_rmse": float(best["macro_component_rmse"]),
            }
        )
    training_seeds = sorted({contract.training_seed for contract in registry.contracts})
    summary: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "contracts": len(registry.contracts),
        "contract_registry": [
            {
                "contract_id": contract.contract_id,
                "panel": contract.panel,
                "split": contract.split,
                "role": contract.role,
                "training_seed": contract.training_seed,
            }
            for contract in registry.contracts
        ],
        "models": list(registry.models),
        "baseline": registry.baseline,
        "metric_definitions": {
            "pooled_row_rmse": "RMSE over all scored rows; long wells receive more weight",
            "macro_well_rmse": "unweighted mean of per-well RMSE",
            "macro_component_rmse": "unweighted mean of geological-component RMSE",
            "harmed_well_rate": f"share of wells whose RMSE is strictly worse than {registry.baseline}",
            "worst_10pct_component_rmse_cvar": (
                "unweighted mean RMSE among the worst ceil(10%) of components"
            ),
            "bootstrap_seed_rank": (
                "rank after resampling whole components with replacement; this measures evaluation-sample "
                "uncertainty, not refit/training-seed uncertainty"
            ),
        },
        "horizon_bins_ft": _horizon_labels(registry.horizon_upper_bounds_ft),
        "worst_component_fraction": registry.worst_component_fraction,
        "component_bootstrap": {
            "draws_per_contract": registry.bootstrap_draws,
            "base_seed": registry.bootstrap_seed,
            "resampling_unit": "geological component",
            "ranking_metric": "macro_component_rmse",
        },
        "training_seed_coverage": {
            "unique_seeds": training_seeds,
            "can_estimate_refit_seed_stability": len(training_seeds) > 1,
            "note": (
                "All saved real-data predictions use one training seed. Refit-seed stability needs "
                "additional registered prediction artifacts. Fold and component-bootstrap stability "
                "are reported now."
                if len(training_seeds) == 1
                else "Multiple training seeds are present in the registry."
            ),
        },
        "quality_checks": {
            "passed": int(pd.DataFrame(checks)["passed"].sum()),
            "failed": int((~pd.DataFrame(checks)["passed"]).sum()),
        },
        "best_macro_component_model_by_contract": contract_best,
        "source_prediction_sha256": source_hashes,
    }
    return RobustnessResult(
        summary=summary,
        quality_checks=pd.DataFrame(checks),
        model_metrics=model_metrics,
        well_metrics=well_metrics,
        component_metrics=component_metrics,
        horizon_metrics=horizon_metrics,
        fold_metrics=fold_metrics,
        rank_stability=rank_stability,
        bootstrap_rank_draws=bootstrap_draws,
    )


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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_robustness_result(
    result: RobustnessResult,
    output_dir: Path,
    *,
    registry_path: Path,
) -> dict[str, Any]:
    """Write compact, deterministic tables and a hash manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", result.summary)
    result.quality_checks.to_csv(output_dir / "quality_checks.csv", index=False)
    result.model_metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    result.well_metrics.to_csv(output_dir / "well_metrics.csv", index=False)
    result.component_metrics.to_csv(output_dir / "component_metrics.csv", index=False)
    result.horizon_metrics.to_csv(output_dir / "horizon_metrics.csv", index=False)
    result.fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    result.rank_stability.to_csv(output_dir / "rank_stability.csv", index=False)
    result.bootstrap_rank_draws.to_csv(
        output_dir / "bootstrap_rank_draws.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    tracked = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "manifest.json")
    manifest = {
        "schema": "rogii_robustness_evaluation_manifest_v1",
        "registry_sha256": _sha256(registry_path),
        "artifacts": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in tracked
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest
