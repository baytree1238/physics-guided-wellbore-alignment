"""Pure evaluation utilities for frozen-arm repeated-refit studies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CONTRACT_FILES = {
    "outer_oof": "outer_oof_predictions.csv.gz",
    "holdout": "untouched_holdout_predictions.csv.gz",
}
ALIGNMENT_COLUMNS = ("row_id", "well", "row_number", "component", "truth")


def _metrics(
    frame: pd.DataFrame,
    parent: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    truth = frame["truth"].to_numpy(float)
    local = frame[["well", "component"]].copy()
    local["parent_se"] = np.square(truth - np.asarray(parent, float))
    local["candidate_se"] = np.square(truth - np.asarray(candidate, float))
    component = local.groupby("component", sort=True)[["parent_se", "candidate_se"]].mean()
    well = local.groupby("well", sort=True)[["parent_se", "candidate_se"]].mean()
    tail_count = max(1, int(np.ceil(0.10 * len(component))))
    parent_tail = np.sort(component["parent_se"].to_numpy(float))[-tail_count:]
    candidate_tail = np.sort(component["candidate_se"].to_numpy(float))[-tail_count:]
    parent_rmse = float(np.sqrt(local["parent_se"].mean()))
    candidate_rmse = float(np.sqrt(local["candidate_se"].mean()))
    parent_macro_component = float(np.sqrt(component["parent_se"]).mean())
    candidate_macro_component = float(np.sqrt(component["candidate_se"]).mean())
    parent_cvar = float(np.sqrt(parent_tail.mean()))
    candidate_cvar = float(np.sqrt(candidate_tail.mean()))
    well_regret = np.sqrt(well["candidate_se"]) - np.sqrt(well["parent_se"])
    return {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "components": int(frame["component"].nunique()),
        "parent_row_rmse": parent_rmse,
        "candidate_row_rmse": candidate_rmse,
        "row_rmse_gain_ft": parent_rmse - candidate_rmse,
        "parent_macro_well_rmse": float(np.sqrt(well["parent_se"]).mean()),
        "candidate_macro_well_rmse": float(np.sqrt(well["candidate_se"]).mean()),
        "parent_macro_component_rmse": parent_macro_component,
        "candidate_macro_component_rmse": candidate_macro_component,
        "macro_component_gain_ft": parent_macro_component - candidate_macro_component,
        "parent_cvar10_component_rmse": parent_cvar,
        "candidate_cvar10_component_rmse": candidate_cvar,
        "cvar10_gain_ft": parent_cvar - candidate_cvar,
        "harmed_well_rate": float((well_regret > 0.0).mean()),
        "maximum_well_regret_ft": float(well_regret.max()),
    }


def _component_rows(
    frame: pd.DataFrame,
    parent: np.ndarray,
    candidate: np.ndarray,
    *,
    refit_seed: int,
    contract: str,
) -> list[dict[str, float | int | str]]:
    truth = frame["truth"].to_numpy(float)
    local = frame[["component"]].copy()
    local["parent_se"] = np.square(truth - np.asarray(parent, float))
    local["candidate_se"] = np.square(truth - np.asarray(candidate, float))
    rows = []
    for component, group in local.groupby("component", sort=True):
        count = int(len(group))
        parent_sse = float(group["parent_se"].sum())
        candidate_sse = float(group["candidate_se"].sum())
        rows.append(
            {
                "contract": contract,
                "refit_seed": int(refit_seed),
                "component": str(component),
                "rows": count,
                "parent_sse": parent_sse,
                "candidate_sse": candidate_sse,
                "parent_rmse": float(np.sqrt(parent_sse / count)),
                "candidate_rmse": float(np.sqrt(candidate_sse / count)),
                "gain_ft": float(np.sqrt(parent_sse / count) - np.sqrt(candidate_sse / count)),
            }
        )
    return rows


def hierarchical_component_seed_bootstrap(
    component_metrics: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Resample refit seeds and whole components, preserving within-component rows."""

    required = {"refit_seed", "component", "rows", "parent_sse", "candidate_sse"}
    missing = required - set(component_metrics.columns)
    if missing:
        raise ValueError(f"component metrics are missing columns: {sorted(missing)}")
    seeds = np.asarray(sorted(component_metrics["refit_seed"].unique()), dtype=np.int64)
    components = np.asarray(sorted(component_metrics["component"].astype(str).unique()), dtype=object)
    if len(seeds) < 3 or len(components) < 2:
        raise ValueError("hierarchical bootstrap needs at least three refits and two components")
    indexed = component_metrics.set_index(["refit_seed", "component"])
    if len(indexed) != len(seeds) * len(components) or indexed.index.duplicated().any():
        raise ValueError("every refit seed must cover every component exactly once")
    stats = np.empty((len(seeds), len(components), 3), dtype=float)
    for seed_index, refit_seed in enumerate(seeds):
        for component_index, component in enumerate(components):
            row = indexed.loc[(int(refit_seed), str(component))]
            stats[seed_index, component_index] = (
                float(row["rows"]),
                float(row["parent_sse"]),
                float(row["candidate_sse"]),
            )

    rng = np.random.default_rng(seed)
    gains = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        sampled_seeds = rng.integers(0, len(seeds), size=len(seeds))
        sampled_components = rng.integers(0, len(components), size=len(components))
        sample = stats[sampled_seeds[:, None], sampled_components[None, :], :].sum(axis=(0, 1))
        gains[draw] = np.sqrt(sample[1] / sample[0]) - np.sqrt(sample[2] / sample[0])
    draw_frame = pd.DataFrame({"draw": np.arange(len(gains), dtype=int), "gain_ft": gains})
    summary = {
        "draws": int(draws),
        "seed": int(seed),
        "refit_seeds": int(len(seeds)),
        "components": int(len(components)),
        "gain_mean": float(gains.mean()),
        "ci95_low": float(np.quantile(gains, 0.025)),
        "ci95_high": float(np.quantile(gains, 0.975)),
        "probability_positive": float((gains > 0.0).mean()),
        "uncertainty_scope": (
            "hierarchical resampling of the three observed refit seeds and whole geological components"
        ),
    }
    return summary, draw_frame


def evaluate_refit_runs(
    run_roots: Iterable[Path],
    *,
    refit_seeds: Iterable[int],
    weight: float,
    bootstrap_draws: int,
    bootstrap_seed: int,
    promotion_rules: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roots = [Path(path) for path in run_roots]
    seeds = [int(value) for value in refit_seeds]
    if len(roots) != len(seeds) or len(set(seeds)) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("run roots must map one-to-one to at least three distinct refit seeds")
    if not 0.0 <= float(weight) <= 0.25:
        raise ValueError("frozen weight must lie in [0, 0.25]")

    seed_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    reference_alignment: dict[str, pd.DataFrame] = {}
    run_manifests: list[dict[str, object]] = []
    for root, refit_seed in zip(roots, seeds):
        summary_path = root / "evidence" / "reproduction_summary.json"
        manifest_path = root / "evidence" / "manifest.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        seed_contract = summary["seed_contract"]
        if int(seed_contract["refit_seed"]) != refit_seed:
            raise RuntimeError(f"refit seed mismatch under {root}")
        # Public summaries must be portable and must not expose the author's
        # local filesystem.  The runner always stores sibling runs under this
        # relative layout; the immutable local registration keeps full paths.
        public_root = Path("runs") / root.name
        run_manifests.append(
            {
                "refit_seed": refit_seed,
                "root": public_root.as_posix(),
                "manifest_path": (public_root / "evidence" / "manifest.json").as_posix(),
            }
        )
        for contract, filename in CONTRACT_FILES.items():
            frame = pd.read_csv(root / "evidence" / filename).sort_values("row_id", kind="mergesort")
            frame = frame.reset_index(drop=True)
            required = set(ALIGNMENT_COLUMNS) | {
                "sequential_final",
                "ridge",
                "trust_region_ridge",
            }
            missing = required - set(frame.columns)
            if missing:
                raise RuntimeError(f"{root}: missing prediction columns {sorted(missing)}")
            alignment = frame.loc[:, ALIGNMENT_COLUMNS]
            if contract not in reference_alignment:
                reference_alignment[contract] = alignment.copy()
            else:
                expected = reference_alignment[contract]
                for name in ALIGNMENT_COLUMNS[:-1]:
                    if not np.array_equal(expected[name].to_numpy(), alignment[name].to_numpy()):
                        raise RuntimeError(f"{contract}: {name} differs across refit seeds")
                if not np.array_equal(
                    expected["truth"].to_numpy(float), alignment["truth"].to_numpy(float)
                ):
                    raise RuntimeError(f"{contract}: truth differs across refit seeds")
            parent = frame["sequential_final"].to_numpy(float)
            ridge = frame["ridge"].to_numpy(float)
            candidate = frame["trust_region_ridge"].to_numpy(float)
            expected_candidate = parent + float(weight) * (ridge - parent)
            maximum_error = float(np.max(np.abs(candidate - expected_candidate)))
            if maximum_error > 1e-10:
                raise RuntimeError(f"{root}: frozen blend algebra differs by {maximum_error:g}")
            metrics = _metrics(frame, parent, candidate)
            seed_rows.append({"contract": contract, "refit_seed": refit_seed, **metrics})
            component_rows.extend(
                _component_rows(
                    frame,
                    parent,
                    candidate,
                    refit_seed=refit_seed,
                    contract=contract,
                )
            )

    seed_metrics = pd.DataFrame(seed_rows).sort_values(
        ["contract", "refit_seed"], kind="mergesort", ignore_index=True
    )
    component_metrics = pd.DataFrame(component_rows).sort_values(
        ["contract", "refit_seed", "component"], kind="mergesort", ignore_index=True
    )
    bootstrap_summaries: dict[str, dict[str, float | int]] = {}
    draw_frames = []
    for offset, contract in enumerate(CONTRACT_FILES):
        local = component_metrics.loc[component_metrics["contract"] == contract]
        bootstrap, draws_frame = hierarchical_component_seed_bootstrap(
            local,
            draws=bootstrap_draws,
            seed=(int(bootstrap_seed) + offset) % (2**32),
        )
        bootstrap_summaries[contract] = bootstrap
        draws_frame.insert(0, "contract", contract)
        draw_frames.append(draws_frame)
    bootstrap_table = pd.concat(draw_frames, ignore_index=True)

    by_contract: dict[str, dict[str, object]] = {}
    for contract in CONTRACT_FILES:
        local = seed_metrics.loc[seed_metrics["contract"] == contract]
        by_contract[contract] = {
            "seed_row_rmse_gains_ft": [float(value) for value in local["row_rmse_gain_ft"]],
            "gain_mean_ft": float(local["row_rmse_gain_ft"].mean()),
            "gain_min_ft": float(local["row_rmse_gain_ft"].min()),
            "gain_max_ft": float(local["row_rmse_gain_ft"].max()),
            "gain_std_ft": float(local["row_rmse_gain_ft"].std(ddof=0)),
            "macro_component_gain_mean_ft": float(local["macro_component_gain_ft"].mean()),
            "cvar10_gain_mean_ft": float(local["cvar10_gain_ft"].mean()),
            "harmed_well_rate_mean": float(local["harmed_well_rate"].mean()),
            "hierarchical_bootstrap": bootstrap_summaries[contract],
        }

    primary = seed_metrics.loc[seed_metrics["contract"] == "outer_oof"]
    holdout = seed_metrics.loc[seed_metrics["contract"] == "holdout"]
    checks = {
        "all_seeds_positive_on_outer_oof": bool((primary["row_rmse_gain_ft"] > 0.0).all()),
        "all_seeds_positive_on_holdout": bool((holdout["row_rmse_gain_ft"] > 0.0).all()),
        "hierarchical_bootstrap_ci95_low_gt": bool(
            bootstrap_summaries["outer_oof"]["ci95_low"]
            > float(promotion_rules["hierarchical_bootstrap_ci95_low_gt"])
        ),
        "mean_macro_component_gain_min_ft": bool(
            primary["macro_component_gain_ft"].mean()
            >= float(promotion_rules["mean_macro_component_gain_min_ft"])
        ),
        "mean_cvar10_gain_min_ft": bool(
            primary["cvar10_gain_ft"].mean()
            >= float(promotion_rules["mean_cvar10_gain_min_ft"])
        ),
    }
    summary = {
        "schema": "rogii_refit_seed_stability_v1",
        "status": "PROMOTE" if all(checks.values()) else "HOLD",
        "primary_contract": "outer_oof",
        "refit_seeds": seeds,
        "frozen_weight": float(weight),
        "candidate_formula": "sequential_final + 0.05 * (ridge - sequential_final)",
        "alignment_verified_across_refits": True,
        "run_manifests": run_manifests,
        "contracts": by_contract,
        "promotion_checks": checks,
        "promotion_rules": promotion_rules,
        "caveat": (
            "The panel is new to the two contracts that selected the 5% weight but is not "
            "historically label-blind; all 773 labels appeared in earlier unrelated experiments."
        ),
    }
    return summary, seed_metrics, component_metrics, bootstrap_table
