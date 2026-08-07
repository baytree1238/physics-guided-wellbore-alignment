"""Leakage-safe audit of the reference PF's initial structural-rate estimate.

The discovery grid is fixed before looking at suffix targets.  Prediction sees
only MD/Z/GR/TVT_input plus typewell TVT/GR; hidden TVT is loaded after every
prediction solely for alignment and scoring.  The exact reference arm is
window=30, median first-difference, shrink=1.

Dev40 is the sole selection panel.  A separate invocation on Exact80 should be
made only for a candidate that passes all preregistered Dev gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEV_PANEL = Path("lightweight_pf_geohmm_evidence/dev40/predictions.npz")
DATA_ROOT = Path("/tmp/rogii-data")
DEV_OUT = Path("pf_initial_state_dev40_evidence")
MODEL_COLUMNS = ("MD", "Z", "GR", "TVT_input")
TYPEWELL_COLUMNS = ("TVT", "GR")
WINDOWS = (15, 30, 60, 120)
ESTIMATORS = ("median_difference", "endpoint", "ols")
SHRINKS = (0.0, 0.5, 1.0)
REFERENCE = "w030_median_difference_s1.0"


@dataclass(frozen=True)
class Arm:
    window: int
    estimator: str
    shrink: float

    @property
    def name(self) -> str:
        return f"w{self.window:03d}_{self.estimator}_s{self.shrink:.1f}"


def registered_arms() -> list[Arm]:
    return [Arm(w, estimator, shrink) for w in WINDOWS for estimator in ESTIMATORS for shrink in SHRINKS]


def compute_arms() -> list[Arm]:
    # Every shrink-zero arm has exactly the same initial state.  Compute that
    # mathematical duplicate once while retaining all 36 registered rows in
    # the score table.
    canonical_zero = Arm(30, "median_difference", 0.0)
    return [canonical_zero] + [arm for arm in registered_arms() if arm.shrink != 0.0]


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(prediction, float) - np.asarray(truth, float)))))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initial_rate(known: pd.DataFrame, arm: Arm) -> float:
    tail = known.tail(int(arm.window))
    md = tail["MD"].to_numpy(float)
    structural = tail["TVT_input"].to_numpy(float) + tail["Z"].to_numpy(float)
    finite = np.isfinite(md) & np.isfinite(structural)
    md, structural = md[finite], structural[finite]
    if len(md) < 3:
        raw = 0.0
    elif arm.estimator == "median_difference":
        delta_md = np.diff(md)
        usable = delta_md > 0.0
        raw = float(np.median(np.diff(structural)[usable] / delta_md[usable])) if int(usable.sum()) >= 3 else 0.0
    elif arm.estimator == "endpoint":
        denominator = float(md[-1] - md[0])
        raw = float((structural[-1] - structural[0]) / denominator) if denominator > 0.0 else 0.0
    elif arm.estimator == "ols":
        centered = md - float(md.mean())
        denominator = float(np.dot(centered, centered))
        raw = float(np.dot(centered, structural - float(structural.mean())) / denominator) if denominator > 0.0 else 0.0
    else:
        raise ValueError(arm.estimator)
    if not np.isfinite(raw):
        raw = 0.0
    return float(arm.shrink) * raw


def _single_path_pf(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    seed: int,
    particles: int,
    starting_rate: float,
) -> tuple[np.ndarray, float]:
    """Reference PF with only the scalar starting rate exposed.

    Operation order and RNG consumption are copied verbatim from
    ``geohmm_no_prior._reference_pf_single``.
    """

    hw = horizontal.loc[:, MODEL_COLUMNS].copy().reset_index(drop=True)
    tw = typewell.loc[:, TYPEWELL_COLUMNS].copy().sort_values("TVT", kind="mergesort").reset_index(drop=True)
    tw_tvt = tw["TVT"].to_numpy(float)
    tw_gr_series = tw["GR"].astype(float)
    tw_gr = tw_gr_series.fillna(float(tw_gr_series.mean())).to_numpy(float)
    if not np.isfinite(tw_gr).all():
        tw_gr = np.nan_to_num(tw_gr, nan=0.0)

    known = hw[hw["TVT_input"].notna()]
    evaluation = hw[hw["TVT_input"].isna()]
    last = known.iloc[-1]
    last_tvt, last_z, previous_md = float(last["TVT_input"]), float(last["Z"]), float(last["MD"])
    typewell_at_known = np.interp(known["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    gr_sigma = float(np.clip(np.nanstd(known["GR"].fillna(0.0).to_numpy(float) - typewell_at_known), 10.0, 60.0))
    if not np.isfinite(gr_sigma) or gr_sigma <= 0.0:
        gr_sigma = 30.0

    rng = np.random.default_rng(int(seed))
    count = int(particles)
    position = last_tvt + last_z + 4.5 * rng.standard_normal(count)
    rate = float(starting_rate) + 0.01 * rng.standard_normal(count)
    weight = np.full(count, 1.0 / count)
    gr_interpolated = hw["GR"].interpolate(limit_direction="both").fillna(float(np.mean(tw_gr)))
    evaluation_md = evaluation["MD"].to_numpy(float)
    evaluation_z = evaluation["Z"].to_numpy(float)
    evaluation_gr = gr_interpolated.to_numpy(float)[evaluation.index.to_numpy(int)]
    prediction = np.empty(len(evaluation), dtype=float)
    log_likelihood = 0.0

    for index in range(len(evaluation)):
        md_step = max(float(evaluation_md[index]) - previous_md, 1.0)
        rate = 0.998 * rate + 0.002 * rng.standard_normal(count)
        position = position + rate * md_step + 0.005 * rng.standard_normal(count)
        tvt_particles = np.clip(position - evaluation_z[index], tw_tvt[0] - 100.0, tw_tvt[-1] + 100.0)
        position = tvt_particles + evaluation_z[index]
        expected_gr = np.interp(tvt_particles, tw_tvt, tw_gr)
        distance = (evaluation_gr[index] - expected_gr) / gr_sigma
        likelihood = np.maximum(np.exp(-0.5 * np.minimum(np.square(distance), 600.0)), 1e-300)
        average_likelihood = float(np.sum(weight * likelihood))
        log_likelihood += float(np.log(max(average_likelihood, 1e-300)))
        weight *= likelihood
        weight_sum = float(weight.sum())
        weight = weight / weight_sum if weight_sum > 0.0 else np.full(count, 1.0 / count)
        effective_count = 1.0 / float(np.sum(np.square(weight)))
        if effective_count < 0.5 * count:
            cumulative = np.cumsum(weight)
            offset = rng.uniform(0.0, 1.0 / count)
            selected = np.clip(np.searchsorted(cumulative, offset + np.arange(count) / count), 0, count - 1)
            position = position[selected] + 0.1 * rng.standard_normal(count)
            rate = rate[selected] + 0.001 * rng.standard_normal(count)
            weight = np.full(count, 1.0 / count)
        prediction[index] = float(np.dot(weight, position - evaluation_z[index]))
        previous_md = float(evaluation_md[index])
    return prediction, float(log_likelihood)


def _ensemble(horizontal: pd.DataFrame, typewell: pd.DataFrame, arm: Arm, seeds: int, particles: int, temperature: float) -> np.ndarray:
    known = horizontal[horizontal["TVT_input"].notna()]
    rate = initial_rate(known, arm)
    members, likelihoods = [], []
    for seed in range(int(seeds)):
        prediction, likelihood = _single_path_pf(
            horizontal, typewell, seed=seed, particles=particles, starting_rate=rate
        )
        members.append(prediction)
        likelihoods.append(likelihood)
    matrix = np.stack(members)
    likelihoods = np.asarray(likelihoods, float)
    weights = np.exp((likelihoods - float(likelihoods.max())) / float(temperature))
    weights /= weights.sum()
    return np.sum(weights[:, None] * matrix, axis=0).astype(np.float32)


def _predict_well(task: tuple[str, str, int, int, float]) -> dict[str, object]:
    well, data_root, seeds, particles, temperature = task
    root = Path(data_root)
    raw_horizontal = pd.read_csv(root / "train" / f"{well}__horizontal_well.csv").sort_values("MD", kind="mergesort", ignore_index=True)
    raw_typewell = pd.read_csv(root / "train" / f"{well}__typewell.csv").sort_values("TVT", kind="mergesort", ignore_index=True)
    horizontal = raw_horizontal.loc[:, MODEL_COLUMNS]
    typewell = raw_typewell.loc[:, TYPEWELL_COLUMNS]
    evaluation = raw_horizontal["TVT_input"].isna().to_numpy()
    started = time.perf_counter()
    predictions = {}
    rates = {}
    known = horizontal[horizontal["TVT_input"].notna()]
    for arm in compute_arms():
        predictions[arm.name] = _ensemble(horizontal, typewell, arm, seeds, particles, temperature)
        rates[arm.name] = initial_rate(known, arm)
    return {
        "well": well,
        "truth": raw_horizontal.loc[evaluation, "TVT"].to_numpy(np.float32),
        "predictions": predictions,
        "rates": rates,
        "seconds": float(time.perf_counter() - started),
    }


def fold_scores(truth: np.ndarray, prediction: np.ndarray, folds: np.ndarray) -> list[float]:
    return [rmse(truth[folds == fold], prediction[folds == fold]) for fold in sorted(np.unique(folds))]


def bootstrap_gain_ci(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    well: np.ndarray,
    *,
    draws: int = 10000,
    seed: int = 260806,
) -> tuple[float, float]:
    unique = np.unique(well.astype(str))
    statistics = []
    for name in unique:
        mask = well.astype(str) == name
        statistics.append(
            (
                int(mask.sum()),
                float(np.sum(np.square(truth[mask] - reference[mask]))),
                float(np.sum(np.square(truth[mask] - candidate[mask]))),
            )
        )
    count = np.asarray([x[0] for x in statistics], np.int64)
    ref_sse = np.asarray([x[1] for x in statistics], float)
    cand_sse = np.asarray([x[2] for x in statistics], float)
    rng = np.random.default_rng(seed)
    gains = np.empty(draws, float)
    for index in range(draws):
        selected = rng.integers(0, len(unique), len(unique))
        denominator = int(count[selected].sum())
        gains[index] = np.sqrt(ref_sse[selected].sum() / denominator) - np.sqrt(cand_sse[selected].sum() / denominator)
    low, high = np.quantile(gains, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEV_PANEL)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEV_OUT)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=8.0)
    args = parser.parse_args()

    with np.load(args.panel, allow_pickle=False) as source:
        panel = {key: source[key].copy() for key in source.files}
    required = {"well", "fold", "truth", "pf"}
    if not required.issubset(panel):
        raise RuntimeError(f"panel missing {sorted(required - set(panel))}")
    wells = np.unique(panel["well"].astype(str))
    tasks = [(well, str(args.data_root), args.seeds, args.particles, args.temperature) for well in wells]

    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = [executor.submit(_predict_well, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            print(f"completed {index}/{len(tasks)}", flush=True)
    wall_seconds = float(time.perf_counter() - started)
    returned = {str(item["well"]): item for item in results}

    computed = {}
    rate_rows = []
    panel_well = panel["well"].astype(str)
    for arm in compute_arms():
        prediction = np.empty(len(panel_well), np.float32)
        for well in wells:
            mask = panel_well == well
            result = returned[well]
            if not np.array_equal(np.asarray(result["truth"], np.float32), panel["truth"][mask].astype(np.float32)):
                raise RuntimeError(f"{well}: truth/order canary failed")
            prediction[mask] = result["predictions"][arm.name]
            rate_rows.append({"well": well, "arm": arm.name, "initial_rate": result["rates"][arm.name]})
        computed[arm.name] = prediction
    zero_prediction = computed[Arm(30, "median_difference", 0.0).name]
    for arm in registered_arms():
        if arm.shrink == 0.0:
            computed[arm.name] = zero_prediction

    truth = panel["truth"].astype(float)
    folds = panel["fold"].astype(int)
    archived = panel["pf"].astype(np.float32)
    reference = computed[REFERENCE]
    canary_max = float(np.max(np.abs(reference.astype(float) - archived.astype(float))))
    canary_bits = bool(np.array_equal(reference.view(np.uint32), archived.view(np.uint32)))
    if not canary_bits:
        raise RuntimeError(f"reference PF canary failed: max_abs={canary_max}")

    reference_rmse = rmse(truth, reference)
    reference_folds = fold_scores(truth, reference, folds)
    rows = []
    for arm in registered_arms():
        prediction = computed[arm.name]
        arm_folds = fold_scores(truth, prediction, folds)
        gains = [left - right for left, right in zip(reference_folds, arm_folds)]
        low, high = bootstrap_gain_ci(truth, reference, prediction, panel_well)
        score = rmse(truth, prediction)
        rows.append(
            {
                "arm": arm.name,
                "window": arm.window,
                "estimator": arm.estimator,
                "shrink": arm.shrink,
                "rmse": score,
                "gain_vs_reference": reference_rmse - score,
                "positive_folds": int(sum(value > 0.0 for value in gains)),
                "min_fold_gain": float(min(gains)),
                "fold_gains": json.dumps(gains),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "passes_gate": bool(sum(value > 0.0 for value in gains) == 5 and low > 0.0 and reference_rmse - score >= 0.05),
            }
        )
    table = pd.DataFrame(rows).sort_values(["rmse", "arm"], ignore_index=True)
    passing = table[table["passes_gate"]]
    selected = None if passing.empty else str(passing.iloc[0]["arm"])
    scope = "Dev40 discovery" if len(wells) == 40 else f"{len(wells)}-well frozen transfer"
    summary = {
        "method": "pf_initial_structural_rate_audit_v1",
        "scope": scope,
        "created_submission_csv": False,
        "registered_grid": {"windows": WINDOWS, "estimators": ESTIMATORS, "shrinks": SHRINKS, "arms": 36, "computed_unique_arms": len(compute_arms())},
        "reference": {"arm": REFERENCE, "rmse": reference_rmse, "float32_bit_exact_vs_archive": canary_bits, "max_abs_vs_archive": canary_max},
        "selection_gate": "Dev only: 5/5 positive folds AND well-bootstrap 95% CI low > 0 AND pooled gain >= 0.05",
        "selected_arm": selected,
        "best_arm": table.iloc[0].to_dict(),
        "target_isolation": {"horizontal_prediction_columns": MODEL_COLUMNS, "typewell_prediction_columns": TYPEWELL_COLUMNS, "suffix_tvt_seen_by_prediction": False},
        "runtime": {"wall_seconds": wall_seconds, "summed_worker_seconds": float(sum(float(item["seconds"]) for item in results)), "workers": int(min(args.workers, len(tasks)))},
        "decision": "TRANSFER_SELECTED_ARM_TO_EXACT80" if selected else "STOP_NO_DEV_CANDIDATE_PASSED",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "scores.csv", index=False)
    pd.DataFrame(rate_rows).to_csv(args.output / "initial_rates.csv", index=False)
    save = {
        "well": panel["well"], "fold": panel["fold"], "truth": panel["truth"],
        "reference_pf": reference,
    }
    for arm in registered_arms():
        save[arm.name] = computed[arm.name]
    np.savez_compressed(args.output / "predictions.npz", **save)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda value: value.item() if hasattr(value, "item") else list(value)) + "\n", encoding="utf-8")
    hashes = {path.name: sha256_file(path) for path in args.output.iterdir() if path.is_file()}
    (args.output / "sha256.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=lambda value: value.item() if hasattr(value, "item") else list(value)), flush=True)


if __name__ == "__main__":
    main()
