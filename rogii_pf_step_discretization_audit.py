"""Audit the particle filter's motion-step discretization on a leakage-safe panel.

Two hypotheses are separated:

1. Continuous-time OU scaling by ``delta_MD``.  In the official row data used
   here every suffix step is exactly 1 ft, making the proposed transform
   algebraically identical to the incumbent update.
2. Replacing measured-depth step by 3-D chord length from X/Y/Z.  This is a
   target-free but approximate motion coordinate and is recomputed on Dev40.

The worker receives only MD/X/Y/Z/GR/TVT_input and typewell TVT/GR.  TVT is read
after prediction solely for scoring/alignment.  No submission is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


PANEL = Path("lightweight_pf_geohmm_evidence/dev40/predictions.npz")
DATA_ROOT = Path("/tmp/rogii-data")
OUT = Path("pf_step_discretization_dev40_evidence")
MODEL_COLUMNS = ("MD", "X", "Y", "Z", "GR", "TVT_input")
TYPEWELL_COLUMNS = ("TVT", "GR")


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(prediction) - np.asarray(truth)))))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _single_path_pf(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    seed: int,
    particles: int,
) -> tuple[np.ndarray, float]:
    hw = horizontal.loc[:, MODEL_COLUMNS].copy().reset_index(drop=True)
    tw = typewell.loc[:, TYPEWELL_COLUMNS].copy().sort_values("TVT", kind="mergesort").reset_index(drop=True)
    tw_tvt = pd.to_numeric(tw["TVT"], errors="coerce").to_numpy(float)
    tw_gr_series = pd.to_numeric(tw["GR"], errors="coerce")
    tw_gr = tw_gr_series.fillna(float(tw_gr_series.mean())).to_numpy(float)
    known = hw[hw["TVT_input"].notna()]
    evaluation = hw[hw["TVT_input"].isna()]
    last = known.iloc[-1]
    last_tvt, last_z = float(last["TVT_input"]), float(last["Z"])
    previous_xyz = last[["X", "Y", "Z"]].to_numpy(float)
    typewell_at_known = np.interp(known["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    gr_sigma = float(np.clip(np.nanstd(known["GR"].fillna(0.0).to_numpy(float) - typewell_at_known), 10.0, 60.0))
    tail = known.tail(30)
    delta_tvt = np.diff(tail["TVT_input"].to_numpy(float))
    delta_z = np.diff(tail["Z"].to_numpy(float))
    delta_md = np.diff(tail["MD"].to_numpy(float))
    usable = delta_md > 0.0
    initial_rate = float(np.median((delta_tvt + delta_z)[usable] / delta_md[usable])) if int(usable.sum()) >= 3 else 0.0

    rng = np.random.default_rng(int(seed))
    count = int(particles)
    position = last_tvt + last_z + 4.5 * rng.standard_normal(count)
    rate = initial_rate + 0.01 * rng.standard_normal(count)
    weight = np.full(count, 1.0 / count)
    gr_interpolated = hw["GR"].interpolate(limit_direction="both").fillna(float(np.mean(tw_gr)))
    evaluation_xyz = evaluation.loc[:, ["X", "Y", "Z"]].to_numpy(float)
    evaluation_z = evaluation["Z"].to_numpy(float)
    evaluation_gr = gr_interpolated.to_numpy(float)[evaluation.index.to_numpy(int)]
    prediction = np.empty(len(evaluation), dtype=np.float64)
    log_likelihood = 0.0
    decay_one = 0.998
    innovation_one = 0.002
    position_noise_one = 0.005
    innovation_denominator = 1.0 - decay_one**2

    for index in range(len(evaluation)):
        step = max(float(np.linalg.norm(evaluation_xyz[index] - previous_xyz)), 0.25)
        decay = decay_one**step
        innovation_scale = innovation_one * np.sqrt((1.0 - decay_one ** (2.0 * step)) / innovation_denominator)
        rate = decay * rate + innovation_scale * rng.standard_normal(count)
        position = position + rate * step + position_noise_one * np.sqrt(step) * rng.standard_normal(count)
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
        previous_xyz = evaluation_xyz[index]
    return prediction, log_likelihood


def _predict_well(task: tuple[str, str, int, int, float]) -> dict[str, object]:
    well, data_root, seeds, particles, temperature = task
    root = Path(data_root)
    raw_horizontal = pd.read_csv(root / "train" / f"{well}__horizontal_well.csv").sort_values("MD", kind="mergesort", ignore_index=True)
    raw_typewell = pd.read_csv(root / "train" / f"{well}__typewell.csv").sort_values("TVT", kind="mergesort", ignore_index=True)
    horizontal = raw_horizontal.loc[:, MODEL_COLUMNS]
    typewell = raw_typewell.loc[:, TYPEWELL_COLUMNS]
    predictions, likelihoods = [], []
    started = time.perf_counter()
    for seed in range(seeds):
        prediction, likelihood = _single_path_pf(horizontal, typewell, seed=seed, particles=particles)
        predictions.append(prediction)
        likelihoods.append(likelihood)
    matrix = np.stack(predictions)
    likelihoods = np.asarray(likelihoods)
    weights = np.exp((likelihoods - likelihoods.max()) / temperature)
    weights /= weights.sum()
    suffix_prediction = np.sum(weights[:, None] * matrix, axis=0)
    evaluation = raw_horizontal["TVT_input"].isna().to_numpy()
    return {
        "well": well,
        "prediction": suffix_prediction.astype(np.float32),
        "truth": raw_horizontal.loc[evaluation, "TVT"].to_numpy(np.float32),
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=8.0)
    args = parser.parse_args()
    with np.load(args.panel, allow_pickle=False) as source:
        panel = {key: source[key].copy() for key in source.files}
    wells = np.unique(panel["well"].astype(str))

    md_steps, chord_steps = [], []
    for well in wells:
        horizontal = pd.read_csv(args.data_root / "train" / f"{well}__horizontal_well.csv").sort_values("MD", kind="mergesort", ignore_index=True)
        evaluation = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
        known = np.flatnonzero(horizontal["TVT_input"].notna().to_numpy())
        rows = np.r_[known[-1], evaluation]
        md_steps.extend(np.diff(horizontal["MD"].to_numpy(float)[rows]))
        xyz = horizontal.loc[rows, ["X", "Y", "Z"]].to_numpy(float)
        chord_steps.extend(np.sqrt(np.sum(np.square(np.diff(xyz, axis=0)), axis=1)))
    md_steps = np.asarray(md_steps)
    chord_steps = np.asarray(chord_steps)
    md_identity = bool(np.array_equal(md_steps, np.ones_like(md_steps)))

    tasks = [(well, str(args.data_root), args.seeds, args.particles, args.temperature) for well in wells]
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = [executor.submit(_predict_well, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    wall = time.perf_counter() - started
    returned = {str(item["well"]): item for item in results}
    chord_prediction = np.empty(len(panel["truth"]), dtype=np.float64)
    for well in wells:
        mask = panel["well"].astype(str) == well
        result = returned[well]
        if not np.array_equal(result["truth"], panel["truth"][mask].astype(np.float32)):
            raise RuntimeError(f"{well}: truth/order canary failed")
        chord_prediction[mask] = result["prediction"]

    truth = panel["truth"].astype(np.float64)
    incumbent = panel["pf"].astype(np.float64)
    hmm = panel["hmm_stride6"].astype(np.float64)
    folds = panel["fold"].astype(int)
    arms = {
        "incumbent_pf": incumbent,
        "delta_md_ou_pf": incumbent.copy() if md_identity else np.full_like(incumbent, np.nan),
        "xyz_chord_ou_pf": chord_prediction,
        "incumbent_pf_plus_20pct_stride6_hmm": 0.8 * incumbent + 0.2 * hmm,
        "xyz_chord_pf_plus_20pct_stride6_hmm": 0.8 * chord_prediction + 0.2 * hmm,
        "incumbent_pf_plus_25pct_stride6_hmm": 0.75 * incumbent + 0.25 * hmm,
        "xyz_chord_pf_plus_25pct_stride6_hmm": 0.75 * chord_prediction + 0.25 * hmm,
    }
    rows = []
    for name, prediction in arms.items():
        if not np.isfinite(prediction).all():
            continue
        fold_scores = [rmse(truth[folds == fold], prediction[folds == fold]) for fold in sorted(np.unique(folds))]
        rows.append(
            {
                "arm": name,
                "rmse": rmse(truth, prediction),
                "gain_vs_incumbent_pf": rmse(truth, incumbent) - rmse(truth, prediction),
                "fold_scores": json.dumps(fold_scores),
            }
        )
    table = pd.DataFrame(rows).sort_values("rmse", ignore_index=True)
    chord_fold_gains = [
        rmse(truth[folds == fold], incumbent[folds == fold]) - rmse(truth[folds == fold], chord_prediction[folds == fold])
        for fold in sorted(np.unique(folds))
    ]
    scope = "Dev40 discovery" if len(wells) == 40 else f"{len(wells)}-well transfer (historically opened)"
    summary = {
        "method": "continuous_time_ou_step_and_xyz_chord_pf_audit_v1",
        "scope": scope,
        "created_submission_csv": False,
        "target_isolation": {
            "horizontal_prediction_columns": list(MODEL_COLUMNS),
            "typewell_prediction_columns": list(TYPEWELL_COLUMNS),
            "suffix_tvt_seen_by_prediction": False,
        },
        "delta_md": {
            "rows": int(len(md_steps)),
            "minimum": float(md_steps.min()),
            "maximum": float(md_steps.max()),
            "fraction_not_exactly_one": float(np.mean(md_steps != 1.0)),
            "ou_transform_algebraically_identical": md_identity,
            "rmse_change": 0.0 if md_identity else None,
        },
        "xyz_chord": {
            "step_quantiles": {str(q): float(np.quantile(chord_steps, q)) for q in (0.0, 0.01, 0.5, 0.99, 1.0)},
            "pf_rmse": rmse(truth, chord_prediction),
            "incumbent_pf_rmse": rmse(truth, incumbent),
            "gain_vs_incumbent_pf": rmse(truth, incumbent) - rmse(truth, chord_prediction),
            "fold_gains": chord_fold_gains,
            "positive_folds": int(sum(value > 0.0 for value in chord_fold_gains)),
        },
        "best_arm": table.iloc[0].to_dict(),
        "runtime": {
            "wall_seconds": wall,
            "summed_worker_seconds": float(sum(float(item["seconds"]) for item in results)),
            "workers": int(min(args.workers, len(tasks))),
        },
        "decision": "HOLD_UNLESS_CHORD_TRANSFER_IS_MATERIAL_AND_STABLE",
        "limitations": [
            f"{scope} is not a fresh pristine confirmation panel.",
            "3-D chord length is a straight-line approximation; MD is the physical along-hole arc length.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "scores.csv", index=False)
    np.savez_compressed(
        args.output / "predictions.npz",
        well=panel["well"], fold=panel["fold"], truth=panel["truth"], incumbent_pf=panel["pf"],
        xyz_chord_ou_pf=chord_prediction.astype(np.float32), hmm_stride6=panel["hmm_stride6"],
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
