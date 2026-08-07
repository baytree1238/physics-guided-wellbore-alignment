from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import inspect

import numpy as np
import pandas as pd
import pytest

from rogii_portfolio.contracts import PipelineConfig
from rogii_portfolio.hgrg import rms
from rogii_portfolio.physics import PhysicsExperts, prepare_scored_well
from rogii_portfolio.pipeline import InferenceCache, _predict_one
from rogii_portfolio.stack import STACK_ARMS, StackPolicy, apply_stack
from rogii_portfolio.switching_state import (
    REGIME_NAMES,
    SwitchingStatePolicy,
    apply_switching_state,
)
from rogii_portfolio.synthetic import make_synthetic_wells


def _policy(**changes: object) -> SwitchingStatePolicy:
    return replace(
        SwitchingStatePolicy(),
        stride_rows=1,
        boundary_ramp_ft=0.0,
        **changes,
    )


def _cleanroom_fixture() -> tuple[InferenceCache, object, object]:
    record = make_synthetic_wells(n_wells=10, rows=100, seed=19)[0]
    config = PipelineConfig(outer_folds=2, inner_folds=2, pf_seeds=2, pf_particles=16)
    prepared = prepare_scored_well(record, config).model_input
    known = np.flatnonzero(np.isfinite(prepared.inference["TVT_input"].to_numpy(float)))
    last = int(known[-1])
    visible = prepared.inference["TVT_input"].to_numpy(float)
    md = prepared.inference["MD"].to_numpy(float)
    tail = known[-20:]
    slope = float(np.median(np.diff(visible[tail]) / np.diff(md[tail])))
    pf_full = visible.copy()
    pf_full[prepared.suffix_rows] = (
        visible[last]
        + slope * prepared.horizon
        + 0.35 * np.sin(prepared.horizon / 11.0)
    )
    hmm_full = pf_full.copy()
    hmm_full[prepared.suffix_rows] += 0.45 * np.cos(prepared.horizon / 13.0)
    physics = PhysicsExperts(
        pf_full=pf_full,
        pf_std_suffix=np.full(len(prepared.suffix_rows), 0.3),
        hmm_full=hmm_full,
        hmm_std_suffix=np.full(len(prepared.suffix_rows), 0.4),
        diagnostics={},
    )
    cache = InferenceCache(
        prepared=prepared,
        physics=physics,
        features=pd.DataFrame(index=np.arange(len(prepared.suffix_rows))),
    )

    class Models:
        def predict(self, features: pd.DataFrame, pf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            del features
            ridge = pf + 0.6 * np.sin(prepared.horizon / 17.0)
            nonlinear = pf - 0.2 * np.cos(prepared.horizon / 19.0)
            return ridge, nonlinear

    class Surface:
        def predict(self, item: object) -> tuple[np.ndarray, float]:
            del item
            structural = pf_full[prepared.suffix_rows] + 0.25 * np.sin(prepared.horizon / 23.0)
            return structural, 4.0

    return cache, Models(), Surface()


def _prediction_hash(predictions: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, values in predictions.items():
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes())
    return digest.hexdigest()


def test_switching_state_is_target_free_aligned_and_capped() -> None:
    rows = 180
    horizon = 2.0 + 4.0 * np.arange(rows, dtype=float)
    hgrg = 10_000.0 + 0.04 * horizon
    phase = np.linspace(0.0, 5.0, rows)
    pf = hgrg + 30.0 * np.sin(phase)
    hmm = hgrg + 22.0 * np.sin(phase + 0.15)
    structural = hgrg + 18.0 * np.sin(phase - 0.10)

    candidate, state, probability, diagnostics = apply_switching_state(
        hgrg=hgrg,
        pf=pf,
        hmm=hmm,
        structural=structural,
        horizon=horizon,
        prefix_cv=5.0,
    )

    assert "truth" not in inspect.signature(apply_switching_state).parameters
    assert candidate.shape == (rows,)
    assert state.shape == (rows, 3)
    assert probability.shape == (rows, len(REGIME_NAMES))
    assert np.isfinite(candidate).all()
    assert np.isfinite(state).all()
    assert np.isfinite(probability).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-12)
    assert np.min(probability) >= 0.0
    assert rms(candidate - hgrg) <= 5.0 + 1e-10
    assert np.max(np.abs(candidate - hgrg)) <= 10.0 + 1e-10
    assert diagnostics["move_rms_ft"] <= 5.0 + 1e-10
    assert diagnostics["move_absmax_ft"] <= 10.0 + 1e-10


def test_cleanroom_default_arm_set_and_prediction_hash_are_unchanged() -> None:
    config = PipelineConfig()
    assert config.switching_state_enabled is False
    assert config.trust_region_ridge_enabled is False
    assert config.trust_region_ridge_weight == 0.05
    assert asdict(config)["extra"] == {}
    assert "switching_state_enabled" not in asdict(config)

    cache, models, surface = _cleanroom_fixture()
    output = _predict_one(cache, models, surface)
    assert tuple(output["predictions"]) == (
        "pf",
        "ridge",
        "nonlinear",
        "incumbent",
        "hgrg",
        "state",
        "meta_state",
        "prefix_boundary",
        "sequential_final",
    )
    assert "switching_state" not in output["diagnostics"]
    assert _prediction_hash(output["predictions"]) == (
        "6a4f8ab98f6ac7bb4da05f60c0c6332394c23d1a9e078b95889552a8506b7e39"
    )


def test_cleanroom_opt_in_adds_candidate_without_changing_frozen_paths() -> None:
    cache, models, surface = _cleanroom_fixture()
    baseline = _predict_one(cache, models, surface)
    config = PipelineConfig(extra={"enable_switching_state": True})
    config.validate()
    experimental = _predict_one(
        cache,
        models,
        surface,
        enable_switching_state=config.switching_state_enabled,
    )

    assert tuple(experimental["predictions"]) == (*tuple(baseline["predictions"]), "switching_state")
    for name, values in baseline["predictions"].items():
        np.testing.assert_array_equal(experimental["predictions"][name], values)
    switching = experimental["predictions"]["switching_state"]
    assert np.isfinite(switching).all()
    assert rms(switching - experimental["predictions"]["hgrg"]) <= 5.0 + 1e-10
    assert "switching_state" in experimental["diagnostics"]
    assert "switching_state" not in STACK_ARMS


def test_cleanroom_trust_region_is_an_opt_in_diagnostic_arm() -> None:
    cache, models, surface = _cleanroom_fixture()
    baseline = _predict_one(cache, models, surface)
    config = PipelineConfig(
        extra={
            "enable_trust_region_ridge": True,
            "trust_region_ridge_weight": 0.05,
        }
    )
    config.validate()
    experimental = _predict_one(
        cache,
        models,
        surface,
        enable_trust_region_ridge=config.trust_region_ridge_enabled,
        trust_region_ridge_weight=config.trust_region_ridge_weight,
    )

    assert tuple(experimental["predictions"]) == (
        *tuple(baseline["predictions"]),
        "trust_region_ridge",
    )
    for name, values in baseline["predictions"].items():
        np.testing.assert_array_equal(experimental["predictions"][name], values)
    expected = baseline["predictions"]["sequential_final"] + 0.05 * (
        baseline["predictions"]["ridge"] - baseline["predictions"]["sequential_final"]
    )
    np.testing.assert_array_equal(experimental["predictions"]["trust_region_ridge"], expected)
    assert experimental["diagnostics"]["trust_region_ridge"] == {
        "ridge_weight": 0.05,
        "sequential_final_weight": 0.95,
    }
    assert "trust_region_ridge" not in STACK_ARMS
    policy = StackPolicy(
        arms=STACK_ARMS,
        weights=np.asarray([0.10, 0.15, 0.20, 0.05]),
        ridge=0.02,
    )
    np.testing.assert_array_equal(
        apply_stack(policy, experimental["predictions"]),
        apply_stack(policy, baseline["predictions"]),
    )


def test_switching_state_config_flag_is_strictly_boolean() -> None:
    with pytest.raises(ValueError, match="must be boolean"):
        PipelineConfig(extra={"enable_switching_state": "true"}).validate()


@pytest.mark.parametrize(
    "extra",
    [
        {"enable_trust_region_ridge": "true"},
        {"trust_region_ridge_weight": "0.05"},
        {"trust_region_ridge_weight": True},
        {"trust_region_ridge_weight": -0.01},
        {"trust_region_ridge_weight": 0.251},
        {"trust_region_ridge_weight": float("nan")},
    ],
)
def test_trust_region_config_rejects_invalid_values(extra: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="trust_region_ridge"):
        PipelineConfig(extra=extra).validate()


def test_fault_up_regime_responds_to_a_positive_step() -> None:
    rng = np.random.default_rng(4)
    rows = 140
    horizon = 5.0 + 5.0 * np.arange(rows, dtype=float)
    hgrg = np.zeros(rows)
    latent = np.zeros(rows)
    latent[70:] = 8.0
    pf = latent + rng.normal(0.0, 0.30, rows)
    hmm = latent + rng.normal(0.0, 0.40, rows)
    structural = latent + rng.normal(0.0, 0.50, rows)

    candidate, _, probability, _ = apply_switching_state(
        hgrg=hgrg,
        pf=pf,
        hmm=hmm,
        structural=structural,
        horizon=horizon,
        prefix_cv=1.0,
        policy=_policy(movement_budget_ft=20.0, row_cap_ft=20.0),
    )

    assert probability[15:60, 0].mean() > 0.90
    fault_window = probability[66:76]
    assert np.max(fault_window[:, 1]) > 0.20
    assert np.max(fault_window[:, 1]) > 5.0 * np.max(fault_window[:, 2])
    assert candidate[90:].mean() > candidate[20:50].mean() + 6.0


def test_uncertain_regime_absorbs_expert_disagreement() -> None:
    rows = 130
    horizon = 5.0 + 5.0 * np.arange(rows, dtype=float)
    hgrg = np.zeros(rows)
    pf = np.zeros(rows)
    hmm = np.zeros(rows)
    structural = np.zeros(rows)
    pf[45:90] = 20.0
    hmm[45:90] = -20.0
    structural[45:90] = 5.0 * np.sin(np.arange(45))

    _, _, probability, diagnostics = apply_switching_state(
        hgrg=hgrg,
        pf=pf,
        hmm=hmm,
        structural=structural,
        horizon=horizon,
        prefix_cv=1.0,
        policy=_policy(movement_budget_ft=20.0, row_cap_ft=20.0),
    )

    assert probability[10:35, 3].mean() < 0.05
    assert probability[52:84, 3].mean() > 0.90
    assert diagnostics["peak_uncertain_probability"] > 0.99


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"hmm": np.zeros(9)}, "not aligned"),
        ({"pf": np.r_[np.zeros(9), np.nan]}, "must be finite"),
        ({"horizon": np.r_[np.arange(9), 8.0]}, "strictly increasing"),
    ],
)
def test_switching_state_rejects_invalid_input(change: dict[str, np.ndarray], message: str) -> None:
    values = {
        "hgrg": np.zeros(10),
        "pf": np.zeros(10),
        "hmm": np.zeros(10),
        "structural": np.zeros(10),
        "horizon": np.arange(10, dtype=float),
    }
    values.update(change)
    with pytest.raises(ValueError, match=message):
        apply_switching_state(**values, prefix_cv=1.0)
