"""Ridge and gradient-boosted residual models on FAST-SAFE features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import PipelineConfig
from .features import SAFE_FEATURES


@dataclass
class FastSafeModels:
    """Fold-fitted residual models that share the frozen feature schema.

    Both estimators predict a correction to the particle-filter path. Keeping
    the common parent explicit makes their outputs directly comparable.
    """

    ridge: object
    nonlinear: HistGradientBoostingRegressor
    feature_names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        truth: np.ndarray,
        pf: np.ndarray,
        config: PipelineConfig,
    ) -> "FastSafeModels":
        """Fit Ridge and nonlinear residual models on one training fold."""

        if tuple(features.columns) != SAFE_FEATURES:
            raise ValueError("model received a feature schema other than the frozen 121 columns")
        x = features.to_numpy(np.float64)
        target = np.asarray(truth, float) - np.asarray(pf, float)
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        ridge.fit(x, target)
        nonlinear = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=config.model_learning_rate,
            max_iter=config.model_max_iter,
            max_leaf_nodes=config.model_max_leaf_nodes,
            l2_regularization=config.model_l2,
            min_samples_leaf=max(8, min(30, len(x) // 20)),
            random_state=config.refit_seed,
        )
        nonlinear.fit(x, target)
        return cls(ridge=ridge, nonlinear=nonlinear, feature_names=SAFE_FEATURES)

    def predict(self, features: pd.DataFrame, pf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return Ridge and nonlinear TVT paths in the frozen feature order."""

        if tuple(features.columns) != self.feature_names:
            raise ValueError("inference feature order differs from training")
        x = features.to_numpy(np.float64)
        pf = np.asarray(pf, float)
        return pf + np.asarray(self.ridge.predict(x), float), pf + np.asarray(self.nonlinear.predict(x), float)


def incumbent_parent(
    *,
    ridge: np.ndarray,
    pf: np.ndarray,
    nonlinear: np.ndarray,
    nonlinear_max_weight: float = 0.00425,
    disagreement_disable_ft: float = 25.0,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Combine Ridge and PF, with a small nonlinear correction at low disagreement."""
    base = 0.30 * np.asarray(ridge, float) + 0.70 * np.asarray(pf, float)
    delta = np.asarray(nonlinear, float) - base
    p95 = float(np.quantile(np.abs(delta), 0.95))
    enabled = p95 <= disagreement_disable_ft
    if enabled:
        gate = nonlinear_max_weight / (1.0 + np.square(np.abs(delta) / 6.0))
        candidate = base + gate * delta
        effective_weight = float(np.mean(gate))
    else:
        candidate = base
        effective_weight = 0.0
    return candidate, {
        "ridge_weight": 0.30,
        "pf_weight": 0.70,
        "nonlinear_enabled": enabled,
        "nonlinear_disagreement_p95_ft": p95,
        "nonlinear_mean_effective_weight": effective_weight,
    }
