"""Data contracts for well records, model inputs and evaluation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


HORIZONTAL_REQUIRED = ("MD", "X", "Y", "Z", "GR", "TVT_input")
TYPEWELL_REQUIRED = ("TVT", "GR")


@dataclass(frozen=True)
class WellRecord:
    """One horizontal well and its reference type-well log.

    ``TVT`` is optional at inference and required only by scoring adapters.
    Model-facing functions receive a copy containing ``TVT_input`` and cannot
    access suffix ``TVT``.
    """

    well_id: str
    horizontal: pd.DataFrame
    typewell: pd.DataFrame

    def validate(self, *, require_truth: bool = False) -> None:
        missing_h = set(HORIZONTAL_REQUIRED) - set(self.horizontal.columns)
        missing_t = set(TYPEWELL_REQUIRED) - set(self.typewell.columns)
        if missing_h or missing_t:
            raise ValueError(
                f"{self.well_id}: missing horizontal={sorted(missing_h)}, "
                f"typewell={sorted(missing_t)}"
            )
        if require_truth and "TVT" not in self.horizontal:
            raise ValueError(f"{self.well_id}: training/scoring requires TVT")
        md = pd.to_numeric(self.horizontal["MD"], errors="coerce").to_numpy(float)
        if len(md) < 40 or not np.isfinite(md).all() or np.any(np.diff(md) <= 0):
            raise ValueError(f"{self.well_id}: MD must be finite and strictly increasing")
        tw_tvt = pd.to_numeric(self.typewell["TVT"], errors="coerce").to_numpy(float)
        finite = np.isfinite(tw_tvt)
        if finite.sum() < 4 or np.any(np.diff(tw_tvt[finite]) <= 0):
            raise ValueError(f"{self.well_id}: invalid type-well TVT grid")

    def model_frame(self) -> pd.DataFrame:
        """Return a copy containing only the permitted inference columns."""
        self.validate()
        allowed = list(HORIZONTAL_REQUIRED)
        return self.horizontal.loc[:, allowed].copy().reset_index(drop=True)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration shared by data splitting, model fitting, and evaluation.

    The top-level fields define the stable clean-room pipeline. Experimental
    switches live in ``extra`` so the default research path remains explicit.
    """

    seed: int = 20260806
    prefix_fraction: float = 0.62
    outer_folds: int = 3
    inner_folds: int = 2
    holdout_fraction: float = 0.20
    pf_seeds: int = 4
    pf_particles: int = 96
    hmm_stride: int = 6
    hmm_checkpoint: int = 48
    model_max_iter: int = 80
    model_max_leaf_nodes: int = 31
    model_learning_rate: float = 0.05
    model_l2: float = 1.0
    spatial_radius_ft: float = 1_500.0
    gr_similarity_threshold: float = 0.985
    similarity_radius_ft: float = 12_000.0
    stack_ridge: float = 0.02
    bootstrap_draws: int = 500
    jobs: int = 1
    mode: str = "smoke"
    extra: dict[str, Any] = field(default_factory=dict)

    def _extra_seed(self, name: str, default: int) -> int:
        value = self.extra.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"extra.{name} must be an integer seed")
        seed = int(value)
        if not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError(f"extra.{name} must be a uint32")
        return seed

    @property
    def split_seed(self) -> int:
        """Seed for panel selection and component assignments.

        It defaults to the historical ``seed`` field so existing configs remain
        byte-for-byte reproducible.  Fresh refit studies can freeze this value
        while changing ``extra.refit_seed`` independently.
        """

        return self._extra_seed("split_seed", self.seed)

    @property
    def refit_seed(self) -> int:
        """Seed for stochastic model fitting, independent of the split."""

        return self._extra_seed("refit_seed", self.seed)

    @property
    def pf_seed_offset(self) -> int:
        """First Monte-Carlo seed used by PF members.

        The default of zero preserves the historical PF ensemble.  A registered
        refit experiment sets a different offset for each independent fit.
        """

        return self._extra_seed("pf_seed_offset", 0)

    @property
    def skip_nested_stack(self) -> bool:
        """Skip weight fitting when validating an already-frozen prediction arm."""

        value = self.extra.get("skip_nested_stack", False)
        if not isinstance(value, bool):
            raise ValueError("extra.skip_nested_stack must be boolean")
        return value

    @property
    def switching_state_enabled(self) -> bool:
        """Whether to emit the experimental switching-state candidate arm."""

        value = self.extra.get("enable_switching_state", False)
        if not isinstance(value, bool):
            raise ValueError("extra.enable_switching_state must be boolean")
        return value

    @property
    def trust_region_ridge_enabled(self) -> bool:
        """Whether to emit the fixed-weight Ridge trust-region arm."""

        value = self.extra.get("enable_trust_region_ridge", False)
        if not isinstance(value, bool):
            raise ValueError("extra.enable_trust_region_ridge must be boolean")
        return value

    @property
    def trust_region_ridge_weight(self) -> float:
        """Return the predeclared Ridge move weight for the diagnostic arm."""

        value = self.extra.get("trust_region_ridge_weight", 0.05)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("extra.trust_region_ridge_weight must be numeric")
        weight = float(value)
        if not np.isfinite(weight) or not 0.0 <= weight <= 0.25:
            raise ValueError("extra.trust_region_ridge_weight must lie in [0, 0.25]")
        return weight

    def validate(self) -> None:
        if not 0.25 <= self.prefix_fraction <= 0.90:
            raise ValueError("prefix_fraction must lie in [0.25, 0.90]")
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("nested validation needs at least two folds per level")
        if not 0.05 <= self.holdout_fraction <= 0.40:
            raise ValueError("holdout_fraction must lie in [0.05, 0.40]")
        if self.pf_seeds < 2 or self.pf_particles < 16:
            raise ValueError("PF ensemble is too small")
        if self.hmm_stride < 1 or self.model_max_iter < 1:
            raise ValueError("invalid HMM/model budget")
        _ = self.switching_state_enabled
        _ = self.trust_region_ridge_enabled
        _ = self.trust_region_ridge_weight
        _ = self.split_seed
        _ = self.refit_seed
        _ = self.pf_seed_offset
        _ = self.skip_nested_stack


@dataclass(frozen=True)
class PreparedWell:
    """Validated inference inputs and suffix indices; contains no target."""

    well_id: str
    inference: pd.DataFrame
    typewell: pd.DataFrame
    suffix_mask: np.ndarray
    suffix_rows: np.ndarray
    horizon: np.ndarray
    row_ids: np.ndarray
    component: str = ""

    def validate(self) -> None:
        n = len(self.inference)
        if len(self.suffix_mask) != n:
            raise ValueError(f"{self.well_id}: prepared arrays are not aligned")
        if not np.array_equal(np.flatnonzero(self.suffix_mask), self.suffix_rows):
            raise ValueError(f"{self.well_id}: suffix index mismatch")
        if len(self.horizon) != len(self.suffix_rows) or len(self.row_ids) != len(self.suffix_rows):
            raise ValueError(f"{self.well_id}: suffix arrays are not aligned")
        if "TVT" in self.inference.columns:
            raise ValueError(f"{self.well_id}: inference frame exposes TVT")
        if self.inference.loc[self.suffix_rows, "TVT_input"].notna().any():
            raise ValueError(f"{self.well_id}: inference suffix exposes TVT_input")


@dataclass(frozen=True)
class ScoredPreparedWell:
    """Inference inputs paired with suffix targets for evaluation."""

    model_input: PreparedWell
    truth_suffix: np.ndarray

    def validate(self) -> None:
        self.model_input.validate()
        truth = np.asarray(self.truth_suffix, float)
        if len(truth) != len(self.model_input.suffix_rows) or not np.isfinite(truth).all():
            raise ValueError(f"{self.model_input.well_id}: invalid scoring suffix")


@dataclass
class PredictionBundle:
    """Aligned model outputs and metadata for one evaluation population.

    Every array has one entry per scored suffix row. ``validate`` protects the
    row identity and finite-value assumptions used by downstream metrics.
    """

    well: np.ndarray
    row_id: np.ndarray
    row_number: np.ndarray
    component: np.ndarray
    horizon: np.ndarray
    truth: np.ndarray
    features: pd.DataFrame
    predictions: dict[str, np.ndarray]
    diagnostics: list[dict[str, Any]]

    def validate(self) -> None:
        n = len(self.truth)
        arrays = (self.well, self.row_id, self.row_number, self.component, self.horizon)
        if not n or any(len(x) != n for x in arrays) or len(self.features) != n:
            raise ValueError("prediction bundle is not row aligned")
        if len(np.unique(self.row_id.astype(str))) != n:
            raise ValueError("prediction bundle has duplicate row IDs")
        if not np.isfinite(self.truth).all():
            raise ValueError("prediction bundle truth is non-finite")
        for name, value in self.predictions.items():
            if len(value) != n or not np.isfinite(value).all():
                raise ValueError(f"invalid prediction arm: {name}")
