"""Adapters for retrained and precomputed historical parent predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .artifacts import sha256_file


REQUIRED_HISTORICAL_COLUMNS = ("id", "incumbent", "ridge", "pf", "hmm")


class ParentProvider(Protocol):
    def predict(self, row_ids: np.ndarray) -> dict[str, np.ndarray]: ...


@dataclass(frozen=True)
class HistoricalArtifactParent:
    """Load expert paths after checking file hash, schema and row IDs."""

    table: pd.DataFrame
    artifact_sha256: str

    @classmethod
    def from_csv(cls, path: Path, *, expected_sha256: str) -> "HistoricalArtifactParent":
        if not expected_sha256:
            raise ValueError("historical parent hash must be pinned")
        observed = sha256_file(path)
        if observed != expected_sha256:
            raise ValueError(f"historical parent SHA-256 mismatch: {observed}")
        table = pd.read_csv(path)
        if tuple(table.columns) != REQUIRED_HISTORICAL_COLUMNS:
            raise ValueError(f"historical expert schema must be {REQUIRED_HISTORICAL_COLUMNS}")
        if table["id"].astype(str).duplicated().any():
            raise ValueError("historical expert artifact contains duplicate IDs")
        numeric = table.loc[:, REQUIRED_HISTORICAL_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise ValueError("historical expert artifact contains non-finite values")
        return cls(table=table.copy(), artifact_sha256=observed)

    def predict(self, row_ids: np.ndarray) -> dict[str, np.ndarray]:
        requested = np.asarray(row_ids).astype(str)
        indexed = self.table.assign(id=self.table["id"].astype(str)).set_index("id")
        if len(np.unique(requested)) != len(requested):
            raise ValueError("requested historical IDs are not unique")
        if not indexed.index.is_unique or not set(requested).issubset(set(indexed.index)):
            raise ValueError("historical artifact does not cover the requested IDs exactly")
        aligned = indexed.loc[requested]
        return {
            name: aligned[name].to_numpy(np.float64)
            for name in REQUIRED_HISTORICAL_COLUMNS[1:]
        }
