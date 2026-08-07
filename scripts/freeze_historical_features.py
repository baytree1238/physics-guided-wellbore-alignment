#!/usr/bin/env python3
"""Freeze the audited baseline SAFE builder into a standalone source module.

This is a maintainer/provenance command, not part of ``make reproduce``.  It
requires the historical notebook and its audit helper only when regenerating
the vendored module.  Normal imports and reproduction use the generated file
and have no path dependency on either source.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = ROOT
DEFAULT_OUTPUT = ROOT / "src" / "rogii_portfolio" / "historical_features.py"


def _header(raw_sha: str, safe_sha: str) -> str:
    return f'''"""Frozen historical 121-column FAST-SAFE feature builder.

GENERATED FILE.  The numerical body is the target-isolated transformation of
the single audited feature-builder cell in ``roggii_baseline.ipynb``.  It is
kept separate from :mod:`rogii_portfolio.features`, whose readable clean-room
builder has the same schema but deliberately simpler formulas.

The historical body reads only ``MD, X, Y, Z, GR, TVT_input`` from the
horizontal file and ``TVT, GR`` from the type well.  It never reads suffix
truth.  Numba's RNG is reset from a stable SHA-256 well seed before each pair
of particle filters; the historical particle/beam constants are unchanged.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit

from .features import SAFE_FEATURES as HISTORICAL_FEATURE_ORDER


HISTORICAL_ORIGINAL_BUILDER_SHA256 = "{raw_sha}"
HISTORICAL_SAFE_SOURCE_SHA256 = "{safe_sha}"
HISTORICAL_BASE_SEED = 42
_FEATURE_BASE_SEED = HISTORICAL_BASE_SEED
_SAFE_HORIZONTAL_OBSERVABLES = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
_SAFE_TYPEWELL_OBSERVABLES = ["TVT", "GR"]
# The frozen notebook body used set arithmetic for its allow-list.  Preserve
# that internal API while retaining one canonical sorted order for matrices.
SAFE_FEATURES = set(HISTORICAL_FEATURE_ORDER)


def _stable_well_seed(base_seed: int, well_id: str) -> int:
    if not isinstance(base_seed, int) or not 0 <= base_seed <= 0xFFFFFFFF:
        raise ValueError("base_seed must be a uint32")
    well = str(well_id)
    if not well or "\\x00" in well:
        raise ValueError("well_id must be non-empty and cannot contain NUL")
    payload = f"{{int(base_seed)}}:{{well}}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


@njit(cache=False)
def _seed_fast_safe_feature_rng(seed: int) -> None:
    np.random.seed(seed)


# Compile the explicit seed reset before any builder call or worker scheduling.
_seed_fast_safe_feature_rng(0)

'''


SUFFIX = r'''

def verify_historical_source_identity() -> bool:
    """Verify that the vendored numerical body still matches its audit hash."""
    source = Path(__file__).read_text(encoding="utf-8")
    begin = "# <" + "BEGIN FROZEN HISTORICAL SAFE SOURCE>\n"
    end = "\n# <" + "END FROZEN HISTORICAL SAFE SOURCE>"
    if source.count(begin) != 1 or source.count(end) != 1:
        raise RuntimeError("frozen historical source markers are missing or ambiguous")
    body = source.split(begin, 1)[1].split(end, 1)[0]
    observed = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if observed != HISTORICAL_SAFE_SOURCE_SHA256:
        raise RuntimeError(
            "vendored historical source changed: "
            f"{observed} != {HISTORICAL_SAFE_SOURCE_SHA256}"
        )
    return True


def _preflight_paths(horizontal_path: Path, typewell_path: Path) -> None:
    horizontal_header = set(pd.read_csv(horizontal_path, nrows=0).columns)
    missing = set(_SAFE_HORIZONTAL_OBSERVABLES) - horizontal_header
    if missing:
        raise ValueError(f"horizontal file is missing observables: {sorted(missing)}")
    typewell_header = set(pd.read_csv(typewell_path, nrows=0).columns)
    missing = set(_SAFE_TYPEWELL_OBSERVABLES) - typewell_header
    if missing:
        raise ValueError(f"type-well file is missing observables: {sorted(missing)}")

    # Read only the allow-listed observables.  A horizontal TVT column may be
    # present in a training CSV, but neither preflight nor the builder loads it.
    horizontal = pd.read_csv(horizontal_path, usecols=_SAFE_HORIZONTAL_OBSERVABLES)
    visible = pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy()
    suffix = np.flatnonzero(~visible)
    if len(suffix) == 0 or int(suffix[0]) < 10:
        raise ValueError("historical builder requires >=10 visible rows and a hidden suffix")
    if not visible[: int(suffix[0])].all() or visible[int(suffix[0]) :].any():
        raise ValueError("TVT_input must be a contiguous visible prefix and hidden suffix")


def build_historical_fast_safe_frame(
    horizontal_path: str | Path,
    typewell_path: str | Path,
) -> pd.DataFrame:
    """Return ``well, id`` and the exact historical 121 SAFE columns.

    The base seed is intentionally frozen to 42.  Changing it would define a
    new stochastic feature artifact rather than reproduce the historical one.
    """
    horizontal_path = Path(horizontal_path)
    typewell_path = Path(typewell_path)
    _preflight_paths(horizontal_path, typewell_path)
    frame = build_well(str(horizontal_path), str(typewell_path), False)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RuntimeError(f"historical feature builder returned no rows for {horizontal_path}")
    expected = {"well", "id", *HISTORICAL_FEATURE_ORDER}
    if set(frame.columns) != expected:
        raise RuntimeError(
            "historical feature schema changed: "
            f"missing={sorted(expected-set(frame.columns))}, "
            f"extra={sorted(set(frame.columns)-expected)}"
        )
    ordered = frame.loc[:, ["well", "id", *HISTORICAL_FEATURE_ORDER]].reset_index(drop=True)
    values = ordered.loc[:, HISTORICAL_FEATURE_ORDER].to_numpy(dtype=np.float32)
    if values.shape[1] != 121 or not np.isfinite(values).all():
        raise RuntimeError("historical feature matrix failed its 121-column finite contract")
    return ordered


def build_historical_fast_safe_features(
    horizontal_path: str | Path,
    typewell_path: str | Path,
) -> pd.DataFrame:
    """Return only the sorted 121-column historical model matrix."""
    return build_historical_fast_safe_frame(horizontal_path, typewell_path).loc[
        :, HISTORICAL_FEATURE_ORDER
    ]


def canonical_historical_feature_hash(frame: pd.DataFrame) -> str:
    """Hash IDs and little-endian float32 feature bytes in canonical order."""
    required = {"id", *HISTORICAL_FEATURE_ORDER}
    if not required.issubset(frame.columns):
        raise ValueError(f"frame is missing columns: {sorted(required-set(frame.columns))}")
    digest = hashlib.sha256()
    for value in frame["id"].astype(str):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    values = np.ascontiguousarray(
        frame.loc[:, HISTORICAL_FEATURE_ORDER].to_numpy(dtype="<f4")
    )
    digest.update(values.tobytes())
    return digest.hexdigest()
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    repository = args.repository.resolve()
    sys.path.insert(0, str(repository))
    from rogii_deterministic_safe_rebuild import locate_builder_cell, make_safe_only_source

    raw = locate_builder_cell(repository / "roggii_baseline.ipynb")
    safe = make_safe_only_source(raw)
    raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    safe_sha = hashlib.sha256(safe.encode("utf-8")).hexdigest()
    output = (
        _header(raw_sha, safe_sha)
        + "# <BEGIN FROZEN HISTORICAL SAFE SOURCE>\n"
        + safe
        + "\n# <END FROZEN HISTORICAL SAFE SOURCE>\n"
        + SUFFIX.lstrip()
    )
    compile(output, str(args.output), "exec")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8", newline="\n")
    print(f"wrote {args.output}")
    print(f"historical_builder_sha256={raw_sha}")
    print(f"safe_source_sha256={safe_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
