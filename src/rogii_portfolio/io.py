"""Load competition wells from ZIP archives or local directories."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from .contracts import WellRecord


class CompetitionStore:
    """Read ROGII wells from an official ZIP, split directory, or flat cache."""

    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        self.split = split
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        if split not in {"train", "test"}:
            raise ValueError("split must be train or test")
        self.archive: zipfile.ZipFile | None = None

    def __enter__(self) -> "CompetitionStore":
        if self.root.is_file():
            self.archive = zipfile.ZipFile(self.root)
        return self

    def __exit__(self, *_: object) -> None:
        if self.archive is not None:
            self.archive.close()
            self.archive = None

    def wells(self) -> list[str]:
        suffix = "__horizontal_well.csv"
        if self.root.is_file():
            archive = self.archive or zipfile.ZipFile(self.root)
            names = [name for name in archive.namelist() if name.startswith(f"{self.split}/") and name.endswith(suffix)]
            if self.archive is None:
                archive.close()
            return sorted(Path(name).name.removesuffix(suffix) for name in names)
        split_root = self.root / self.split
        search_root = split_root if split_root.exists() else self.root
        return sorted(path.name.removesuffix(suffix) for path in search_root.glob(f"*{suffix}"))

    def _read(self, well: str, kind: str) -> pd.DataFrame:
        filename = f"{well}__{kind}.csv"
        if self.root.is_file():
            if self.archive is None:
                raise RuntimeError("ZIP store must be used as a context manager")
            member = f"{self.split}/{filename}"
            with self.archive.open(member) as stream:
                return pd.read_csv(io.BytesIO(stream.read()))
        split_path = self.root / self.split / filename
        path = split_path if split_path.exists() else self.root / filename
        return pd.read_csv(path)

    def load(self, well: str) -> WellRecord:
        return WellRecord(well, self._read(well, "horizontal_well"), self._read(well, "typewell"))

    def load_all(self, *, limit: int | None = None) -> list[WellRecord]:
        names = self.wells()
        if limit is not None:
            names = names[: int(limit)]
        records = [self.load(well) for well in names]
        if not records:
            raise RuntimeError(f"no {self.split} wells found under {self.root}")
        return records
