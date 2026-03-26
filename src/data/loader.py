from pathlib import Path
from typing import Union

import pandas as pd

from src.data.normalize import DailyUnitConfig, normalize_daily


PathLike = Union[str, Path]


class LabelLoader:
    """Load label table and normalize date columns for downstream dataset build."""

    def __init__(self, path: PathLike):
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)

        if "asof_date" in df.columns:
            df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce")
            df = df.sort_values("asof_date")
        elif "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
            df = df.sort_values("trade_date")

        return df.reset_index(drop=True)


class DailyDataLoader:
    """Load single-stock daily parquet and apply schema normalization."""

    def __init__(
        self,
        data_dir: PathLike,
        *,
        unit_config: DailyUnitConfig | None = None,
        duplicate_policy: str = "raise",
    ):
        self.data_dir = Path(data_dir)
        self.unit_config = unit_config
        self.duplicate_policy = duplicate_policy

    def load_one(self, ts_code: str) -> pd.DataFrame:
        file_path = self.data_dir / f"{ts_code}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Daily data file not found: {file_path}")

        df = pd.read_parquet(file_path)
        return normalize_daily(
            df,
            unit_config=self.unit_config,
            duplicate_policy=self.duplicate_policy,
        )
