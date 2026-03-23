from pathlib import Path
from typing import Union

import pandas as pd


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
    """Load single-stock daily K-line parquet from raw storage."""

    def __init__(self, data_dir: PathLike):
        self.data_dir = Path(data_dir)

    def load_one(self, ts_code: str) -> pd.DataFrame:
        file_path = self.data_dir / f"{ts_code}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Daily data file not found: {file_path}")

        df = pd.read_parquet(file_path)
        if "trade_date" not in df.columns:
            raise KeyError(f"Missing required column 'trade_date' in {file_path}")

        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = df.dropna(subset=["trade_date"]).sort_values("trade_date")
        return df.reset_index(drop=True)
