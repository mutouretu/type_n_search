from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import DailyDataLoader


def test_load_parquet_success(tmp_path):
    data_dir = tmp_path / "daily"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=3, freq="D"),
            "open": [10.0, 10.2, 10.3],
            "close": [10.1, 10.3, 10.4],
        }
    )
    df.to_parquet(data_dir / "000001.SZ.parquet", index=False)

    loader = DailyDataLoader(data_dir)
    out = loader.load_one("000001.SZ")

    assert not out.empty
    assert "trade_date" in out.columns


def test_load_parquet_file_not_found(tmp_path):
    loader = DailyDataLoader(tmp_path)

    with pytest.raises(FileNotFoundError):
        loader.load_one("NOT_EXIST.SZ")
