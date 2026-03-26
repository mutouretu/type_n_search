from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import DailyDataLoader
from src.data.validator import (
    validate_daily_df,
    validate_daily_quality,
    validate_label_daily_alignment,
    validate_labels_df,
)


def _make_daily(start: str = "2025-01-01", n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.date_range(start, periods=n, freq="D"),
            "open": [10.0 + i for i in range(n)],
            "high": [10.5 + i for i in range(n)],
            "low": [9.5 + i for i in range(n)],
            "close": [10.2 + i for i in range(n)],
            "vol": [1000.0 + i for i in range(n)],
            "amount": [2000.0 + i for i in range(n)],
        }
    )


def test_validate_daily_df_success():
    out = validate_daily_df(_make_daily())
    assert pd.api.types.is_datetime64_any_dtype(out["trade_date"])


def test_validate_daily_quality_invalid_ohlc():
    df = _make_daily()
    df.loc[0, "high"] = 9.0

    with pytest.raises(ValueError, match="OHLC inconsistency"):
        validate_daily_quality(df)


def test_validate_labels_df_requirements():
    labels = pd.DataFrame(
        {
            "sample_id": ["000001.SZ_2025-01-03"],
            "ts_code": ["000001.SZ"],
            "asof_date": ["2025-01-03"],
            "label": [1],
        }
    )
    out = validate_labels_df(labels)
    assert pd.api.types.is_datetime64_any_dtype(out["asof_date"])


def test_validate_label_daily_alignment(tmp_path):
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    _make_daily(n=5).to_parquet(daily_dir / "000001.SZ.parquet", index=False)

    labels = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ"],
            "asof_date": ["2025-01-03", "2025-01-03", "2025-01-10"],
            "label": [1, 0, 1],
        }
    )

    report = validate_label_daily_alignment(
        labels,
        daily_data_dir=daily_dir,
        min_history=4,
        daily_loader=DailyDataLoader(daily_dir),
        raise_on_error=False,
    )

    assert report["ok"] is False
    assert len(report["issues"]["missing_daily_file"]) == 1
    assert len(report["issues"]["asof_not_covered"]) == 1
    assert len(report["issues"]["insufficient_history"]) == 1
