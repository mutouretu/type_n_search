from __future__ import annotations

import pandas as pd
import pytest

from src.data.normalize import DailyUnitConfig, normalize_daily


def test_normalize_daily_date_sort_and_units_default():
    df = pd.DataFrame(
        {
            "trade_date": ["2025-01-03", "2025-01-01", "2025-01-02"],
            "open": [10, 9, 9.5],
            "high": [10.5, 9.5, 10],
            "low": [9.8, 8.9, 9.2],
            "close": [10.2, 9.2, 9.8],
            "vol": [1, 2, 3],
            "amount": [4, 5, 6],
            "pct_chg": [1.0, 2.0, 3.0],
            "extra_col": [7, 8, 9],
        }
    )

    out = normalize_daily(df)

    assert out["trade_date"].tolist() == list(pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]))
    assert out["vol"].tolist() == [200.0, 300.0, 100.0]  # lots -> shares
    assert out["amount"].tolist() == [5000.0, 6000.0, 4000.0]  # thousand yuan -> yuan
    assert "extra_col" not in out.columns


def test_normalize_daily_custom_units_no_conversion():
    df = pd.DataFrame(
        {
            "trade_date": ["2025-01-01"],
            "open": [10],
            "high": [10.5],
            "low": [9.5],
            "close": [10.1],
            "vol": [1234],
            "amount": [5678],
        }
    )

    out = normalize_daily(
        df,
        unit_config=DailyUnitConfig(vol_unit="shares", amount_unit="yuan"),
    )

    assert out["vol"].iloc[0] == 1234
    assert out["amount"].iloc[0] == 5678


def test_normalize_daily_duplicate_trade_date_raise():
    df = pd.DataFrame(
        {
            "trade_date": ["2025-01-01", "2025-01-01"],
            "open": [10, 11],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.1, 11.1],
            "vol": [100, 200],
        }
    )

    with pytest.raises(ValueError, match="duplicate trade_date"):
        normalize_daily(df, duplicate_policy="raise")


def test_normalize_daily_duplicate_trade_date_keep_last():
    df = pd.DataFrame(
        {
            "trade_date": ["2025-01-01", "2025-01-01"],
            "open": [10, 11],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.1, 11.1],
            "vol": [1, 2],
        }
    )

    out = normalize_daily(df, duplicate_policy="last")
    assert len(out) == 1
    assert out["open"].iloc[0] == 11
