from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.window_builder import build_window_by_asof_date


def _make_daily_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 10 + np.cumsum(rng.normal(0, 0.05, size=n))
    open_ = close * (1 + rng.normal(0, 0.003, size=n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.01, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.01, size=n))
    vol = rng.integers(100000, 150000, size=n).astype(float)

    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "vol": vol,
        }
    )


def test_build_window_success():
    df = _make_daily_df(200)
    asof_date = df["trade_date"].iloc[150]

    window = build_window_by_asof_date(df, asof_date=asof_date, window_size=60, min_history=80)

    assert window is not None
    assert len(window) == 60
    assert window["trade_date"].max() <= asof_date


def test_build_window_insufficient_history():
    df = _make_daily_df(40)
    asof_date = df["trade_date"].iloc[-1]

    window = build_window_by_asof_date(df, asof_date=asof_date, window_size=60, min_history=80)

    assert window is None
