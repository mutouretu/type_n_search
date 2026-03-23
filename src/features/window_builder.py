from __future__ import annotations

from typing import Optional, Union

import pandas as pd


DateLike = Union[str, pd.Timestamp]


def build_window_by_asof_date(
    df: pd.DataFrame,
    asof_date: DateLike,
    window_size: int,
    min_history: int,
) -> Optional[pd.DataFrame]:
    """Build a fixed-length historical window ending at ``asof_date`` without future data."""
    if df is None or df.empty:
        return None
    if "trade_date" not in df.columns:
        raise KeyError("Missing required column: trade_date")
    if window_size <= 0 or min_history <= 0:
        raise ValueError("window_size and min_history must be positive")

    working = df.copy()
    working["trade_date"] = pd.to_datetime(working["trade_date"], errors="coerce")
    working = working.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    if working.empty:
        return None

    asof_ts = pd.to_datetime(asof_date, errors="coerce")
    if pd.isna(asof_ts):
        raise ValueError(f"Invalid asof_date: {asof_date}")

    history = working[working["trade_date"] <= asof_ts]
    if len(history) < min_history:
        return None

    window = history.tail(window_size).copy()
    if len(window) < window_size:
        return None

    return window.reset_index(drop=True)
