from __future__ import annotations

import pandas as pd


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight indicators used by both dataset build and scan inference."""
    if "trade_date" not in df.columns:
        raise KeyError("Missing required column: trade_date")

    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)

    if "close" in out.columns:
        close = pd.to_numeric(out["close"], errors="coerce")
        out["ret_1d"] = close.pct_change()
        out["ma_5"] = close.rolling(5, min_periods=1).mean()
        out["ma_20"] = close.rolling(20, min_periods=1).mean()

    return out
