from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


EPS = 1e-12


def _safe_div(numerator: float, denominator: float) -> float:
    """Safe division that returns NaN when denominator is invalid."""
    if denominator is None or np.isnan(denominator) or abs(denominator) <= EPS:
        return float("nan")
    return float(numerator / denominator)


def _pick_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    """Pick first existing column name from candidates, case-insensitive."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        key = name.lower()
        if key in lower_map:
            return lower_map[key]
    if required:
        raise KeyError(f"Missing required columns, candidates={list(candidates)}")
    return None


def build_tabular_features(window_df: pd.DataFrame) -> Dict[str, float]:
    """Extract tabular features from one daily-bar window."""
    if window_df is None or window_df.empty:
        raise ValueError("window_df is empty")

    high_col = _pick_col(window_df, ["high"])
    low_col = _pick_col(window_df, ["low"])
    close_col = _pick_col(window_df, ["close"])
    vol_col = _pick_col(window_df, ["vol", "volume"])

    df = window_df.copy().reset_index(drop=True)
    for col in [high_col, low_col, close_col, vol_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    vol = df[vol_col]

    low_min = float(low.min())
    high_max = float(high.max())
    if np.isnan(low_min) or low_min <= EPS:
        base_range_pct = float("nan")
    else:
        base_ratio = _safe_div(high_max, low_min)
        base_range_pct = base_ratio - 1.0 if not np.isnan(base_ratio) else float("nan")

    vol_mean = float(vol.mean())
    vol_std = float(vol.std(ddof=0))
    base_vol_cv = _safe_div(vol_std, vol_mean)

    recent20_vol = float(vol.tail(20).mean())
    prev60_slice = vol.iloc[-80:-20] if len(vol) >= 80 else vol.iloc[:-20]
    prev60_vol = float(prev60_slice.mean()) if not prev60_slice.empty else float("nan")
    vol_shrink_ratio = _safe_div(recent20_vol, prev60_vol)

    close_last = float(close.iloc[-1])
    close_6 = float(close.iloc[-6]) if len(close) >= 6 else float("nan")
    close_11 = float(close.iloc[-11]) if len(close) >= 11 else float("nan")
    rr5_ratio = _safe_div(close_last, close_6)
    rr10_ratio = _safe_div(close_last, close_11)
    recent_return_5 = rr5_ratio - 1.0 if not np.isnan(rr5_ratio) else float("nan")
    recent_return_10 = rr10_ratio - 1.0 if not np.isnan(rr10_ratio) else float("nan")

    high_prev20 = float(high.iloc[-21:-1].max()) if len(high) >= 21 else float(high.tail(20).max())
    high_prev60 = float(high.iloc[-61:-1].max()) if len(high) >= 61 else float(high.tail(60).max())
    b20_ratio = _safe_div(close_last, high_prev20)
    b60_ratio = _safe_div(close_last, high_prev60)
    breakout_distance_20 = b20_ratio - 1.0 if not np.isnan(b20_ratio) else float("nan")
    breakout_distance_60 = b60_ratio - 1.0 if not np.isnan(b60_ratio) else float("nan")

    vol_last = float(vol.iloc[-1])
    past20_vol = vol.iloc[-21:-1] if len(vol) >= 21 else vol.iloc[:-1]
    past20_mean = float(past20_vol.mean()) if not past20_vol.empty else float("nan")
    volume_spike_ratio = _safe_div(vol_last, past20_mean)

    ma10 = close.rolling(window=10, min_periods=10).mean()
    ma20 = close.rolling(window=20, min_periods=20).mean()
    above_ma10_ratio = float((close[ma10.notna()] > ma10[ma10.notna()]).mean()) if ma10.notna().any() else float("nan")
    above_ma20_ratio = float((close[ma20.notna()] > ma20[ma20.notna()]).mean()) if ma20.notna().any() else float("nan")

    ret = close.pct_change()
    volatility_20 = float(ret.tail(20).std(ddof=0))

    return {
        "base_range_pct": float(base_range_pct),
        "base_vol_cv": float(base_vol_cv),
        "vol_shrink_ratio": float(vol_shrink_ratio),
        "recent_return_5": float(recent_return_5),
        "recent_return_10": float(recent_return_10),
        "breakout_distance_20": float(breakout_distance_20),
        "breakout_distance_60": float(breakout_distance_60),
        "volume_spike_ratio": float(volume_spike_ratio),
        "above_ma10_ratio": float(above_ma10_ratio),
        "above_ma20_ratio": float(above_ma20_ratio),
        "volatility_20": float(volatility_20),
    }
