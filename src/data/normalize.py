from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


VolUnit = Literal["shares", "lots"]
AmountUnit = Literal["yuan", "thousand_yuan"]
DuplicatePolicy = Literal["raise", "first", "last"]


@dataclass(frozen=True)
class DailyUnitConfig:
    """Unit config for normalizing external daily data into internal contract.

    Notes:
    - Tushare daily commonly uses: `vol` in lots(手), `amount` in thousand yuan(千元).
    - Internal contract prefers: `vol` in shares(股), `amount` in yuan(元).
    - Keep this mapping explicit and configurable to avoid hidden assumptions.
    """

    vol_unit: VolUnit = "lots"
    amount_unit: AmountUnit = "thousand_yuan"
    vol_lot_size: float = 100.0
    amount_thousand_size: float = 1000.0


def _convert_volume(vol: pd.Series, config: DailyUnitConfig) -> pd.Series:
    out = pd.to_numeric(vol, errors="coerce")
    if config.vol_unit == "lots":
        out = out * config.vol_lot_size
    return out


def _convert_amount(amount: pd.Series, config: DailyUnitConfig) -> pd.Series:
    out = pd.to_numeric(amount, errors="coerce")
    if config.amount_unit == "thousand_yuan":
        out = out * config.amount_thousand_size
    return out


def normalize_daily(
    df: pd.DataFrame,
    *,
    unit_config: DailyUnitConfig | None = None,
    duplicate_policy: DuplicatePolicy = "raise",
    derive_optional_fields: bool = False,
) -> pd.DataFrame:
    """Normalize one-stock daily data to internal schema for downstream pipeline.

    This function sits at the boundary between external source format (e.g. Tushare-like
    parquet columns/units) and internal `data_contract` schema.

    Behavior:
    - Keeps internal core columns when present: `trade_date, open, high, low, close, vol`.
    - Keeps optional columns when present: `amount, pct_chg`.
    - Parses `trade_date` to datetime, drops invalid date rows, sorts ascending.
    - Handles duplicate `trade_date` by configured policy.
    - Converts units by `DailyUnitConfig` (default: vol 手->股, amount 千元->元).
    - Optionally derives `pre_close`, `change`, `pct_chg_calc` from `close`.

    Notes:
    - `pre_close`/`change` are not required input fields.
    - Missing OHLCV columns are tolerated here; strict checks belong to validator.
    """

    if unit_config is None:
        unit_config = DailyUnitConfig()

    if "trade_date" not in df.columns:
        raise KeyError("daily data missing required column: trade_date")

    keep_cols = ["trade_date", "open", "high", "low", "close", "vol", "amount", "pct_chg"]
    exist_cols = [c for c in keep_cols if c in df.columns]

    out = df[exist_cols].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.dropna(subset=["trade_date"]).sort_values("trade_date")

    if duplicate_policy == "raise":
        dup_mask = out["trade_date"].duplicated(keep=False)
        if dup_mask.any():
            dup_dates = out.loc[dup_mask, "trade_date"].dt.strftime("%Y-%m-%d").unique().tolist()
            preview = ", ".join(dup_dates[:5])
            raise ValueError(f"duplicate trade_date detected: {preview}")
    elif duplicate_policy == "first":
        out = out.drop_duplicates(subset=["trade_date"], keep="first")
    elif duplicate_policy == "last":
        out = out.drop_duplicates(subset=["trade_date"], keep="last")
    else:
        raise ValueError(f"unknown duplicate_policy: {duplicate_policy}")

    numeric_cols = [c for c in ["open", "high", "low", "close", "vol", "amount", "pct_chg"] if c in out]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "vol" in out.columns:
        out["vol"] = _convert_volume(out["vol"], unit_config)

    if "amount" in out.columns:
        out["amount"] = _convert_amount(out["amount"], unit_config)

    if derive_optional_fields and "close" in out.columns:
        close = pd.to_numeric(out["close"], errors="coerce")
        out["pre_close"] = close.shift(1)
        out["change"] = close - out["pre_close"]
        with pd.option_context("mode.use_inf_as_na", True):
            out["pct_chg_calc"] = out["change"] / out["pre_close"] * 100.0

    return out.reset_index(drop=True)
