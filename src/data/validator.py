from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader import DailyDataLoader


def _raise_or_return(errors: list[str], raise_on_error: bool = True) -> list[str]:
    if errors and raise_on_error:
        raise ValueError("; ".join(errors))
    return errors


def validate_labels_df(
    labels_df: pd.DataFrame,
    *,
    require_sample_id: bool = True,
    allowed_labels: tuple[int, ...] = (0, 1),
    raise_on_error: bool = True,
) -> pd.DataFrame:
    """Validate label schema and values for pipeline/notebook reuse.

    Required columns: `ts_code`, `asof_date`, `label`, and optionally `sample_id`.
    Returns a normalized copy with `asof_date` as datetime if validation passes.
    """

    required = ["ts_code", "asof_date", "label"]
    if require_sample_id:
        required = ["sample_id", *required]

    errors: list[str] = []
    missing = [c for c in required if c not in labels_df.columns]
    if missing:
        errors.append(f"labels missing required columns: {missing}")
        _raise_or_return(errors, raise_on_error)
        return labels_df

    out = labels_df.copy()
    out["asof_date"] = pd.to_datetime(out["asof_date"], errors="coerce")

    if out["asof_date"].isna().any():
        cnt = int(out["asof_date"].isna().sum())
        errors.append(f"labels has invalid asof_date rows: {cnt}")

    if require_sample_id and out["sample_id"].duplicated().any():
        cnt = int(out["sample_id"].duplicated().sum())
        errors.append(f"labels has duplicate sample_id rows: {cnt}")

    if out["label"].isna().any():
        cnt = int(out["label"].isna().sum())
        errors.append(f"labels has null label rows: {cnt}")

    invalid_label = ~out["label"].isin(list(allowed_labels)) & out["label"].notna()
    if invalid_label.any():
        values = sorted(out.loc[invalid_label, "label"].astype(str).unique().tolist())
        errors.append(f"labels has invalid label values: {values}, allowed={list(allowed_labels)}")

    _raise_or_return(errors, raise_on_error)
    return out


def validate_daily_df(daily_df: pd.DataFrame, *, raise_on_error: bool = True) -> pd.DataFrame:
    """Validate required daily schema and normalize `trade_date` dtype."""

    required = ["trade_date", "open", "high", "low", "close", "vol"]
    errors: list[str] = []

    missing = [c for c in required if c not in daily_df.columns]
    if missing:
        errors.append(f"daily missing required columns: {missing}")
        _raise_or_return(errors, raise_on_error)
        return daily_df

    out = daily_df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    if out["trade_date"].isna().any():
        cnt = int(out["trade_date"].isna().sum())
        errors.append(f"daily has invalid trade_date rows: {cnt}")

    _raise_or_return(errors, raise_on_error)
    return out


def validate_daily_quality(daily_df: pd.DataFrame, *, raise_on_error: bool = True) -> pd.DataFrame:
    """Validate daily data quality constraints for OHLCV and timeline integrity."""

    out = validate_daily_df(daily_df, raise_on_error=True).copy()
    errors: list[str] = []

    if not out["trade_date"].is_monotonic_increasing:
        errors.append("daily trade_date is not ascending")

    if out["trade_date"].duplicated().any():
        cnt = int(out["trade_date"].duplicated().sum())
        errors.append(f"daily has duplicate trade_date rows: {cnt}")

    for col in ["open", "high", "low", "close", "vol", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["open", "high", "low", "close"]:
        bad = out[col].isna() | (out[col] <= 0)
        if bad.any():
            errors.append(f"daily has invalid {col} rows: {int(bad.sum())}")

    if ((out["high"] < out["low"]) | (out["high"] < out["open"]) | (out["high"] < out["close"])).any():
        errors.append("daily has OHLC inconsistency: high below low/open/close")

    if ((out["low"] > out["open"]) | (out["low"] > out["close"])).any():
        errors.append("daily has OHLC inconsistency: low above open/close")

    vol_bad = out["vol"].isna() | (out["vol"] < 0)
    if vol_bad.any():
        errors.append(f"daily has invalid vol rows: {int(vol_bad.sum())}")

    if "amount" in out.columns:
        amt_bad = out["amount"].isna() | (out["amount"] < 0)
        if amt_bad.any():
            errors.append(f"daily has invalid amount rows: {int(amt_bad.sum())}")

    _raise_or_return(errors, raise_on_error)
    return out


def validate_label_daily_alignment(
    labels_df: pd.DataFrame,
    *,
    daily_data_dir: str | Path,
    min_history: int,
    daily_loader: DailyDataLoader | None = None,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Validate label-daily alignment across file existence, date coverage and history.

    Checks:
    - each `ts_code` in labels has corresponding parquet
    - each `(ts_code, asof_date)` is covered by daily up to asof date
    - history length up to asof date >= `min_history`
    """

    labels = validate_labels_df(labels_df, require_sample_id=False, raise_on_error=True)
    data_dir = Path(daily_data_dir)
    loader = daily_loader or DailyDataLoader(data_dir)

    issues: dict[str, list[dict[str, Any]]] = {
        "missing_daily_file": [],
        "asof_not_covered": [],
        "insufficient_history": [],
    }

    cache: dict[str, pd.DataFrame | None] = {}

    for row in labels.itertuples(index=False):
        ts_code = str(getattr(row, "ts_code"))
        asof_date = pd.to_datetime(getattr(row, "asof_date"), errors="coerce")

        if pd.isna(asof_date):
            continue

        if ts_code not in cache:
            try:
                daily_df = loader.load_one(ts_code)
                daily_df = validate_daily_quality(daily_df, raise_on_error=True)
                cache[ts_code] = daily_df
            except FileNotFoundError:
                cache[ts_code] = None
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"failed to validate daily for {ts_code}: {exc}") from exc

        daily_df = cache[ts_code]
        if daily_df is None:
            issues["missing_daily_file"].append({"ts_code": ts_code})
            continue

        dates = daily_df["trade_date"]
        if dates.empty:
            issues["asof_not_covered"].append(
                {"ts_code": ts_code, "asof_date": asof_date.strftime("%Y-%m-%d"), "reason": "empty_daily"}
            )
            continue

        if asof_date < dates.iloc[0] or asof_date > dates.iloc[-1]:
            issues["asof_not_covered"].append(
                {
                    "ts_code": ts_code,
                    "asof_date": asof_date.strftime("%Y-%m-%d"),
                    "daily_start": dates.iloc[0].strftime("%Y-%m-%d"),
                    "daily_end": dates.iloc[-1].strftime("%Y-%m-%d"),
                }
            )
            continue

        hist_len = int((dates <= asof_date).sum())
        if hist_len < min_history:
            issues["insufficient_history"].append(
                {
                    "ts_code": ts_code,
                    "asof_date": asof_date.strftime("%Y-%m-%d"),
                    "history_len": hist_len,
                    "required": min_history,
                }
            )

    error_msgs: list[str] = []
    if issues["missing_daily_file"]:
        miss_codes = sorted({x["ts_code"] for x in issues["missing_daily_file"]})
        error_msgs.append(f"missing daily parquet for ts_code: {miss_codes}")
    if issues["asof_not_covered"]:
        preview = issues["asof_not_covered"][:3]
        error_msgs.append(f"asof_date not covered by daily range, examples={preview}")
    if issues["insufficient_history"]:
        preview = issues["insufficient_history"][:3]
        error_msgs.append(f"insufficient history for min_history={min_history}, examples={preview}")

    _raise_or_return(error_msgs, raise_on_error)

    return {
        "ok": len(error_msgs) == 0,
        "issues": issues,
        "num_labels": len(labels),
    }
