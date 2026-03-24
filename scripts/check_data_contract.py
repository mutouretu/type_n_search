from __future__ import annotations

from pathlib import Path
from typing import Any

import argparse
import json

import pandas as pd


REQUIRED_LABEL_COLS = ["ts_code", "asof_date", "label"]
RECOMMENDED_LABEL_COLS = ["sample_id", "label_source", "confidence"]

REQUIRED_DAILY_COLS = ["trade_date", "open", "high", "low", "close", "vol"]
OPTIONAL_DAILY_COLS = ["amount", "turnover_rate", "pct_chg"]


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def build_sample_id(ts_code: str, asof_date: Any) -> str:
    dt = pd.to_datetime(asof_date, errors="coerce")
    if pd.isna(dt):
        return f"{ts_code}_INVALID_DATE"
    return f"{ts_code}_{dt.strftime('%Y-%m-%d')}"


def load_labels(labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        raise FileNotFoundError(f"labels file not found: {labels_path}")
    return pd.read_csv(labels_path)


def validate_label_columns(df: pd.DataFrame) -> list[str]:
    errors = []
    for col in REQUIRED_LABEL_COLS:
        if col not in df.columns:
            errors.append(f"labels missing required column: {col}")
    return errors


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "asof_date" in out.columns:
        out["asof_date"] = safe_to_datetime(out["asof_date"])

    if "sample_id" not in out.columns and {"ts_code", "asof_date"}.issubset(out.columns):
        out["sample_id"] = [
            build_sample_id(ts_code, asof_date)
            for ts_code, asof_date in zip(out["ts_code"], out["asof_date"])
        ]

    return out


def check_label_values(df: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "num_rows": len(df),
        "num_null_ts_code": 0,
        "num_null_asof_date": 0,
        "num_null_label": 0,
        "num_invalid_label": 0,
        "duplicate_sample_id_count": 0,
        "duplicate_ts_code_asof_date_count": 0,
        "label_distribution": {},
        "warnings": [],
        "errors": [],
    }

    if "ts_code" in df.columns:
        result["num_null_ts_code"] = int(df["ts_code"].isna().sum())

    if "asof_date" in df.columns:
        result["num_null_asof_date"] = int(df["asof_date"].isna().sum())

    if "label" in df.columns:
        result["num_null_label"] = int(df["label"].isna().sum())
        valid_mask = df["label"].isin([0, 1])
        result["num_invalid_label"] = int((~valid_mask & df["label"].notna()).sum())
        result["label_distribution"] = {
            str(k): int(v) for k, v in df["label"].value_counts(dropna=False).to_dict().items()
        }

    if "sample_id" in df.columns:
        result["duplicate_sample_id_count"] = int(df["sample_id"].duplicated().sum())

    if {"ts_code", "asof_date"}.issubset(df.columns):
        result["duplicate_ts_code_asof_date_count"] = int(
            df.duplicated(subset=["ts_code", "asof_date"]).sum()
        )

    if result["num_null_ts_code"] > 0:
        result["errors"].append("labels contain null ts_code")
    if result["num_null_asof_date"] > 0:
        result["errors"].append("labels contain invalid or null asof_date")
    if result["num_null_label"] > 0:
        result["errors"].append("labels contain null label")
    if result["num_invalid_label"] > 0:
        result["errors"].append("labels contain invalid label values (must be 0/1)")
    if result["duplicate_sample_id_count"] > 0:
        result["warnings"].append("labels contain duplicate sample_id")
    if result["duplicate_ts_code_asof_date_count"] > 0:
        result["warnings"].append("labels contain duplicate (ts_code, asof_date)")

    return result


def load_daily_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"daily file not found: {path}")
    return pd.read_parquet(path)


def validate_daily_columns(df: pd.DataFrame) -> list[str]:
    errors = []
    for col in REQUIRED_DAILY_COLS:
        if col not in df.columns:
            errors.append(f"daily missing required column: {col}")
    return errors


def check_daily_values(df: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "num_rows": len(df),
        "num_invalid_trade_date": 0,
        "duplicate_trade_date_count": 0,
        "num_nonpositive_open": 0,
        "num_nonpositive_high": 0,
        "num_nonpositive_low": 0,
        "num_nonpositive_close": 0,
        "num_negative_vol": 0,
        "warnings": [],
        "errors": [],
    }

    if "trade_date" in df.columns:
        trade_date = safe_to_datetime(df["trade_date"])
        result["num_invalid_trade_date"] = int(trade_date.isna().sum())
        valid_trade_df = df.loc[trade_date.notna()].copy()
        valid_trade_df["trade_date"] = trade_date[trade_date.notna()].values
        result["duplicate_trade_date_count"] = int(valid_trade_df["trade_date"].duplicated().sum())

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            count = int((pd.to_numeric(df[col], errors="coerce") <= 0).sum())
            result[f"num_nonpositive_{col}"] = count

    if "vol" in df.columns:
        vol = pd.to_numeric(df["vol"], errors="coerce")
        result["num_negative_vol"] = int((vol < 0).sum())

    if result["num_invalid_trade_date"] > 0:
        result["errors"].append("daily contains invalid trade_date")
    if result["duplicate_trade_date_count"] > 0:
        result["warnings"].append("daily contains duplicate trade_date")

    for col in ["open", "high", "low", "close"]:
        if result.get(f"num_nonpositive_{col}", 0) > 0:
            result["errors"].append(f"daily contains non-positive {col}")

    if result["num_negative_vol"] > 0:
        result["errors"].append("daily contains negative vol")

    return result


def check_label_daily_alignment(
    labels_df: pd.DataFrame,
    raw_daily_dir: Path,
    min_history: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "num_label_rows": len(labels_df),
        "num_missing_daily_files": 0,
        "missing_daily_files": [],
        "num_asof_before_first_trade": 0,
        "asof_before_first_trade_samples": [],
        "num_history_too_short": 0,
        "history_too_short_samples": [],
        "warnings": [],
        "errors": [],
    }

    if not {"ts_code", "asof_date"}.issubset(labels_df.columns):
        result["errors"].append("labels missing ts_code/asof_date, cannot check alignment")
        return result

    for _, row in labels_df.iterrows():
        ts_code = row["ts_code"]
        asof_date = row["asof_date"]

        if pd.isna(ts_code) or pd.isna(asof_date):
            continue

        daily_path = raw_daily_dir / f"{ts_code}.parquet"
        if not daily_path.exists():
            result["num_missing_daily_files"] += 1
            result["missing_daily_files"].append(str(daily_path))
            continue

        try:
            df = load_daily_file(daily_path)
            col_errors = validate_daily_columns(df)
            if col_errors:
                result["errors"].extend([f"{ts_code}: {msg}" for msg in col_errors])
                continue

            df = df.copy()
            df["trade_date"] = safe_to_datetime(df["trade_date"])
            df = df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)

            if df.empty:
                result["errors"].append(f"{ts_code}: daily parquet is empty after trade_date cleanup")
                continue

            first_trade = df["trade_date"].iloc[0]
            if asof_date < first_trade:
                result["num_asof_before_first_trade"] += 1
                result["asof_before_first_trade_samples"].append(
                    {
                        "ts_code": ts_code,
                        "asof_date": str(asof_date.date()),
                        "first_trade_date": str(first_trade.date()),
                    }
                )
                continue

            hist_len = int((df["trade_date"] <= asof_date).sum())
            if hist_len < min_history:
                result["num_history_too_short"] += 1
                result["history_too_short_samples"].append(
                    {
                        "ts_code": ts_code,
                        "asof_date": str(asof_date.date()),
                        "history_len": hist_len,
                        "required_min_history": min_history,
                    }
                )

        except Exception as e:
            result["errors"].append(f"{ts_code}: failed to inspect daily file: {e}")

    if result["num_missing_daily_files"] > 0:
        result["errors"].append("some label samples do not have corresponding daily parquet files")
    if result["num_asof_before_first_trade"] > 0:
        result["warnings"].append("some asof_date are earlier than first available trade_date")
    if result["num_history_too_short"] > 0:
        result["warnings"].append("some samples do not meet min_history requirement")

    return result


def summarize_daily_directory(raw_daily_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "raw_daily_dir": str(raw_daily_dir),
        "num_parquet_files": 0,
        "sample_files": [],
        "errors": [],
    }

    if not raw_daily_dir.exists():
        result["errors"].append(f"raw daily directory not found: {raw_daily_dir}")
        return result

    files = sorted(raw_daily_dir.glob("*.parquet"))
    result["num_parquet_files"] = len(files)
    result["sample_files"] = [f.name for f in files[:10]]
    return result


def run_check(labels_path: Path, raw_daily_dir: Path, min_history: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "labels_path": str(labels_path),
        "raw_daily_dir": str(raw_daily_dir),
        "min_history": min_history,
        "label_contract": {},
        "daily_directory": {},
        "alignment": {},
        "ok": True,
    }

    report["daily_directory"] = summarize_daily_directory(raw_daily_dir)

    try:
        labels_df = load_labels(labels_path)
    except Exception as e:
        report["label_contract"] = {"errors": [str(e)]}
        report["ok"] = False
        return report

    col_errors = validate_label_columns(labels_df)
    labels_df = normalize_labels(labels_df)
    value_result = check_label_values(labels_df)

    report["label_contract"] = {
        "column_errors": col_errors,
        **value_result,
        "recommended_columns_present": {
            col: (col in labels_df.columns) for col in RECOMMENDED_LABEL_COLS
        },
    }

    alignment_result = check_label_daily_alignment(labels_df, raw_daily_dir, min_history)
    report["alignment"] = alignment_result

    has_errors = False
    if col_errors:
        has_errors = True
    if report["daily_directory"].get("errors"):
        has_errors = True
    if report["label_contract"].get("errors"):
        has_errors = True
    if report["alignment"].get("errors"):
        has_errors = True

    report["ok"] = not has_errors
    return report


def print_report(report: dict[str, Any]) -> None:
    print("=" * 80)
    print("DATA CONTRACT CHECK REPORT")
    print("=" * 80)

    print(f"labels_path   : {report['labels_path']}")
    print(f"raw_daily_dir : {report['raw_daily_dir']}")
    print(f"min_history   : {report['min_history']}")
    print(f"overall_ok    : {report['ok']}")
    print()

    daily_dir = report["daily_directory"]
    print("[Daily Directory]")
    print(f"num_parquet_files: {daily_dir.get('num_parquet_files')}")
    if daily_dir.get("sample_files"):
        print(f"sample_files     : {daily_dir.get('sample_files')}")
    if daily_dir.get("errors"):
        print(f"errors           : {daily_dir.get('errors')}")
    print()

    label = report["label_contract"]
    print("[Labels]")
    print(f"num_rows                       : {label.get('num_rows')}")
    print(f"label_distribution            : {label.get('label_distribution')}")
    print(f"num_null_ts_code              : {label.get('num_null_ts_code')}")
    print(f"num_null_asof_date            : {label.get('num_null_asof_date')}")
    print(f"num_null_label                : {label.get('num_null_label')}")
    print(f"num_invalid_label             : {label.get('num_invalid_label')}")
    print(f"duplicate_sample_id_count     : {label.get('duplicate_sample_id_count')}")
    print(f"duplicate_ts_code_asof_date   : {label.get('duplicate_ts_code_asof_date_count')}")
    print(f"recommended_columns_present   : {label.get('recommended_columns_present')}")
    if label.get("column_errors"):
        print(f"column_errors                 : {label.get('column_errors')}")
    if label.get("warnings"):
        print(f"warnings                      : {label.get('warnings')}")
    if label.get("errors"):
        print(f"errors                        : {label.get('errors')}")
    print()

    alignment = report["alignment"]
    print("[Alignment]")
    print(f"num_label_rows                : {alignment.get('num_label_rows')}")
    print(f"num_missing_daily_files       : {alignment.get('num_missing_daily_files')}")
    print(f"num_asof_before_first_trade   : {alignment.get('num_asof_before_first_trade')}")
    print(f"num_history_too_short         : {alignment.get('num_history_too_short')}")
    if alignment.get("warnings"):
        print(f"warnings                      : {alignment.get('warnings')}")
    if alignment.get("errors"):
        print(f"errors                        : {alignment.get('errors')}")
    print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Check dataset contract for stock pattern ML.")
    parser.add_argument(
        "--labels-path",
        type=str,
        default="data/labels/labels.csv",
        help="Path to labels CSV",
    )
    parser.add_argument(
        "--raw-daily-dir",
        type=str,
        default="data/raw/daily",
        help="Directory containing daily parquet files",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=160,
        help="Minimum history length required by dataset builder",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional path to save full report as JSON",
    )
    args = parser.parse_args()

    labels_path = Path(args.labels_path)
    raw_daily_dir = Path(args.raw_daily_dir)

    report = run_check(
        labels_path=labels_path,
        raw_daily_dir=raw_daily_dir,
        min_history=args.min_history,
    )
    print_report(report)

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"json report saved to: {out_path}")

    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()