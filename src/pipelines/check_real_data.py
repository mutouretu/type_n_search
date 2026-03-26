from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import DailyDataLoader
from src.data.normalize import DailyUnitConfig
from src.data.validator import validate_label_daily_alignment, validate_labels_df


def run_real_data_check(
    *,
    labels_path: str = "data/labels/labels.csv",
    daily_dir: str = "data/raw/daily",
    min_history: int = 80,
    unit_config: DailyUnitConfig | None = None,
    duplicate_policy: str = "raise",
    raise_on_fail: bool = True,
) -> Dict[str, Any]:
    labels_df = pd.read_csv(labels_path)
    labels_df = validate_labels_df(labels_df, require_sample_id=False, raise_on_error=True)

    report = validate_label_daily_alignment(
        labels_df,
        daily_data_dir=daily_dir,
        min_history=min_history,
        daily_loader=DailyDataLoader(
            daily_dir,
            unit_config=unit_config or DailyUnitConfig(),
            duplicate_policy=duplicate_policy,
        ),
        raise_on_error=False,
    )

    if raise_on_fail and not report["ok"]:
        issues = report["issues"]
        summary = (
            f"real data check failed: "
            f"missing_daily_file={len(issues['missing_daily_file'])}, "
            f"asof_not_covered={len(issues['asof_not_covered'])}, "
            f"insufficient_history={len(issues['insufficient_history'])}"
        )
        raise ValueError(summary)

    return report


def _print_report(report: Dict[str, Any], preview_n: int = 5) -> None:
    issues = report["issues"]
    print("real data check result")
    print(f"  ok: {report['ok']}")
    print(f"  num_labels: {report['num_labels']}")
    print(f"  missing_daily_file: {len(issues['missing_daily_file'])}")
    print(f"  asof_not_covered: {len(issues['asof_not_covered'])}")
    print(f"  insufficient_history: {len(issues['insufficient_history'])}")
    if not report["ok"]:
        print(f"  preview_n: {preview_n}")
        print(f"  missing_daily_file_examples: {issues['missing_daily_file'][:preview_n]}")
        print(f"  asof_not_covered_examples: {issues['asof_not_covered'][:preview_n]}")
        print(f"  insufficient_history_examples: {issues['insufficient_history'][:preview_n]}")


def main(
    labels_path: str,
    daily_dir: str,
    min_history: int,
    *,
    vol_unit: str,
    amount_unit: str,
    duplicate_policy: str,
    save_json: str,
    preview_n: int,
) -> Dict[str, Any]:
    unit_config = DailyUnitConfig(vol_unit=vol_unit, amount_unit=amount_unit)
    report = run_real_data_check(
        labels_path=labels_path,
        daily_dir=daily_dir,
        min_history=min_history,
        unit_config=unit_config,
        duplicate_policy=duplicate_policy,
        raise_on_fail=False,
    )

    _print_report(report, preview_n=preview_n)
    if save_json:
        out_path = Path(save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"  report_json: {out_path}")

    if not report["ok"]:
        issues = report["issues"]
        raise ValueError(
            "real data check failed: "
            f"missing_daily_file={len(issues['missing_daily_file'])}, "
            f"asof_not_covered={len(issues['asof_not_covered'])}, "
            f"insufficient_history={len(issues['insufficient_history'])}"
        )

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check real data labels/daily alignment")
    parser.add_argument("--labels-path", default="data/labels/labels.csv")
    parser.add_argument("--daily-dir", default="data/raw/daily")
    parser.add_argument("--min-history", type=int, default=80)
    parser.add_argument("--vol-unit", default="lots", choices=["lots", "shares"])
    parser.add_argument(
        "--amount-unit",
        default="thousand_yuan",
        choices=["thousand_yuan", "yuan"],
    )
    parser.add_argument(
        "--duplicate-policy",
        default="raise",
        choices=["raise", "first", "last"],
    )
    parser.add_argument("--save-json", default="")
    parser.add_argument("--preview-n", type=int, default=5)
    args = parser.parse_args()
    main(
        args.labels_path,
        args.daily_dir,
        args.min_history,
        vol_unit=args.vol_unit,
        amount_unit=args.amount_unit,
        duplicate_policy=args.duplicate_policy,
        save_json=args.save_json,
        preview_n=args.preview_n,
    )
