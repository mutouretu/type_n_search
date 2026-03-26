from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_builder import DatasetBuilder
from src.data.loader import DailyDataLoader
from src.data.normalize import DailyUnitConfig
from src.data.validator import validate_label_daily_alignment, validate_labels_df


def _load_yaml(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _precheck_or_raise(
    labels_path: str,
    raw_daily_dir: str,
    min_history: int,
    *,
    unit_config: DailyUnitConfig,
    duplicate_policy: str,
) -> None:
    labels_df = pd.read_csv(labels_path)
    labels_df = validate_labels_df(labels_df, require_sample_id=False, raise_on_error=True)
    report = validate_label_daily_alignment(
        labels_df,
        daily_data_dir=raw_daily_dir,
        min_history=min_history,
        daily_loader=DailyDataLoader(
            raw_daily_dir,
            unit_config=unit_config,
            duplicate_policy=duplicate_policy,
        ),
        raise_on_error=False,
    )
    if not report["ok"]:
        issues = report["issues"]
        raise ValueError(
            "real data precheck failed: "
            f"missing_daily_file={len(issues['missing_daily_file'])}, "
            f"asof_not_covered={len(issues['asof_not_covered'])}, "
            f"insufficient_history={len(issues['insufficient_history'])}"
        )


def main(config_path: str = "configs/data.yaml") -> Dict[str, Any]:
    config = _load_yaml(config_path)

    labels_path = config.get("labels_path", "data/labels/labels.csv")
    raw_daily_dir = config.get("raw_daily_dir", config.get("data_dir", "data/raw/daily"))
    min_history = int(config.get("min_history", 80))
    duplicate_policy = str(config.get("duplicate_policy", "raise"))
    unit_config = DailyUnitConfig(
        vol_unit=str(config.get("vol_unit", "lots")),
        amount_unit=str(config.get("amount_unit", "thousand_yuan")),
        vol_lot_size=float(config.get("vol_lot_size", 100.0)),
        amount_thousand_size=float(config.get("amount_thousand_size", 1000.0)),
    )

    _precheck_or_raise(
        labels_path=labels_path,
        raw_daily_dir=raw_daily_dir,
        min_history=min_history,
        unit_config=unit_config,
        duplicate_policy=duplicate_policy,
    )

    builder = DatasetBuilder(
        labels_path=labels_path,
        data_dir=config.get("data_dir", raw_daily_dir),
        output_dir=config.get("output_dir", config.get("processed_dir", "data/processed")),
        window_size=int(config.get("window_size", 120)),
        min_history=min_history,
        train_ratio=float(config.get("train_ratio", 0.7)),
        valid_ratio=float(config.get("valid_ratio", 0.15)),
        sequence_cols=config.get("sequence_cols"),
        save_sequence=bool(config.get("save_sequence", True)),
        unit_config=unit_config,
        duplicate_policy=duplicate_policy,
        max_skip_ratio=(
            float(config["max_skip_ratio"]) if "max_skip_ratio" in config else None
        ),
    )

    result = builder.build()
    print("Dataset build finished:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dataset artifacts.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config yaml")
    args = parser.parse_args()
    main(config_path=args.config)
