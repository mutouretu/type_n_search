from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_builder import DatasetBuilder
from scripts.check_data_contract import validate_or_raise


def _load_yaml(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def main(config_path: str = "configs/data.yaml") -> Dict[str, Any]:
    config = _load_yaml(config_path)

    labels_path = config.get("labels_path", "data/labels/labels.csv")
    raw_daily_dir = config.get("raw_daily_dir", config.get("data_dir", "data/raw/daily"))
    min_history = int(config.get("min_history", 80))

    validate_or_raise(
        labels_path=Path(labels_path),
        raw_daily_dir=Path(raw_daily_dir),
        min_history=min_history,
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
