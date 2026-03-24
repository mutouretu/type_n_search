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

from src.data.loader import DailyDataLoader
from src.data.schema import build_sample_id
from src.features.feature_builder_tabular import build_tabular_features
from src.features.indicators import add_basic_indicators
from src.features.window_builder import build_window_by_asof_date
from src.inference.postprocess import sort_predictions
from src.inference.predictor import TabularPredictor


def _load_yaml(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def main(config_path: str = "configs/infer.yaml") -> pd.DataFrame:
    config = _load_yaml(config_path)

    raw_daily_dir = Path(config.get("raw_daily_dir", "data/raw/daily"))
    model_dir = Path(config.get("model_dir", "outputs/models/baseline_lr"))
    output_path = Path(config.get("output_path", "outputs/predictions/scan_predictions.csv"))
    window_size = int(config.get("window_size", 120))
    min_history = int(config.get("min_history", 160))

    if not raw_daily_dir.exists():
        raise FileNotFoundError(f"raw_daily_dir not found: {raw_daily_dir}")

    loader = DailyDataLoader(raw_daily_dir)
    meta_rows: list[dict[str, str]] = []
    feat_rows: list[dict[str, float]] = []

    parquet_files = sorted(raw_daily_dir.glob("*.parquet"))
    for path in parquet_files:
        ts_code = path.stem
        try:
            daily_df = loader.load_one(ts_code)
            daily_df = add_basic_indicators(daily_df)

            asof_date = pd.to_datetime(daily_df["trade_date"].iloc[-1], errors="coerce")
            if pd.isna(asof_date):
                print(f"[skip] {ts_code}: invalid latest trade_date")
                continue

            window_df = build_window_by_asof_date(
                daily_df,
                asof_date=asof_date,
                window_size=window_size,
                min_history=min_history,
            )
            if window_df is None:
                print(f"[skip] {ts_code}: insufficient history")
                continue

            features = build_tabular_features(window_df)
            asof_date_str = asof_date.strftime("%Y-%m-%d")
            sample_id = build_sample_id(ts_code=ts_code, asof_date=asof_date_str)

            meta_rows.append(
                {
                    "sample_id": sample_id,
                    "ts_code": ts_code,
                    "asof_date": asof_date_str,
                }
            )
            feat_rows.append(features)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {ts_code}: {exc}")

    if not meta_rows:
        raise ValueError("No predictable samples found in raw_daily_dir")

    meta_df = pd.DataFrame(meta_rows)
    feat_df = pd.DataFrame(feat_rows)

    predictor = TabularPredictor.from_dir(model_dir)
    scores = predictor.predict_proba(feat_df)

    pred_df = meta_df.copy()
    pred_df["score"] = scores
    pred_df = sort_predictions(pred_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)
    print(f"scan finished: {len(pred_df)} samples -> {output_path}")

    return pred_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run latest-date tabular scan over daily parquet files.")
    parser.add_argument("--config", default="configs/infer.yaml", help="Path to infer config yaml")
    args = parser.parse_args()
    main(config_path=args.config)
