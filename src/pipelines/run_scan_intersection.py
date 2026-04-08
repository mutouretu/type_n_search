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

from src.inference.postprocess import resolve_dated_output_path
from src.pipelines.run_scan import main as run_single_scan


def _load_yaml(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _prepare_ranked(df: pd.DataFrame, score_col: str, rank_col: str) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out[rank_col] = out.index + 1
    return out.rename(columns={"score": score_col})


def main(config_path: str = "configs/infer_intersection.yaml") -> pd.DataFrame:
    config = _load_yaml(config_path)
    lgbm_config = str(config.get("lgbm_config", "configs/infer_lgbm.yaml"))
    xgb_config = str(config.get("xgb_config", "configs/infer_xgb.yaml"))
    output_path = Path(config.get("output_path", "outputs/predictions/scan_predictions_intersection.csv"))
    top_n = int(config.get("top_n", 200))

    lgbm_df = _prepare_ranked(run_single_scan(lgbm_config), "score_lgbm", "rank_lgbm")
    xgb_df = _prepare_ranked(run_single_scan(xgb_config), "score_xgb", "rank_xgb")

    merged = lgbm_df.merge(
        xgb_df,
        on=["sample_id", "ts_code", "asof_date"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No intersection candidates found between LGBM and XGB scans")

    merged["score_mean"] = (merged["score_lgbm"] + merged["score_xgb"]) / 2.0
    merged["score_min"] = merged[["score_lgbm", "score_xgb"]].min(axis=1)
    merged["rank_sum"] = merged["rank_lgbm"] + merged["rank_xgb"]

    merged = merged.sort_values(
        ["score_mean", "score_min", "rank_sum"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    if top_n > 0:
        merged = merged.head(top_n).copy()

    latest_asof_date = str(merged["asof_date"].max()) if not merged.empty else None
    output_path = resolve_dated_output_path(output_path, latest_asof_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"intersection scan finished: {len(merged)} samples -> {output_path}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LGBM+XGB intersection scan and export top candidates.")
    parser.add_argument("--config", default="configs/infer_intersection.yaml", help="Path to intersection infer config yaml")
    args = parser.parse_args()
    main(config_path=args.config)
