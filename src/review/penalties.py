from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.review.scoring import sigmoid_decay_factor

KEY_COLUMNS = ["sample_id", "ts_code", "asof_date"]


def _resolve_path(project_root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root / p


def _load_runup_values(
    rows: pd.DataFrame,
    *,
    raw_data_dir: Path,
    window: int,
) -> pd.DataFrame:
    runup_rows: List[Dict[str, Any]] = []
    for ts_code, group in rows[KEY_COLUMNS].drop_duplicates().groupby("ts_code"):
        path = raw_data_dir / f"{ts_code}.parquet"
        runup_by_date: Dict[str, float] = {}
        if path.exists():
            daily = pd.read_parquet(path, columns=["trade_date", "close"]).copy()
            daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            daily = daily.dropna(subset=["trade_date"]).sort_values("trade_date")
            close = pd.to_numeric(daily["close"], errors="coerce")
            rolling_low = close.rolling(window, min_periods=window).min().replace(0, float("nan"))
            daily["runup"] = close / rolling_low - 1.0
            runup_by_date = daily.set_index("trade_date")["runup"].to_dict()
        for _, row in group.iterrows():
            runup_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "ts_code": row["ts_code"],
                    "asof_date": row["asof_date"],
                    "runup_value": runup_by_date.get(str(row["asof_date"]), pd.NA),
                }
            )
    return pd.DataFrame(runup_rows)


def _apply_runup_post_penalty(result: pd.DataFrame, config: Dict[str, Any], project_root: Path) -> pd.DataFrame:
    if not bool(config.get("enabled", False)):
        return result

    raw_data_dir = _resolve_path(project_root, config["raw_data_dir"])
    window = int(config.get("window", 150))
    threshold = float(config.get("threshold", 0.35))
    sharpness = float(config.get("sharpness", 20.0))
    score_col = str(config.get("score_col", "baseline_score"))
    output_score_col = str(config.get("output_score_col", "adjusted_score"))
    runup_col = str(config.get("runup_col", f"runup_{window}"))

    if score_col not in result.columns:
        raise ValueError(f"Runup post penalty score_col not found: {score_col}")

    runup = _load_runup_values(result, raw_data_dir=raw_data_dir, window=window)
    out = result.merge(runup, on=KEY_COLUMNS, how="left")
    out = out.rename(columns={"runup_value": runup_col})
    out["runup_penalty_threshold"] = threshold
    out["runup_penalty_sharpness"] = sharpness
    out["runup_penalty_factor"] = sigmoid_decay_factor(out[runup_col], threshold=threshold, sharpness=sharpness)
    out[output_score_col] = pd.to_numeric(out[score_col], errors="coerce") * out["runup_penalty_factor"]
    return out


def apply_post_penalties(result: pd.DataFrame, config: Dict[str, Any], project_root: Path) -> pd.DataFrame:
    """Apply optional review-stage penalties without touching training/inference pipelines."""
    if not isinstance(config, dict):
        return result

    out = result.copy()
    runup_penalty_cfg = config.get("runup", {})
    if isinstance(runup_penalty_cfg, dict):
        out = _apply_runup_post_penalty(out, runup_penalty_cfg, project_root)
    return out


__all__ = ["apply_post_penalties"]
