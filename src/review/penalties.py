from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.review.scoring import sigmoid_boost_factor, sigmoid_decay_factor

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


def _load_volume_values(
    rows: pd.DataFrame,
    *,
    raw_data_dir: Path,
    ma_window: int,
    short_window: int,
    spike_ratio: float,
) -> pd.DataFrame:
    volume_rows: List[Dict[str, Any]] = []
    for ts_code, group in rows[KEY_COLUMNS].drop_duplicates().groupby("ts_code"):
        path = raw_data_dir / f"{ts_code}.parquet"
        values_by_date: Dict[str, Dict[str, float]] = {}
        if path.exists():
            daily = pd.read_parquet(path, columns=["trade_date", "vol"]).copy()
            daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            daily = daily.dropna(subset=["trade_date"]).sort_values("trade_date")
            vol = pd.to_numeric(daily["vol"], errors="coerce")
            vol_base = vol.rolling(ma_window, min_periods=ma_window).mean().shift(1).replace(0, float("nan"))
            ratio_1d = vol / vol_base
            ratio_short = vol.rolling(short_window, min_periods=short_window).mean() / vol_base
            spike = ratio_1d >= spike_ratio
            streak = spike.groupby((spike != spike.shift()).cumsum()).cumcount() + 1
            streak = streak.where(spike, 0)
            daily["volume_ratio"] = ratio_short
            daily["volume_spike_streak"] = streak
            values_by_date = daily.set_index("trade_date")[["volume_ratio", "volume_spike_streak"]].to_dict("index")
        for _, row in group.iterrows():
            values = values_by_date.get(str(row["asof_date"]), {})
            volume_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "ts_code": row["ts_code"],
                    "asof_date": row["asof_date"],
                    "volume_ratio": values.get("volume_ratio", pd.NA),
                    "volume_spike_streak": values.get("volume_spike_streak", pd.NA),
                }
            )
    return pd.DataFrame(volume_rows)


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


def _apply_volume_post_penalty(result: pd.DataFrame, config: Dict[str, Any], project_root: Path) -> pd.DataFrame:
    if not bool(config.get("enabled", False)):
        return result

    raw_data_dir = _resolve_path(project_root, config["raw_data_dir"])
    ma_window = int(config.get("ma_window", 20))
    short_window = int(config.get("short_window", 3))
    score_col = str(config.get("score_col", "adjusted_score"))
    output_score_col = str(config.get("output_score_col", score_col))

    strength_cfg = config.get("strength", {})
    if not isinstance(strength_cfg, dict):
        strength_cfg = {}
    strength_threshold = float(strength_cfg.get("threshold", 1.8))
    strength_sharpness = float(strength_cfg.get("sharpness", 3.0))
    max_boost = float(strength_cfg.get("max_boost", 0.15))

    streak_cfg = config.get("streak", {})
    if not isinstance(streak_cfg, dict):
        streak_cfg = {}
    spike_ratio = float(streak_cfg.get("spike_ratio", strength_threshold))
    streak_threshold = float(streak_cfg.get("threshold", 3.0))
    streak_sharpness = float(streak_cfg.get("sharpness", 1.5))

    ratio_col = str(config.get("ratio_col", f"volume_ratio_{short_window}d_{ma_window}"))
    streak_col = str(config.get("streak_col", "volume_spike_streak"))
    factor_col = str(config.get("factor_col", "volume_penalty_factor"))
    keep_component_factors = bool(config.get("keep_component_factors", False))

    if score_col not in result.columns:
        raise ValueError(f"Volume post penalty score_col not found: {score_col}")

    volume = _load_volume_values(
        result,
        raw_data_dir=raw_data_dir,
        ma_window=ma_window,
        short_window=short_window,
        spike_ratio=spike_ratio,
    )
    out = result.merge(volume, on=KEY_COLUMNS, how="left")
    out = out.rename(columns={"volume_ratio": ratio_col, "volume_spike_streak": streak_col})

    strength_factor = sigmoid_boost_factor(
        out[ratio_col],
        threshold=strength_threshold,
        sharpness=strength_sharpness,
        max_boost=max_boost,
    )
    streak_factor = sigmoid_decay_factor(
        out[streak_col],
        threshold=streak_threshold,
        sharpness=streak_sharpness,
    )
    out[factor_col] = strength_factor * streak_factor
    if keep_component_factors:
        out["volume_strength_boost_factor"] = strength_factor
        out["volume_streak_decay_factor"] = streak_factor
    out["volume_strength_threshold"] = strength_threshold
    out["volume_streak_threshold"] = streak_threshold
    out[output_score_col] = pd.to_numeric(out[score_col], errors="coerce") * out[factor_col]
    return out


def apply_post_penalties(result: pd.DataFrame, config: Dict[str, Any], project_root: Path) -> pd.DataFrame:
    """Apply optional review-stage penalties without touching training/inference pipelines."""
    if not isinstance(config, dict):
        return result

    out = result.copy()
    runup_penalty_cfg = config.get("runup", {})
    if isinstance(runup_penalty_cfg, dict):
        out = _apply_runup_post_penalty(out, runup_penalty_cfg, project_root)
    volume_penalty_cfg = config.get("volume", {})
    if isinstance(volume_penalty_cfg, dict):
        out = _apply_volume_post_penalty(out, volume_penalty_cfg, project_root)
    return out


__all__ = ["apply_post_penalties"]
