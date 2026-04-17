from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.review.scoring import sigmoid_boost_factor, sigmoid_decay_factor

KEY_COLUMNS = ["sample_id", "ts_code", "asof_date"]


def _resolve_path(project_root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root / p


def _apply_factor_weight(factor: pd.Series, weight: float) -> pd.Series:
    """Interpolate a multiplicative factor toward neutral 1.0 by weight."""
    numeric_factor = pd.to_numeric(factor, errors="coerce")
    return 1.0 + float(weight) * (numeric_factor - 1.0)


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


def _load_base_stability_values(
    rows: pd.DataFrame,
    *,
    raw_data_dir: Path,
    window: int,
    fast_ma_col: str,
    slow_ma_col: str,
    trend_ma_col: str,
    prior_lag: int,
    recent_weight: float,
    prior_weight: float,
) -> pd.DataFrame:
    stability_rows: List[Dict[str, Any]] = []
    parquet_columns = list(dict.fromkeys(["trade_date", fast_ma_col, slow_ma_col, trend_ma_col]))
    for ts_code, group in rows[KEY_COLUMNS].drop_duplicates().groupby("ts_code"):
        path = raw_data_dir / f"{ts_code}.parquet"
        values_by_date: Dict[str, Dict[str, float]] = {}
        if path.exists():
            daily = pd.read_parquet(path, columns=parquet_columns).copy()
            daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            daily = daily.dropna(subset=["trade_date"]).sort_values("trade_date")
            fast_ma = pd.to_numeric(daily[fast_ma_col], errors="coerce")
            slow_ma = pd.to_numeric(daily[slow_ma_col], errors="coerce").replace(0, float("nan"))
            trend_ma = pd.to_numeric(daily[trend_ma_col], errors="coerce").replace(0, float("nan"))

            # Shift by one day so the base score describes the pre-breakout box, not the trigger day.
            ma_gap = ((fast_ma - slow_ma) / slow_ma).shift(1)
            daily["base_ma_gap_l2"] = (ma_gap.pow(2).rolling(window, min_periods=window).mean()).pow(0.5)
            daily["base_ma60_recent_slope_pct"] = trend_ma.shift(1) / trend_ma.shift(window + 1) - 1.0
            daily["base_ma60_prior_slope_pct"] = trend_ma.shift(window + 1) / trend_ma.shift(prior_lag + 1) - 1.0
            daily["base_ma60_trend_abs"] = (
                recent_weight * daily["base_ma60_recent_slope_pct"].abs()
                + prior_weight * daily["base_ma60_prior_slope_pct"].abs()
            )
            values_by_date = daily.set_index("trade_date")[
                [
                    "base_ma_gap_l2",
                    "base_ma60_recent_slope_pct",
                    "base_ma60_prior_slope_pct",
                    "base_ma60_trend_abs",
                ]
            ].to_dict("index")
        for _, row in group.iterrows():
            values = values_by_date.get(str(row["asof_date"]), {})
            stability_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "ts_code": row["ts_code"],
                    "asof_date": row["asof_date"],
                    "base_ma_gap_l2": values.get("base_ma_gap_l2", pd.NA),
                    "base_ma60_recent_slope_pct": values.get("base_ma60_recent_slope_pct", pd.NA),
                    "base_ma60_prior_slope_pct": values.get("base_ma60_prior_slope_pct", pd.NA),
                    "base_ma60_trend_abs": values.get("base_ma60_trend_abs", pd.NA),
                }
            )
    return pd.DataFrame(stability_rows)


def _load_box_breakout_values(
    rows: pd.DataFrame,
    *,
    raw_data_dir: Path,
    window: int,
    high_col: str,
    close_col: str,
) -> pd.DataFrame:
    breakout_rows: List[Dict[str, Any]] = []
    for ts_code, group in rows[KEY_COLUMNS].drop_duplicates().groupby("ts_code"):
        path = raw_data_dir / f"{ts_code}.parquet"
        values_by_date: Dict[str, Dict[str, float]] = {}
        if path.exists():
            daily = pd.read_parquet(path, columns=["trade_date", high_col, close_col]).copy()
            daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            daily = daily.dropna(subset=["trade_date"]).sort_values("trade_date")
            high = pd.to_numeric(daily[high_col], errors="coerce")
            close = pd.to_numeric(daily[close_col], errors="coerce")
            box_high = high.shift(1).rolling(window, min_periods=window).max().replace(0, float("nan"))
            daily["box_high"] = box_high
            daily["box_breakout_pct"] = close / box_high - 1.0
            values_by_date = daily.set_index("trade_date")[["box_high", "box_breakout_pct"]].to_dict("index")
        for _, row in group.iterrows():
            values = values_by_date.get(str(row["asof_date"]), {})
            breakout_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "ts_code": row["ts_code"],
                    "asof_date": row["asof_date"],
                    "box_high": values.get("box_high", pd.NA),
                    "box_breakout_pct": values.get("box_breakout_pct", pd.NA),
                }
            )
    return pd.DataFrame(breakout_rows)


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
    weight = float(config.get("weight", 1.0))

    if score_col not in result.columns:
        raise ValueError(f"Runup post penalty score_col not found: {score_col}")

    runup = _load_runup_values(result, raw_data_dir=raw_data_dir, window=window)
    out = result.merge(runup, on=KEY_COLUMNS, how="left")
    out = out.rename(columns={"runup_value": runup_col})
    out["runup_penalty_threshold"] = threshold
    out["runup_penalty_sharpness"] = sharpness
    raw_factor = sigmoid_decay_factor(out[runup_col], threshold=threshold, sharpness=sharpness)
    out["runup_penalty_factor"] = _apply_factor_weight(raw_factor, weight)
    out["runup_penalty_raw_factor"] = raw_factor
    out["runup_penalty_weight"] = weight
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
    weight = float(config.get("weight", 1.0))

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
    raw_factor = strength_factor * streak_factor
    out[factor_col] = _apply_factor_weight(raw_factor, weight)
    out["volume_penalty_weight"] = weight
    if keep_component_factors:
        out["volume_strength_boost_factor"] = strength_factor
        out["volume_streak_decay_factor"] = streak_factor
        out["volume_raw_factor"] = raw_factor
    out["volume_strength_threshold"] = strength_threshold
    out["volume_streak_threshold"] = streak_threshold
    out[output_score_col] = pd.to_numeric(out[score_col], errors="coerce") * out[factor_col]
    return out


def _apply_base_stability_post_penalty(result: pd.DataFrame, config: Dict[str, Any], project_root: Path) -> pd.DataFrame:
    if not bool(config.get("enabled", False)):
        return result

    raw_data_dir = _resolve_path(project_root, config["raw_data_dir"])
    window = int(config.get("window", 60))
    fast_ma_col = str(config.get("fast_ma_col", "ma_bfq_20"))
    slow_ma_col = str(config.get("slow_ma_col", "ma_bfq_60"))
    trend_ma_col = str(config.get("trend_ma_col", slow_ma_col))
    score_col = str(config.get("score_col", "adjusted_score"))
    output_score_col = str(config.get("output_score_col", score_col))
    factor_col = str(config.get("factor_col", "base_stability_factor"))
    keep_component_factors = bool(config.get("keep_component_factors", False))

    ma_gap_cfg = config.get("ma_gap", {})
    if not isinstance(ma_gap_cfg, dict):
        ma_gap_cfg = {}
    ma_gap_threshold = float(ma_gap_cfg.get("threshold", 0.04))
    ma_gap_sharpness = float(ma_gap_cfg.get("sharpness", 80.0))

    trend_cfg = config.get("ma_trend", {})
    if not isinstance(trend_cfg, dict):
        trend_cfg = {}
    prior_lag = int(trend_cfg.get("prior_lag", window * 2))
    recent_weight = float(trend_cfg.get("recent_weight", 0.7))
    prior_weight = float(trend_cfg.get("prior_weight", 0.3))
    trend_threshold = float(trend_cfg.get("threshold", 0.08))
    trend_sharpness = float(trend_cfg.get("sharpness", 25.0))

    if score_col not in result.columns:
        raise ValueError(f"Base stability post penalty score_col not found: {score_col}")

    stability = _load_base_stability_values(
        result,
        raw_data_dir=raw_data_dir,
        window=window,
        fast_ma_col=fast_ma_col,
        slow_ma_col=slow_ma_col,
        trend_ma_col=trend_ma_col,
        prior_lag=prior_lag,
        recent_weight=recent_weight,
        prior_weight=prior_weight,
    )
    out = result.merge(stability, on=KEY_COLUMNS, how="left")
    ma_gap_factor = sigmoid_decay_factor(
        out["base_ma_gap_l2"],
        threshold=ma_gap_threshold,
        sharpness=ma_gap_sharpness,
    )
    trend_factor = sigmoid_decay_factor(
        out["base_ma60_trend_abs"],
        threshold=trend_threshold,
        sharpness=trend_sharpness,
    )
    out[factor_col] = ma_gap_factor * trend_factor
    if keep_component_factors:
        out["base_ma_gap_factor"] = ma_gap_factor
        out["base_ma60_trend_factor"] = trend_factor
    out["base_ma_gap_threshold"] = ma_gap_threshold
    out["base_ma60_trend_threshold"] = trend_threshold
    out[output_score_col] = pd.to_numeric(out[score_col], errors="coerce") * out[factor_col]
    return out


def _apply_box_breakout_post_penalty(result: pd.DataFrame, config: Dict[str, Any], project_root: Path) -> pd.DataFrame:
    if not bool(config.get("enabled", False)):
        return result

    raw_data_dir = _resolve_path(project_root, config["raw_data_dir"])
    window = int(config.get("window", 60))
    high_col = str(config.get("high_col", "high"))
    close_col = str(config.get("close_col", "close"))
    score_col = str(config.get("score_col", "adjusted_score"))
    output_score_col = str(config.get("output_score_col", score_col))
    factor_col = str(config.get("factor_col", "box_breakout_factor"))

    strength_cfg = config.get("strength", {})
    if not isinstance(strength_cfg, dict):
        strength_cfg = {}
    strength_threshold = float(strength_cfg.get("threshold", -0.01))
    strength_sharpness = float(strength_cfg.get("sharpness", 80.0))
    min_factor = float(config.get("min_factor", 0.3))
    max_factor = float(config.get("max_factor", 1.2))
    weight = float(config.get("weight", 1.0))

    if score_col not in result.columns:
        raise ValueError(f"Box breakout post penalty score_col not found: {score_col}")

    breakout = _load_box_breakout_values(
        result,
        raw_data_dir=raw_data_dir,
        window=window,
        high_col=high_col,
        close_col=close_col,
    )
    out = result.merge(breakout, on=KEY_COLUMNS, how="left")
    strength = sigmoid_boost_factor(
        out["box_breakout_pct"],
        threshold=strength_threshold,
        sharpness=strength_sharpness,
        max_boost=1.0,
        missing_value=0.0,
    )
    # sigmoid_boost_factor returns [1, 2]; normalize it to [0, 1] before scaling.
    normalized_strength = strength - 1.0
    raw_factor = min_factor + (max_factor - min_factor) * normalized_strength
    out[factor_col] = _apply_factor_weight(raw_factor, weight)
    out["box_breakout_raw_factor"] = raw_factor
    out["box_breakout_weight"] = weight
    out["box_breakout_threshold"] = strength_threshold
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
    base_stability_penalty_cfg = config.get("base_stability", {})
    if isinstance(base_stability_penalty_cfg, dict):
        out = _apply_base_stability_post_penalty(out, base_stability_penalty_cfg, project_root)
    box_breakout_penalty_cfg = config.get("box_breakout", {})
    if isinstance(box_breakout_penalty_cfg, dict):
        out = _apply_box_breakout_post_penalty(out, box_breakout_penalty_cfg, project_root)
    return out


__all__ = ["apply_post_penalties"]
