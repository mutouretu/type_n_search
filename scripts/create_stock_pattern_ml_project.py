from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd


@dataclass
class TypeNConfig:
    start_date: str = "20250101"
    end_date: str = pd.Timestamp.today().strftime("%Y%m%d")
    cache_dir: str = "data_cache/daily_basic_parquet"

    # 基底区参数
    base_lookback: int = 100
    base_max_range_pct: float = 0.35
    base_max_vol_cv: float = 0.80
    base_vol_shrink_ratio_max: float = 0.80

    # 启动参数
    impulse_lookback: int = 20
    breakout_lookback: int = 90
    min_impulse_pct: float = 0.07
    min_impulse_vol_ratio: float = 2.0
    min_turnover_rate: float = 3.0

    # 回撤参数
    min_pullback_days: int = 2
    max_pullback_days: int = 8
    min_pullback_pct: float = 0.03
    max_pullback_pct: float = 0.15
    max_pullback_vol_ratio: float = 0.75

    # 结构参数
    max_close_below_breakout_close: float = 0.06
    min_above_ma10_ratio: float = 0.97
    min_above_platform_ratio: float = 0.98

    # 粗筛参数
    prefilter_lookback: int = 25
    prefilter_recent_days: int = 20
    prefilter_min_pct_chg: float = 7.0
    prefilter_min_turnover: float = 5.0
    prefilter_vol_ratio: float = 1.8

    # 其他过滤
    min_price: float = 5.0
    max_price: float = 120.0

    # 输出
    output_csv: str = "type_n_candidates_from_cache.csv"


class ParquetCacheManager:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)

    def list_codes(self) -> List[str]:
        if not self.cache_dir.exists():
            return []
        return [p.stem for p in self.cache_dir.glob("*.parquet")]

    def load(self, ts_code: str) -> pd.DataFrame:
        path = self.cache_dir / f"{ts_code}.parquet"
        if not path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(path)
        if not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date").reset_index(drop=True)
        return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for n in [5, 10, 20, 30, 60, 120]:
        df[f"ma{n}"] = df["close"].rolling(n).mean()
        df[f"vma{n}"] = df["vol"].rolling(n).mean()

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    df["hhv90_prev"] = df["high"].shift(1).rolling(90).max()
    df["llv100"] = df["low"].rolling(100).min()
    df["hhv100"] = df["high"].rolling(100).max()

    df["vol_mean_20"] = df["vol"].rolling(20).mean()
    df["vol_mean_100"] = df["vol"].rolling(100).mean()
    df["vol_std_100"] = df["vol"].rolling(100).std()
    df["vol_cv_100"] = df["vol_std_100"] / (df["vol_mean_100"] + 1e-9)

    df["base_range_pct_100"] = (df["hhv100"] - df["llv100"]) / (df["llv100"] + 1e-9)
    df["vol_shrink_ratio"] = df["vol_mean_20"] / (df["vol_mean_100"] + 1e-9)

    return df


def is_limit_like(row: pd.Series, ts_code: str) -> bool:
    pct = row.get("pct_chg", np.nan)
    if pd.isna(pct):
        return False

    if ts_code.startswith("300") or ts_code.startswith("688"):
        return pct >= 19.0
    return pct >= 9.5


def quick_pre_filter(df: pd.DataFrame, cfg: TypeNConfig) -> bool:
    if df.empty or len(df) < max(10, cfg.prefilter_lookback):
        return False

    df = df.copy().sort_values("trade_date").reset_index(drop=True)
    df["vma5"] = df["vol"].rolling(5).mean()

    recent = df.iloc[-cfg.prefilter_recent_days:].copy()
    if recent.empty:
        return False

    latest_close = recent.iloc[-1]["close"]
    if pd.isna(latest_close) or latest_close < cfg.min_price or latest_close > cfg.max_price:
        return False

    cond1 = (recent["pct_chg"] >= cfg.prefilter_min_pct_chg).any()

    cond2 = False
    if "turnover_rate" in recent.columns:
        cond2 = (recent["turnover_rate"] >= cfg.prefilter_min_turnover).any()

    cond3 = (recent["vol"] > recent["vma5"] * cfg.prefilter_vol_ratio).any()

    return bool(cond1 or cond2 or cond3)


def detect_type_n(df: pd.DataFrame, ts_code: str, cfg: TypeNConfig) -> Optional[Dict]:
    if df.empty:
        return None

    df = df.copy()
    df = df[
        (df["trade_date"] >= pd.to_datetime(cfg.start_date)) &
        (df["trade_date"] <= pd.to_datetime(cfg.end_date))
    ].sort_values("trade_date").reset_index(drop=True)

    if len(df) < max(cfg.base_lookback + 30, 160):
        return None

    df = add_indicators(df)
    latest = df.iloc[-1]

    if pd.isna(latest["close"]) or latest["close"] < cfg.min_price or latest["close"] > cfg.max_price:
        return None

    recent = df.iloc[-cfg.impulse_lookback:].copy()
    impulse_candidates = []

    for i in recent.index:
        row = df.loc[i]

        if i < cfg.base_lookback:
            continue

        base_start = i - cfg.base_lookback
        base_df = df.iloc[base_start:i]
        if len(base_df) < cfg.base_lookback:
            continue

        base_range_ok = base_df["high"].max() / (base_df["low"].min() + 1e-9) - 1 <= cfg.base_max_range_pct
        vol_cv_ok = (base_df["vol"].std() / (base_df["vol"].mean() + 1e-9)) <= cfg.base_max_vol_cv
        vol_shrink_ok = (base_df["vol"].tail(20).mean() / (base_df["vol"].mean() + 1e-9)) <= cfg.base_vol_shrink_ratio_max

        pct_ok = row["pct_chg"] >= cfg.min_impulse_pct * 100 or is_limit_like(row, ts_code)
        vol_ratio = row["vol"] / (base_df["vol"].tail(20).mean() + 1e-9)
        vol_ok = vol_ratio >= cfg.min_impulse_vol_ratio

        turnover_ok = True
        if "turnover_rate" in row.index and not pd.isna(row["turnover_rate"]):
            turnover_ok = row["turnover_rate"] >= cfg.min_turnover_rate

        prev_high = df.iloc[max(0, i - cfg.breakout_lookback):i]["high"].max()
        breakout_ok = row["close"] > prev_high * 0.98

        if base_range_ok and vol_cv_ok and vol_shrink_ok and pct_ok and vol_ok and breakout_ok and turnover_ok:
            impulse_candidates.append({
                "idx": i,
                "trade_date": row["trade_date"],
                "impulse_close": row["close"],
                "impulse_high": row["high"],
                "impulse_vol": row["vol"],
                "impulse_pct_chg": row["pct_chg"],
                "impulse_vol_ratio": vol_ratio,
                "platform_high_before": prev_high,
                "base_range_pct": base_df["high"].max() / (base_df["low"].min() + 1e-9) - 1,
                "base_vol_cv": base_df["vol"].std() / (base_df["vol"].mean() + 1e-9),
            })

    if not impulse_candidates:
        return None

    impulse = impulse_candidates[-1]
    impulse_idx = impulse["idx"]

    post_df = df.iloc[impulse_idx + 1:].copy()
    if len(post_df) < cfg.min_pullback_days or len(post_df) > 15:
        return None

    sub_df = df.iloc[impulse_idx:]
    first_leg_high = sub_df["high"].max()
    first_leg_high_idx = sub_df["high"].idxmax()

    if first_leg_high_idx >= len(df) - 1:
        return None

    pullback_df = df.iloc[first_leg_high_idx + 1:].copy()
    if len(pullback_df) < cfg.min_pullback_days or len(pullback_df) > cfg.max_pullback_days:
        return None

    current_close = latest["close"]
    pullback_pct = (first_leg_high - current_close) / (first_leg_high + 1e-9)
    if pullback_pct < cfg.min_pullback_pct or pullback_pct > cfg.max_pullback_pct:
        return None

    pullback_mean_vol = pullback_df["vol"].mean()
    if pullback_mean_vol / (impulse["impulse_vol"] + 1e-9) > cfg.max_pullback_vol_ratio:
        return None

    if current_close < impulse["impulse_close"] * (1 - cfg.max_close_below_breakout_close):
        return None

    if not pd.isna(latest["ma10"]) and current_close < latest["ma10"] * cfg.min_above_ma10_ratio:
        return None

    platform_high_before = impulse["platform_high_before"]
    if current_close < platform_high_before * cfg.min_above_platform_ratio:
        return None

    if current_close >= first_leg_high * 0.98:
        return None

    last2 = df.iloc[-2:]
    if ((last2["pct_chg"] > 7.0) & (last2["vol"] > last2["vma20"] * 2)).any():
        return None

    score = 0.0
    score += max(0, 20 - impulse["base_range_pct"] * 40)
    score += max(0, 20 - impulse["base_vol_cv"] * 20)
    score += min(20, impulse["impulse_vol_ratio"] * 4)
    score += max(0, 20 - pullback_pct * 100)
    score += max(0, 20 - (pullback_mean_vol / (impulse["impulse_vol"] + 1e-9)) * 20)

    return {
        "ts_code": ts_code,
        "trade_date": latest["trade_date"].strftime("%Y-%m-%d"),
        "close": round(float(current_close), 2),
        "impulse_date": impulse["trade_date"].strftime("%Y-%m-%d"),
        "impulse_pct_chg": round(float(impulse["impulse_pct_chg"]), 2),
        "impulse_vol_ratio": round(float(impulse["impulse_vol_ratio"]), 2),
        "first_leg_high": round(float(first_leg_high), 2),
        "pullback_pct": round(float(pullback_pct * 100), 2),
        "pullback_days": int(len(pullback_df)),
        "pullback_vol_ratio": round(float(pullback_mean_vol / (impulse["impulse_vol"] + 1e-9)), 2),
        "ma10": round(float(latest["ma10"]), 2) if not pd.isna(latest["ma10"]) else np.nan,
        "ma20": round(float(latest["ma20"]), 2) if not pd.isna(latest["ma20"]) else np.nan,
        "platform_high_before": round(float(platform_high_before), 2),
        "score": round(float(score), 2),
    }


def main():
    cfg = TypeNConfig()
    cache = ParquetCacheManager(cfg.cache_dir)

    codes = cache.list_codes()
    if not codes:
        print(f"缓存目录为空或不存在: {cfg.cache_dir}")
        return

    print(f"缓存目录: {cfg.cache_dir}")
    print(f"缓存股票数: {len(codes)}")

    pre_candidates = []
    results = []

    for idx, ts_code in enumerate(codes):
        try:
            df = cache.load(ts_code)
            if df.empty:
                continue

            if quick_pre_filter(df, cfg):
                pre_candidates.append(ts_code)

            if (idx + 1) % 500 == 0 or (idx + 1) == len(codes):
                print(f"[粗筛] {idx + 1}/{len(codes)}，候选数={len(pre_candidates)}")

        except Exception as e:
            print(f"[粗筛WARN] {ts_code} failed: {e}")

    print(f"粗筛后候选数: {len(pre_candidates)}")

    for idx, ts_code in enumerate(pre_candidates):
        try:
            df = cache.load(ts_code)
            hit = detect_type_n(df, ts_code, cfg)
            if hit is not None:
                results.append(hit)

            if (idx + 1) % 100 == 0 or (idx + 1) == len(pre_candidates):
                print(f"[精筛] {idx + 1}/{len(pre_candidates)}，命中数={len(results)}")

        except Exception as e:
            print(f"[精筛WARN] {ts_code} failed: {e}")

    if not results:
        print("No TypeN candidates found.")
        return

    out = pd.DataFrame(results).sort_values(
        by=["score", "pullback_pct", "impulse_vol_ratio"],
        ascending=[False, True, False]
    ).reset_index(drop=True)

    out.to_csv(cfg.output_csv, index=False, encoding="utf-8-sig")
    print(f"已输出: {cfg.output_csv}")
    print(out.head(50))


if __name__ == "__main__":
    main()