"""Generate richer mock daily data and labels for baseline model comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
RAW_DAILY_DIR = ROOT / "data" / "raw" / "daily"
LABELS_PATH = ROOT / "data" / "labels" / "labels.csv"

SEED = 20260324
N_DAYS = 320
SAMPLES_PER_STOCK = 6

PATTERN_A = "sideways_breakout"
PATTERN_B = "random_chop"
PATTERN_C = "slow_uptrend"
PATTERN_D = "high_vol_noise"
PATTERN_E = "fake_breakout"
PATTERNS = [PATTERN_A, PATTERN_B, PATTERN_C, PATTERN_D, PATTERN_E]


def build_sample_id(ts_code: str, asof_date: str) -> str:
    dt = pd.to_datetime(asof_date, errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"invalid asof_date: {asof_date}")
    return f"{ts_code}_{dt.strftime('%Y-%m-%d')}"


def _base_dates(n_days: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=n_days)


def _gen_close_sideways_breakout(rng: np.random.Generator, n_days: int, start: float) -> np.ndarray:
    close = np.empty(n_days, dtype=float)
    close[0] = start

    for i in range(1, n_days):
        if i < 120:
            ret = rng.normal(0.0001, 0.0045)
        elif i < 220:
            ret = rng.normal(0.0, 0.0022)
        elif i < 255:
            ret = rng.normal(0.0050, 0.0075)
        else:
            ret = rng.normal(0.0010, 0.0065)
        close[i] = max(close[i - 1] * (1.0 + ret), 1.0)
    return close


def _gen_close_random_chop(rng: np.random.Generator, n_days: int, start: float) -> np.ndarray:
    close = np.empty(n_days, dtype=float)
    close[0] = start
    for i in range(1, n_days):
        ret = rng.normal(0.0002, 0.0140)
        close[i] = max(close[i - 1] * (1.0 + ret), 1.0)
    return close


def _gen_close_slow_uptrend(rng: np.random.Generator, n_days: int, start: float) -> np.ndarray:
    close = np.empty(n_days, dtype=float)
    close[0] = start
    for i in range(1, n_days):
        ret = rng.normal(0.0012, 0.0050)
        close[i] = max(close[i - 1] * (1.0 + ret), 1.0)
    return close


def _gen_close_high_vol_noise(rng: np.random.Generator, n_days: int, start: float) -> np.ndarray:
    close = np.empty(n_days, dtype=float)
    close[0] = start
    for i in range(1, n_days):
        shock = rng.normal(0.0, 0.0250)
        if rng.random() < 0.06:
            shock += rng.normal(0.0, 0.05)
        close[i] = max(close[i - 1] * (1.0 + shock), 1.0)
    return close


def _gen_close_fake_breakout(rng: np.random.Generator, n_days: int, start: float) -> np.ndarray:
    close = np.empty(n_days, dtype=float)
    close[0] = start
    for i in range(1, n_days):
        if i < 200:
            ret = rng.normal(0.0001, 0.0040)
        elif i < 235:
            ret = rng.normal(0.0040, 0.0080)
        elif i < 280:
            ret = rng.normal(-0.0036, 0.0100)
        else:
            ret = rng.normal(-0.0004, 0.0070)
        close[i] = max(close[i - 1] * (1.0 + ret), 1.0)
    return close


def _gen_close_by_pattern(
    rng: np.random.Generator,
    pattern: str,
    n_days: int,
    start: float,
) -> np.ndarray:
    if pattern == PATTERN_A:
        return _gen_close_sideways_breakout(rng, n_days, start)
    if pattern == PATTERN_B:
        return _gen_close_random_chop(rng, n_days, start)
    if pattern == PATTERN_C:
        return _gen_close_slow_uptrend(rng, n_days, start)
    if pattern == PATTERN_D:
        return _gen_close_high_vol_noise(rng, n_days, start)
    if pattern == PATTERN_E:
        return _gen_close_fake_breakout(rng, n_days, start)
    raise ValueError(f"unknown pattern: {pattern}")


def _gen_volume(rng: np.random.Generator, pattern: str, n_days: int, base_vol: float) -> np.ndarray:
    vol = rng.normal(loc=base_vol, scale=base_vol * 0.12, size=n_days)
    vol = np.maximum(vol, base_vol * 0.25)

    if pattern == PATTERN_A:
        vol[120:220] *= rng.uniform(0.65, 0.78)
        vol[220:255] *= rng.uniform(1.9, 2.8)
    elif pattern == PATTERN_E:
        vol[200:235] *= rng.uniform(1.5, 2.2)
        vol[235:280] *= rng.uniform(0.95, 1.25)
    elif pattern == PATTERN_D:
        burst_days = rng.choice(np.arange(30, n_days - 30), size=10, replace=False)
        vol[burst_days] *= rng.uniform(1.8, 3.0)

    return vol.astype(float)


def _build_one_stock(
    ts_code: str,
    pattern: str,
    rng: np.random.Generator,
    n_days: int,
) -> pd.DataFrame:
    trade_dates = _base_dates(n_days)
    start_price = float(rng.uniform(6.0, 28.0))
    close = _gen_close_by_pattern(rng, pattern, n_days=n_days, start=start_price)

    open_ = close * (1.0 + rng.normal(0.0, 0.004, size=n_days))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.018, size=n_days))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.018, size=n_days))

    base_vol = float(rng.uniform(1.2e5, 4.2e5))
    vol = _gen_volume(rng, pattern=pattern, n_days=n_days, base_vol=base_vol)
    amount = vol * close * 100.0

    pct_chg = pd.Series(close).pct_change().fillna(0.0).to_numpy(dtype=float)

    df = pd.DataFrame(
        {
            "trade_date": trade_dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "vol": vol,
            "amount": amount,
            "pct_chg": pct_chg,
        }
    )
    return df


def _rolling_features_for_label(df: pd.DataFrame, idx: int) -> dict[str, float]:
    if idx < 160:
        return {
            "r10": 0.0,
            "r5": 0.0,
            "v_spike": 1.0,
            "range60": 0.0,
            "dist20": 0.0,
        }

    win = df.iloc[idx - 119 : idx + 1].copy()
    close = win["close"].to_numpy(dtype=float)
    vol = win["vol"].to_numpy(dtype=float)
    high = win["high"].to_numpy(dtype=float)
    low = win["low"].to_numpy(dtype=float)

    r10 = close[-1] / max(close[-11], 1e-8) - 1.0
    r5 = close[-1] / max(close[-6], 1e-8) - 1.0
    v20 = float(np.mean(vol[-20:]))
    v_last = float(vol[-1])
    v_spike = v_last / max(v20, 1e-8)
    range60 = float(np.max(high[-60:]) / max(np.min(low[-60:]), 1e-8) - 1.0)
    dist20 = float(close[-1] / max(np.max(high[-20:]), 1e-8) - 1.0)

    return {
        "r10": r10,
        "r5": r5,
        "v_spike": v_spike,
        "range60": range60,
        "dist20": dist20,
    }


def _choose_label(
    rng: np.random.Generator,
    pattern: str,
    feats: dict[str, float],
    idx: int,
) -> tuple[int, float]:
    r10 = feats["r10"]
    r5 = feats["r5"]
    v_spike = feats["v_spike"]
    range60 = feats["range60"]

    score = 0.0
    if pattern == PATTERN_A:
        score += 2.2
        if 220 <= idx <= 265:
            score += 1.2
    elif pattern == PATTERN_E:
        score -= 1.0
    elif pattern == PATTERN_D:
        score -= 0.8
    elif pattern == PATTERN_C:
        score += 0.8

    score += 6.0 * r10 + 3.0 * r5
    score += 0.9 * (v_spike - 1.0)
    score -= 1.2 * max(range60 - 0.18, 0.0)
    score += rng.normal(0.0, 0.55)

    label = 1 if score > 0.8 else 0
    confidence = 0.95 if abs(score - 0.8) > 0.9 else 0.90
    return label, confidence


def _make_stock_codes() -> list[str]:
    sz = [f"{i:06d}.SZ" for i in range(1, 21)]
    sh = [f"{600000 + i:06d}.SH" for i in range(1, 21)]
    return sz + sh


def _assign_patterns(codes: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for i, code in enumerate(codes):
        out[code] = PATTERNS[i % len(PATTERNS)]
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)

    RAW_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)

    stock_codes = _make_stock_codes()
    pattern_map = _assign_patterns(stock_codes)

    label_rows: list[dict[str, object]] = []

    for ts_code in stock_codes:
        pattern = pattern_map[ts_code]
        stock_seed = int(rng.integers(1, 10_000_000))
        stock_rng = np.random.default_rng(stock_seed)

        daily_df = _build_one_stock(
            ts_code=ts_code,
            pattern=pattern,
            rng=stock_rng,
            n_days=N_DAYS,
        )

        out_path = RAW_DAILY_DIR / f"{ts_code}.parquet"
        daily_df.to_parquet(out_path, index=False)

        candidate_indices = np.arange(170, N_DAYS - 5)

        if pattern == PATTERN_A:
            focus = candidate_indices[(candidate_indices >= 220) & (candidate_indices <= 275)]
            other = candidate_indices[(candidate_indices < 220) | (candidate_indices > 275)]
            n_focus = min(5, len(focus))
            picked_focus = stock_rng.choice(focus, size=n_focus, replace=False)
            picked_other = stock_rng.choice(other, size=SAMPLES_PER_STOCK - n_focus, replace=False)
            picked_idx = np.concatenate([picked_focus, picked_other])
        else:
            picked_idx = stock_rng.choice(candidate_indices, size=SAMPLES_PER_STOCK, replace=False)

        picked_idx = np.sort(picked_idx)

        for idx in picked_idx:
            asof_date = pd.to_datetime(daily_df.iloc[int(idx)]["trade_date"]).strftime("%Y-%m-%d")
            feats = _rolling_features_for_label(daily_df, int(idx))
            label, confidence = _choose_label(stock_rng, pattern=pattern, feats=feats, idx=int(idx))
            label_rows.append(
                {
                    "sample_id": build_sample_id(ts_code=ts_code, asof_date=asof_date),
                    "ts_code": ts_code,
                    "asof_date": asof_date,
                    "label": int(label),
                    "label_source": "mock",
                    "confidence": float(confidence),
                }
            )

    labels_df = pd.DataFrame(label_rows)

    # Keep >=200 samples while avoiding extreme class imbalance.
    pos_df = labels_df[labels_df["label"] == 1]
    neg_df = labels_df[labels_df["label"] == 0]
    if min(len(pos_df), len(neg_df)) > 0:
        ratio = max(len(pos_df), len(neg_df)) / min(len(pos_df), len(neg_df))
        if ratio > 1.8:
            if len(pos_df) > len(neg_df):
                pos_df = pos_df.sample(n=int(len(neg_df) * 1.6), random_state=SEED)
            else:
                neg_df = neg_df.sample(n=int(len(pos_df) * 1.6), random_state=SEED + 1)
            labels_df = pd.concat([pos_df, neg_df], axis=0).sample(frac=1.0, random_state=SEED + 2)

    labels_df = labels_df.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
    labels_df["asof_date"] = pd.to_datetime(labels_df["asof_date"], errors="coerce")
    labels_df = labels_df.dropna(subset=["asof_date"]).sort_values(["asof_date", "ts_code"]).reset_index(drop=True)
    labels_df["asof_date"] = labels_df["asof_date"].dt.strftime("%Y-%m-%d")

    LABELS_PATH.write_text(labels_df.to_csv(index=False), encoding="utf-8")

    pos_n = int((labels_df["label"] == 1).sum())
    neg_n = int((labels_df["label"] == 0).sum())

    print(f"generated stocks: {len(stock_codes)}")
    print(f"generated labels: {len(labels_df)}")
    print(f"label distribution: pos={pos_n}, neg={neg_n}")
    print(f"daily dir: {RAW_DAILY_DIR}")
    print(f"labels path: {LABELS_PATH}")


if __name__ == "__main__":
    main()
