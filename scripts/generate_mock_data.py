from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path('.')
RAW_DAILY_DIR = ROOT / 'data' / 'raw' / 'daily'


def _make_sideways_breakout_series(rng: np.random.Generator, n: int, start_price: float) -> np.ndarray:
    close = np.empty(n, dtype=float)

    # Stage 1: gentle drift
    close[:100] = start_price + np.cumsum(rng.normal(0.002, 0.03, size=100))

    # Stage 2: sideways base
    base_center = float(close[99])
    close[100:180] = base_center + rng.normal(0.0, 0.06, size=80)

    # Stage 3: breakout leg
    close[180:200] = np.linspace(base_center * 1.01, base_center * 1.18, 20)

    # Stage 4: hold with mild trend
    close[200:] = close[199] + np.cumsum(rng.normal(0.01, 0.04, size=n - 200))

    return np.maximum(close, 1.0)


def _make_random_walk_series(rng: np.random.Generator, n: int, start_price: float) -> np.ndarray:
    steps = rng.normal(0.0, 0.08, size=n)
    close = start_price + np.cumsum(steps)
    return np.maximum(close, 1.0)


def _build_one_stock(ts_code: str, seed: int, n_days: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    trade_dates = pd.bdate_range('2024-05-01', periods=n_days)

    if ts_code in {'000001.SZ', '000002.SZ'}:
        start_price = 11.0 if ts_code == '000001.SZ' else 8.8
        close = _make_sideways_breakout_series(rng, n_days, start_price=start_price)
    else:
        close = _make_random_walk_series(rng, n_days, start_price=13.5)

    open_ = close * (1.0 + rng.normal(0.0, 0.008, size=n_days))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.02, size=n_days))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.02, size=n_days))

    vol = rng.integers(90_000, 190_000, size=n_days).astype(float)
    if ts_code == '000001.SZ':
        vol[180:200] *= 2.8
    elif ts_code == '000002.SZ':
        vol[180:200] *= 2.5

    amount = vol * close * 100.0
    pct_chg = pd.Series(close).pct_change().fillna(0.0).to_numpy()

    return pd.DataFrame(
        {
            'trade_date': trade_dates,
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'vol': vol,
            'amount': amount,
            'pct_chg': pct_chg,
        }
    )


def main() -> None:
    RAW_DAILY_DIR.mkdir(parents=True, exist_ok=True)

    stocks = {
        '000001.SZ': 20250321,
        '000002.SZ': 20250322,
        '600000.SH': 20250323,
    }

    for ts_code, seed in stocks.items():
        df = _build_one_stock(ts_code=ts_code, seed=seed, n_days=220)
        out_path = RAW_DAILY_DIR / f'{ts_code}.parquet'
        df.to_parquet(out_path, index=False)
        print(f'saved: {out_path} rows={len(df)}')

    print('mock daily parquet generation done.')


if __name__ == '__main__':
    main()
