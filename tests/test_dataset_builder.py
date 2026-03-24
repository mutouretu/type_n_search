from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.dataset_builder import DatasetBuilder


def _make_daily(n: int = 220) -> pd.DataFrame:
    close = 10 + np.linspace(0, 1, n)
    open_ = close * 0.999
    high = close * 1.01
    low = close * 0.99
    vol = np.full(n, 120000.0)
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "vol": vol,
            "amount": vol * close,
            "pct_chg": pd.Series(close).pct_change().fillna(0.0),
        }
    )


def test_build_dataset_minimal(tmp_path):
    labels_path = tmp_path / "labels.csv"
    data_dir = tmp_path / "daily"
    out_dir = tmp_path / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)

    _make_daily().to_parquet(data_dir / "000001.SZ.parquet", index=False)
    _make_daily().to_parquet(data_dir / "000002.SZ.parquet", index=False)

    labels = pd.DataFrame(
        {
            "sample_id": ["000001.SZ_2024-06-30", "000002.SZ_2024-06-30"],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "asof_date": ["2024-06-30", "2024-06-30"],
            "label": [1, 0],
            "label_source": ["test", "test"],
            "confidence": [1.0, 1.0],
        }
    )
    labels.to_csv(labels_path, index=False)

    builder = DatasetBuilder(
        labels_path=str(labels_path),
        data_dir=str(data_dir),
        output_dir=str(out_dir),
        window_size=120,
        min_history=80,
        save_sequence=False,
    )
    result = builder.build()

    meta_df = pd.read_parquet(result["sample_meta"])
    x_df = pd.read_parquet(result["X_tabular"])
    y = np.load(result["y"])

    assert len(meta_df) > 0
    assert len(x_df) > 0
    assert len(y) > 0
    assert len(meta_df) == len(x_df) == len(y)
