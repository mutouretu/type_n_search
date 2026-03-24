from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.feature_builder_tabular import build_tabular_features


def test_build_tabular_features_basic():
    n = 120
    x = np.linspace(0, 1, n)
    close = 10 + x
    open_ = close * 0.998
    high = close * 1.01
    low = close * 0.99
    vol = 120000 + 20000 * np.sin(np.linspace(0, 3.14, n))

    window_df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "vol": vol,
        }
    )

    features = build_tabular_features(window_df)

    assert isinstance(features, dict)
    assert len(features) > 0
    assert all(isinstance(v, float) for v in features.values())
    assert all(np.isfinite(v) for v in features.values())
