from __future__ import annotations

import pandas as pd


def sort_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Sort prediction rows by score descending."""
    return df.sort_values("score", ascending=False).reset_index(drop=True)
