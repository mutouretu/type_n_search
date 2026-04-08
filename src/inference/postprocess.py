from __future__ import annotations

from pathlib import Path

import pandas as pd


def sort_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Sort prediction rows by score descending."""
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def resolve_dated_output_path(output_path: str | Path, asof_date: str | None) -> Path:
    """Append asof_date to output filename stem when available."""
    path = Path(output_path)
    if not asof_date:
        return path
    return path.with_name(f"{path.stem}_{asof_date}{path.suffix}")
