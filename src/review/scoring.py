from __future__ import annotations

import math

import pandas as pd


def sigmoid_decay_factor(values: pd.Series, *, threshold: float, sharpness: float, missing_value: float = 1.0) -> pd.Series:
    """Return a 0-1 factor that decays quickly after values exceed threshold."""
    numeric_values = pd.to_numeric(values, errors="coerce")

    def _calc(value: float) -> float:
        if pd.isna(value):
            return missing_value
        exponent = max(min(sharpness * (float(value) - threshold), 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(exponent))

    return numeric_values.map(_calc)


__all__ = ["sigmoid_decay_factor"]
