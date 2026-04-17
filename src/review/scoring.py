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


def sigmoid_rise_factor(values: pd.Series, *, threshold: float, sharpness: float, missing_value: float = 0.0) -> pd.Series:
    """Return a 0-1 factor that rises quickly after values exceed threshold."""
    numeric_values = pd.to_numeric(values, errors="coerce")

    def _calc(value: float) -> float:
        if pd.isna(value):
            return missing_value
        exponent = max(min(-sharpness * (float(value) - threshold), 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(exponent))

    return numeric_values.map(_calc)


def sigmoid_boost_factor(
    values: pd.Series,
    *,
    threshold: float,
    sharpness: float,
    max_boost: float,
    missing_value: float = 1.0,
) -> pd.Series:
    """Return a multiplicative boost in [1, 1 + max_boost]."""
    rise = sigmoid_rise_factor(values, threshold=threshold, sharpness=sharpness, missing_value=0.0)
    factor = 1.0 + float(max_boost) * rise
    return factor.where(pd.to_numeric(values, errors="coerce").notna(), missing_value)


__all__ = ["sigmoid_boost_factor", "sigmoid_decay_factor", "sigmoid_rise_factor"]
