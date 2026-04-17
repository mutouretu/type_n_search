import pandas as pd

from src.review.scoring import sigmoid_boost_factor, sigmoid_decay_factor, sigmoid_rise_factor


def test_sigmoid_decay_factor_decreases_after_threshold() -> None:
    factors = sigmoid_decay_factor(pd.Series([0.2, 0.4, 0.6, None]), threshold=0.4, sharpness=20)

    assert factors.iloc[0] > factors.iloc[1] > factors.iloc[2]
    assert factors.iloc[1] == 0.5
    assert factors.iloc[3] == 1.0


def test_sigmoid_boost_factor_increases_after_threshold() -> None:
    factors = sigmoid_boost_factor(pd.Series([1.0, 1.8, 3.0, None]), threshold=1.8, sharpness=3, max_boost=0.2)

    assert factors.iloc[0] < factors.iloc[1] < factors.iloc[2]
    assert factors.iloc[1] == 1.1
    assert factors.iloc[2] <= 1.2
    assert factors.iloc[3] == 1.0


def test_sigmoid_rise_factor_handles_missing_values() -> None:
    factors = sigmoid_rise_factor(pd.Series([1.0, None]), threshold=1.8, sharpness=3, missing_value=0.25)

    assert factors.iloc[0] < 0.5
    assert factors.iloc[1] == 0.25
