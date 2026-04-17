import pandas as pd

from src.review.scoring import sigmoid_decay_factor


def test_sigmoid_decay_factor_decreases_after_threshold() -> None:
    factors = sigmoid_decay_factor(pd.Series([0.2, 0.4, 0.6, None]), threshold=0.4, sharpness=20)

    assert factors.iloc[0] > factors.iloc[1] > factors.iloc[2]
    assert factors.iloc[1] == 0.5
    assert factors.iloc[3] == 1.0
