from __future__ import annotations

from typing import Dict

import numpy as np

from src.training.metrics import compute_binary_metrics


class Evaluator:
    """Simple evaluator wrapper aligned with Trainer's y_true/y_score interface."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def evaluate(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
        return compute_binary_metrics(y_true=y_true, y_score=y_score, threshold=self.threshold)
