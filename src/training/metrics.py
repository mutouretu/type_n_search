from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute common binary classification metrics from true labels and scores."""
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_score_arr = np.asarray(y_score).reshape(-1)

    if len(y_true_arr) != len(y_score_arr):
        raise ValueError("y_true and y_score must have the same length")

    y_pred = (y_score_arr >= threshold).astype(int)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "auc": float("nan"),
    }

    try:
        if np.unique(y_true_arr).size >= 2:
            metrics["auc"] = float(roc_auc_score(y_true_arr, y_score_arr))
    except Exception:  # noqa: BLE001
        metrics["auc"] = float("nan")

    return metrics
