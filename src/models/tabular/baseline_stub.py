from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression

from src.models.base import BaseModel


class LogisticRegressionBaseline(BaseModel):
    """Minimal LogisticRegression wrapper as tabular baseline model."""
    model_name = "logistic_regression"

    def __init__(self, **kwargs: Any):
        self.model = LogisticRegression(**kwargs)

    def fit(self, X: Any, y: Any) -> "LogisticRegressionBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str) -> "LogisticRegressionBaseline":
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        with load_path.open("rb") as f:
            sklearn_model = pickle.load(f)

        instance = cls()
        instance.model = sklearn_model
        return instance
