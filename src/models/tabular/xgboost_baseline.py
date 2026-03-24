from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from xgboost import XGBClassifier

from src.models.base import BaseModel


class XGBoostBaseline(BaseModel):
    """Minimal XGBoost wrapper as tabular baseline model."""

    model_name = "xgboost"

    def __init__(self, **kwargs: Any):
        self.model = XGBClassifier(**kwargs)

    def fit(self, X: Any, y: Any) -> "XGBoostBaseline":
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
    def load(cls, path: str) -> "XGBoostBaseline":
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        with load_path.open("rb") as f:
            xgb_model = pickle.load(f)

        instance = cls()
        instance.model = xgb_model
        return instance
