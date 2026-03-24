from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb

from src.models.base import BaseModel


class LightGBMBaseline(BaseModel):
    """Minimal LightGBM wrapper as tabular baseline model."""

    model_name = "lightgbm"

    def __init__(self, **kwargs: Any):
        self.model = lgb.LGBMClassifier(**kwargs)

    def fit(self, X: Any, y: Any) -> "LightGBMBaseline":
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
    def load(cls, path: str) -> "LightGBMBaseline":
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        with load_path.open("rb") as f:
            lgb_model = pickle.load(f)

        instance = cls()
        instance.model = lgb_model
        return instance
