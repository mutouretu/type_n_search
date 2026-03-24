from __future__ import annotations

from typing import Any

from src.models.base import BaseModel
from src.models.tabular.baseline_stub import LogisticRegressionBaseline


def build_model(name: str, **kwargs: Any) -> BaseModel:
    """Build model instance by name."""
    model_name = name.strip().lower()

    if model_name == "logistic_regression":
        return LogisticRegressionBaseline(**kwargs)

    raise ValueError(f"Unknown model name: {name}")


def load_model(model_name: str, model_path: str) -> BaseModel:
    """Load persisted model instance by registered model name."""
    name = model_name.strip().lower()

    if name == "logistic_regression":
        return LogisticRegressionBaseline.load(model_path)

    raise ValueError(f"Unknown model name: {model_name}")
