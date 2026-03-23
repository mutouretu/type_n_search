from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Minimal model interface for training, inference, and persistence."""

    @abstractmethod
    def fit(self, X: Any, y: Any) -> "BaseModel":
        """Fit model with features and targets."""

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Predict labels or scores for input features."""

    def predict_proba(self, X: Any) -> Any:
        """Optional probability prediction interface."""
        raise NotImplementedError("predict_proba is not implemented for this model")

    def save(self, path: str) -> None:
        """Optional model persistence interface."""
        raise NotImplementedError("save is not implemented for this model")
