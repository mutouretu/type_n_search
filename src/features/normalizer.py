from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler


class TabularNormalizer:
    """Minimal tabular feature normalizer using StandardScaler."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "TabularNormalizer":
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(X)

    def save(self, path: str) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: str) -> "TabularNormalizer":
        load_path = Path(path)
        with load_path.open("rb") as f:
            scaler = pickle.load(f)

        normalizer = cls()
        normalizer.scaler = scaler
        return normalizer
