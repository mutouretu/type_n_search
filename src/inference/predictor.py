from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from src.features.normalizer import TabularNormalizer
from src.models.factory import load_model


PathLike = Union[str, Path]


class TabularPredictor:
    """Minimal tabular predictor wrapper with model + feature normalizer."""

    def __init__(self, model, normalizer: TabularNormalizer):
        self.model = model
        self.normalizer = normalizer

    @classmethod
    def from_dir(cls, model_dir: PathLike) -> "TabularPredictor":
        model_dir_path = Path(model_dir)
        model_path = model_dir_path / "model.pkl"
        model_meta_path = model_dir_path / "model_meta.json"
        normalizer_path = model_dir_path / "normalizer.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")
        if not model_meta_path.exists():
            raise FileNotFoundError(f"Missing model metadata file: {model_meta_path}")
        if not normalizer_path.exists():
            raise FileNotFoundError(f"Missing normalizer file: {normalizer_path}")

        with model_meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        model_name = str(meta.get("model_name", "")).strip()
        if not model_name:
            raise ValueError("Invalid model metadata: missing model_name")

        model = load_model(model_name=model_name, model_path=str(model_path))
        normalizer = TabularNormalizer.load(str(normalizer_path))

        return cls(model=model, normalizer=normalizer)

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        """Predict one-dimensional score array from pure tabular feature dataframe."""
        if X_df is None or X_df.empty:
            raise ValueError("X_df is empty")
        forbidden_cols = {"sample_id", "ts_code", "asof_date"}
        overlap = forbidden_cols.intersection({str(c) for c in X_df.columns})
        if overlap:
            raise ValueError(f"X_df contains non-feature columns: {sorted(overlap)}")

        X_num = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X_arr = X_num.to_numpy(dtype=float)
        X_scaled = self.normalizer.transform(X_arr)

        try:
            proba = np.asarray(self.model.predict_proba(X_scaled))
            scores = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.squeeze()
        except (AttributeError, NotImplementedError):
            scores = np.asarray(self.model.predict(X_scaled)).squeeze()

        return np.asarray(scores, dtype=float).reshape(-1)
