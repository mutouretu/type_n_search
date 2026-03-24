from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class Trainer:
    """Minimal trainer for fit/evaluate/save workflow."""

    def __init__(self, model: Any, evaluator: Any, output_dir: str, model_name: Optional[str] = None):
        self.model = model
        self.evaluator = evaluator
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        sample_ids_valid: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Train model, evaluate on valid set, and save artifacts."""
        y_train_arr = np.asarray(y_train)
        y_valid_arr = np.asarray(y_valid)

        self.model.fit(X_train, y_train_arr)

        y_score = self._predict_score(X_valid)
        y_pred = (y_score >= 0.5).astype(int)

        metrics = self._evaluate(y_valid_arr, y_score, y_pred)

        model_path = self.output_dir / "model.pkl"
        self._save_model(model_path)
        self._save_model_meta(self.output_dir / "model_meta.json")

        metrics_path = self.output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        sample_ids = (
            np.asarray(sample_ids_valid).reshape(-1)
            if sample_ids_valid is not None
            else np.arange(len(y_valid_arr))
        )

        pred_df = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "y_true": y_valid_arr,
                "y_score": y_score,
                "y_pred": y_pred,
            }
        )
        pred_path = self.output_dir / "valid_predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        return {k: float(v) for k, v in metrics.items()}

    def _predict_score(self, X_valid: Any) -> np.ndarray:
        try:
            proba = np.asarray(self.model.predict_proba(X_valid))
            score = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.squeeze()
        except (AttributeError, NotImplementedError):
            score = np.asarray(self.model.predict(X_valid)).squeeze()

        return np.asarray(score, dtype=float).reshape(-1)

    def _evaluate(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        if self.evaluator is None:
            acc = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0
            return {"accuracy": acc}

        if hasattr(self.evaluator, "evaluate"):
            result = self.evaluator.evaluate(y_true, y_score)
            return self._to_float_dict(result)

        if callable(self.evaluator):
            result = self.evaluator(y_true, y_score)
            return self._to_float_dict(result)

        acc = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0
        return {"accuracy": acc}

    def _save_model(self, path: Path) -> None:
        if hasattr(self.model, "save"):
            self.model.save(str(path))
            return
        if hasattr(self.model, "save_model"):
            self.model.save_model(str(path))
            return

        with path.open("wb") as f:
            pickle.dump(self.model, f)

    def _to_float_dict(self, obj: Any) -> Dict[str, float]:
        if isinstance(obj, dict):
            return {str(k): float(v) for k, v in obj.items()}
        return {"metric": float(obj)}

    def _save_model_meta(self, path: Path) -> None:
        if self.model_name:
            name = str(self.model_name)
        elif hasattr(self.model, "model_name"):
            name = str(getattr(self.model, "model_name"))
        else:
            name = self.model.__class__.__name__ if self.model is not None else "unknown"

        with path.open("w", encoding="utf-8") as f:
            json.dump({"model_name": name}, f, ensure_ascii=False, indent=2)
