from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.features.normalizer import TabularNormalizer
from src.inference.predictor import TabularPredictor
from src.models.tabular.baseline_stub import LogisticRegressionBaseline


def _build_model_artifacts(model_dir):
    X = np.array(
        [
            [0.1, 1.0],
            [0.2, 1.1],
            [1.0, 0.2],
            [1.1, 0.1],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1], dtype=int)

    normalizer = TabularNormalizer()
    Xn = normalizer.fit_transform(X)

    model = LogisticRegressionBaseline(max_iter=200)
    model.fit(Xn, y)

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir / "model.pkl"))
    normalizer.save(str(model_dir / "normalizer.pkl"))
    (model_dir / "model_meta.json").write_text(
        json.dumps({"model_name": "logistic_regression"}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_predictor_load_success(tmp_path):
    model_dir = tmp_path / "model_artifacts"
    _build_model_artifacts(model_dir)

    predictor = TabularPredictor.from_dir(model_dir)

    assert predictor is not None


def test_predictor_reject_non_feature_columns(tmp_path):
    model_dir = tmp_path / "model_artifacts"
    _build_model_artifacts(model_dir)
    predictor = TabularPredictor.from_dir(model_dir)

    bad_df = pd.DataFrame(
        {
            "sample_id": ["a"],
            "ts_code": ["000001.SZ"],
            "f1": [0.3],
            "f2": [0.9],
        }
    )

    with pytest.raises(ValueError):
        predictor.predict_proba(bad_df)


def test_predictor_empty_input(tmp_path):
    model_dir = tmp_path / "model_artifacts"
    _build_model_artifacts(model_dir)
    predictor = TabularPredictor.from_dir(model_dir)

    with pytest.raises(ValueError):
        predictor.predict_proba(pd.DataFrame())
