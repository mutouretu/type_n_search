from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.normalizer import TabularNormalizer
from src.models.factory import build_model
from src.training.evaluator import Evaluator
from src.training.trainer import Trainer


def _load_yaml(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _load_processed_data(config: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    data_dir = Path(config.get("data_dir", "data/processed"))
    meta_path = Path(config.get("sample_meta_path", data_dir / "sample_meta.parquet"))
    tabular_path = Path(config.get("x_tabular_path", data_dir / "X_tabular.parquet"))
    y_path = Path(config.get("y_path", data_dir / "y.npy"))

    meta_df = pd.read_parquet(meta_path)
    tabular_df = pd.read_parquet(tabular_path)
    y_array = np.load(y_path)

    return meta_df, tabular_df, y_array


def _align_tabular_by_sample_id(meta_df: pd.DataFrame, tabular_df: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" not in meta_df.columns:
        raise KeyError("sample_meta is missing required column: sample_id")
    if "sample_id" not in tabular_df.columns:
        raise KeyError("X_tabular is missing required column: sample_id")

    missing_ids = set(meta_df["sample_id"]) - set(tabular_df["sample_id"])
    if missing_ids:
        raise KeyError(f"X_tabular missing sample_id(s): count={len(missing_ids)}")

    aligned = (
        tabular_df.set_index("sample_id")
        .reindex(meta_df["sample_id"])
        .reset_index(drop=True)
    )
    return aligned


def main(config_path: str = "configs/train.yaml") -> Dict[str, float]:
    config = _load_yaml(config_path)

    meta_df, tabular_df, y_array = _load_processed_data(config)
    if len(meta_df) != len(y_array):
        raise ValueError("sample_meta and y length mismatch")
    if "split" not in meta_df.columns:
        raise KeyError("sample_meta is missing required column: split")

    X_df = _align_tabular_by_sample_id(meta_df, tabular_df)
    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    train_mask = meta_df["split"].astype(str) == "train"
    valid_mask = meta_df["split"].astype(str) == "valid"
    test_mask = meta_df["split"].astype(str) == "test"

    if train_mask.sum() == 0:
        raise ValueError("No train samples found in sample_meta split column")
    if valid_mask.sum() == 0:
        # Fallback 1: use test split as validation if available.
        if test_mask.sum() > 0:
            valid_mask = test_mask
            print("[info] No valid split found. Fallback to test split for validation.")
        else:
            # Fallback 2: carve out a small tail from train as validation.
            train_indices = np.where(train_mask.to_numpy())[0]
            if len(train_indices) < 2:
                raise ValueError("Not enough samples to create fallback validation split")

            n_valid_fallback = max(1, int(round(len(train_indices) * 0.2)))
            if n_valid_fallback >= len(train_indices):
                n_valid_fallback = 1

            valid_indices = train_indices[-n_valid_fallback:]
            train_indices_new = train_indices[:-n_valid_fallback]
            if len(train_indices_new) == 0:
                raise ValueError("Fallback split left no samples for training")

            train_mask = pd.Series(False, index=meta_df.index)
            valid_mask = pd.Series(False, index=meta_df.index)
            train_mask.iloc[train_indices_new] = True
            valid_mask.iloc[valid_indices] = True
            print(
                "[info] No valid/test split found. "
                f"Fallback to train tail as validation: train={train_mask.sum()} valid={valid_mask.sum()}."
            )

    X_train = X_df.loc[train_mask].to_numpy(dtype=float)
    y_train = np.asarray(y_array)[train_mask.to_numpy()]
    X_valid = X_df.loc[valid_mask].to_numpy(dtype=float)
    y_valid = np.asarray(y_array)[valid_mask.to_numpy()]
    sample_ids_valid = meta_df.loc[valid_mask, "sample_id"].to_numpy()

    output_dir = Path(config.get("output_dir", "artifacts/train"))

    normalizer = TabularNormalizer()
    X_train = normalizer.fit_transform(X_train)
    X_valid = normalizer.transform(X_valid)
    normalizer.save(str(output_dir / "normalizer.pkl"))

    model_name = str(config.get("model_name", "logistic_regression"))
    model_params = config.get("model_params", {})
    if not isinstance(model_params, dict):
        model_params = {}
    if model_name.strip().lower() == "logistic_regression" and "max_iter" not in model_params:
        model_params["max_iter"] = 1000

    model = build_model(model_name, **model_params)
    evaluator = Evaluator(threshold=float(config.get("threshold", 0.5)))
    trainer = Trainer(
        model=model,
        evaluator=evaluator,
        output_dir=str(output_dir),
        model_name=model_name,
    )

    metrics = trainer.run(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        sample_ids_valid=sample_ids_valid,
    )

    print("Training finished. Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tabular baseline model.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to train config yaml")
    args = parser.parse_args()
    main(config_path=args.config)
