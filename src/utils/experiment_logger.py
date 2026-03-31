from __future__ import annotations

from datetime import datetime
import json
import platform
from pathlib import Path
from typing import Any, Dict
import subprocess
import sys

import pandas as pd


SUMMARY_COLUMNS = [
    "exp_id",
    "created_at",
    "model_name",
    "dataset_name",
    "data_config_path",
    "train_config_path",
    "label_source",
    "split_method",
    "n_train",
    "n_valid",
    "n_test",
    "n_features",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "model_output_dir",
    "experiment_dir",
    "git_commit",
    "hostname",
    "python_version",
]


def load_experiment_summary(summary_path: str = "experiments/summary.csv") -> pd.DataFrame:
    path = Path(summary_path)
    if not path.exists():
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.read_csv(path)


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _infer_dataset_name(data_config_path: str | None, train_config_path: str) -> str:
    if data_config_path:
        stem = Path(data_config_path).stem
        if stem.startswith("data_"):
            return stem.replace("data_", "", 1)
        return stem
    return Path(train_config_path).stem


def _infer_split_method(meta_df: pd.DataFrame) -> str:
    if not {"split", "asof_date"}.issubset(meta_df.columns):
        return "unknown"

    df = meta_df.copy()
    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce")
    if df["asof_date"].isna().all():
        return "unknown"

    by_split = {}
    for split in ["train", "valid", "test"]:
        d = df.loc[df["split"].astype(str) == split, "asof_date"]
        d = d.dropna()
        if d.empty:
            continue
        by_split[split] = (d.min(), d.max())

    if "train" in by_split and "valid" in by_split:
        if by_split["train"][1] > by_split["valid"][0]:
            return "unknown"
    if "valid" in by_split and "test" in by_split:
        if by_split["valid"][1] > by_split["test"][0]:
            return "unknown"
    if by_split:
        return "time_by_asof_date"
    return "unknown"


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:  # noqa: BLE001
        return None


def _to_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def log_training_experiment(
    *,
    train_config: Dict[str, Any],
    train_config_path: str,
    data_config: Dict[str, Any] | None,
    data_config_path: str | None,
    model_name: str,
    metrics: Dict[str, Any],
    meta_df: pd.DataFrame,
    tabular_df: pd.DataFrame,
    model_output_dir: str,
    processed_data_dir: str,
    experiments_root: str = "experiments",
) -> Dict[str, Any]:
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exp_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{exp_ts}_{Path(train_config_path).stem}"

    experiments_dir = Path(experiments_root)
    exp_dir = experiments_dir / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = _infer_dataset_name(data_config_path, train_config_path)
    label_file = (data_config or {}).get("labels_path")
    split_method = _infer_split_method(meta_df)

    split_series = meta_df.get("split", pd.Series(dtype=str)).astype(str)
    label_series = pd.to_numeric(meta_df.get("label", pd.Series(dtype=float)), errors="coerce")
    asof_series = pd.to_datetime(meta_df.get("asof_date", pd.Series(dtype=str)), errors="coerce")

    n_train = int((split_series == "train").sum())
    n_valid = int((split_series == "valid").sum())
    n_test = int((split_series == "test").sum())

    feat_excluded = [c for c in ["sample_id"] if c in tabular_df.columns]
    feature_columns = [c for c in tabular_df.columns if c not in feat_excluded]
    n_features = len(feature_columns)

    def _pos_ratio(mask: pd.Series | None = None) -> float | None:
        s = label_series
        if mask is not None:
            s = s[mask]
        s = s.dropna()
        if s.empty:
            return None
        return float((s == 1).mean())

    label_sources = []
    if "label_source" in meta_df.columns:
        label_sources = sorted(meta_df["label_source"].dropna().astype(str).unique().tolist())
    label_source = "|".join(label_sources) if label_sources else "unknown"

    has_sequence = (Path(processed_data_dir) / "X_sequence.npy").exists()
    has_tabular = (Path(processed_data_dir) / "X_tabular.parquet").exists()

    metrics_clean = {k: _safe_float(v) for k, v in metrics.items()}
    _to_json(exp_dir / "metrics.json", metrics_clean)

    config_snapshot = {
        "train_config_path": train_config_path,
        "data_config_path": data_config_path,
        "train_config": train_config,
        "data_config": data_config or {},
    }
    _to_json(exp_dir / "config_snapshot.json", config_snapshot)

    data_snapshot = {
        "n_samples": int(len(meta_df)),
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "positive_ratio_all": _pos_ratio(),
        "positive_ratio_train": _pos_ratio(split_series == "train"),
        "positive_ratio_valid": _pos_ratio(split_series == "valid"),
        "positive_ratio_test": _pos_ratio(split_series == "test"),
        "date_min": None if asof_series.dropna().empty else str(asof_series.min().date()),
        "date_max": None if asof_series.dropna().empty else str(asof_series.max().date()),
        "n_unique_ts_code": int(meta_df["ts_code"].nunique()) if "ts_code" in meta_df.columns else None,
    }
    _to_json(exp_dir / "data_snapshot.json", data_snapshot)

    feature_snapshot = {
        "n_features": n_features,
        "feature_columns": feature_columns,
        "has_sequence_features": bool(has_sequence),
        "has_tabular_features": bool(has_tabular),
        "excluded_non_feature_columns": feat_excluded,
    }
    _to_json(exp_dir / "feature_snapshot.json", feature_snapshot)

    experiment_meta = {
        "exp_id": exp_id,
        "created_at": created_at,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "data_config_path": data_config_path,
        "train_config_path": train_config_path,
        "model_output_dir": model_output_dir,
        "processed_data_dir": processed_data_dir,
        "label_file": label_file,
        "split_method": split_method,
        "notes": "",
    }
    _to_json(exp_dir / "experiment.json", experiment_meta)
    (exp_dir / "notes.txt").write_text("auto-generated experiment record\n", encoding="utf-8")

    summary_path = experiments_dir / "summary.csv"
    summary_df = load_experiment_summary(str(summary_path))
    summary_row = {
        "exp_id": exp_id,
        "created_at": created_at,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "data_config_path": data_config_path,
        "train_config_path": train_config_path,
        "label_source": label_source,
        "split_method": split_method,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "n_features": n_features,
        "accuracy": metrics_clean.get("accuracy"),
        "precision": metrics_clean.get("precision"),
        "recall": metrics_clean.get("recall"),
        "f1": metrics_clean.get("f1"),
        "auc": metrics_clean.get("auc"),
        "model_output_dir": model_output_dir,
        "experiment_dir": str(exp_dir),
        "git_commit": _git_commit(),
        "hostname": platform.node(),
        "python_version": sys.version.split()[0],
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
    summary_df = summary_df.reindex(columns=SUMMARY_COLUMNS)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    return {
        "exp_id": exp_id,
        "experiment_dir": str(exp_dir),
        "summary_path": str(summary_path),
    }

