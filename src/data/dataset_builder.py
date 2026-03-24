from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.data.loader import DailyDataLoader, LabelLoader
from src.data.schema import SampleMeta, build_sample_id
from src.features.feature_builder_tabular import build_tabular_features
from src.features.indicators import add_basic_indicators
from src.features.window_builder import build_window_by_asof_date


class DatasetBuilder:
    """Build training dataset artifacts from labels and daily K-line files."""

    def __init__(
        self,
        labels_path: str = "data/labels/labels.csv",
        data_dir: str = "data/raw/daily",
        output_dir: str = "data/processed",
        window_size: int = 120,
        min_history: int = 80,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.15,
        sequence_cols: Optional[Iterable[str]] = None,
        save_sequence: bool = True,
    ):
        self.labels_path = labels_path
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.window_size = window_size
        self.min_history = min_history
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.sequence_cols = list(sequence_cols) if sequence_cols is not None else [
            "open",
            "high",
            "low",
            "close",
            "vol",
        ]
        self.save_sequence = save_sequence

        self.label_loader = LabelLoader(labels_path)
        self.daily_loader = DailyDataLoader(data_dir)

    def build(self) -> Dict[str, Any]:
        """Run end-to-end dataset build and save standard output artifacts."""
        labels_df = self.label_loader.load()
        labels_df = self._normalize_label_schema(labels_df)
        self._validate_labels_df(labels_df)

        labels_df["asof_date"] = pd.to_datetime(labels_df["asof_date"], errors="coerce")
        labels_df = (
            labels_df.dropna(subset=["asof_date", "ts_code"])
            .sort_values("asof_date")
            .reset_index(drop=True)
        )

        split_map = self._build_time_split(labels_df["asof_date"])

        meta_records: List[Dict[str, Any]] = []
        tabular_records: List[Dict[str, float]] = []
        y_values: List[int] = []
        seq_values: List[np.ndarray] = []

        for row in labels_df.itertuples(index=False):
            ts_code = str(getattr(row, "ts_code"))
            asof_date = pd.to_datetime(getattr(row, "asof_date"), errors="coerce")
            sample_id_raw = getattr(row, "sample_id", None)
            sample_id = sample_id_raw if isinstance(sample_id_raw, str) and sample_id_raw else ""
            if not sample_id:
                sample_id = build_sample_id(ts_code, asof_date.strftime("%Y-%m-%d"))

            try:
                daily_df = self.daily_loader.load_one(ts_code)
                self._validate_daily_df_core(daily_df, ts_code)
                daily_df = add_basic_indicators(daily_df)

                window_df = build_window_by_asof_date(
                    daily_df,
                    asof_date=asof_date,
                    window_size=self.window_size,
                    min_history=self.min_history,
                )
                if window_df is None:
                    print(f"[skip] {sample_id}: insufficient history/window")
                    continue

                tabular_feat = build_tabular_features(window_df)
                # Sequence features are intentionally MVP here.
                # If save_sequence=True and sequence construction fails, this sample is skipped entirely
                # to keep sample alignment consistent across meta/tabular/y/sequence.
                seq_feat = self._build_sequence_features(window_df) if self.save_sequence else None

                label = int(getattr(row, "label"))
                split = split_map.get(asof_date.normalize(), "train")
                meta = SampleMeta(
                    sample_id=sample_id,
                    ts_code=ts_code,
                    asof_date=asof_date.strftime("%Y-%m-%d"),
                    window_start=window_df["trade_date"].iloc[0].strftime("%Y-%m-%d"),
                    window_end=window_df["trade_date"].iloc[-1].strftime("%Y-%m-%d"),
                    label=label,
                    label_source=str(getattr(row, "label_source", "labels.csv")),
                    confidence=float(getattr(row, "confidence", 1.0)),
                    split=split,
                )

                meta_records.append(asdict(meta))
                tabular_records.append({"sample_id": sample_id, **tabular_feat})
                y_values.append(label)
                if self.save_sequence and seq_feat is not None:
                    seq_values.append(seq_feat)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {sample_id}: {exc}")

        if not meta_records:
            raise RuntimeError("No valid samples built from labels.")

        meta_df = pd.DataFrame(meta_records)
        tabular_df = pd.DataFrame(tabular_records)
        y_array = np.asarray(y_values, dtype=np.int64)

        self._validate_alignment(meta_df, tabular_df, y_array, seq_values)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        meta_path = self.output_dir / "sample_meta.parquet"
        tabular_path = self.output_dir / "X_tabular.parquet"
        y_path = self.output_dir / "y.npy"
        sequence_path = self.output_dir / "X_sequence.npy"

        meta_df.to_parquet(meta_path, index=False)
        tabular_df.to_parquet(tabular_path, index=False)
        np.save(y_path, y_array)

        result: Dict[str, Any] = {
            "sample_meta": str(meta_path),
            "X_tabular": str(tabular_path),
            "y": str(y_path),
            "num_samples": len(meta_df),
        }

        if self.save_sequence and seq_values:
            seq_array = np.stack(seq_values, axis=0)
            np.save(sequence_path, seq_array)
            result["X_sequence"] = str(sequence_path)

        return result

    def _normalize_label_schema(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize possible label column aliases to standard schema."""
        rename_map: Dict[str, str] = {}
        lower_map = {c.lower(): c for c in labels_df.columns}
        mapping = {
            "ts_code": ["ts_code", "code", "ticker"],
            "asof_date": ["asof_date", "trade_date", "date"],
            "label": ["label", "y", "target"],
        }

        for expected, candidates in mapping.items():
            for name in candidates:
                if name in lower_map:
                    rename_map[lower_map[name]] = expected
                    break

        out = labels_df.rename(columns=rename_map).copy()
        missing = [c for c in ["ts_code", "asof_date", "label"] if c not in out.columns]
        if missing:
            raise KeyError(f"labels missing required columns: {missing}")
        return out

    def _validate_labels_df(self, labels_df: pd.DataFrame) -> None:
        required = {"ts_code", "asof_date", "label"}
        missing = [c for c in required if c not in labels_df.columns]
        if missing:
            raise KeyError(f"labels missing required columns: {missing}")

    def _build_time_split(self, asof_series: pd.Series) -> Dict[pd.Timestamp, str]:
        """Split unique dates in chronological order into train/valid/test."""
        dates = sorted(pd.Series(asof_series).dt.normalize().dropna().unique().tolist())
        n = len(dates)

        if n == 0:
            return {}
        if n < 3:
            return {d: "train" for d in dates}

        n_train = min(n - 2, max(1, int(n * self.train_ratio)))
        n_valid = min(n - n_train - 1, max(1, int(n * self.valid_ratio)))

        train_dates = set(dates[:n_train])
        valid_dates = set(dates[n_train : n_train + n_valid])
        test_dates = set(dates[n_train + n_valid :])

        split_map: Dict[pd.Timestamp, str] = {}
        for d in dates:
            if d in train_dates:
                split_map[d] = "train"
            elif d in valid_dates:
                split_map[d] = "valid"
            elif d in test_dates:
                split_map[d] = "test"

        return split_map

    def _validate_daily_df_core(self, daily_df: pd.DataFrame, ts_code: str) -> None:
        required_cols = ["trade_date", "high", "low", "close", "vol"]
        missing = [c for c in required_cols if c not in daily_df.columns]
        if missing:
            raise KeyError(f"{ts_code} daily data missing required columns: {missing}")

    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Backward-compatible wrapper; use shared indicator implementation."""
        return add_basic_indicators(df)

    def _build_sequence_features(self, window_df: pd.DataFrame) -> np.ndarray:
        """Build MVP sequence matrix from selected numeric columns."""
        cols = [c for c in self.sequence_cols if c in window_df.columns]
        if not cols:
            raise ValueError("No sequence feature columns found in window data")

        seq_df = window_df[cols].copy().apply(pd.to_numeric, errors="coerce")
        seq_df = seq_df.ffill().bfill().fillna(0.0)
        return seq_df.to_numpy(dtype=np.float32)

    def _validate_alignment(
        self,
        meta_df: pd.DataFrame,
        tabular_df: pd.DataFrame,
        y_array: np.ndarray,
        seq_values: List[np.ndarray],
    ) -> None:
        if not (len(meta_df) == len(tabular_df) == len(y_array)):
            raise RuntimeError("Sample size mismatch among meta/tabular/y")

        if not meta_df["sample_id"].equals(tabular_df["sample_id"]):
            raise RuntimeError("sample_id mismatch between sample_meta and X_tabular")

        if self.save_sequence and len(seq_values) != len(meta_df):
            raise RuntimeError("Sample size mismatch between sequence features and meta")
