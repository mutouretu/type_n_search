from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def _to_int_dict(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.to_dict().items()}


def inspect_dataset(
    sample_meta_path: Path,
    x_tabular_path: Path,
    y_path: Path,
    top_n: int,
) -> dict[str, Any]:
    """Inspect processed dataset artifacts and return structured diagnostics."""
    _require_file(sample_meta_path)
    _require_file(x_tabular_path)
    _require_file(y_path)

    sample_meta = pd.read_parquet(sample_meta_path)
    x_tabular = pd.read_parquet(x_tabular_path)
    y = np.load(y_path)
    y_series = pd.Series(y, name="label")

    shape = {
        "sample_meta_rows": int(len(sample_meta)),
        "x_tabular_rows": int(len(x_tabular)),
        "y_length": int(len(y)),
    }
    shape["is_consistent"] = (
        shape["sample_meta_rows"] == shape["x_tabular_rows"] == shape["y_length"]
    )
    if not shape["is_consistent"]:
        raise ValueError(
            "Shape mismatch: "
            f"sample_meta={shape['sample_meta_rows']}, "
            f"x_tabular={shape['x_tabular_rows']}, "
            f"y={shape['y_length']}"
        )

    if "sample_id" not in sample_meta.columns:
        raise KeyError("sample_meta is missing required column: sample_id")
    if "sample_id" not in x_tabular.columns:
        raise KeyError("X_tabular is missing required column: sample_id")

    x_sample_id_unique = bool(x_tabular["sample_id"].is_unique)
    sample_id_aligned = bool(sample_meta["sample_id"].equals(x_tabular["sample_id"]))

    feature_cols = [c for c in x_tabular.columns if c != "sample_id"]
    if not feature_cols:
        raise ValueError("X_tabular has no feature columns (except sample_id)")

    x_features = x_tabular[feature_cols].apply(pd.to_numeric, errors="coerce")

    label_distribution = _to_int_dict(y_series.value_counts(dropna=False).sort_index())
    pos_count = int((y_series == 1).sum())
    neg_count = int((y_series == 0).sum())
    pos_ratio = float(pos_count / len(y_series)) if len(y_series) > 0 else float("nan")

    split_distribution: dict[str, Any] = {
        "available": "split" in sample_meta.columns,
        "counts": {},
        "label_distribution_by_split": {},
    }
    if "split" in sample_meta.columns:
        split_counts = sample_meta["split"].fillna("<NA>").value_counts(dropna=False).sort_index()
        split_distribution["counts"] = _to_int_dict(split_counts)

        split_label_dist: dict[str, dict[str, int]] = {}
        temp = sample_meta[["split"]].copy()
        temp["label"] = y_series.values
        for split_name, grp in temp.groupby(temp["split"].fillna("<NA>"), dropna=False):
            split_label_dist[str(split_name)] = _to_int_dict(
                grp["label"].value_counts(dropna=False).sort_index()
            )
        split_distribution["label_distribution_by_split"] = split_label_dist

    missing_count = x_features.isna().sum()
    missing_ratio = missing_count / len(x_features)
    missing_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "missing_count": missing_count.values.astype(int),
            "missing_ratio": missing_ratio.values.astype(float),
        }
    ).sort_values("missing_ratio", ascending=False)
    total_missing_count = int(missing_count.sum())
    total_cells = int(x_features.shape[0] * x_features.shape[1])
    global_missing_ratio = (
        float(total_missing_count / total_cells) if total_cells > 0 else float("nan")
    )

    unique_counts = x_features.nunique(dropna=False)
    constant_cols = sorted(unique_counts[unique_counts <= 1].index.tolist())

    variances = x_features.var(ddof=0)
    var_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "variance": variances.values.astype(float),
        }
    ).sort_values("variance", ascending=True)

    describe_df = x_features.describe(percentiles=[0.25, 0.5, 0.75]).T
    describe_cols = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    for c in describe_cols:
        if c not in describe_df.columns:
            describe_df[c] = np.nan
    describe_df = describe_df[describe_cols]
    describe_df = describe_df.fillna(0.0)

    result: dict[str, Any] = {
        "shape_check": shape,
        "split_distribution": split_distribution,
        "label_distribution": {
            "distribution": label_distribution,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "positive_ratio": pos_ratio,
        },
        "feature_info": {
            "num_features": int(len(feature_cols)),
            "feature_columns": feature_cols,
            "has_sample_id": "sample_id" in x_tabular.columns,
            "x_sample_id_unique": x_sample_id_unique,
            "sample_id_aligned": sample_id_aligned,
        },
        "missingness": {
            "global_missing_count": total_missing_count,
            "global_missing_ratio": global_missing_ratio,
            "per_feature": {
                row["feature"]: {
                    "missing_count": int(row["missing_count"]),
                    "missing_ratio": float(row["missing_ratio"]),
                }
                for _, row in missing_df.iterrows()
            },
            "top_missing": [
                {
                    "feature": str(row["feature"]),
                    "missing_count": int(row["missing_count"]),
                    "missing_ratio": float(row["missing_ratio"]),
                }
                for _, row in missing_df.head(top_n).iterrows()
            ],
        },
        "variance": {
            "constant_columns": constant_cols,
            "top_low_variance": [
                {
                    "feature": str(row["feature"]),
                    "variance": float(row["variance"]),
                }
                for _, row in var_df.head(top_n).iterrows()
            ],
        },
        "numeric_summary": {
            idx: {k: float(v) for k, v in row.items()}
            for idx, row in describe_df.iterrows()
        },
    }

    return result


def print_report(result: dict[str, Any], top_n: int) -> None:
    print("[Shape Check]")
    shape = result["shape_check"]
    print(f"sample_meta_rows: {shape['sample_meta_rows']}")
    print(f"x_tabular_rows  : {shape['x_tabular_rows']}")
    print(f"y_length        : {shape['y_length']}")
    print(f"consistent      : {shape['is_consistent']}")
    print()

    print("[Split Distribution]")
    split = result["split_distribution"]
    print(f"split_available: {split['available']}")
    if split["available"]:
        print(f"split_counts: {split['counts']}")
        print("split_label_distribution:")
        for k, v in split["label_distribution_by_split"].items():
            print(f"  {k}: {v}")
    print()

    print("[Label Distribution]")
    label = result["label_distribution"]
    print(f"distribution  : {label['distribution']}")
    print(f"positive      : {label['positive_count']}")
    print(f"negative      : {label['negative_count']}")
    print(f"positive_ratio: {label['positive_ratio']:.6f}")
    print()

    print("[Feature Info]")
    finfo = result["feature_info"]
    print(f"num_features       : {finfo['num_features']}")
    print(f"has_sample_id      : {finfo['has_sample_id']}")
    print(f"x_sample_id_unique : {finfo['x_sample_id_unique']}")
    print(f"sample_id_aligned  : {finfo['sample_id_aligned']}")
    print(f"feature_columns    : {finfo['feature_columns']}")
    print()

    print("[Feature Missingness]")
    print(
        f"global_missing_count={result['missingness']['global_missing_count']}, "
        f"global_missing_ratio={result['missingness']['global_missing_ratio']:.6f}"
    )
    for row in result["missingness"]["top_missing"][:top_n]:
        print(
            f"{row['feature']}: missing_count={row['missing_count']}, "
            f"missing_ratio={row['missing_ratio']:.6f}"
        )
    print()

    print("[Low Variance Features]")
    print(
        f"constant_columns ({len(result['variance']['constant_columns'])}): "
        f"{result['variance']['constant_columns']}"
    )
    for row in result["variance"]["top_low_variance"][:top_n]:
        print(f"{row['feature']}: variance={row['variance']:.12f}")
    print()

    print("[Numeric Summary]")
    num_summary = result["numeric_summary"]
    for i, (feat, stats) in enumerate(num_summary.items()):
        if i >= top_n:
            break
        print(
            f"{feat}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
            f"min={stats['min']:.6f}, p25={stats['25%']:.6f}, "
            f"p50={stats['50%']:.6f}, p75={stats['75%']:.6f}, max={stats['max']:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect processed dataset artifacts.")
    parser.add_argument(
        "--sample-meta-path",
        default="data/processed/sample_meta.parquet",
        help="Path to sample_meta.parquet",
    )
    parser.add_argument(
        "--x-tabular-path",
        default="data/processed/X_tabular.parquet",
        help="Path to X_tabular.parquet",
    )
    parser.add_argument(
        "--y-path",
        default="data/processed/y.npy",
        help="Path to y.npy",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save full analysis result as json",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top N features for missingness/variance summary",
    )
    args = parser.parse_args()

    result = inspect_dataset(
        sample_meta_path=Path(args.sample_meta_path),
        x_tabular_path=Path(args.x_tabular_path),
        y_path=Path(args.y_path),
        top_n=args.top_n,
    )
    print_report(result, top_n=args.top_n)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\njson saved to: {out_path}")


if __name__ == "__main__":
    main()
