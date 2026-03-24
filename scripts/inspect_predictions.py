from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _score_summary(scores: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(scores, errors="coerce")
    return {
        "mean": _to_float(s.mean()),
        "std": _to_float(s.std(ddof=0)),
        "min": _to_float(s.min()),
        "max": _to_float(s.max()),
        "q25": _to_float(s.quantile(0.25)),
        "q50": _to_float(s.quantile(0.50)),
        "q75": _to_float(s.quantile(0.75)),
        "q90": _to_float(s.quantile(0.90)),
        "q95": _to_float(s.quantile(0.95)),
        "q99": _to_float(s.quantile(0.99)),
    }


def _pick_score_col(df: pd.DataFrame) -> str | None:
    if "y_score" in df.columns:
        return "y_score"
    if "score" in df.columns:
        return "score"
    return None


def _rows_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        cleaned: dict[str, Any] = {}
        for k, v in rec.items():
            if isinstance(v, (np.floating, np.integer)):
                cleaned[k] = v.item()
            else:
                cleaned[k] = v
        records.append(cleaned)
    return records


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NaN/Inf to None for strict JSON compatibility."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def _analyze_validation(df: pd.DataFrame, score_col: str, top_n: int, threshold: float) -> dict[str, Any]:
    if "y_true" not in df.columns:
        raise ValueError("Validation mode requires column: y_true")

    y_true = pd.to_numeric(df["y_true"], errors="coerce")
    y_score = pd.to_numeric(df[score_col], errors="coerce")

    if "y_pred" in df.columns:
        y_pred = pd.to_numeric(df["y_pred"], errors="coerce")
    else:
        y_pred = (y_score >= threshold).astype(int)

    valid_mask = y_true.notna() & y_score.notna() & y_pred.notna()
    y_true_v = y_true[valid_mask].astype(int)
    y_score_v = y_score[valid_mask].astype(float)
    y_pred_v = y_pred[valid_mask].astype(int)

    if len(y_true_v) == 0:
        raise ValueError("No valid rows for validation analysis after numeric conversion")

    metrics = {
        "accuracy": _to_float(accuracy_score(y_true_v, y_pred_v)),
        "precision": _to_float(precision_score(y_true_v, y_pred_v, zero_division=0)),
        "recall": _to_float(recall_score(y_true_v, y_pred_v, zero_division=0)),
        "f1": _to_float(f1_score(y_true_v, y_pred_v, zero_division=0)),
        "auc": float("nan"),
    }
    try:
        if y_true_v.nunique() >= 2:
            metrics["auc"] = _to_float(roc_auc_score(y_true_v, y_score_v))
    except Exception:
        metrics["auc"] = float("nan")
    warnings: list[str] = []
    if y_true_v.nunique() < 2:
        warnings.append("validation set contains only one class; auc may be nan")

    work_df = df.copy()
    work_df["_y_true"] = y_true
    work_df["_y_score"] = y_score
    work_df["_y_pred"] = y_pred

    base_cols = [c for c in ["sample_id", "y_true", score_col, "y_pred"] if c in work_df.columns]
    if "y_true" in base_cols:
        base_cols[base_cols.index("y_true")] = "_y_true"
    if score_col in base_cols:
        base_cols[base_cols.index(score_col)] = "_y_score"
    if "y_pred" in base_cols:
        base_cols[base_cols.index("y_pred")] = "_y_pred"

    high_df = work_df.sort_values("_y_score", ascending=False).head(top_n)
    low_df = work_df.sort_values("_y_score", ascending=True).head(top_n)

    fp_df = work_df[(work_df["_y_true"] == 0) & (work_df["_y_pred"] == 1)].sort_values("_y_score", ascending=False).head(top_n)
    fn_df = work_df[(work_df["_y_true"] == 1) & (work_df["_y_pred"] == 0)].sort_values("_y_score", ascending=True).head(top_n)

    pos_mean = _to_float(y_score_v[y_true_v == 1].mean()) if (y_true_v == 1).any() else float("nan")
    neg_mean = _to_float(y_score_v[y_true_v == 0].mean()) if (y_true_v == 0).any() else float("nan")

    rename_map = {"_y_true": "y_true", "_y_score": "y_score", "_y_pred": "y_pred"}
    return {
        "mode": "validation",
        "basic_info": {
            "num_rows": int(len(df)),
            "columns": list(df.columns),
            "score_column": score_col,
            "threshold": float(threshold),
        },
        "score_distribution": _score_summary(y_score_v),
        "validation_metrics": metrics,
        "warnings": warnings,
        "label_score_mean": {
            "positive_mean_score": pos_mean,
            "negative_mean_score": neg_mean,
        },
        "top_high_score": _rows_to_records(high_df[base_cols].rename(columns=rename_map)),
        "top_low_score": _rows_to_records(low_df[base_cols].rename(columns=rename_map)),
        "false_positives": _rows_to_records(fp_df[base_cols].rename(columns=rename_map)),
        "false_negatives": _rows_to_records(fn_df[base_cols].rename(columns=rename_map)),
    }


def _analyze_scan(df: pd.DataFrame, score_col: str, top_n: int) -> dict[str, Any]:
    score = pd.to_numeric(df[score_col], errors="coerce")
    valid = score.notna()
    if valid.sum() == 0:
        raise ValueError("No valid score values in prediction file")

    work_df = df.copy()
    work_df["_score"] = score

    display_cols = [c for c in ["sample_id", "ts_code", "asof_date", score_col] if c in work_df.columns]
    if score_col in display_cols:
        display_cols[display_cols.index(score_col)] = "_score"

    high_df = work_df.sort_values("_score", ascending=False).head(top_n)
    low_df = work_df.sort_values("_score", ascending=True).head(top_n)

    return {
        "mode": "scan",
        "basic_info": {
            "num_rows": int(len(df)),
            "columns": list(df.columns),
            "score_column": score_col,
        },
        "score_distribution": _score_summary(score[valid]),
        "top_high_score": _rows_to_records(high_df[display_cols].rename(columns={"_score": "score"})),
        "top_low_score": _rows_to_records(low_df[display_cols].rename(columns={"_score": "score"})),
    }


def print_report(result: dict[str, Any], top_n: int) -> None:
    print("[Basic Info]")
    b = result["basic_info"]
    print(f"mode        : {result['mode']}")
    print(f"num_rows    : {b['num_rows']}")
    print(f"score_column: {b['score_column']}")
    print(f"columns     : {b['columns']}")
    if "threshold" in b:
        print(f"threshold   : {b['threshold']}")
    print()

    print("[Score Distribution]")
    sd = result["score_distribution"]
    for k in ["mean", "std", "min", "max", "q25", "q50", "q75", "q90", "q95", "q99"]:
        print(f"{k}: {sd[k]:.6f}")
    print()

    if result["mode"] == "validation":
        print("[Validation Metrics]")
        vm = result["validation_metrics"]
        for k in ["accuracy", "precision", "recall", "f1", "auc"]:
            print(f"{k}: {vm[k]:.6f}" if not np.isnan(vm[k]) else f"{k}: nan")
        print()
        warnings = result.get("warnings", [])
        if warnings:
            print("[Warnings]")
            for w in warnings:
                print(f"- {w}")
            print()

        print("[Label Score Mean]")
        lsm = result["label_score_mean"]
        for k, v in lsm.items():
            print(f"{k}: {v:.6f}" if not np.isnan(v) else f"{k}: nan")
        print()

    print(f"[Top High Score Samples] top_n={top_n}")
    for row in result["top_high_score"][:top_n]:
        print(row)
    print()

    print(f"[Top Low Score Samples] top_n={top_n}")
    for row in result["top_low_score"][:top_n]:
        print(row)
    print()

    if result["mode"] == "validation":
        print(f"[False Positives] top_n={top_n}")
        for row in result["false_positives"][:top_n]:
            print(row)
        print()

        print(f"[False Negatives] top_n={top_n}")
        for row in result["false_negatives"][:top_n]:
            print(row)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect prediction csv outputs.")
    parser.add_argument("--pred-path", required=True, help="Path to prediction csv")
    parser.add_argument("--output-json", default="", help="Optional path to save analysis json")
    parser.add_argument("--top-n", type=int, default=20, help="Top N samples for summaries")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for y_pred fallback")
    args = parser.parse_args()

    pred_path = Path(args.pred_path)
    _require_file(pred_path)

    df = pd.read_csv(pred_path)
    score_col = _pick_score_col(df)

    if "y_true" in df.columns and score_col is not None:
        result = _analyze_validation(df=df, score_col=score_col, top_n=args.top_n, threshold=args.threshold)
    elif "y_true" not in df.columns and "score" in df.columns:
        result = _analyze_scan(df=df, score_col="score", top_n=args.top_n)
    else:
        raise ValueError(
            "Cannot determine prediction mode. "
            "Validation mode requires y_true and (y_score or score); "
            "scan mode requires score without y_true."
        )

    print_report(result, top_n=args.top_n)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        safe_result = _sanitize_for_json(result)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(safe_result, f, ensure_ascii=False, indent=2)
        print(f"json saved to: {out_path}")


if __name__ == "__main__":
    main()
