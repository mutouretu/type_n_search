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


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def _detect_score_col(df: pd.DataFrame, explicit_col: str | None = None) -> str:
    if explicit_col:
        if explicit_col not in df.columns:
            raise KeyError(f"Requested score column not found: {explicit_col}")
        return explicit_col
    if "y_score" in df.columns:
        return "y_score"
    if "score" in df.columns:
        return "score"
    raise KeyError("Cannot detect score column. Expected one of: y_score, score")


def _detect_mode(df: pd.DataFrame, score_col: str) -> str:
    if "y_true" in df.columns:
        return "validation"
    if score_col in df.columns:
        return "scan"
    raise ValueError("Cannot detect mode")


def _score_summary(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce")
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


def _rows_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, (np.floating, np.integer)):
                clean[k] = v.item()
            else:
                clean[k] = v
        out.append(clean)
    return out


def _prepare_alignment(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if "sample_id" in df_a.columns and "sample_id" in df_b.columns:
        merged = df_a.merge(df_b, on=["sample_id"], suffixes=("_a", "_b"), how="inner")
        return merged, "sample_id"

    key_cols = ["ts_code", "asof_date"]
    if all(c in df_a.columns for c in key_cols) and all(c in df_b.columns for c in key_cols):
        merged = df_a.merge(df_b, on=key_cols, suffixes=("_a", "_b"), how="inner")
        return merged, "ts_code+asof_date"

    raise KeyError(
        "Cannot align two prediction files. Need either shared sample_id or shared ts_code+asof_date"
    )


def _get_record_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in ["sample_id", "ts_code", "asof_date", "score_a", "score_b", "avg_score", "delta"]:
        if c in df.columns:
            cols.append(c)
    if not cols:
        cols = list(df.columns)
    return cols


def _safe_metrics(y_true: pd.Series, y_score: pd.Series, threshold: float) -> dict[str, float]:
    y_t = pd.to_numeric(y_true, errors="coerce")
    y_s = pd.to_numeric(y_score, errors="coerce")
    valid = y_t.notna() & y_s.notna()
    y_t = y_t[valid].astype(int)
    y_s = y_s[valid].astype(float)

    if len(y_t) == 0:
        raise ValueError("No valid rows for metrics")

    y_p = (y_s >= threshold).astype(int)
    out = {
        "accuracy": _to_float(accuracy_score(y_t, y_p)),
        "precision": _to_float(precision_score(y_t, y_p, zero_division=0)),
        "recall": _to_float(recall_score(y_t, y_p, zero_division=0)),
        "f1": _to_float(f1_score(y_t, y_p, zero_division=0)),
        "auc": float("nan"),
    }
    if y_t.nunique() >= 2:
        try:
            out["auc"] = _to_float(roc_auc_score(y_t, y_s))
        except Exception:
            out["auc"] = float("nan")
    return out


def _bucket_future_stats(fut: pd.DataFrame, bucket: set[str]) -> dict[str, float | int | None]:
    if not bucket:
        return {"count": 0, "mean": None, "median": None, "std": None}
    part = fut[fut["__join_key"].isin(bucket)]["future_return"]
    if len(part) == 0:
        return {"count": 0, "mean": None, "median": None, "std": None}
    return {
        "count": int(len(part)),
        "mean": _to_float(part.mean()),
        "median": _to_float(part.median()),
        "std": _to_float(part.std(ddof=0)),
    }


def _build_future_summary(
    future_path: str,
    key_type: str,
    top_a_keys: set[str],
    top_b_keys: set[str],
) -> dict[str, Any]:
    p = Path(future_path)
    _require_file(p)
    fut = pd.read_csv(p)

    if "future_return" not in fut.columns:
        return {"warning": "future_return column not found; skip future return summary"}

    if key_type == "sample_id":
        if "sample_id" not in fut.columns:
            return {"warning": "future file missing sample_id; skip future return summary"}
        fut["__join_key"] = fut["sample_id"].astype(str)
    else:
        req = ["ts_code", "asof_date"]
        if not all(c in fut.columns for c in req):
            return {
                "warning": "future file missing ts_code/asof_date; skip future return summary"
            }
        fut["__join_key"] = fut["ts_code"].astype(str) + "|" + fut["asof_date"].astype(str)

    fut["future_return"] = pd.to_numeric(fut["future_return"], errors="coerce")
    fut = fut.dropna(subset=["future_return"]).copy()

    overlap = top_a_keys & top_b_keys
    only_a = top_a_keys - top_b_keys
    only_b = top_b_keys - top_a_keys

    return {
        "topN_a": _bucket_future_stats(fut, top_a_keys),
        "topN_b": _bucket_future_stats(fut, top_b_keys),
        "overlap_topN": _bucket_future_stats(fut, overlap),
        "a_only_topN": _bucket_future_stats(fut, only_a),
        "b_only_topN": _bucket_future_stats(fut, only_b),
    }


def compare_models(
    pred_a_path: Path,
    pred_b_path: Path,
    name_a: str,
    name_b: str,
    top_n: int,
    threshold: float,
    score_col_a: str | None,
    score_col_b: str | None,
    future_return_path: str,
) -> dict[str, Any]:
    _require_file(pred_a_path)
    _require_file(pred_b_path)

    df_a = pd.read_csv(pred_a_path)
    df_b = pd.read_csv(pred_b_path)

    score_a_col = _detect_score_col(df_a, score_col_a)
    score_b_col = _detect_score_col(df_b, score_col_b)

    mode_a = _detect_mode(df_a, score_a_col)
    mode_b = _detect_mode(df_b, score_b_col)
    if mode_a != mode_b:
        raise ValueError(f"Mode mismatch: {name_a}={mode_a}, {name_b}={mode_b}")

    merged, key_type = _prepare_alignment(df_a, df_b)
    if len(merged) == 0:
        raise ValueError("No overlapped rows after alignment")

    if key_type == "sample_id":
        merged["__join_key"] = merged["sample_id"].astype(str)
    else:
        merged["__join_key"] = merged["ts_code"].astype(str) + "|" + merged["asof_date"].astype(str)

    merged["score_a"] = pd.to_numeric(merged[score_a_col + "_a"], errors="coerce")
    merged["score_b"] = pd.to_numeric(merged[score_b_col + "_b"], errors="coerce")
    merged = merged.dropna(subset=["score_a", "score_b"]).copy()
    if len(merged) == 0:
        raise ValueError("No rows with valid scores after alignment")

    pearson = _to_float(merged["score_a"].corr(merged["score_b"], method="pearson"))
    spearman = _to_float(merged["score_a"].corr(merged["score_b"], method="spearman"))

    top_a_df = merged.sort_values("score_a", ascending=False).head(top_n).copy()
    top_b_df = merged.sort_values("score_b", ascending=False).head(top_n).copy()
    top_a_keys = set(top_a_df["__join_key"].astype(str))
    top_b_keys = set(top_b_df["__join_key"].astype(str))
    overlap = top_a_keys & top_b_keys

    overlap_df = merged[merged["__join_key"].isin(overlap)].copy()
    overlap_df["avg_score"] = (overlap_df["score_a"] + overlap_df["score_b"]) / 2.0
    overlap_df = overlap_df.sort_values("avg_score", ascending=False)

    merged["delta"] = merged["score_a"] - merged["score_b"]
    a_better_df = merged.sort_values("delta", ascending=False).head(top_n)
    b_better_df = merged.sort_values("delta", ascending=True).head(top_n)

    only_a_df = top_a_df[~top_a_df["__join_key"].isin(top_b_keys)]
    only_b_df = top_b_df[~top_b_df["__join_key"].isin(top_a_keys)]

    out: dict[str, Any] = {
        "mode": mode_a,
        "basic_info": {
            "name_a": name_a,
            "name_b": name_b,
            "rows_a": int(len(df_a)),
            "rows_b": int(len(df_b)),
            "rows_aligned": int(len(merged)),
            "columns_a": list(df_a.columns),
            "columns_b": list(df_b.columns),
            "score_col_a": score_a_col,
            "score_col_b": score_b_col,
            "alignment_key": key_type,
        },
        "score_distribution": {
            name_a: _score_summary(merged["score_a"]),
            name_b: _score_summary(merged["score_b"]),
        },
        "score_correlation": {
            "pearson": pearson,
            "spearman": spearman,
        },
        "top_n_overlap": {
            "top_n": int(top_n),
            "overlap_count": int(len(overlap)),
            "overlap_ratio": float(len(overlap) / top_n) if top_n > 0 else 0.0,
            "overlap_sample_ids": sorted(list(overlap))[:top_n],
        },
        "top_samples": {
            f"top_{name_a}": _rows_to_records(top_a_df[_get_record_cols(top_a_df)]),
            f"top_{name_b}": _rows_to_records(top_b_df[_get_record_cols(top_b_df)]),
        },
        "disagreement": {
            "only_in_top_a": _rows_to_records(only_a_df[_get_record_cols(only_a_df)]),
            "only_in_top_b": _rows_to_records(only_b_df[_get_record_cols(only_b_df)]),
            "a_more_bullish": _rows_to_records(a_better_df[_get_record_cols(a_better_df)]),
            "b_more_bullish": _rows_to_records(b_better_df[_get_record_cols(b_better_df)]),
        },
    }

    if mode_a == "validation":
        y_true_col_a = "y_true_a" if "y_true_a" in merged.columns else "y_true"
        y_true_col_b = "y_true_b" if "y_true_b" in merged.columns else "y_true"
        y_true_a = pd.to_numeric(merged[y_true_col_a], errors="coerce")
        y_true_b = pd.to_numeric(merged[y_true_col_b], errors="coerce")
        y_true = y_true_a.copy()

        label_mismatch = int(((y_true_a != y_true_b) & y_true_a.notna() & y_true_b.notna()).sum())

        y_pred_a = (
            pd.to_numeric(merged["y_pred_a"], errors="coerce")
            if "y_pred_a" in merged.columns
            else (merged["score_a"] >= threshold).astype(int)
        )
        y_pred_b = (
            pd.to_numeric(merged["y_pred_b"], errors="coerce")
            if "y_pred_b" in merged.columns
            else (merged["score_b"] >= threshold).astype(int)
        )

        metrics_a = _safe_metrics(y_true, merged["score_a"], threshold)
        metrics_b = _safe_metrics(y_true, merged["score_b"], threshold)

        valid = y_true.notna() & y_pred_a.notna() & y_pred_b.notna()
        yt = y_true[valid].astype(int)
        pa = y_pred_a[valid].astype(int)
        pb = y_pred_b[valid].astype(int)

        mis_a = pa != yt
        mis_b = pb != yt

        common_mis = merged.loc[valid].loc[mis_a & mis_b]
        only_a_mis = merged.loc[valid].loc[mis_a & (~mis_b)]
        only_b_mis = merged.loc[valid].loc[mis_b & (~mis_a)]

        fp_a = merged.loc[valid].loc[(yt == 0) & (pa == 1)].sort_values("score_a", ascending=False).head(top_n)
        fn_a = merged.loc[valid].loc[(yt == 1) & (pa == 0)].sort_values("score_a", ascending=True).head(top_n)
        fp_b = merged.loc[valid].loc[(yt == 0) & (pb == 1)].sort_values("score_b", ascending=False).head(top_n)
        fn_b = merged.loc[valid].loc[(yt == 1) & (pb == 0)].sort_values("score_b", ascending=True).head(top_n)

        pos_mask = yt == 1
        neg_mask = yt == 0

        def _label_score_stat(s: pd.Series, m: pd.Series) -> float:
            return _to_float(s[m].mean()) if m.any() else float("nan")

        label_summary = {
            name_a: {
                "positive_mean_score": _label_score_stat(merged.loc[valid, "score_a"], pos_mask),
                "negative_mean_score": _label_score_stat(merged.loc[valid, "score_a"], neg_mask),
            },
            name_b: {
                "positive_mean_score": _label_score_stat(merged.loc[valid, "score_b"], pos_mask),
                "negative_mean_score": _label_score_stat(merged.loc[valid, "score_b"], neg_mask),
            },
        }
        label_summary[name_a]["pos_neg_gap"] = _to_float(
            label_summary[name_a]["positive_mean_score"] - label_summary[name_a]["negative_mean_score"]
        )
        label_summary[name_b]["pos_neg_gap"] = _to_float(
            label_summary[name_b]["positive_mean_score"] - label_summary[name_b]["negative_mean_score"]
        )

        metric_winner = {}
        for m in ["accuracy", "f1", "auc"]:
            a_val = metrics_a[m]
            b_val = metrics_b[m]
            if math.isnan(a_val) and math.isnan(b_val):
                metric_winner[m] = "tie"
            elif math.isnan(a_val):
                metric_winner[m] = name_b
            elif math.isnan(b_val):
                metric_winner[m] = name_a
            elif a_val > b_val:
                metric_winner[m] = name_a
            elif b_val > a_val:
                metric_winner[m] = name_b
            else:
                metric_winner[m] = "tie"

        out["validation"] = {
            "threshold": float(threshold),
            "label_mismatch_count": label_mismatch,
            "metrics": {name_a: metrics_a, name_b: metrics_b},
            "metric_winner": metric_winner,
            "misclassification": {
                f"false_positive_{name_a}": _rows_to_records(fp_a[_get_record_cols(fp_a)]),
                f"false_negative_{name_a}": _rows_to_records(fn_a[_get_record_cols(fn_a)]),
                f"false_positive_{name_b}": _rows_to_records(fp_b[_get_record_cols(fp_b)]),
                f"false_negative_{name_b}": _rows_to_records(fn_b[_get_record_cols(fn_b)]),
                "common_misclassified": _rows_to_records(common_mis[_get_record_cols(common_mis)]),
                f"only_{name_a}_misclassified": _rows_to_records(only_a_mis[_get_record_cols(only_a_mis)]),
                f"only_{name_b}_misclassified": _rows_to_records(only_b_mis[_get_record_cols(only_b_mis)]),
            },
            "label_wise_score_summary": label_summary,
        }

    if mode_a == "scan":
        out["scan"] = {
            "top_n": int(top_n),
            "consensus_top": _rows_to_records(overlap_df[_get_record_cols(overlap_df)]),
            "a_only_top": _rows_to_records(only_a_df[_get_record_cols(only_a_df)]),
            "b_only_top": _rows_to_records(only_b_df[_get_record_cols(only_b_df)]),
            "bucket_counts": {
                "consensus_top": int(len(overlap)),
                "a_only_top": int(len(top_a_keys - top_b_keys)),
                "b_only_top": int(len(top_b_keys - top_a_keys)),
            },
        }

    if future_return_path:
        out["future_return_summary"] = _build_future_summary(
            future_path=future_return_path,
            key_type=key_type,
            top_a_keys=top_a_keys,
            top_b_keys=top_b_keys,
        )

    return out


def print_report(result: dict[str, Any], name_a: str, name_b: str, top_n: int) -> None:
    print("[Basic Info]")
    b = result["basic_info"]
    print(f"mode           : {result['mode']}")
    print(f"rows_a         : {b['rows_a']}")
    print(f"rows_b         : {b['rows_b']}")
    print(f"rows_aligned   : {b['rows_aligned']}")
    print(f"score_col_a    : {b['score_col_a']}")
    print(f"score_col_b    : {b['score_col_b']}")
    print(f"alignment_key  : {b['alignment_key']}")
    print()

    print("[Score Distribution]")
    for n in [name_a, name_b]:
        s = result["score_distribution"][n]
        print(
            f"{n}: mean={s['mean']:.6f}, std={s['std']:.6f}, min={s['min']:.6f}, max={s['max']:.6f}, "
            f"q25={s['q25']:.6f}, q50={s['q50']:.6f}, q75={s['q75']:.6f}, q90={s['q90']:.6f}, q95={s['q95']:.6f}, q99={s['q99']:.6f}"
        )
    print()

    print("[Score Correlation]")
    c = result["score_correlation"]
    print(f"pearson : {c['pearson']:.6f}" if c["pearson"] is not None else "pearson : None")
    print(f"spearman: {c['spearman']:.6f}" if c["spearman"] is not None else "spearman: None")
    print()

    print("[Top-N Overlap]")
    t = result["top_n_overlap"]
    print(f"top_n         : {t['top_n']}")
    print(f"overlap_count : {t['overlap_count']}")
    print(f"overlap_ratio : {t['overlap_ratio']:.6f}")
    print(f"overlap_ids   : {t['overlap_sample_ids']}")
    print()

    print(f"[Top High Score Samples] {name_a} top_n={top_n}")
    for row in result["top_samples"].get(f"top_{name_a}", [])[:top_n]:
        print(row)
    print()

    print(f"[Top High Score Samples] {name_b} top_n={top_n}")
    for row in result["top_samples"].get(f"top_{name_b}", [])[:top_n]:
        print(row)
    print()

    print("[Disagreement Samples]")
    print(f"only_in_top_a ({name_a}):")
    for row in result["disagreement"]["only_in_top_a"][:top_n]:
        print(row)
    print(f"only_in_top_b ({name_b}):")
    for row in result["disagreement"]["only_in_top_b"][:top_n]:
        print(row)
    print(f"delta_max ({name_a} more bullish):")
    for row in result["disagreement"]["a_more_bullish"][:top_n]:
        print(row)
    print(f"delta_min ({name_b} more bullish):")
    for row in result["disagreement"]["b_more_bullish"][:top_n]:
        print(row)
    print()

    if result["mode"] == "validation":
        v = result["validation"]
        print("[Validation Metrics]")
        for n in [name_a, name_b]:
            m = v["metrics"][n]
            auc_text = "nan" if np.isnan(m["auc"]) else f"{m['auc']:.6f}"
            print(
                f"{n}: accuracy={m['accuracy']:.6f}, precision={m['precision']:.6f}, "
                f"recall={m['recall']:.6f}, f1={m['f1']:.6f}, "
                f"auc={auc_text}"
            )
        print(f"metric_winner: {v['metric_winner']}")
        print(f"label_mismatch_count: {v['label_mismatch_count']}")
        print()

        print("[Misclassification Comparison]")
        for k, rows in v["misclassification"].items():
            print(f"{k}: {len(rows)}")
        print()

        print("[Label-wise Score Summary]")
        print(v["label_wise_score_summary"])
        print()

    if result["mode"] == "scan":
        s = result["scan"]
        print("[Scan Candidate Comparison]")
        print(f"bucket_counts: {s['bucket_counts']}")
        print()

        print("[Consensus Top Samples]")
        for row in s["consensus_top"][:top_n]:
            print(row)
        print()

    if "future_return_summary" in result:
        print("[Future Return Summary]")
        print(result["future_return_summary"])
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two prediction csv files.")
    parser.add_argument("--pred-a-path", required=True, help="Path to model A prediction csv")
    parser.add_argument("--pred-b-path", required=True, help="Path to model B prediction csv")
    parser.add_argument("--name-a", default="model_a", help="Display name for model A")
    parser.add_argument("--name-b", default="model_b", help="Display name for model B")
    parser.add_argument("--output-json", default="", help="Optional path to save comparison json")
    parser.add_argument("--top-n", type=int, default=20, help="Top N for overlap and examples")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for y_pred fallback")
    parser.add_argument("--future-return-path", default="", help="Optional future return csv")
    parser.add_argument(
        "--future-key-cols",
        default="sample_id,ts_code,asof_date",
        help="Reserved option for future extension",
    )
    parser.add_argument("--score-col-a", default="", help="Optional explicit score column for A")
    parser.add_argument("--score-col-b", default="", help="Optional explicit score column for B")
    args = parser.parse_args()

    result = compare_models(
        pred_a_path=Path(args.pred_a_path),
        pred_b_path=Path(args.pred_b_path),
        name_a=args.name_a,
        name_b=args.name_b,
        top_n=int(args.top_n),
        threshold=float(args.threshold),
        score_col_a=args.score_col_a or None,
        score_col_b=args.score_col_b or None,
        future_return_path=args.future_return_path,
    )

    print_report(result, name_a=args.name_a, name_b=args.name_b, top_n=int(args.top_n))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved json: {out_path}")


if __name__ == "__main__":
    main()
