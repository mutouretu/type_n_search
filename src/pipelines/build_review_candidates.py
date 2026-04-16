from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


KEY_COLUMNS = ["sample_id", "ts_code", "asof_date"]


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _resolve_path(project_root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root / p


def _require_columns(df: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _read_predictions(path: Path, asof_date: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, KEY_COLUMNS, path)
    out = df.copy()
    out["asof_date"] = out["asof_date"].astype(str)
    if asof_date:
        out = out[out["asof_date"] == str(asof_date)].copy()
    return out.reset_index(drop=True)


def _prepare_baseline(
    path: Path,
    *,
    name: str,
    score_col: str,
    asof_date: str | None,
    top_n: int,
) -> pd.DataFrame:
    df = _read_predictions(path, asof_date=asof_date)
    _require_columns(df, [score_col], path)
    if top_n > 0:
        df = df.head(top_n).copy()
    df["baseline_rank"] = range(1, len(df) + 1)
    df = df[KEY_COLUMNS + ["baseline_rank", score_col]].copy()
    return df.rename(columns={score_col: "baseline_score"}).assign(baseline_name=name)


def _prepare_specialist(
    path: Path,
    *,
    name: str,
    score_col: str,
    asof_date: str | None,
    top_n: int,
    pass_rank: int,
) -> pd.DataFrame:
    df = _read_predictions(path, asof_date=asof_date)
    _require_columns(df, [score_col], path)
    if top_n > 0:
        df = df.head(top_n).copy()
    rank_col = f"{name}_rank"
    score_out_col = f"{name}_score"
    pass_col = f"pass_{name}"
    df[rank_col] = range(1, len(df) + 1)
    df[pass_col] = df[rank_col] <= pass_rank
    return df[KEY_COLUMNS + [rank_col, score_col, pass_col]].rename(columns={score_col: score_out_col})


def build_review_candidates(config_path: str | Path) -> pd.DataFrame:
    config = _load_yaml(config_path)
    project_root = _resolve_path(Path.cwd(), config.get("project_root", "."))
    asof_date = config.get("asof_date")
    output_path = _resolve_path(project_root, config.get("output_path", "outputs/predictions/review_candidates.csv"))
    output_top_n = int(config.get("output_top_n", 0))

    baseline_cfg = config.get("baseline", {})
    if not isinstance(baseline_cfg, dict):
        raise ValueError("Config field 'baseline' must be a mapping")
    baseline_name = str(baseline_cfg.get("name", "baseline"))
    baseline_path = _resolve_path(project_root, baseline_cfg["path"])
    baseline_score_col = str(baseline_cfg.get("score_col", "score_mean"))
    baseline_top_n = int(baseline_cfg.get("top_n", 0))

    result = _prepare_baseline(
        baseline_path,
        name=baseline_name,
        score_col=baseline_score_col,
        asof_date=asof_date,
        top_n=baseline_top_n,
    )
    if result.empty:
        raise ValueError("Baseline predictions are empty after filtering")

    specialist_cfgs = config.get("specialists", [])
    if not isinstance(specialist_cfgs, list):
        raise ValueError("Config field 'specialists' must be a list")

    pass_cols: List[str] = []
    rank_cols: List[str] = []
    for item in specialist_cfgs:
        if not isinstance(item, dict):
            raise ValueError("Each specialist config must be a mapping")
        name = str(item["name"])
        path = _resolve_path(project_root, item["path"])
        score_col = str(item.get("score_col", "score_mean"))
        top_n = int(item.get("top_n", 0))
        pass_rank = int(item.get("pass_rank", top_n if top_n > 0 else 0))
        specialist = _prepare_specialist(
            path,
            name=name,
            score_col=score_col,
            asof_date=asof_date,
            top_n=top_n,
            pass_rank=pass_rank,
        )
        result = result.merge(specialist, on=KEY_COLUMNS, how="left")
        pass_col = f"pass_{name}"
        rank_col = f"{name}_rank"
        result[pass_col] = result[pass_col].where(result[pass_col].notna(), False).astype(bool)
        pass_cols.append(pass_col)
        rank_cols.append(rank_col)

    result["pass_count"] = result[pass_cols].sum(axis=1) if pass_cols else 0
    if rank_cols:
        result["specialist_rank_sum"] = result[rank_cols].fillna(10**9).sum(axis=1)
    else:
        result["specialist_rank_sum"] = 0

    result = result.sort_values(
        ["pass_count", "baseline_score", "baseline_rank", "specialist_rank_sum"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    result["review_rank"] = range(1, len(result) + 1)

    front_cols = KEY_COLUMNS + ["review_rank", "baseline_name", "baseline_rank", "baseline_score", "pass_count"]
    other_cols = [col for col in result.columns if col not in front_cols]
    result = result[front_cols + other_cols]
    if output_top_n > 0:
        result = result.head(output_top_n).copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(
        "review candidates built: "
        f"{len(result)} samples, specialists={len(specialist_cfgs)} -> {output_path}"
    )
    if pass_cols:
        print(result["pass_count"].value_counts().sort_index(ascending=False).to_string())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build human-review candidates from baseline and specialist predictions.")
    parser.add_argument("--config", default="configs/review_candidates.yaml", help="Path to review candidate config yaml")
    args = parser.parse_args()
    build_review_candidates(args.config)


if __name__ == "__main__":
    main()
