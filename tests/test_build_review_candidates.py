from pathlib import Path

import pandas as pd
import yaml

from src.pipelines.build_review_candidates import build_review_candidates


def test_build_review_candidates_merges_specialist_passes(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    baseline_path = pred_dir / "baseline.csv"
    runup_path = pred_dir / "runup.csv"
    output_path = pred_dir / "review.csv"

    pd.DataFrame(
        [
            {"sample_id": "a_2026-03-27", "ts_code": "a", "asof_date": "2026-03-27", "score_mean": 0.9},
            {"sample_id": "b_2026-03-27", "ts_code": "b", "asof_date": "2026-03-27", "score_mean": 0.8},
            {"sample_id": "c_2026-03-27", "ts_code": "c", "asof_date": "2026-03-27", "score_mean": 0.7},
        ]
    ).to_csv(baseline_path, index=False)
    pd.DataFrame(
        [
            {"sample_id": "b_2026-03-27", "ts_code": "b", "asof_date": "2026-03-27", "score_mean": 0.95},
            {"sample_id": "c_2026-03-27", "ts_code": "c", "asof_date": "2026-03-27", "score_mean": 0.85},
        ]
    ).to_csv(runup_path, index=False)

    config_path = tmp_path / "review.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project_root": str(tmp_path),
                "asof_date": "2026-03-27",
                "output_path": str(output_path),
                "baseline": {
                    "name": "baseline_type_n",
                    "path": str(baseline_path),
                    "score_col": "score_mean",
                    "top_n": 3,
                },
                "specialists": [
                    {
                        "name": "runup",
                        "path": str(runup_path),
                        "score_col": "score_mean",
                        "top_n": 2,
                        "pass_rank": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    out = build_review_candidates(config_path)

    assert output_path.exists()
    assert out.loc[0, "ts_code"] == "b"
    assert out.loc[0, "pass_runup"]
    assert out.loc[0, "pass_count"] == 1
    assert not out[out["ts_code"] == "a"].iloc[0]["pass_runup"]
    assert {"review_rank", "baseline_rank", "runup_rank", "runup_score"}.issubset(out.columns)
