from dataclasses import dataclass
from typing import Optional


@dataclass
class SampleMeta:
    """
    Standard metadata schema for one stock sample in a fixed historical window.

    This structure is used for:
    - Training: carry supervised target (`label`) and its provenance.
    - Inference: keep deterministic sample identity and time window context.
    - Dataset construction: provide consistent indexing, split tagging, and QA fields.
    """

    sample_id: str
    ts_code: str
    asof_date: str
    window_start: str
    window_end: str
    label: int
    label_source: str
    confidence: float
    split: Optional[str] = None


def build_sample_id(ts_code: str, asof_date: str) -> str:
    """Build stable sample id in the format: `000001.SZ_2025-03-21`."""
    return f"{ts_code}_{asof_date}"
