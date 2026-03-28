"""Shared paths and text normalization for paper manual-labeling artifacts."""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

FULL_LLM_REFINED = PROJECT_ROOT / "data/labels/action_labels_llm_clean_refined.csv"
GOLD_DIR = PROJECT_ROOT / "data/annotation_rounds/final_gold_dataset"
R001_REFINED = (
    PROJECT_ROOT
    / "data/annotation_rounds/r001_gold_exports/r001_refined_combined/r001_refined_combined.csv"
)
R002_REFINED = (
    PROJECT_ROOT
    / "data/annotation_rounds/r002_gold_exports/r002_refined_combined/r002_refined_combined.csv"
)
R001_MERGED = (
    PROJECT_ROOT
    / "data/annotation_rounds/r001_merged_validation/round1_validated_rows_with_decisions.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data/analysis/paper_manual_labeling"


def normalize_narration_for_analysis(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[.,;:!?]+$", "", t)
    t = re.sub(r"#\w+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def canonical_action(name: str) -> str:
    s = (name or "").strip()
    if s == "Task Operation":
        return "Essential Operation"
    return s
