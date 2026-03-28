#!/usr/bin/env python3
"""
Estimate how much of the full LLM corpus is covered by human-validated keys.

A row is counted as covered if (normalize(narration_text), canonical LLM action)
matches at least one refined annotation row with verdict Gold or Bad, using the
same normalization as merge_round1_validated_sources.py.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Set, Tuple

from .common import (
    FULL_LLM_REFINED,
    OUTPUT_DIR,
    R001_REFINED,
    R002_REFINED,
    canonical_action,
    normalize_narration_for_analysis,
)

ValidatedKey = Tuple[str, str]


def _load_validated_keys() -> Set[ValidatedKey]:
    keys: Set[ValidatedKey] = set()

    def ingest_row(narr: str, llm_action: str, verdict: str) -> None:
        v = (verdict or "").strip()
        if v not in ("Gold", "Bad"):
            return
        nn = normalize_narration_for_analysis(narr)
        act = canonical_action(llm_action)
        if nn and act:
            keys.add((nn, act))

    for path in (R001_REFINED, R002_REFINED):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            if "verdict" in r.fieldnames:
                vf = "verdict"
            elif "status_main" in r.fieldnames:
                vf = "status_main"
            else:
                raise ValueError(f"No verdict column in {path}")
            for row in r:
                ingest_row(row.get("narration_text", ""), row.get("action", ""), row.get(vf, ""))

    return keys


def run_propagation() -> Dict[str, Any]:
    keys = _load_validated_keys()
    matched = 0
    total = 0
    with FULL_LLM_REFINED.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            nn = normalize_narration_for_analysis(row.get("narration_text", ""))
            act = canonical_action(row.get("action", ""))
            if (nn, act) in keys:
                matched += 1
    frac = matched / total if total else 0.0
    payload = {
        "full_corpus_rows": total,
        "validated_unique_keys": len(keys),
        "rows_matching_a_validated_key": matched,
        "fraction_of_corpus_covered": frac,
        "inputs": {
            "full_llm": str(FULL_LLM_REFINED),
            "r001_refined": str(R001_REFINED),
            "r002_refined": str(R002_REFINED),
        },
        "note": (
            "Coverage counts rows whose (normalized narration, LLM action) was explicitly "
            "validated as Gold or Bad. It does not count Skip/Delete or narration-only rules."
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "propagation_coverage.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
