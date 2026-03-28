"""Run: python -m scripts.paper_manual_labeling (from repo root, PYTHONPATH=. or pip -e .)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/paper_manual_labeling/__main__.py` from repo root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper manual-labeling artifacts")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("all", "harvest", "propagation", "figures"),
    )
    args = parser.parse_args()

    if args.command in ("all", "harvest"):
        from scripts.paper_manual_labeling.harvest import run_harvest, write_paper_stats_md

        payload = run_harvest()
        md_path = write_paper_stats_md(payload)
        print("Wrote data/analysis/paper_manual_labeling/paper_stats.json and", md_path)

    if args.command in ("all", "propagation"):
        from scripts.paper_manual_labeling.propagation import run_propagation

        p = run_propagation()
        print(
            "Propagation coverage:",
            f"{p['rows_matching_a_validated_key']:,} / {p['full_corpus_rows']:,}",
            f"({100 * p['fraction_of_corpus_covered']:.2f}%)",
        )

    if args.command in ("all", "figures"):
        import json

        from scripts.paper_manual_labeling.figures import run_figures

        stats_path = _ROOT / "data/analysis/paper_manual_labeling/paper_stats.json"
        if not stats_path.is_file():
            from scripts.paper_manual_labeling.harvest import run_harvest

            run_harvest()
        paper_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        s = run_figures(paper_stats)
        print("Figures:", s.get("new_figures"))
        if s.get("mirrored_r001_figures"):
            print("Mirrored", len(s["mirrored_r001_figures"]), "existing PNGs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
