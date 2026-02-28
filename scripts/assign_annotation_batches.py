#!/usr/bin/env python3
"""
Assign annotation batches for Label Studio using power allocation.

Uses power allocation (p^alpha) to oversample rare classes while keeping
video counts per batch manageable. The output CSV contains ALL rows from
the input — assigned rows get batch/round values, unassigned rows are blank.

See doc/BATCH_ASSIGNMENT_STRATEGY.md for scientific justification.

Usage (Round 1):
  python scripts/assign_annotation_batches.py \
      --labels-csv data/labels/action_labels_llm_clean_refined.csv \
      --num-batches 12 \
      --batch-size 2500 \
      --alpha 0.5 \
      --round-id r001 \
      --seed 42 \
      --output-csv data/labels/action_labels_llm_clean_refined.csv

Usage (Round 2 — assigns from unassigned rows only):
  python scripts/assign_annotation_batches.py \
      --labels-csv data/labels/action_labels_llm_clean_refined.csv \
      --num-batches 12 \
      --batch-size 2500 \
      --alpha 0.5 \
      --round-id r002 \
      --seed 2026 \
      --output-csv data/labels/action_labels_llm_clean_refined.csv
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ACTIVE_CLASSES = [
    "Object Transfer",
    "Essential Operation",
    "Stationary",
    "Locomotion",
    "Search",
]


def normalize_no_articles(text: str) -> str:
    """Lowercase, remove hashtags/punctuation/articles, collapse spaces."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"#\w+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\b(the|a|an|some)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def build_frequency_key(series: pd.Series, mode: str) -> pd.Series:
    """
    Build text key for frequency grouping.
    - raw: original narration text normalized to lowercase/space.
    - no_articles: regex-cleaned text removing articles.
    - nouns_verbs: aggressive key keeping only NOUN/PROPN/VERB lemmas.
    """
    s = series.fillna("").astype(str)

    if mode == "raw":
        return s.str.lower().str.strip()

    if mode == "no_articles":
        return s.apply(normalize_no_articles)

    if mode == "nouns_verbs":
        try:
            import spacy
        except ImportError as e:
            raise RuntimeError(
                "spacy is required for --frequency-key nouns_verbs. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            ) from e

        model = "en_core_web_sm"
        if not spacy.util.is_package(model):
            raise RuntimeError(
                "Missing spaCy model 'en_core_web_sm'. "
                "Run: python -m spacy download en_core_web_sm"
            )
        nlp = spacy.load(model, disable=["parser", "ner"])
        texts = s.apply(lambda x: re.sub(r"#\w+", " ", x)).tolist()
        out: list[str] = []
        for doc in nlp.pipe(texts, batch_size=2000, n_process=1):
            toks = [tok.lemma_.lower() for tok in doc if tok.pos_ in {"NOUN", "PROPN", "VERB"}]
            out.append(" ".join(toks).strip())
        return pd.Series(out, index=series.index)

    raise ValueError(f"Unknown frequency key mode: {mode}")


def compute_power_allocation(
    class_counts: dict[str, int], batch_size: int, alpha: float
) -> dict[str, int]:
    """
    Compute per-class targets using power allocation: n_h = N * p_h^alpha / sum(p_j^alpha).
    """
    total = sum(class_counts.values())
    proportions = {cls: count / total for cls, count in class_counts.items()}

    powered = {cls: p ** alpha for cls, p in proportions.items()}
    powered_sum = sum(powered.values())

    # Compute raw targets
    raw = {cls: batch_size * powered[cls] / powered_sum for cls in ACTIVE_CLASSES}

    # Floor and distribute remainders
    targets = {cls: int(math.floor(raw[cls])) for cls in ACTIVE_CLASSES}
    remainders = {cls: raw[cls] - targets[cls] for cls in ACTIVE_CLASSES}
    leftover = batch_size - sum(targets.values())

    for cls in sorted(remainders, key=remainders.get, reverse=True):
        if leftover <= 0:
            break
        targets[cls] += 1
        leftover -= 1

    return targets


def compute_video_class_profiles(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """For each video, count how many rows of each class are available."""
    profiles: dict[str, dict[str, int]] = {}
    for video_uid, group in df.groupby("video_uid"):
        counts = group["action"].value_counts().to_dict()
        profiles[str(video_uid)] = {cls: counts.get(cls, 0) for cls in ACTIVE_CLASSES}
    return profiles


def assign_videos_to_batches(
    profiles: dict[str, dict[str, int]],
    num_batches: int,
    class_targets: dict[str, int],
) -> dict[int, list[str]]:
    """
    Greedy video assignment: Search-rich videos first, assign to batch
    with largest class deficit.
    """
    sorted_videos = sorted(
        profiles.keys(),
        key=lambda v: (profiles[v].get("Search", 0), sum(profiles[v].values())),
        reverse=True,
    )

    batch_videos: dict[int, list[str]] = {b: [] for b in range(1, num_batches + 1)}
    batch_available: dict[int, dict[str, int]] = {
        b: {cls: 0 for cls in ACTIVE_CLASSES} for b in range(1, num_batches + 1)
    }

    for video_uid in sorted_videos:
        vp = profiles[video_uid]
        if sum(vp.values()) == 0:
            continue

        best_batch = None
        best_score = float("-inf")

        for b in range(1, num_batches + 1):
            score = 0.0
            for cls in ACTIVE_CLASSES:
                deficit = class_targets[cls] - batch_available[b][cls]
                if deficit > 0 and vp.get(cls, 0) > 0:
                    score += min(deficit, vp[cls])
            # Tie-break: prefer batch with fewer total rows
            total_available = sum(batch_available[b].values())
            score -= total_available * 0.0001

            if score > best_score:
                best_score = score
                best_batch = b

        if best_batch is not None:
            batch_videos[best_batch].append(video_uid)
            for cls in ACTIVE_CLASSES:
                batch_available[best_batch][cls] += vp.get(cls, 0)

    return batch_videos


def sample_rows_per_batch(
    df: pd.DataFrame,
    batch_videos: dict[int, list[str]],
    class_targets: dict[str, int],
    seed: int,
    prioritize_frequent: bool = False,
) -> pd.DataFrame:
    """
    For each batch, sample rows from its assigned videos
    to hit per-class targets.
    """
    all_sampled: list[pd.DataFrame] = []

    for batch_id, video_list in sorted(batch_videos.items()):
        if not video_list:
            continue

        batch_pool = df[df["video_uid"].isin(video_list)].copy()
        batch_rows: list[pd.DataFrame] = []

        for cls in ACTIVE_CLASSES:
            cls_pool = batch_pool[batch_pool["action"] == cls]
            target = class_targets[cls]
            n = min(target, len(cls_pool))
            if n > 0:
                if prioritize_frequent and "_narr_freq" in cls_pool.columns:
                    # Deterministic highest-frequency first (very aggressive mode support)
                    sampled = cls_pool.sort_values(
                        by=["_narr_freq", "video_uid", "timestamp_sec"],
                        ascending=[False, True, True],
                    ).head(n)
                else:
                    sampled = cls_pool.sample(n=n, random_state=seed + batch_id * 100)
                batch_rows.append(sampled)

        if batch_rows:
            batch_df = pd.concat(batch_rows, ignore_index=False)
            batch_df["_batch"] = batch_id
            all_sampled.append(batch_df)

    if not all_sampled:
        return pd.DataFrame()

    return pd.concat(all_sampled, ignore_index=False)


def print_summary(
    result: pd.DataFrame, num_batches: int, class_targets: dict[str, int]
) -> None:
    """Print batch summary table."""
    class_short = {
        "Object Transfer": "OT",
        "Essential Operation": "EO",
        "Stationary": "Stat",
        "Locomotion": "Loco",
        "Search": "Search",
    }

    header = f"{'Batch':>6} | {'Total':>5}"
    for cls in ACTIVE_CLASSES:
        header += f" | {class_short[cls]:>6}"
    header += f" | {'Videos':>6}"
    print("\n" + header)
    print("-" * len(header))

    for batch_id in range(1, num_batches + 1):
        batch_rows = result[result["_batch"] == batch_id]
        if batch_rows.empty:
            continue
        label = f"B{batch_id:02d}"
        line = f"{label:>6} | {len(batch_rows):>5}"
        for cls in ACTIVE_CLASSES:
            count = int((batch_rows["action"] == cls).sum())
            target = class_targets[cls]
            marker = " " if count >= target else "*"
            line += f" | {count:>5}{marker}"
        n_videos = batch_rows["video_uid"].nunique()
        line += f" | {n_videos:>6}"
        print(line)

    # Targets row
    line = f"{'Target':>6} | {sum(class_targets.values()):>5}"
    for cls in ACTIVE_CLASSES:
        line += f" | {class_targets[cls]:>6}"
    line += f" |      -"
    print("-" * len(header))
    print(line)

    # Totals row
    line = f"{'Total':>6} | {len(result):>5}"
    for cls in ACTIVE_CLASSES:
        count = int((result["action"] == cls).sum())
        line += f" | {count:>6}"
    n_videos = result["video_uid"].nunique()
    line += f" | {n_videos:>6}"
    print(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assign annotation batches using power allocation (p^alpha)"
    )
    parser.add_argument(
        "--labels-csv",
        default="data/labels/action_labels_llm_clean_refined.csv",
    )
    parser.add_argument("--num-batches", type=int, default=12)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2500,
        help="Total rows per batch (default: 2500)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Power allocation exponent: 0=equal, 0.5=sqrt, 1=proportional (default: 0.5)",
    )
    parser.add_argument("--round-id", default="r001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prioritize-frequent",
        action="store_true",
        help="Prioritize high-frequency narrations during per-class sampling.",
    )
    parser.add_argument(
        "--min-narration-frequency",
        type=int,
        default=1,
        help="Keep rows whose narration key frequency >= N (default: 1 = no filter).",
    )
    parser.add_argument(
        "--frequency-key",
        choices=["raw", "no_articles", "nouns_verbs"],
        default="raw",
        help="Key used to compute narration frequency. Use nouns_verbs for very aggressive grouping.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/labels/action_labels_llm_clean_refined.csv",
        help="Output CSV (can be same as input to update in-place)",
    )
    args = parser.parse_args()

    labels_path = Path(args.labels_csv)
    if not labels_path.exists():
        print(f"Error: Labels CSV not found: {labels_path}")
        return 1

    print(f"Loading labels: {labels_path}")
    df = pd.read_csv(labels_path)
    print(f"Total rows: {len(df):,}")

    # Ensure batch/round columns exist and can store string labels like B01/r002
    if "batch" not in df.columns:
        df["batch"] = pd.NA
    if "round" not in df.columns:
        df["round"] = pd.NA
    df["batch"] = df["batch"].astype("string")
    df["round"] = df["round"].astype("string")

    # Pool: active classes with no existing batch assignment
    pool = df[
        df["action"].isin(ACTIVE_CLASSES)
        & df["batch"].isna()
    ].copy()
    print(f"Unassigned active-class rows (pool): {len(pool):,}")

    if pool.empty:
        print("Error: No unassigned rows available.")
        return 1

    # Optional narration-frequency filtering/prioritization
    if args.prioritize_frequent or args.min_narration_frequency > 1:
        print(
            f"\nBuilding narration frequency key (mode={args.frequency_key}) "
            f"for prioritize/min-frequency filtering..."
        )
        pool["_freq_key"] = build_frequency_key(pool["narration_text"], args.frequency_key)
        freq = pool.groupby("_freq_key").size()
        pool["_narr_freq"] = pool["_freq_key"].map(freq).astype(int)
        print(
            f"  Frequency groups: {len(freq):,} | "
            f"max freq: {int(freq.max()):,} | median freq: {float(freq.median()):.1f}"
        )

        if args.min_narration_frequency > 1:
            before = len(pool)
            pool = pool[pool["_narr_freq"] >= args.min_narration_frequency].copy()
            after = len(pool)
            print(
                f"  Applied min frequency >= {args.min_narration_frequency}: "
                f"{before:,} -> {after:,} rows"
            )
            if pool.empty:
                print("Error: Pool became empty after min-narration-frequency filter.")
                return 1

    # Compute class distribution in pool
    class_counts = {cls: int((pool["action"] == cls).sum()) for cls in ACTIVE_CLASSES}
    print(f"\nPool class distribution:")
    for cls in ACTIVE_CLASSES:
        print(f"  {cls:<20s}: {class_counts[cls]:>7,} ({class_counts[cls]/len(pool)*100:.1f}%)")

    # Compute power allocation targets
    class_targets = compute_power_allocation(class_counts, args.batch_size, args.alpha)
    total_per_batch = sum(class_targets.values())
    total_all = total_per_batch * args.num_batches

    print(f"\nPower allocation (alpha={args.alpha}):")
    print(f"  Per batch ({total_per_batch} total):")
    for cls in ACTIVE_CLASSES:
        pct = class_targets[cls] / total_per_batch * 100
        boost = (class_targets[cls] / total_per_batch) / (class_counts[cls] / len(pool))
        print(f"    {cls:<20s}: {class_targets[cls]:>5} ({pct:.1f}%, {boost:.2f}x boost)")
    print(f"  Total across {args.num_batches} batches: {total_all:,}")

    # Check feasibility
    print(f"\nFeasibility check:")
    feasible = True
    for cls in ACTIVE_CLASSES:
        needed = class_targets[cls] * args.num_batches
        available = class_counts[cls]
        status = "OK" if available >= needed else "SHORT"
        if status == "SHORT":
            feasible = False
        print(f"  {cls:<20s}: {available:>7,} available, {needed:>6,} needed [{status}]")

    if not feasible:
        print("\nWarning: Some classes may be short. Will sample as many as available.")

    # Step 1: Video class profiles
    profiles = compute_video_class_profiles(pool)
    print(f"\nVideos in pool: {len(profiles):,}")

    # Step 2: Assign videos to batches
    print(f"Assigning videos to {args.num_batches} batches...")
    batch_videos = assign_videos_to_batches(
        profiles, args.num_batches, class_targets
    )

    for b in range(1, args.num_batches + 1):
        n = len(batch_videos[b])
        search_avail = sum(profiles[v].get("Search", 0) for v in batch_videos[b])
        print(f"  Batch {b:>2}: {n:>4} videos, {search_avail:>4} Search available")

    # Step 3: Sample rows
    print(f"\nSampling rows...")
    sampled = sample_rows_per_batch(
        pool,
        batch_videos,
        class_targets,
        args.seed,
        prioritize_frequent=args.prioritize_frequent,
    )

    if sampled.empty:
        print("Error: No rows sampled.")
        return 1

    # Print summary
    print_summary(sampled, args.num_batches, class_targets)

    # Step 4: Format batch labels as "B01"–"B12" to avoid substring collisions
    # (Label Studio "contains" filter: "2" would match both "2" and "12")
    sampled["_batch_label"] = sampled["_batch"].apply(lambda x: f"B{x:02d}")

    # Write back to full DataFrame
    assigned_indices = sampled.index
    df.loc[assigned_indices, "batch"] = sampled.loc[assigned_indices, "_batch_label"]
    df.loc[assigned_indices, "round"] = args.round_id

    # Summary counts
    assigned_total = df["batch"].notna().sum()
    unassigned_total = df["batch"].isna().sum()
    print(f"\nFull CSV summary:")
    print(f"  Total rows:      {len(df):,}")
    print(f"  Assigned:        {assigned_total:,} (batch/round populated)")
    print(f"  Unassigned:      {unassigned_total:,} (available for future rounds)")

    # Save full CSV
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved full CSV to {out_path}")

    # Save assigned-only CSV
    assigned_df = df[df["batch"].notna()].copy()
    assigned_df = assigned_df.sort_values(
        ["round", "batch", "video_uid", "timestamp_sec"]
    ).reset_index(drop=True)
    assigned_path = out_path.with_name(
        out_path.stem + "_assigned_only" + out_path.suffix
    )
    assigned_df.to_csv(assigned_path, index=False)
    print(f"Saved assigned-only CSV to {assigned_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
