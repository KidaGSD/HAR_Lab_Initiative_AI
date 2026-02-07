"""
Run the Stage 2.5 "LLM Strict Correction" logic outside Jupyter.

Why:
- Jupyter kernels may stop when your laptop disconnects.
- This script can be run under nohup/tmux so it keeps running on SSH.

What it does:
- Reads a labels CSV.
- Finds rows where action == "Unknown" (configurable via --target-action).
- Uses Qwen2.5 (via vLLM) with t-3 context to relabel them.
- STRICT RULE: final label cannot be "Error / Correction" or "Unknown".
- Does NOT modify `status`.
- Adds "[CtxFixed]" prefix to `reasoning` for changed rows.
- Prints a full change report at the end and writes a change log CSV.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from vllm import LLM, SamplingParams


# Allowed final taxonomy for automatic fixes.
# NOTE: Excludes both "Error / Correction" and "Unknown".
FORCED_TAXONOMY = [
    "Locomotion",
    "Essential Operation",
    "Object Transfer",
    "Search",
    "Stationary",
]

# If the model output is invalid / unparseable, we fall back to context, otherwise to this.
FALLBACK_LABEL = "Obeject Transfer"


SYSTEM_PROMPT_CONTEXT_TEMPLATE = """You are an expert at analyzing human behavior logs.
You will see a sequence of 4 actions (t-3, t-2, t-1, Current).
The Current Action is labeled as 'Unknown', which is INVALID for this task.
You MUST re-classify the action into one of the valid categories below.

Valid Taxonomy: {taxonomy_json}

Instructions:
- If it looks like a deliberate placement (e.g., dropping ONTO surface, putting down, repetitive motion), label as 'Object Transfer'.
- Otherwise, you can label with any of the taxonomy labels
- Do NOT output 'Error / Correction' or 'Unknown'.

Output JSON:
{{
  "new_label": "Category",
  "reasoning": "Why context matters..."
}}
"""


def _safe_parse_json_from_text(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        code_block = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block:
            return json.loads(code_block.group(1))
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        return None
    return None


def get_context_prompt(df: pd.DataFrame, idx: int, window: int = 3) -> str:
    current_row = df.loc[idx]
    uid = current_row["video_uid"]
    video_df = df[df["video_uid"] == uid].sort_values("timestamp_sec")
    video_df_indices = video_df.index.tolist()

    try:
        pos = video_df_indices.index(idx)
    except ValueError:
        return ""

    start_pos = max(0, pos - window)
    context_indices = video_df_indices[start_pos:pos]

    context_str = ""
    for i, ctx_idx in enumerate(context_indices):
        row = df.loc[ctx_idx]
        offset = len(context_indices) - i
        context_str += f"t-{offset}: {row['narration_text']} (Label: {row['action']})\n"

    return context_str


def fallback_label_from_context(df: pd.DataFrame, idx: int, window: int = 3) -> str:
    """Pick a deterministic fallback label based on recent context (t-1, t-2, t-3)."""
    current_row = df.loc[idx]
    uid = current_row["video_uid"]
    video_df = df[df["video_uid"] == uid].sort_values("timestamp_sec")
    video_df_indices = video_df.index.tolist()

    try:
        pos = video_df_indices.index(idx)
    except ValueError:
        return FALLBACK_LABEL

    start_pos = max(0, pos - window)
    context_indices = list(reversed(video_df_indices[start_pos:pos]))  # t-1 first

    for ctx_idx in context_indices:
        lbl = str(df.at[ctx_idx, "action"]).strip()
        if lbl in FORCED_TAXONOMY:
            return lbl

    return FALLBACK_LABEL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output", type=str, default="", help="Output CSV path (default: overwrite input)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct-AWQ", help="HF model id/path")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--gpu-mem", type=float, default=0.5, help="vLLM gpu_memory_utilization")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--context-window", type=int, default=3, help="How many previous lines to include (t-3)")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of error rows to process (0 = all)")
    parser.add_argument("--log-changes", type=str, default="", help="Change log CSV path (default: alongside output)")
    parser.add_argument(
        "--target-action",
        type=str,
        default="Unknown",
        help="Only relabel rows where action == this value (default: Unknown).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    df = pd.read_csv(input_path)

    errors_to_check = df[df["action"] == args.target_action].index.tolist()
    if args.limit and args.limit > 0:
        errors_to_check = errors_to_check[: args.limit]

    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Found {len(errors_to_check)} rows with action == {args.target_action!r} to force-fix.")

    if not errors_to_check:
        print("Nothing to do.")
        if output_path != input_path:
            df.to_csv(output_path, index=False)
            print(f"Saved (unchanged) to {output_path}")
        return

    system_prompt = SYSTEM_PROMPT_CONTEXT_TEMPLATE.format(taxonomy_json=json.dumps(FORCED_TAXONOMY))

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
    )
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    prompts: list[str] = []
    indices: list[int] = []

    tok = llm.get_tokenizer()
    for idx in errors_to_check:
        row = df.loc[idx]
        context_str = get_context_prompt(df, idx, window=args.context_window)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Sequence:\n{context_str}\n\nCurrent Action: {row['narration_text']}\n\nOutput JSON:",
            },
        ]
        text_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text_prompt)
        indices.append(idx)

    print(f"Generating {len(prompts)} responses...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    changes: list[dict[str, Any]] = []
    invalid_or_failed = 0

    for i, output in enumerate(outputs):
        idx = indices[i]
        raw_text = output.outputs[0].text.strip() if output.outputs else ""
        data = _safe_parse_json_from_text(raw_text) or {}
        new_label = (data.get("new_label") or "").strip()
        reasoning = (data.get("reasoning") or "").strip()

        # Hard enforcement: never allow Error / Correction or Unknown.
        if new_label in ("Error / Correction", "Unknown", ""):
            new_label = fallback_label_from_context(df, idx, window=args.context_window)
            invalid_or_failed += 1

        # Enforce allowed taxonomy.
        if new_label not in FORCED_TAXONOMY:
            new_label = fallback_label_from_context(df, idx, window=args.context_window)
            invalid_or_failed += 1

        old_label = df.at[idx, "action"]
        if new_label != old_label:
            df.at[idx, "action"] = new_label
            df.at[idx, "reasoning"] = f"[CtxFixed] {reasoning}".strip()
            changes.append(
                {
                    "row_index": int(idx),
                    "video_uid": df.at[idx, "video_uid"],
                    "timestamp_sec": df.at[idx, "timestamp_sec"],
                    "narration_text": df.at[idx, "narration_text"],
                    "old_action": old_label,
                    "new_action": new_label,
                    "reasoning": reasoning,
                }
            )

    # Save outputs.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Change log.
    log_path = Path(args.log_changes) if args.log_changes else output_path.with_suffix(".changes.csv")
    pd.DataFrame(changes).to_csv(log_path, index=False)

    # Report.
    print("\n=== Change report ===")
    print(f"Total changed rows: {len(changes)} / {len(errors_to_check)}")
    if invalid_or_failed:
        print(f"Forced to 'Unknown' due to invalid label/parse: {invalid_or_failed}")
    if changes:
        # Print all changes at the end (as requested).
        for c in changes:
            print(f"[Fix {c['row_index']}] {c['old_action']} -> {c['new_action']}")
        # And a compact summary table.
        summary = (
            pd.DataFrame(changes)
            .groupby(["old_action", "new_action"])
            .size()
            .sort_values(ascending=False)
        )
        print("\n(old_action -> new_action) counts:")
        print(summary)

    remaining = int((df["action"] == "Error / Correction").sum())
    remaining_unknown = int((df["action"] == "Unknown").sum())
    print(f"\nSaved to {output_path}")
    print(f"Change log saved to {log_path}")
    print(f"Remaining 'Error / Correction' labels in file: {remaining}")
    print(f"Remaining 'Unknown' labels in file: {remaining_unknown}")


if __name__ == "__main__":
    main()

