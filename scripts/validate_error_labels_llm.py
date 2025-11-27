#!/usr/bin/env python3
"""
Re-validate Error/Correction labels using Qwen LLM.

This script uses Qwen3-8B to intelligently distinguish between:
- Intentional actions mislabeled as errors (e.g., "drops X on table")
- Genuine errors that should remain as Error/Correction

Based on scripts/label_with_qwen.py but with a specialized validation prompt.
"""

import json
import pandas as pd
import argparse
from pathlib import Path
import os
import re

# Try importing vllm
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("WARNING: vLLM not available. This script requires vLLM to run.")

# --- Configuration ---
INPUT_CSV = 'data/labels/action_labels_llm_clean.csv'
OUTPUT_CSV = 'data/labels/action_labels_llm_validated.csv'

# Validation Prompt (Specialized for Error/Correction re-evaluation)
VALIDATION_SYSTEM_PROMPT = """You are an expert at analyzing human actions from narration text.

Your task is to determine if an action labeled as "Error / Correction" is actually:
1. **Object Transfer**: Intentional placement/movement of objects (e.g., "drops bowl on table")
2. **Essential Operation**: Core manual task (e.g., "adjusts the cutting board")
3. **Error / Correction**: Genuine accident/mistake (e.g., "spills water", "drops glass")

Key Guidelines:
- "drops X on [surface]" is usually INTENTIONAL placement → Object Transfer
- "adjusts X" is usually deliberate → Object Transfer or Essential Operation
- "spills", "fumbles", "accidentally" indicate genuine errors
- Broken items (glass, plate) are usually errors unless placed intentionally

Respond with JSON:
{
  "is_error": true/false,
  "correct_label": "Object Transfer" | "Essential Operation" | "Error / Correction",
  "reasoning": "Brief explanation"
}

Examples:

Narration: "C drops the bowl on the table."
Output: {"is_error": false, "correct_label": "Object Transfer", "reasoning": "Intentional placement on surface."}

Narration: "C adjusts the cutting board."
Output: {"is_error": false, "correct_label": "Object Transfer", "reasoning": "Deliberate repositioning."}

Narration: "C drops the glass cup."
Output: {"is_error": true, "correct_label": "Error / Correction", "reasoning": "Fragile item dropped, likely accident."}

Narration: "C spills water on the counter."
Output: {"is_error": true, "correct_label": "Error / Correction", "reasoning": "Spilling is unintentional."}
"""

USER_PROMPT_TEMPLATE = """Scenario: {scenario}
Original Action: "{narration}"
Current Label: Error / Correction

Is this truly an error, or should it be relabeled?

Output JSON:"""


def load_error_labels(input_path: str) -> pd.DataFrame:
    """Load CSV and filter for Error/Correction labels only"""
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Filter for Error/Correction labels
    error_df = df[df['action'] == 'Error / Correction'].copy()
    
    print(f"Total rows: {len(df):,}")
    print(f"Error/Correction labels to validate: {len(error_df):,}")
    
    return df, error_df


def main(args):
    if not VLLM_AVAILABLE:
        print("Error: vLLM is required. Please install it:")
        print("  pip install vllm>=0.8.5")
        return
    
    # 1. Load Data
    print("Loading data...")
    full_df, error_df = load_error_labels(args.input)
    
    if args.limit:
        error_df = error_df.head(args.limit)
    
    print(f"Processing {len(error_df)} Error/Correction labels...")
    
    # 2. Initialize Model
    print(f"Initializing Qwen model: {args.model}")
    try:
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            tensor_parallel_size=args.gpus,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        print("Try lowering --gpu_memory_utilization or --max_model_len")
        return

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=1024,  # Shorter since we just need validation
        stop=["\n\n\n"]
    )
    
    # 3. Process in Batches
    BATCH_SIZE = 1000  # Smaller batches for validation
    total_processed = 0
    
    # Track corrections
    corrections = []
    
    print(f"Processing in batches of {BATCH_SIZE}...")
    
    import re
    tokenizer = llm.get_tokenizer()
    
    for i in range(0, len(error_df), BATCH_SIZE):
        batch_data = error_df.iloc[i : i + BATCH_SIZE]
        print(f"Processing batch {i} to {i + len(batch_data)}...")
        
        # Prepare Prompts
        prompts = []
        for idx, row in batch_data.iterrows():
            user_content = USER_PROMPT_TEMPLATE.format(
                narration=row['narration_text'],
                scenario=row['scenario']
            )
            
            messages = [
                {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            
            prompts.append(prompt)
        
        # Generate
        outputs = llm.generate(prompts, sampling_params)
        
        # Parse Results
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            
            idx = batch_data.index[j]
            original_row = batch_data.iloc[j]
            
            # Parse JSON
            new_label = 'Error / Correction'  # Default to keeping it
            is_error = True
            reasoning = ''
            
            try:
                # Extract JSON
                decoder = json.JSONDecoder()
                valid_objs = []
                
                for k in range(len(generated_text)):
                    if generated_text[k] == '{':
                        try:
                            obj, _ = decoder.raw_decode(generated_text[k:])
                            if isinstance(obj, dict):
                                valid_objs.append(obj)
                        except json.JSONDecodeError:
                            continue
                
                if valid_objs:
                    data_dict = valid_objs[-1]
                    is_error = data_dict.get('is_error', True)
                    new_label = data_dict.get('correct_label', 'Error / Correction')
                    reasoning = data_dict.get('reasoning', '')
                    
                    # Validate the new label is in our taxonomy
                    valid_labels = [
                        'Locomotion', 'Essential Operation', 'Object Transfer',
                        'Search', 'Error / Correction', 'Stationary'
                    ]
                    if new_label not in valid_labels:
                        new_label = 'Error / Correction'
                        reasoning = f"Invalid label returned: {new_label}. Keeping as Error."
            
            except Exception as e:
                new_label = 'Error / Correction'
                reasoning = f"Parse error: {str(e)}"
            
            # Record correction if label changed
            if new_label != 'Error / Correction':
                corrections.append({
                    'index': idx,
                    'video_uid': original_row['video_uid'],
                    'timestamp_sec': original_row['timestamp_sec'],
                    'narration_text': original_row['narration_text'],
                    'scenario': original_row['scenario'],
                    'old_label': 'Error / Correction',
                    'new_label': new_label,
                    'new_reasoning': f"[LLM Validated] {reasoning}",
                    'llm_output': generated_text
                })
        
        total_processed += len(batch_data)
    
    # 4. Apply corrections to full dataframe
    print(f"\nApplying {len(corrections)} corrections...")
    
    for correction in corrections:
        idx = correction['index']
        full_df.at[idx, 'action'] = correction['new_label']
        full_df.at[idx, 'reasoning'] = correction['new_reasoning']
    
    # 5. Save results
    print(f"Saving validated labels to {args.output}...")
    full_df.to_csv(args.output, index=False)
    
    # 6. Generate report
    report_path = args.output.replace('.csv', '_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\\n")
        f.write("ERROR LABEL VALIDATION REPORT (LLM-Based)\\n")
        f.write("=" * 80 + "\\n\\n")
        
        f.write(f"Error labels processed: {total_processed:,}\\n")
        f.write(f"Labels corrected: {len(corrections):,} ({len(corrections)/total_processed*100:.1f}%)\\n")
        f.write(f"Labels kept as Error: {total_processed - len(corrections):,}\\n\\n")
        
        # Count new labels
        new_label_counts = {}
        for c in corrections:
            new_label_counts[c['new_label']] = new_label_counts.get(c['new_label'], 0) + 1
        
        f.write("Corrected to:\\n")
        for label, count in sorted(new_label_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {label}: {count:,}\\n")
        
        f.write("\\n" + "=" * 80 + "\\n")
        f.write("SAMPLE CORRECTIONS (First 20)\\n")
        f.write("=" * 80 + "\\n\\n")
        
        for i, c in enumerate(corrections[:20], 1):
            f.write(f"{i}. {c['narration_text']}\\n")
            f.write(f"   Scenario: {c['scenario']}\\n")
            f.write(f"   {c['old_label']} → {c['new_label']}\\n")
            f.write(f"   Reasoning: {c['new_reasoning']}\\n\\n")
    
    print(f"Saved validation report to {report_path}")
    
    # Print summary
    print("\\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Error labels validated: {total_processed:,}")
    print(f"Labels corrected: {len(corrections):,} ({len(corrections)/total_processed*100:.1f}%)")
    print(f"Labels kept as Error: {total_processed - len(corrections):,}")
    
    if new_label_counts:
        print("\\nCorrected to:")
        for label, count in sorted(new_label_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {count:,}")
    
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-validate Error/Correction labels using Qwen LLM"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_CSV,
        help="Input CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_CSV,
        help="Output CSV file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model path (HuggingFace)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples for testing"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=2,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use (default: 0.9). Lower this if OOM occurs."
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Maximum context length (default: 8192). Lower this (e.g., 4096) to save memory."
    )
    
    args = parser.parse_args()
    main(args)
