import json
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import os

# Try importing vllm, but don't fail if not installed (for local dev)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# --- Configuration ---
NARRATIONS_PATH = 'data/ego4d_data/v2/annotations/narration.json'
SCENARIO_LABELS_PATH = 'data/labels/scenario_labels.csv'
OUTPUT_PATH = 'data/labels/action_labels_llm.csv'

# Simple classification prompt - just return the label
# IMU-Focused Low-Level Labeling Prompt
SYSTEM_PROMPT = """You are an expert at analyzing human behavior from narration text. 
Your goal is to classify the action into a category that corresponds to a distinct IMU/Motion signature.

Taxonomy (Choose exactly one):
1. Locomotion: Body moving through space (walk, run, climb, stand up). High Body Accel.
2. Essential Operation: Core manual task (cut, wash, mix, screw, type). High Hand Accel, Irregular.
3. Object Transfer: Logistics (pick up, put down, open drawer, take). Short bursts of Hand Accel.
4. Search: Visual search or monitoring (looking for item, checking time, inspecting). High Head Rotation, Low Hand Accel.
5. Error / Correction: Explicit failure, fumbling, dropping, spilling. Jerky/Irregular motion.
6. Stationary: Idle, waiting, talking, sitting. Low Energy.

Respond with a JSON object:
{
  "label": "Category Name",
  "reasoning": "Brief explanation"
}

Examples:
Scenario: Cooking
Action: "C cuts the carrot"
Output: {"label": "Essential Operation", "reasoning": "Core task, high hand activity."}

Scenario: Cooking
Action: "C takes the knife from the drawer"
Output: {"label": "Object Transfer", "reasoning": "Logistics/Setup step."}

Scenario: Cooking
Action: "C looks for the salt"
Output: {"label": "Search", "reasoning": "Head movement (visual search), hands mostly still."}

Scenario: Any
Action: "C checks the time"
Output: {"label": "Search", "reasoning": "Head movement (checking), passive hands."}

Scenario: Cleaning
Action: "C drops the bowl"
Output: {"label": "Error / Correction", "reasoning": "Jerky/Failure motion."}

Scenario: Any
Action: "C talks to X"
Output: {"label": "Stationary", "reasoning": "Low body/hand motion."}
"""

USER_PROMPT_TEMPLATE = """Scenario: {scenario}
Action: {narration}

Output JSON:"""

def load_data():
    print("Loading metadata...")
    # Load Scenarios (Context)
    scenario_df = pd.read_csv(SCENARIO_LABELS_PATH).set_index('video_uid')
    
    # Load Narrations
    print(f"Loading narrations from {NARRATIONS_PATH}...")
    with open(NARRATIONS_PATH, 'r') as f:
        all_narrations = json.load(f)
        
    # Filter for target videos
    target_uids = set(scenario_df.index)
    
    data_to_process = []
    
    for uid, video_data in all_narrations.items():
        if uid not in target_uids:
            continue
            
        scenario = scenario_df.loc[uid, 'scenario']
        
        # Handle nested structure
        if 'narration_pass_1' in video_data and 'narrations' in video_data['narration_pass_1']:
            narr_list = video_data['narration_pass_1']['narrations']
        elif 'narration_pass_2' in video_data and 'narrations' in video_data['narration_pass_2']:
            narr_list = video_data['narration_pass_2']['narrations']
        else:
            continue
            
        for item in narr_list:
            data_to_process.append({
                'video_uid': uid,
                'timestamp_sec': item['timestamp_sec'],
                'narration_text': item['narration_text'],
                'scenario': scenario
            })
            
    print(f"Found {len(data_to_process)} narrations to label.")
    return data_to_process

def main(args):
    # 1. Load Data
    print("Loading data...")
    data = load_data()
    if args.limit:
        data = data[:args.limit]
    print(f"Loaded {len(data)} narrations.")

    # 2. Initialize Model
    print(f"Initializing Qwen model: {args.model}")
    llm = LLM(model=args.model, trust_remote_code=True, tensor_parallel_size=args.gpus)
    # 2. Initialize Model
    print(f"Initializing Qwen model: {args.model}")
    llm = LLM(model=args.model, trust_remote_code=True, tensor_parallel_size=args.gpus)
    sampling_params = SamplingParams(
        temperature=0.6, 
        top_p=0.95,
        max_tokens=2048,  # Increased to 2048 to prevent truncation of thinking + JSON
        stop=["\n\n\n"]   # Relaxed stop tokens
    )
    
    # ... (batch loop) ...

        # Parse Results
        results = []
        import re # Import regex
        
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            
            # Parse JSON (Robust Regex Method)
            try:
                # 1. Try finding a code block first ```json ... ```
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', generated_text, re.DOTALL)
                if not json_match:
                    # 2. Try finding any JSON-like object { ... }
                    # This regex looks for the last balanced brace pair if possible, 
                    # or just the last { ... } block.
                    # Qwen3 often puts the JSON at the very end.
                    json_match = re.search(r'(\{.*\})', generated_text, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1)
                    data_dict = json.loads(json_str)
                    label = data_dict.get('label', 'Unknown')
                    reasoning = data_dict.get('reasoning', '')
                else:
                    raise ValueError("No JSON pattern found")
                
            except Exception as e:
                label = 'Unknown'
                reasoning = f"Error: {str(e)}"
            
            item = batch_data[j]
            results.append({
                'video_uid': item['video_uid'],
                'timestamp_sec': item['timestamp_sec'],
                'narration_text': item['narration_text'],
                'scenario': item['scenario'],
                'action': label,
                'reasoning': reasoning,
                'llm_raw_output': generated_text
            })
            
        # Save Batch
        df_batch = pd.DataFrame(results)
        df_batch.to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
        print(f"Saved batch to {OUTPUT_PATH}")
        total_processed += len(batch_data)

    print(f"\nDone! Processed {total_processed} items.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B", help="Model path (HuggingFace)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    args = parser.parse_args()
    main(args)
