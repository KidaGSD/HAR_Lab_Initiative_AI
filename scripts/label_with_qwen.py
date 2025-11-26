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
    # ... (same as before) ...
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
    sampling_params = SamplingParams(
        temperature=0.6, # Recommended for thinking mode
        top_p=0.95,
        max_tokens=1024,  # Increased for thinking content
        stop=["\n\n", "Scenario:", "Action:"] 
    )
    
    # 3. Process in Batches
    BATCH_SIZE = 5000
    total_processed = 0
    
    # Initialize output file with header if it doesn't exist
    if not os.path.exists(OUTPUT_PATH):
        pd.DataFrame(columns=['video_uid', 'timestamp_sec', 'narration_text', 'scenario', 'action', 'reasoning', 'llm_raw_output']).to_csv(OUTPUT_PATH, index=False)
    
    print(f"Processing in batches of {BATCH_SIZE}...")
    
    for i in range(0, len(data), BATCH_SIZE):
        batch_data = data[i : i + BATCH_SIZE]
        print(f"Processing batch {i} to {i + len(batch_data)}...")
        
        # Prepare Prompts
        prompts = []
        for item in batch_data:
            prompt = f"""{SYSTEM_PROMPT}

{USER_PROMPT_TEMPLATE.format(
    narration=item['narration_text'],
    scenario=item['scenario']
)}"""
            prompts.append(prompt)
            
        # Generate
        outputs = llm.generate(prompts, sampling_params)
        
        # Parse Results
        results = []
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            
            # Parse JSON (robust to <think> blocks)
            try:
                # Find the LAST valid JSON block
                end = generated_text.rfind('}') + 1
                if end == 0:
                    raise ValueError("No JSON end found")
                
                start = generated_text.rfind('{', 0, end)
                if start == -1:
                    raise ValueError("No JSON start found")
                    
                json_str = generated_text[start:end]
                data_dict = json.loads(json_str)
                
                label = data_dict.get('label', 'Unknown')
                reasoning = data_dict.get('reasoning', '')
                
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
