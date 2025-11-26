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
# Simple classification prompt - just return the label
SYSTEM_PROMPT = """You are an expert at classifying human actions from narration text. You must respond with ONLY ONE WORD from these options:
- Locomotion (walking, running, climbing, moving body)
- Manual Work (using hands to manipulate objects, tools, or environment)
- Scanning (looking, searching visually)
- Stationary (sitting, standing still, waiting, talking)
- Unknown (only if completely ambiguous or irrelevant)

Your goal is to use the SCENARIO CONTEXT to interpret the action.

Examples of Contextual Reasoning:

1. Verb "Move"
   Scenario: Cooking
   Action: "C moves the pan"
   Label: Manual Work

   Scenario: Cooking
   Action: "C moves to the sink"
   Label: Locomotion

   Scenario: Walking Outdoors
   Action: "C moves down the walkway"
   Label: Locomotion

2. Verb "Check"
   Scenario: Relaxing
   Action: "C checks the phone"
   Label: Scanning

   Scenario: Car Repair
   Action: "C checks the tire pressure"
   Label: Manual Work

3. Verb "Stand"
   Scenario: Talking
   Action: "C stands by the door"
   Label: Stationary

   Scenario: Any
   Action: "C stands up"
   Label: Locomotion

4. Clear Actions
   Scenario: Any
   Action: "C walks down the hall"
   Label: Locomotion

   Scenario: Any
   Action: "C takes the bowl"
   Label: Manual Work

5. Unknown / Irrelevant
   Scenario: Any
   Action: "C is visible"
   Label: Unknown

   Scenario: Any
   Action: "Camera moves"
   Label: Unknown

Respond with ONLY the category name."""

USER_PROMPT_TEMPLATE = """Scenario: {scenario}
Action: {narration}

Category:"""

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
    if not VLLM_AVAILABLE:
        print("Error: vllm is not installed. Please install it on the GPU server:")
        print("pip install vllm")
        return

    # 1. Load Data
    data = load_data()
    
    if args.limit:
        data = data[:args.limit]
        print(f"Limiting to first {args.limit} samples.")
        
    # 2. Initialize Model
    print(f"Initializing Qwen model: {args.model}")
    llm = LLM(model=args.model, trust_remote_code=True, tensor_parallel_size=args.gpus)
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=100,  
        stop=["\n", "Narration:", "Scenario:"]  # Stop at newlines or next prompt
    )
    
    # 3. Prepare Prompts as simple strings
    prompts = []
    for item in data:
        # Simple combined prompt
        prompt = f"""{SYSTEM_PROMPT}

{USER_PROMPT_TEMPLATE.format(
    narration=item['narration_text'],
    scenario=item['scenario']
)}"""
        prompts.append(prompt)
        
    # 4. Generate
    print("Generating labels...")
    if len(prompts) > 0:
        print(f"DEBUG: Type of first prompt: {type(prompts[0])}")
        print(f"DEBUG: First prompt content: {prompts[0]!r}")
        
    outputs = llm.generate(prompts, sampling_params)
    
    # 5. Parse Results
    results = []
    valid_labels = {'Locomotion', 'Manual Work', 'Scanning', 'Stationary', 'Unknown'}
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        
        # Basic cleaning
        label = generated_text.split('\n')[0].strip()
        
        # Validation
        if label not in valid_labels:
            # Try to find valid label in text
            found = False
            for v in valid_labels:
                if v.lower() in label.lower():
                    label = v
                    found = True
                    break
            if not found:
                label = 'Unknown'
        
        item = data[i]
        results.append({
            'video_uid': item['video_uid'],
            'timestamp_sec': item['timestamp_sec'],
            'narration_text': item['narration_text'],
            'scenario': item['scenario'],
            'action': label,
            'llm_raw_output': generated_text
        })
        
    # 6. Save
    df = pd.DataFrame(results)
    
    # Filter out Unknowns for the final dataset? 
    # Or keep them for analysis? Let's keep them but save separate files.
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved all labels to {OUTPUT_PATH}")
    
    # Clean version
    df_clean = df[df['action'] != 'Unknown']
    clean_path = OUTPUT_PATH.replace('.csv', '_clean.csv')
    df_clean.to_csv(clean_path, index=False)
    print(f"Saved clean labels ({len(df_clean)}) to {clean_path}")
    
    # Stats
    print("\nLabel Distribution:")
    print(df['action'].value_counts(normalize=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-14B-Chat-AWQ", help="Model path (HuggingFace)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    args = parser.parse_args()
    main(args)
