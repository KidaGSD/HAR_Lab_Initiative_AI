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

# Qwen Prompt Template
PROMPT_TEMPLATE = """You are an expert annotator for IMU sensor data. Your job is to classify the user's action into a single category based on the narration and context.

Categories:
1. Locomotion: Global body movement (walking, stepping, standing up).
2. Manual Work: Hand-object interaction (cutting, holding, typing).
3. Scanning: Visual search ONLY (looking, checking). No hand interaction.
4. Stationary: Passive posture (sitting, waiting). No movement.
5. Unknown: Ambiguous or unclear.

Rules:
- "Move [object]" is Manual Work. "Move to [place]" is Locomotion.
- "Check [object]" is Scanning unless it implies touching/fixing.
- If the narration implies BOTH walking and carrying, prioritize Manual Work if the hand motion is dominant, or Unknown if unclear.
- Be conservative. If unsure, output Unknown.

Examples:
Narration: "C walks to the sink" (Context: Cooking) -> Locomotion
Narration: "C cuts the carrot" (Context: Cooking) -> Manual Work
Narration: "C moves the pan" (Context: Cooking) -> Manual Work
Narration: "C looks for the salt" (Context: Cooking) -> Scanning
Narration: "C waits for water to boil" (Context: Cooking) -> Stationary
Narration: "C moves" (Context: Unknown) -> Unknown

Task:
Narration: "{narration}"
Context: "{scenario}"
Label:"""

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
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10) # Deterministic
    
    # 3. Prepare Prompts
    prompts = []
    for item in data:
        prompt = PROMPT_TEMPLATE.format(
            narration=item['narration_text'],
            scenario=item['scenario']
        )
        prompts.append(prompt)
        
    # 4. Generate
    print("Generating labels...")
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
