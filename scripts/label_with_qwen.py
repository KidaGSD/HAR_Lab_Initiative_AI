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
    sampling_params = SamplingParams(
        temperature=0.6, 
        top_p=0.95,
        max_tokens=2048,  # Increased to 2048 to prevent truncation of thinking + JSON
        stop=["\n\n\n"]   # Relaxed stop tokens
    )
    
    # 3. Process in Batches
    BATCH_SIZE = 5000
    total_processed = 0
    
    # Initialize output files with header if they don't exist
    CLEAN_OUTPUT_PATH = OUTPUT_PATH.replace('.csv', '_clean.csv')
    
    processed_keys = set()
    if os.path.exists(OUTPUT_PATH):
        try:
            # Read existing file to find processed items
            # We only need video_uid and timestamp_sec to identify unique items
            df_existing = pd.read_csv(OUTPUT_PATH, usecols=['video_uid', 'timestamp_sec'])
            for _, row in df_existing.iterrows():
                key = f"{row['video_uid']}_{float(row['timestamp_sec']):.4f}"
                processed_keys.add(key)
            print(f"Found {len(processed_keys)} already processed items. Resuming...")
        except Exception as e:
            print(f"Warning: Could not read existing file to resume: {e}")

    if not os.path.exists(OUTPUT_PATH):
        pd.DataFrame(columns=['video_uid', 'timestamp_sec', 'narration_text', 'scenario', 'action', 'reasoning', 'thinking_process', 'llm_raw_output']).to_csv(OUTPUT_PATH, index=False)
        
    if not os.path.exists(CLEAN_OUTPUT_PATH):
        pd.DataFrame(columns=['video_uid', 'timestamp_sec', 'narration_text', 'scenario', 'action', 'reasoning']).to_csv(CLEAN_OUTPUT_PATH, index=False)
    
    # Filter data to skip processed items
    data_to_run = []
    for item in data:
        key = f"{item['video_uid']}_{float(item['timestamp_sec']):.4f}"
        if key not in processed_keys:
            data_to_run.append(item)
            
    print(f"Remaining items to process: {len(data_to_run)} (Skipped {len(data) - len(data_to_run)})")
    data = data_to_run
    
    print(f"Processing in batches of {BATCH_SIZE}...")
    
    import re # Import regex
    
    for i in range(0, len(data), BATCH_SIZE):
        batch_data = data[i : i + BATCH_SIZE]
        print(f"Processing batch {i} to {i + len(batch_data)}...")
        
        # Prepare Prompts using Chat Template (Critical for Qwen3 Thinking Mode)
        prompts = []
        tokenizer = llm.get_tokenizer()
        
        for item in batch_data:
            user_content = USER_PROMPT_TEMPLATE.format(
                narration=item['narration_text'],
                scenario=item['scenario']
            )
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            
            # Use apply_chat_template to ensure <think> tokens are handled correctly
            # enable_thinking=True is default for Qwen3, but explicit is safer if supported by template
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    # enable_thinking=True # Uncomment if tokenizer supports it explicitly in kwargs, 
                                         # otherwise it's often default or part of the system prompt handling
                )
            except TypeError:
                # Fallback if enable_thinking kwarg causes error in older transformers
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
            prompts.append(prompt)
            
        # Generate
        outputs = llm.generate(prompts, sampling_params)
        
        # Parse Results
        results = []
        
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            
            # Robust JSON Extraction
            label = 'Unknown'
            reasoning = ''
            thinking_process = ''
            
            try:
                # 0. Extract Thinking Process (<think>...</think>)
                think_match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL)
                if think_match:
                    thinking_process = think_match.group(1).strip()
                else:
                    # Fallback: If no tags, assume everything before the first '{' is thinking
                    first_brace = generated_text.find('{')
                    if first_brace > 0:
                        thinking_process = generated_text[:first_brace].strip()

                # 1. Try extracting from code block first
                code_block = re.search(r'```json\s*(\{.*?\})\s*```', generated_text, re.DOTALL)
                if code_block:
                    json_str = code_block.group(1)
                    data_dict = json.loads(json_str)
                    label = data_dict.get('label', 'Unknown')
                    reasoning = data_dict.get('reasoning', '')
                else:
                    # 2. Scan for valid JSON objects using raw_decode
                    decoder = json.JSONDecoder()
                    valid_objs = []
                    
                    # Search for '{' characters
                    for k in range(len(generated_text)):
                        if generated_text[k] == '{':
                            try:
                                obj, idx = decoder.raw_decode(generated_text[k:])
                                # Check if it looks like our label object
                                if isinstance(obj, dict) and 'label' in obj:
                                    valid_objs.append(obj)
                            except json.JSONDecodeError:
                                continue
                    
                    if valid_objs:
                        # Take the last valid object found (likely the final answer)
                        data_dict = valid_objs[-1]
                        label = data_dict.get('label', 'Unknown')
                        reasoning = data_dict.get('reasoning', '')
                    else:
                        # Fallback: Try to find just the label if JSON failed
                        if '"label":' in generated_text:
                            # Very hacky fallback for partial JSON
                            label_match = re.search(r'"label":\s*"([^"]+)"', generated_text)
                            if label_match:
                                label = label_match.group(1)
                                reasoning = "Extracted via regex fallback"
                            else:
                                raise ValueError("No JSON found")
                        else:
                            raise ValueError("No JSON found")

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
                'thinking_process': thinking_process,
                'llm_raw_output': generated_text
            })
            
        # Save Batch (Full)
        df_batch = pd.DataFrame(results)
        df_batch.to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
        
        # Save Batch (Clean/Lightweight - KEEPS UNKNOWNS)
        df_clean_batch = df_batch[['video_uid', 'timestamp_sec', 'narration_text', 'scenario', 'action', 'reasoning']]
        df_clean_batch.to_csv(CLEAN_OUTPUT_PATH, mode='a', header=False, index=False)
        
        print(f"Saved batch to {OUTPUT_PATH} and {CLEAN_OUTPUT_PATH}")
        total_processed += len(batch_data)

    print(f"\nDone! Processed {total_processed} items.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model path (HuggingFace)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs to use")
    args = parser.parse_args()
    main(args)
