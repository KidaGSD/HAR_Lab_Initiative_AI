LLM-Based Labeling Strategy (v3: IMU-Focused Granular Labels)
The Goal
To generate labels that are not only semantically meaningful but also distinguishable by IMU sensors.

The Taxonomy (6 Classes)
We merged "Search" and "Scanning" because both involve head movement with passive hands.

Locomotion: Moving body through space (walk, run, climb).
IMU Signature: High Body Accel/Gyro, rhythmic.
Essential Operation: Performing the core task (cut, wash, mix).
IMU Signature: High Hand Accel, irregular/complex.
Object Transfer: Logistics/Setup (pick up, put down).
IMU Signature: Short bursts of Hand Accel, distinct from continuous operation.
Search: Visual search or monitoring (looking for item, checking time).
IMU Signature: High Head Rotation (Gyro), Low Hand Accel.
Error / Correction: Explicit failure, fumbling, dropping.
IMU Signature: Jerky/Sudden motion, breaks in rhythm.
Stationary: Idle, waiting, talking.
IMU Signature: Low energy on all sensors.
Implementation (Qwen Prompt)
We ask Qwen to output a JSON object with the label and reasoning:

{
  "label": "Search",
  "reasoning": "Head movement (visual search), hands mostly still."
}
3. Prompt Design (Few-Shot)
You are an expert annotator for IMU sensor data. Your job is to classify the user's action into a single category based on the narration and context.
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
Narration: "C moves" (Context: Unknown) -> Unknown (Too ambiguous)
Task:
Narration: "{narration}"
Context: "{scenario}"
Label:
4. Implementation Plan
Environment: Install vllm and transformers on the GPU server.
1. Model Selection
Model: Qwen/Qwen3-14B (Latest generation with "Thinking Mode").
Why: Superior reasoning capabilities for complex contextual disambiguation.
Engine: vllm (requires version >= 0.8.5).
Batch Processing: Process all ~300k narrations in batches.
Validation: Manually review 100 random LLM labels vs Keyword labels to quantify improvement.
Advantages
Context Sensitivity: Distinguishes "move object" vs "move self".
High Precision: Explicit "Unknown" class filters noise.
Scalable: vllm can process thousands of tokens per second on RTX 6000.
