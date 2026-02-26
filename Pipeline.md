# Hierarchical IMU Activity Recognition: Project Master Plan

## 1. Project Vision & Objective

### The Goal: Privacy-Preserving AR Assistance
We are building a system for AR glasses that can **detect when a user is "stuck" or confused** during a physical task (e.g., cooking, carpentry) and offer assistance.

**The Constraint**: We want to do this **without** always-on cameras, which drain battery and raise privacy concerns.
**The Solution**: Use **IMU (Head Motion)** as the primary always-on sensor to track activity, only engaging high-power/privacy-sensitive sensors (or asking the user) when assistance is likely needed.

### The Core Logic
1.  **Track Activity**: Continuously monitor *what* the user is doing (Scenario + Atomic Action) using low-power IMU.
2.  **Detect "Stuck"**: Analyze the *sequence* of actions. If the user is looping, hesitating, or performing errors, they are likely stuck.
3.  **Intervene**: Trigger the "Assistant" (LLM/Camera) only at these critical moments.

---

## 2. The Model: Hierarchical Understanding

To achieve this, our model must understand activity at two levels simultaneously:

### Level 1: The "What" (Scenario)
*   **Goal**: Identify the broad context.
*   **Classes**: `Cooking`, `Carpentry`, `Mechanical Repair`, `Gardening`, etc. (9 Classes).
*   **Why**: Knowing the scenario narrows down the expected steps and valid actions.

### Level 2: The "How" (Atomic Action)
*   **Goal**: Identify the specific physical movement in real-time.
*   **Classes**:
    1.  `Locomotion`: Moving between stations.
    2.  `Essential Operation`: Doing the work (chopping, hammering).
    3.  `Object Transfer`: Moving tools/ingredients.
    4.  `Search`: Looking for items (distinctive head scanning).
    5.  `Error / Correction`: Fumbling, dropping, fixing mistakes.
    6.  `Stationary`: Idle/Thinking.
*   **Why**: The *sequence* of these actions reveals the user's state. A smooth workflow is `Transfer -> Operation -> Transfer`. A stuck workflow might be `Search -> Search -> Stationary -> Error`.

---

## 3. Future Roadmap: The LLM Reasoning Layer

The current model outputs a stream of (Scenario, Action) pairs. The next phase is to feed this stream into an LLM to determine **"Assistance Needed"**.

### The "Stuck" Detector
We will fine-tune or prompt an LLM (e.g., Llama 3, Qwen) to analyze the last 60 seconds of action history.

**Input to LLM**:
> **Context**: Cooking
> **Action Sequence**:
> 00:00 - Locomotion (Walk to fridge)
> 00:05 - Search (Looking inside)
> 00:15 - Search (Still looking)
> 00:25 - Stationary (Thinking)
> 00:30 - Locomotion (Walk to pantry)
> 00:35 - Search (Looking in pantry)

**LLM Output**:
> **State**: Confused / Missing Ingredient
> **Confidence**: High
> **Action**: Ask user "Are you looking for something?"

### Why this approach?
-   **Efficiency**: The LLM only processes text tokens (action labels), not video frames.
-   **Privacy**: No video is sent to the cloud unless the user accepts help.
-   **Context**: The LLM can use the "Scenario" to know that "Searching" for 30s while "Cooking" is bad, but "Searching" while "Cleaning" might be normal.

---

## 4. Current Technical Pipeline

### 4.1 Data Acquisition
-   **Source**: Ego4D (IMU + Gaze).
-   **Script**: `scripts/download_sensors_direct.py`
-   **Status**: Downloads raw CSVs from S3.

### 4.2 Data Processing
-   **Script**: `scripts/process_imu_data.py`
-   **Logic**:
    -   Aligns IMU/Gaze to 50Hz.
    -   Windows data into 1.0s chunks.
    -   **Critical**: Handles duplicate timestamps and NaNs to ensure clean training data.
-   **Output**: `seq.npz` files (Trajectory + Gaze + Timestamps).

### 4.3 Labeling (The Hybrid Strategy)
We created a "Gold Standard" dataset using a hybrid approach:
1.  **Map**: Convert Ego4D narrations to our 6 classes using keywords.
2.  **Refine (LLM)**: Use Qwen-14B to fix ambiguities (e.g., "reaches" -> Transfer).
3.  **Validate Errors**: Specifically re-check "Error" labels to distinguish *intentional drops* from *accidental drops*.
-   **Result**: `data/labels/action_labels_llm_validated.csv`

### 4.4 Training (Hierarchical Model)
-   **Script**: `scripts/train_hierarchical.py`
-   **Architecture**:
    -   **LLE (Low-Level Encoder)**: CNN-GRU for 1s windows.
    -   **HLA (High-Level Aggregator)**: GRU for 30s context.
-   **Loss**: Jointly optimizes Scenario Classification (Cross-Entropy) and Action Classification (Masked Cross-Entropy).
-   **Features**:
    -   Multi-GPU support (`nn.DataParallel`).
    -   WandB logging.
    -   Early stopping based on Scenario F1.

### 4.5 Server Workflow
1.  **Setup**: `./setup_training.sh` (Installs deps + downloads data).
2.  **Validate**: `python scripts/validate_training_data.py` (Checks for corruption).
3.  **Train**: `./run_training.sh` (Runs in `tmux` on GPUs 6 & 7).



```
\`\`\`  
\<View style="margin-top: 1em;"\>

  \<Header value="Action prediction:"/\>

  \<Text name="action\_text" value="$action"/\>

  \<Header value="Help: LLM reasoning:"/\>

  \<Text name="reasoning" value="$reasoning"/\>

\</View\>

\<Header value="Choose label"/\>

\<View style="display: flex; justify-content: center; gap: 60px; margin-top: 1.5em;"\>

  \<Choices name="status\_main" toName="narration" choice="single" showInline="true"\>

    \<Choice value="Gold" hotkey="1"/\>

    \<Choice value="Bad" hotkey="2"/\>

  \</Choices\>

\</View\>

\<View style="display: flex; justify-content: center; gap: 24px; margin-top: 2em;"\>

  \<Choices name="status\_secondary" toName="narration" choice="single" showInline="true"\>

    \<Choice value="Skip" hotkey="3"/\>

    \<Choice value="Delete Row" hotkey="4"/\>

  \</Choices\>

\</View\>

\<Choices name="corrected\_action" toName="narration" choice="single" showInline="false"\>

  \<Choice value="Essential Operation"/\>

  \<Choice value="Object Transfer"/\>

  \<Choice value="Search"/\>

  \<Choice value="Stationary"/\>

  \<Choice value="Locomotion"/\>

\</Choices\>  
\`\`\`
```

