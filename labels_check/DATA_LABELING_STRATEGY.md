# Data Labeling & Verification Strategy

## Overview
Workflow for transforming raw LLM-generated action labels ("Silver" quality) into a verified dataset ("Gold" quality). We are going to prioritize fixing broken data first, then statistically verifying the rest.

## 1. Pipeline Stages

### Stage 1: Base Generation (Silver)
*   **Tool:** `scripts/label_with_qwen.py`
*   **Input:** Raw Ego4D narrations.
*   **Process:** 
    *   Run Qwen-2.5-14B (or similar) on the full dataset (~355k rows).
    *   Output includes `reasoning` which flags "Regex fallback" or failures.
*   **Output:** `action_labels_llm_clean.csv`
*   **Default Status:** All new rows are marked as **`silver`**.
*   **Performance Benchmark (Feb 3, 2026):**
    *   **Model:** Qwen/Qwen2.5-14B-Instruct-AWQ
    *   **Hardware:** 2x GPUs (60% VRAM constraint)
    *   **Throughput:** ~77 samples/sec (Total: 355k rows in 1h 17m)

### Stage 1.5: Rule-Based Refinement (Automated)
*   **Tool:** `scripts/refine_error_labels.py`
*   **Goal:** Automatically correct obvious misclassifications using regex rules (e.g., "drops X on table" -> Object Transfer).
*   **Input:** `action_labels_llm_clean.csv`
*   **Output:** `action_labels_llm_clean_refined.csv`
*   **Process:** Scans "Error / Correction" labels for patterns indicating intentional actions.
*   **Current Statistics (Feb 3, 2026):**

| Metric | Stage 1 (Raw) | Stage 1.5 (Refined) | Delta |
| :--- | :--- | :--- | :--- |
| **Total Rows** | 355,581 | 355,581 | - |
| **Unknowns** | 83 | 83 | 0 |
| **Regex Fallback (Gen)** | 0 | 0 | 0 |
| **Error / Correction** | 17,784 | 12,497 | -5,287 |
| **Refined (Auto-Fix)** | 0 | 5,287 | +5,287 |

### Stage 1.75 (Optional): LLM Strict Correction (Automated)
*   **Tool:** `scripts/run_stage2_5_llm_strict.py`
*   **Goal:** Automatically “repair” a problematic label bucket (e.g., `"Error / Correction"` or `"Unknown"`) by forcing the LLM to choose from a taxonomy that **excludes** both `"Error / Correction"` **and** `"Unknown"`.
*   **Input/Output:** Overwrites `action_labels_llm_clean_refined.csv` in-place and writes a per-row change log:
    *   `data/labels/action_labels_llm_clean_refined.changes.csv`
*   **Key behaviors (current script):**
    *   **Never outputs** `"Error / Correction"` or `"Unknown"` (invalid outputs fall back deterministically to recent context labels, else `Stationary`).
    *   Does **not** modify the `status` column.
    *   Prefixes updated rows’ reasoning with `"[CtxFixed]"`.
*   **Run Report (Feb 5, 2026 — earlier script version that still allowed `Unknown`):**
    *   **Model:** Qwen/Qwen2.5-14B-Instruct-AWQ (vLLM, `tensor_parallel_size=2`, `gpu_memory_utilization=0.5`, `enforce_eager=True`)
    *   **Rows processed (Error / Correction):** 12,493
    *   **Wall time:** ~7m 13s (from vLLM init `17:35:17` to termination `17:42:30`)
    *   **Effective throughput:** ~29 rows/sec (12,493 / 433s, includes model init)
    *   **Output distribution (old → new):**
        *   Error / Correction → Object Transfer: 6,681
        *   Error / Correction → Unknown: 3,750
        *   Error / Correction → Essential Operation: 1,294
        *   Error / Correction → Stationary: 532
        *   Error / Correction → Locomotion: 205
        *   Error / Correction → Search: 31
*   **Recommended usage:**
    *   To remove `"Error / Correction"`: run with `--target-action "Error / Correction"`
    *   To remove `"Unknown"`: run with default `--target-action "Unknown"`

### Stage 2: Targeted Fixing (Gold)
> **⚠️ DONT FORGET TO PULL THE REFINED CSV FILE, SOMEONE MIGHT HAVE MADE SOME CHECKS AS WELL**
*   **Tool:** `labels_check/stage2_targeted_fixing.ipynb`
*   **Input:** `action_labels_llm_clean_refined.csv`
*   **Goal:** Repair the rows known to be low-quality (Unknowns, Regex fallbacks, Errors).
*   **Workflow:**
    1.  **Filter:** System selects rows where `action == "Unknown"` OR `reasoning` contains "fallback".
    2.  **Batching:** User selects a batch size `k` (e.g., 50).
    3.  **Interface:** Displays Context + Narrations.
    4.  **Human Action:** User manually selects the correct label from the dropdown.
    5.  **Result:** Row updated to correct label, status set to **`gold`**.

### Stage 3: Statistical Verification (Audit)
> **⚠️ DONT FORGET TO PULL THE REFINED CSV FILE, SOMEONE MIGHT HAVE MADE SOME CHECKS AS WELL**
*   **Tool:** `labels_check/stage3_auditing.ipynb`
*   **Goal:** Estimate the reliability of the "Silver" dataset without fixing every row.
*   **Workflow:**
    1.  **Sampling:** System selects `N` random `silver` rows.
    2.  **Interface:** Displays Video Context + Text Label.
    3.  **Human Action:**
        *   **Good:** Label is correct.
        *   **Bad:** Label is incorrect. (No correction needed).
    4.  **Result:** 
        *   Calculates **Accuracy %** (Confidence Score).
        *   Rows marked "Good" become `gold`. Rows marked "Bad" are flagged (status `bad`).

### Stage 3b: VLM Automated Audit (VLM-Gold)
*   **Tool:** `scripts/verify_with_vlm.py` (To be created)
*   **Goal:** Scale verification to the massive dataset.
*   **Workflow:**
    1.  Iterate through `silver` rows.
    2.  Query VLM: "Is this action plausible in the image?"
    3.  **Result:**
        *   **Plausible:** Update status to **`vlm_gold`**.
        *   **Implausible:** Flag for review.

### Stage 4: VLM Action Duration (Full-Video Labeling)
*   **Idea:** Current labels are *point-in-time* (one row per narration, ~1 action per second). The goal is to have *segment-level* labels: for each action, a start and end time so that **entire videos** are covered by labeled segments (no gaps).
*   **Tool (to implement):** `labels_check/stage4_VLM_actions_duration.ipynb`
*   **Input:** Rows with **gold** (or high-confidence) labels: `video_uid`, `timestamp_sec`, `action`, `narration_text`.
*   **Process:**
    1.  For each gold-labeled timestamp, extract a short video clip (e.g. ±15 s) or keyframes.
    2.  Ask a VLM: “Given this clip and the action label at t=X, when does this action start and end (in seconds)? Output start_sec, end_sec.”
    3.  Merge/overlap handling: adjacent or overlapping segments (same action) can be merged; gaps can be filled with a “background” or next action.
*   **Output:** A segment table: `(video_uid, start_sec, end_sec, action)` so that the union of segments covers the video (or the narrated part).
*   **Caveats:** VLM temporal resolution is coarse; need a clear prompt and possibly multiple keyframes per segment. Cost/speed: running on ~355k clips is heavy—consider batching and filtering (e.g. only gold, or only certain scenarios first).

#### Design variant: precise window from current narration to next
*   **Window:** For each gold row at time `t`, define the segment as `[t, t_next)` where `t_next` is the timestamp of the **next narration** in the same video. That gives a precise, gap-free duration (one action per narration interval).
*   **Sampling:** Extract **1 or 2 frames per second** within that interval (so we can assign an action, or nothing, to every second). Feed these frames to the VLM.
*   **Prompt:** "The action at the start is \<action\>. These frames cover from X s to Y s (1–2 frames per second). At which second (or between which two frames) does this action stop?" → get `end_sec`; segment is `[t, end_sec]`.
*   **Pros:** Clear boundaries, no overlap with next action; 1–2 fps supports second-level labels while keeping token count manageable.

#### Frames vs video clip
*   **Frames (1–2 per second):** Each call = N images (1–2 per second of the interval). Goal: one action (or nothing) per second. VLMs (e.g. Qwen2-VL) support multiple images in one prompt. Typically **faster and cheaper** than video because (1) no video decoding, (2) no temporal encoding. Downside: no explicit motion; model infers from context.
*   **Video clip:** One call = one short video (e.g. 5–30 s). **Slower and heavier**: video VLMs often use more tokens (temporal patches) and need more VRAM. Better for "when does the action stop?" if the model is trained on video. For many open VLMs, video is 2–5× more expensive per second than 1–2 fps images; start with frames, then try video on a subset if needed.

#### Scale: “Rows if we labeled every second”
If we discretize time at **1 second** and assign one label per second for the entire dataset:

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Current rows (narrations)** | 355,580 | One per narration timestamp |
| **Unique videos** | 1,491 | |
| **Total video duration (proxy)** | ~1,902,561 sec | Per video: `max(timestamp_sec) - min(timestamp_sec) + 1`, summed. True total duration may be slightly higher if narrations don’t cover full video. |
| **Rows if 1 label per second** | **~1.9M** | ≈ 5.4× current row count |
| **Avg. video length (proxy)** | ~1,276 sec (~21 min) | |
| **Avg. narrations per video** | ~239 | |

So moving from “one label per narration” to “one label per second” would grow the dataset to on the order of **~2 million rows** (or the same information as a segment table covering ~1.9M seconds). The VLM-duration stage would produce **segments**; converting segments to 1-sec rows is then straightforward if needed.

## 2. Data Schema
The `action_labels.csv` file will contain the following columns:

| Column | Description |
| :--- | :--- |
| `video_uid` | Ego4D Video ID |
| `timestamp_sec` | Timestamp of the action |
| `scenario` | Context (e.g., Cooking) |
| `action_text` | Raw narration |
| `pred_class` | The taxonomy class |
| `status` | **`silver`**, **`gold`** (human fixed/verified), **`vlm_gold`**, **`bad`** |
| `reasoning` | LLM explanation or "Extracted via regex fallback" |

## 3. Metrics

1.  **Dataset Coverage:**
    $$ Coverage = \frac{\text{Count(Gold)} + \text{Count(VLM-Gold)}}{\text{Total Dataset Size}} $$

2.  **Reliability Score (from Stage 3):**
    $$ Accuracy = \frac{\text{Count(Good)}}{\text{Count(Good)} + \text{Count(Bad)}} $$
