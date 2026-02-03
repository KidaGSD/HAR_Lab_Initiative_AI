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

### Stage 2: Targeted Fixing (Gold)
*   **Tool:** `notebooks/fix_unsure.ipynb` (To be created)
*   **Goal:** Repair the rows known to be low-quality (Unknowns, Regex fallbacks, Errors).
*   **Workflow:**
    1.  **Filter:** System selects rows where `action == "Unknown"` OR `reasoning` contains "fallback".
    2.  **Batching:** User selects a batch size `k` (e.g., 50).
    3.  **Interface:** Displays Context + Narrations.
    4.  **Human Action:** User manually selects the correct label from the dropdown.
    5.  **Result:** Row updated to correct label, status set to **`gold`**.

### Stage 3: Statistical Verification (Audit)
*   **Tool:** `notebooks/verify_stats.ipynb` (To be created)
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

### Stage 4: VLM Automated Audit (VLM-Gold)
*   **Tool:** `scripts/verify_with_vlm.py` (To be created)
*   **Goal:** Scale verification to the massive dataset.
*   **Workflow:**
    1.  Iterate through `silver` rows.
    2.  Query VLM: "Is this action plausible in the image?"
    3.  **Result:**
        *   **Plausible:** Update status to **`vlm_gold`**.
        *   **Implausible:** Flag for review.

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
