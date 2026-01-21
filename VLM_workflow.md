# VLM Workflow (Super Simple v0)

Goal: use the VLM **only when needed**, and decide fast whether to **shutdown camera**, **ask user**, or **prepare help** (via a separate LLM).

## Inputs (from IMU pipeline)
- `scenario_pred`: string label (e.g., `"Cooking"`) — updated every 30s
- `action_pred`: string label (e.g., `"Manipulation"`) — updated every 1s
- `scenario_conf`, `action_conf`: floats in \([0,1]\)
- `anomaly_reason`: structured reason from Stage 2 (why we turned on camera)
  - examples:
    - `"low_context_prob: P(action|scenario)=0.01 < 0.05"`
    - `"safety_rule: outdoors+stationary>10s"`

## When do we turn on the camera?
Only when IMU anomaly logic triggers (e.g., low \(P(\text{Action} \mid \text{Scenario})\) or safety rule).

## VLM Decision Strategy (2-step)

### Step A — Label plausibility check (fast)
Ask the VLM:
- “Given this image, are the IMU labels (`scenario_pred`, `action_pred`) **plausible**?”

VLM returns JSON fields (minimum):
- `labels_plausibility`: one of `plausible | implausible | unclear`
- `labels_reasoning`: short explanation (why plausible/implausible/unclear)

### Step B — Only if `labels_plausibility == plausible`: scene report
If the labels seem plausible, we assume the IMU anomaly might be a **real issue** (not a pure false positive).
Ask the VLM:
- “Describe what is happening + any safety risk.”

VLM returns JSON fields (minimum):
- `scene_summary`: 1–3 sentences
- `risk_level`: `none | low | medium | high`
- `risk_reason`: short text

## Routing (the output of this workflow)
We output a single `decision` plus a payload for downstream steps.

### 1) If `labels_plausibility == implausible`
Interpretation: likely **IMU false positive**, BUT we still sanity-check for obvious danger.
- If VLM indicates **no concern**: `decision = shutdown_camera`
- If VLM indicates **possible danger / unclear**: `decision = ask_user`

### 2) If `labels_plausibility == plausible`
Interpretation: IMU anomaly is likely meaningful in context.
- If `risk_level == none` and nothing looks wrong: `decision = shutdown_camera`
- If `risk_level == low/medium` or VLM is uncertain: `decision = ask_user`
- If `risk_level == high`: `decision = give_help`

## What we pass to the LLM (only if needed)
If `decision` is `ask_user` or `give_help`, create an LLM prompt payload containing:
- IMU: `scenario_pred`, `action_pred`, confidences
- `anomaly_reason`
- VLM: `labels_plausibility`, `labels_reasoning`, `scene_summary`, `risk_level`, `risk_reason`

## Minimal JSON contract (recommended)
The VLM should return something like:
- `labels_plausibility`
- `labels_reasoning`
- `scene_summary` (optional if implausible + shutdown)
- `risk_level` (optional if implausible + shutdown)
- `risk_reason` (optional if implausible + shutdown)

And our workflow returns:
- `decision`: `shutdown_camera | ask_user | give_help`
- `why`: short string (one-line reason)
- `llm_payload`: object (only when decision != shutdown)


