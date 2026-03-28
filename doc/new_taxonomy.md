# HiVD-HAR Final Design Plan (End-to-End)

## Target: Sense of Space Workshop @ CVPR 2026

---

## 1. Core Problems and Solution Strategy

### 1.1 Two Bottlenecks (Empirically observed in the baseline paper)

| Problem | Current Status | Root Cause |
|---|---|---|
| **Generalization Gap** | Scenario F1: 0.610 val -> 0.450 test (-26%); Action F1: 0.395 val -> 0.242 test (-37%) | Training/test domain shift + model overfitting |
| **Action Classification Ceiling** | 6-class semantic F1=0.21 -> 4-class motion-based F1=0.39 (85% gain after merging OT+EO) | Head IMU cannot directly observe fine-grained hand-object physics |

### 1.2 Solution Path

**Core insight**: A video teacher can **see hand-object interactions**, so it can separate OT vs EO. Head IMU contains **weak but non-zero cues** (anticipatory head motion during reaching, gait rhythm during locomotion). Distillation matters because it helps the IMU student exploit these weak correlations.

**Our plan**:
1. **Data cleaning**: remove known ambiguous narrations and reduce label noise -> improve training signal quality.
2. **Hierarchical cross-modal distillation**: distill at both 1s window level (actions) and 30s sequence level (scenarios) -> **a gap in prior literature**.
3. **Confidence-gated queue + Scenario-conditional KD**: targeted mechanisms for the OT/EO boundary.
4. **Honest reporting**: quantify what distillation can improve and what it cannot.

### 1.3 Novelty (Literature-Verified, No Prior Equivalent)

| Prior work | Cross-Modal Distill | Hierarchical | Multi-Level Distill | OT/EO-focused |
|---|---|---|---|---|
| COMODO (2025) | Yes | NO | NO (single-scale 5s, 31 scenarios) | NO |
| EgoCHARM (2025) | NO | YES (1s->30s) | N/A | NO (only 3 trivial classes) |
| IMU2CLIP (2023) | Contrastive only | NO | NO | NO |
| **HiVD-HAR (ours)** | **Yes** | **Yes** | **Yes (L_win + L_seq)** | **Yes** |

**Concrete differences**:
- COMODO only performs scenario-level distillation (31 classes), not fine-grained action distillation.
- EgoCHARM has hierarchy but no cross-modal distillation; lower-level classes are only Stationary/Walking/Running.
- We distill at two temporal scales and connect them with scenario-conditional action KD.

---

## 2. Data Cleaning Strategy

### 2.1 Data Status

| Dataset | Rows | Description |
|---|---|---|
| Full LLM-labeled set | 355,580 | `action_labels_llm_clean_refined.csv`, 5-class |
| Gold-validated | 13,931 | `r001_refined.csv`, human-validated correct labels |
| Bad (LLM mislabeled) | 1,984 | Rows identified as wrong by human validation |

**Overall LLM accuracy**: 87.5% (Gold / (Gold + Bad))

**Per-class LLM accuracy**:
- Essential Operation: 94.9%
- Object Transfer: 93.5%
- Search: 80.3%
- Locomotion: 78.2%
- **Stationary: 74.0%** (worst, often corrected to EO or OT)

### 2.2 Data Types Requiring Cleaning

#### A. Narrations with inconsistent LLM labels (highest priority)

Same narration assigned different action classes by LLM across instances:

- **3,391 narrations** have inconsistent labels
- Affecting **51,347 rows** (14.4%)
- Among them, agreement < 80%: **31,663 rows** (8.9%), recommended for removal

#### B. Ambiguous verbs

**"moves" (15,375 rows, 4.3%)** - most ambiguous verb:
```
Locomotion: 5,281 (34.4%)
Object Transfer: 4,867 (31.6%)
Stationary: 3,293 (21.4%)
Essential Operation: 1,818 (11.8%)
```
-> Recommendation: keep, but overwrite with Gold-validated labels where available; mark remaining as low-confidence.

**"looks" (14,617 rows, 4.1%)** - Search/Stationary confusion:
```
Search: 12,686 (86.8%)
Stationary: 1,753 (12.0%)
```
-> Recommendation: keep (mostly consistent), but formalize rules for "looks at phone"-type cases labeled as Stationary.

#### C. Known erroneous narrations

Human validation identified LLM mislabels for 1,959 unique narrations (affecting 21,954 rows):
- For validated rows: replace with `corrected_action`.
- For unvalidated rows with the same narration: propagate correction if narration-level agreement > 90%.

#### D. Other noise

| Type | Rows | Handling |
|---|---|---|
| `#O` (other person's action) | 35,691 (10.0%) | **Remove** - we focus on camera wearer actions |
| `#unsure` tag | 15,142 (4.3%) | **Keep but down-weight** - noisy but still informative |
| `#Summary` segments | 22 | **Remove** - not action narrations |
| Empty/minimal | 8 | **Remove** |
| `Error`/`Invalid` labels | 5 | **Remove** |

### 2.3 Three Cleaning Levels

| Plan | Removed Rows | Remaining Rows | Removal Scope |
|---|---|---|---|
| **Conservative** | ~14,204 (4.0%) | ~341,376 | Bad + inconsistent overlap |
| **Moderate (recommended)** | ~67,381 (18.9%) | ~288,199 | Conservative + #O + inconsistent (<80%) + Summary |
| **Aggressive** | ~82,523 (23.2%) | ~273,057 | Moderate + #unsure + all instances of bad narrations |

**Recommended: Moderate plan**. Post-cleaning distribution:

| Action Class | Rows After Cleaning | Ratio |
|---|---|---|
| Object Transfer | ~130,000 | ~45.1% |
| Essential Operation | ~58,000 | ~20.1% |
| Stationary | ~57,000 | ~19.8% |
| Locomotion | ~35,000 | ~12.1% |
| Search | ~8,200 | ~2.8% |

### 2.4 Gold Data Cleaning

From 13,931 Gold rows, remove:
- 41 multi-label narrations (same narration labeled as different classes by different annotators) -> remove 82 rows
- Remaining: **13,849 Clean Gold**

### 2.5 Execution Steps

```python
# Step 1: Load full dataset
full_data = load_csv('action_labels_llm_clean_refined.csv')  # 355,580 rows

# Step 2: Normalize narration
normalize(narr) = lowercase -> remove #tags -> remove trailing punctuation -> strip

# Step 3: Remove #O (other person) rows
full_data = full_data[~starts_with('#O')]  # -35,691

# Step 4: Remove #Summary and empty
full_data = full_data[~contains('#summary') & len(narr) >= 3]  # -30

# Step 5: Remove inconsistent narrations (agreement < 80%)
narr_action_dist = groupby(normalize(narr)).action.value_counts()
inconsistent_narrs = {n for n, dist if max(dist)/sum(dist) < 0.8}
full_data = full_data[~normalize(narr).isin(inconsistent_narrs)]  # -31,663

# Step 6: Propagate labels for Gold-corrected narrations
for narr in gold_corrections:
    if consistency(narr, corrected_action) > 0.9:
        full_data.loc[normalize(narr) == narr, 'action'] = corrected_action

# Step 7: Mark #unsure as low confidence (keep rows, down-weight during training)
full_data['confidence'] = 1.0
full_data.loc[contains_unsure, 'confidence'] = 0.5

# Final: ~288K clean rows + 13,849 Gold rows
```

---

## 3. Model Architecture (v4 SCUQD)

### 3.1 Overall Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Training Stage (paired data)              │
│                                                              │
│  VIDEO TEACHER (Frozen)          IMU STUDENT (Trainable)     │
│  ┌─────────────────┐            ┌──────────────────────────┐ │
│  │ Pre-extracted    │            │ 8-ch IMU, 50Hz, 1s=50    │ │
│  │ SlowFast (2304d) │            │ samples                  │ │
│  │ or EgoVLP (768d) │            │                          │ │
│  │                  │            │  ┌────────────────────┐  │ │
│  │ clip feat -> MLP │            │  │ LLE (Low-Level Enc)│  │ │
│  │ -> z_T^w (128d)  │            │  │ Multi-Dilation CNN │  │ │
│  │                  │            │  │ + SE + BiGRU + Attn│  │ │
│  │ seq feat -> MLP  │            │  │ -> e_t (128d)      │  │ │
│  │ -> z_T^s (128d)  │            │  └────────┬───────────┘  │ │
│  └────────┬─────────┘            │           │              │ │
│           │                      │  30×e_t -> HLA           │ │
│           │                      │  ┌────────────────────┐  │ │
│           │                      │  │ 3-layer Transformer│  │ │
│           │                      │  │ CLS -> h_cls (128d)│  │ │
│           │                      │  │ h_t (128d) / window│  │ │
│           │                      │  └────────┬───────────┘  │ │
│           │                      │           │              │ │
│           │  L_queue             │  ┌────────┴───────────┐  │ │
│           ├──────────────────────┤  │ Context-Aware Head  │  │ │
│           │  L_cond              │  │ a_loc = W_loc·e_t   │  │ │
│           ├──────────────────────┤  │ a_ctx = W_ctx·h_t   │  │ │
│           │                      │  │ g = σ(MLP([e;h]))   │  │ │
│           │                      │  │ a = (1-g)·a_loc     │  │ │
│           │                      │  │   + g·a_ctx         │  │ │
│           │                      │  ├────────────────────┤  │ │
│           │                      │  │ Scenario Head       │  │ │
│           │                      │  │ s = W_s·h_cls       │  │ │
│           │                      │  └────────────────────┘  │ │
│           │                      └──────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    Inference Stage (IMU only)                │
│                                                              │
│  8-ch IMU -> LLE -> e_t -> Action head (5 classes, context) │
│              └-> 30×e_t -> HLA -> h_cls -> Scenario head    │
│                                                              │
│  Projectors dropped, Teacher dropped                         │
│  Deployment params: ~1.2M                                    │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 LLE Detailed Specification

**Input**: x in R^{8×50}

**Multi-Dilation CNN** (3 blocks):
```
Stem: Conv1d(8, 64, k=5, p=2) -> BN -> GELU

Block 1: 3 parallel branches
  Conv1d(64, 32, k=5, d=1, p=2)
  Conv1d(64, 32, k=5, d=2, p=4)
  Conv1d(64, 32, k=5, d=4, p=8)
  -> Concat -> 96 channels -> SE(r=8) -> Conv1d(96, 128, k=1)

Block 2: 3 parallel branches
  Conv1d(128, 64, k=5, d=1/2/4)
  -> Concat -> 192 channels -> SE(r=8) -> Conv1d(192, 128, k=1)

Block 3: 3 parallel branches
  Conv1d(128, 64, k=3, d=1/2/4)
  -> Concat -> 192 channels -> SE(r=8) -> Conv1d(192, 128, k=1)
```

**BiGRU + Attention Pooling**:
```
BiGRU(input=128, hidden=96, layers=1)  ->  output: (50, 192)
Attention pooling:
  α_t = softmax(w^T · tanh(W_a · h_t + b_a))
  e = Σ α_t · h_t  ->  (192,)
FC: 192 -> 128  ->  e_t in R^{128}
```

**LLE parameter count**: ~900K

### 3.3 HLA Detailed Specification

**Input**: E = [e_1, ..., e_30] in R^{30×128}

```
Learned positional embeddings: P in R^{30×128}
CLS token: cls in R^{128}
E' = [cls; E + P]  in R^{31×128}

Transformer Encoder × 3 layers:
  d_model = 128
  n_heads = 4
  d_ff = 512
  dropout = 0.1
  activation = GELU
  norm_first = True (Pre-LN)

Output:
  h_cls = E'_out[0]    -> sequence embedding (128d)
  h_t = E'_out[1:31]   -> contextualized window embeddings (30×128d)
```

**HLA parameter count**: ~350K

### 3.4 Context-Aware Action Head (Key Improvement)

**Motivation**: The v3 action head only uses local embedding e_t (1s info), which cannot reliably disambiguate OT/EO. v4 fuses local and 30s context:

```
a_t^loc = W_loc · e_t           (128->5, from LLE)
a_t^ctx = W_ctx · h_t           (128->5, from HLA Transformer)
g_t = sigmoid(MLP([e_t; h_t]))  (MLP: 256->64->1)
a_t = (1 - g_t) · a_t^loc + g_t · a_t^ctx
```

**Why it helps**:
- For Locomotion/Search/Stationary: local signal is usually enough, so gate g_t -> 0 (favor local).
- For OT/EO ambiguity: 30s context adds scenario cues (Cooking is more likely EO; Walking Outdoors more likely OT), so gate g_t -> 1 (favor context).
- The gate learns automatically when context is needed.

### 3.5 Distillation Projectors (Training only, removed at inference)

```
Teacher projector: MLP(d_T -> 512 -> 128) + L2_norm -> z_T^w / z_T^s
Student projector: MLP(128 -> 256 -> 128) + L2_norm -> z_S^w / z_S^s

where d_T = 2304 (SlowFast) or 768 (EgoVLP)
```

### 3.6 Total Parameters

| Component | Params |
|---|---|
| LLE (CNN + SE + BiGRU + Attn + FC) | ~900K |
| HLA (Transformer × 3 + pos + CLS) | ~350K |
| Context-aware gate | ~17K |
| Action head + Scenario head | ~1.7K |
| **Deployment total** | **~1.27M** |
| Distill projectors (training only) | ~200K |
| Teacher projector (training only) | ~300K (SlowFast) |

---

## 4. Loss Functions

### 4.1 Total Loss

```
L_total = L_task + λ_d · L_distill_total

where L_task = L_scenario + 0.7 · L_action
      L_distill_total = L_queue + 0.6 · L_cond
```

### 4.2 Task Loss

**L_scenario**: FocalLoss(γ=2.0) + inverse-frequency class weights
```
Scenario weights: Cooking=1.0, Cleaning=1.9, MechRepair=1.9, PlayInstr=3.1,
                  Carpentry=3.4, WalkOutdoors=3.4, DeskWork=6.2, Gardening=10.3
```

**L_action**: FocalLoss(γ=2.0) + class weights + OT/EO asymmetric soft labels
```
Action weights: OT=1.0, EO=1.7, Stationary=2.3, Locomotion=4.4, Search=8.8

OT/EO special handling:
  when teacher confidence w_t is low (teacher also uncertain between OT and EO),
  apply asymmetric smoothing on hard labels:
    if y in {OT, EO}:
      ε = 0.3 · (1 - w_t)    # lower confidence -> stronger smoothing
      label = (1-ε)·one_hot(y) + ε·one_hot(other_of_{OT,EO})
    else:
      label = one_hot(y)     # keep hard label for other classes
```

### 4.3 Confidence-Gated Queue Distillation (L_queue)

**Teacher confidence**:
```
p_T^a = teacher action distribution (5-dim softmax)
w_t = 1 - H(p_T^a) / log(5)         # normalized entropy, 0=uncertain, 1=certain
m_t = p_T_top1 - p_T_top2           # top-2 margin
```

**Class-balanced sub-queues** (384 entries/class, 1920 total):
```
Enqueue condition: w_t >= 0.6 AND m_t >= 0.2
Queued content: z_T^w (teacher projected embedding, 128d, L2-normalized)
Queued class: c_t = argmax(p_T^a)
```

**Distribution alignment**:
```
P_T(i | Q_c) = softmax(z_T^w · q_i / τ_T),   τ_T = 0.1
P_S(i | Q_c) = softmax(z_S^w · q_i / τ_S),   τ_S = 0.05

L_queue = (1/N_valid) · Σ_{t in valid_samples} w_t · KL(P_T || P_S)
```

**Key differences from COMODO**:
1. **Confidence gating**: COMODO enqueues everything; we only enqueue high-confidence samples to avoid OT/EO noise contamination.
2. **Weighted KD**: each sample's KD loss is weighted by teacher confidence; uncertain samples are learned less.
3. **Class-balanced queue**: COMODO uses a single FIFO queue; we use per-class sub-queues to avoid OT+EO majority domination.

### 4.4 Scenario-Conditional Action KD (L_cond) - Core Innovation

**Motivation**: When teacher is uncertain between OT and EO, scenario-level context can help. In Cooking, P(EO|Cooking) > P(OT|Cooking); in Walking Outdoors, P(OT|Walking) >> P(EO|Walking).

**Implementation**:
```
# Maintain scenario->action conditional table Π in R^{8×5}
# Updated with EMA during training
Π[s, a] = P_teacher(action=a | scenario=s)

# For each sample t:
u_t = Σ_s p_T^scenario(s) · Π[s, :]       # scenario-weighted action prior (5-dim)

# Mixing weight from teacher uncertainty
ρ_t = min(0.7, H(p_T^action) / log(5))    # more uncertain -> rely more on scenario prior

# Soft target fusion
q_t = (1 - ρ_t) · p_T^action + ρ_t · u_t  # smoothed, scenario-aware action target

# Loss
L_cond = (1/N) · Σ_t KL(q_t || p_S^action)
```

**Why better than v3 coherence loss**:
- Coherence passes "which window is important" -> indirect, weak for OT/EO.
- Scenario-conditional KD directly tells the student: "under this scenario, this action is more likely EO than OT" -> target the real error mode.
- Simpler implementation (no extra window-to-sequence similarity module).
- Better interpretability (Π itself is a contribution: cross-modal conditional structure).

### 4.5 Hyperparameter Summary (7 Key Parameters)

| Symbol | Description | Value | Source |
|---|---|---|---|
| β_task | Action task weight | 0.7 | Tuned from baseline β=0.5 |
| λ_d | Distillation total weight | 1.0 (Phase 1) / 0.3 (Phase 2) | COMODO reference |
| τ_T | Teacher temperature | 0.1 | COMODO |
| τ_S | Student temperature | 0.05 | COMODO |
| w_thresh | Queue confidence threshold | 0.6 | New, to validate |
| m_thresh | Queue margin threshold | 0.2 | New, to validate |
| ρ_max | Max scenario prior mixing | 0.7 | New |

---

## 5. Video Teacher Feature Preparation

### 5.1 Available Pre-extracted Features

| Model | Dim | Access | Priority |
|---|---|---|---|
| **SlowFast R101** | 2304 | `ego4d --datasets slowfast8x8_r101_k400` | **Primary** |
| **EgoVLP** | 256/768 | Google Drive download | Backup |
| **Omnivore Swin-L** | 1536 | `ego4d --datasets omnivore_video_swinl` | Backup |

### 5.2 Feature Alignment

```
SlowFast temporal resolution: ~1.87 fps (30fps video, stride=16 frames)
Each feature corresponds to ~0.53 seconds

Alignment:
1. For each (video_uid, timestamp_sec):
   feature_idx = round(timestamp_sec * 30 / 16)
   clip_feature = slowfast_features[video_uid][feature_idx]  # (2304,)

2. For each 30s sequence:
   seq_feature = mean(clip_features[start:start+30])  # (2304,)

3. Store in HDF5:
   h5['clip_features'][video_uid][timestamp] = clip_feature
   h5['seq_features'][video_uid][seq_start] = seq_feature
```

### 5.3 Teacher Quality Audit

**Required experiment**: train a linear probe on pre-extracted video features to verify teacher separability of 5 action classes:
```
Linear(2304, 5) on SlowFast features -> report 5-class action F1
Expected: >70% F1 (video can observe hand-object interactions)

If teacher F1 < 50%: distillation is not meaningful; switch teacher/features
```

---

## 6. Training Pipeline

### Phase 1: Joint Distillation (~288K cleaned data)

```
Data: 288K clean LLM-labeled samples
Loss: L_total = L_task + λ_d · (L_queue + 0.6·L_cond)
λ_d = 1.0
Epochs: 30
Optimizer: AdamW(lr=1e-4, wd=1e-5)
Schedule: 5-epoch warmup -> cosine decay
Batch: 128 sequences (each = 30 windows)
Grad clip: 1.0
Early stopping: patience=15, monitor val combined F1

Data augmentation (IMU):
  - Jittering: Gaussian σ=0.02, p=0.5
  - Scaling: [0.9, 1.1], p=0.5
  - Time masking: zero 5-sample spans, p=0.3
```

### Phase 2: Gold Fine-tuning (13,849 Clean Gold samples)

```
Data: 13,849 Gold-validated samples only
Loss: L_task + 0.3 · (L_queue + 0.6·L_cond)
Epochs: 15
Optimizer: AdamW(lr=5e-5)
Schedule: Cosine decay
Early stopping: patience=10
```

---

## 7. Benchmarking and Comparison Strategy

### 7.1 Main Table

| Model | Scen F1 (val) | Scen F1 (test) | Act F1 (val) | Act F1 (test) | Params |
|---|---|---|---|---|---|
| Baseline-4class (main.tex) | 0.610 | 0.450 | 0.395* | 0.242* | 1.16M |
| Baseline-5class (new labels) | ? | ? | ? | ? | 1.16M |
| + L_queue only | ? | ? | ? | ? | ~1.3M |
| **HiVD-HAR (full)** | ? | ? | ? | ? | ~1.3M |
| IMU2CLIP | 0.603 | 0.447 | 0.359* | 0.258* | 3.97M |

*4-class action F1 (OT+EO merged into Manipulation)

### 7.2 Key Diagnostic Experiments

**Experiment 1: Identifiability Test (SVM ceiling)**
```
Train SVM for OT vs EO binary classification on raw IMU statistical features
Features: mean, std, max, min, energy, zero-crossing per channel (48 features)
Report: accuracy, F1, confusion matrix
Expected: ~55-65% (above 50% chance but far from perfect)
Purpose: establish physical upper-bound intuition for distillation
```

**Experiment 2: Teacher Quality Audit**
```
Train linear probe on video features for 5-class action classification
Report: accuracy, F1, per-class metrics
Expected: >70% F1
Purpose: verify teacher signal quality
```

**Experiment 3: Data Cleaning Impact**
```
Compare baseline trained on raw 355K vs cleaned 288K
Report: class-wise F1 shifts, especially OT/EO/Stationary
Purpose: quantify data cleaning contribution
```

**Experiment 4: Per-Class Confusion Analysis**
```
Confusion matrices: baseline-5class vs HiVD-HAR
Focus: OT<->EO and Search<->Stationary confusion changes
Purpose: localize where distillation helps
```

**Experiment 5: Generalization Gap**
```
For each model variant, compute gap_ratio = (val_F1 - test_F1) / val_F1
Key question: does distillation reduce the generalization gap?
```

### 7.3 Ablation

| Ablation | L_task | L_queue | L_cond | Gate |
|---|---|---|---|---|
| A1: Baseline-5class | ✓ | | | |
| A2: + Queue distillation | ✓ | ✓ | | |
| A3: + Scenario-conditional KD | ✓ | | ✓ | |
| A4: + Context gate | ✓ | | | ✓ |
| **A5: Full HiVD-HAR** | **✓** | **✓** | **✓** | **✓** |

Use 3 seeds and report mean ± std.

---

## 8. Paper Structure (8 pages)

### Title
"Hierarchical Vision-to-IMU Distillation for Multi-Level Egocentric Activity Recognition"

### Outline

1. **Introduction** (1 page): AR-glasses motivation, two key problems, contributions summary
2. **Related Work** (0.75 page): HAR hierarchy (EgoCHARM), cross-modal KD (COMODO, C2KD), identified gap
3. **Method** (2.5 pages):
   - 3.1 Hierarchical architecture (LLE + HLA + context-aware head)
   - 3.2 Confidence-gated queue distillation
   - 3.3 Scenario-conditional action KD
   - 3.4 Training pipeline (data cleaning + two phases)
4. **Dataset** (1 page): LLM pipeline, 17K Gold, cleaning strategy, validation findings
5. **Experiments** (2 pages): main table, identifiability test, teacher audit, ablations, per-class analysis
6. **Discussion** (0.5 page): what distillation can and cannot solve, honest limitations
7. **Conclusion** (0.25 page)

### Three Contributions

1. **Method**: first hierarchical cross-modal distillation framework for egocentric HAR, distilling at both action (1s) and scenario (30s) levels with confidence-gated queue and scenario-conditional KD
2. **Dataset**: first large-scale human-validated annotation package for egocentric IMU HAR — 17K Gold labels + 344K LLM labels + a practical cleaning pipeline
3. **Analysis**: empirical boundary analysis of cross-modal transfer — quantifying what distillation can improve vs cannot improve

---

## 9. Timeline (9-day Sprint)

### Day 1 (Feb 27): Data and Teacher Preparation
- [ ] Download Ego4D pre-extracted features (SlowFast or EgoVLP)
- [ ] Check coverage: how many of 1,596 videos have usable features?
- [ ] Run data cleaning script (Moderate plan) -> output `clean_training_data.csv`
- [ ] Precompute and store HDF5: clip features + sequence features
- [ ] Run Teacher Quality Audit (linear probe on video features)

### Day 2 (Feb 28): Data Pipeline
- [ ] Build paired DataLoader: `(IMU_window, video_feature, action_label, scenario_label, confidence)`
- [ ] Create train/val/test splits (by video)
- [ ] Run Identifiability Test (SVM on raw IMU -> OT vs EO)
- [ ] Run Data Cleaning Impact comparison

### Day 3-4 (Mar 1-2): Model Implementation
- [ ] Implement LLE (Multi-Dilation CNN + SE + BiGRU + Attention Pooling)
- [ ] Implement HLA (3-layer Transformer + CLS)
- [ ] Implement Context-Aware Action Head (gated fusion)
- [ ] Implement Confidence-Gated Class Queue
- [ ] Implement `L_queue` + `L_cond`
- [ ] Implement full training loop (2-phase schedule)

### Day 5-6 (Mar 3-4): Training and Ablation
- [ ] Phase 1: train on 288K clean data (~30 epochs)
- [ ] Phase 2: fine-tune on Gold data (~15 epochs)
- [ ] Run ablation A1-A5 (3 seeds = 15 runs)
- [ ] If no visible gain by end of Day 5 -> simplify to soft-label KD as fallback

### Day 7 (Mar 5): Analysis
- [ ] Generate all result tables
- [ ] Confusion matrices (before/after distillation)
- [ ] Generalization gap analysis
- [ ] Per-class F1 breakdown

### Day 8-9 (Mar 6-7): Paper Writing
- [ ] Write 8-page paper in CVPR format
- [ ] Generate figures (architecture, confusion matrices, ablations)
- [ ] Double-blind anonymization
- [ ] Submit to OpenReview

---

## 10. Risks and Fallback

| Risk | Probability | Mitigation |
|---|---|---|
| Pre-extracted features do not cover our videos | MEDIUM | Switch to EgoVLP or extract features for subset |
| Distillation does not improve 5-class action F1 | HIGH (especially OT<->EO) | Report honestly. Frame as "empirical transfer boundary"; still publishable |
| Training does not converge | LOW | Fall back to soft-label KD (simplest distillation) |
| Not enough time for full ablation | MEDIUM | Prioritize main results + A1 vs A5 + identifiability |
| Paper not finished in time | LOW-MEDIUM | Switch to non-proceedings track (March 21) |

### Day 5 Checkpoint (Critical Decision Gate)

If by end of Day 5:
- **Clear gain (>=2% macro F1)** -> continue full plan
- **Small gain (0.5-2%)** -> reduce ablations, focus on paper quality and analysis depth
- **No gain / negative gain** -> stop complex losses immediately; pivot to baseline-5class + dataset contribution + identifiability analysis

---

## 11. Expected Results (Honest Estimate)

| Metric | Baseline-5class | HiVD-HAR (Expected) | Source of gain |
|---|---|---|---|
| Scenario F1 (val) | ~0.60 | ~0.61-0.63 | Distillation regularization |
| Scenario F1 (test) | ~0.44 | ~0.45-0.47 | Reduced generalization gap |
| Action F1 (val) | ~0.30-0.35 | ~0.33-0.38 | Queue KD + context gate |
| Action F1 (test) | ~0.20-0.25 | ~0.23-0.28 | Scenario-conditional KD |
| OT F1 (test) | ~0.25 | ~0.28-0.32 | Context + locomotion subset |
| EO F1 (test) | ~0.15 | ~0.17-0.20 | Scenario prior (Cooking -> EO) |
| Generalization gap | 26-37% | 20-30% | Distillation as regularizer |

**What will not happen**:
- OT/EO will not become an easy boundary (physical observability limit remains)
- Action F1 will not exceed 0.45 on test (IMU ceiling)
- This is not a "we solved it" paper; it is a "we quantified the boundary and showed a partial, defensible solution" paper
