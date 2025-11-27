# Research Design: Hierarchical IMU Activity Recognition

## 1. Objective

Develop a hierarchical activity recognition system using head-mounted IMU data to classify:
- **High-Level (HL)**: Scenario context (9 classes, 30-second windows)
- **Low-Level (LL)**: Motion primitives (6 classes, 1-second windows)

The system employs semi-supervised learning where both encoder levels train concurrently using only high-level scenario labels.

---

## 2. Architecture

### 2.1 Hierarchical Model Design

**Low-Level Encoder (LLE)**
- Input: 1-second IMU window (6 channels × 50 Hz = 300 features)
- Architecture: CNN-GRU with variable dilation convolutions
- Output: 32-dimensional motion embedding
- Parameters: ~22,000

**High-Level Architecture (HLA)**
- Input: Sequence of 30 LLE embeddings (representing 30 seconds)
- Architecture: 2-layer GRU
- Output: 9-class scenario prediction
- Parameters: ~63,000

### 2.2 Training Strategy

Both LLE and HLA train concurrently using only high-level scenario labels. The LLE learns generalizable low-level motion patterns as a byproduct of optimizing for scenario classification. This semi-supervised approach eliminates the need for extensive low-level annotation.

**Loss Function**: Weighted cross-entropy on scenario predictions
**Backpropagation**: Gradients flow through both HLA and LLE
**Post-Training**: LLE is frozen and probed with a linear layer for low-level classification

---

## 3. Scenario Selection

### 3.1 Selection Criteria

Scenarios must exhibit distinct motion signatures detectable by head-mounted IMU:
1. **Locomotion intensity**: Static to continuous movement
2. **Head motion patterns**: Stable gaze vs. active scanning
3. **Periodicity**: Rhythmic vs. aperiodic
4. **Vertical motion**: Sitting, standing, jumping

### 3.2 Proposed 9-Class Taxonomy

| Scenario | Locomotion | Head Motion | Periodicity | Videos (with IMU) |
|:---------|:-----------|:------------|:------------|:------------------|
| Cleaning | Intermittent walking | Downward + scanning | Wiping/sweeping cycles | **313** videos |
| Mechanical Repair | Static | Focused on object | Tool manipulation | **294** videos |
| Cooking | Minimal | Focused downward | Episodic arm motion | **267** videos |
| Walking Outdoors | Continuous locomotion | Forward gaze | Step frequency | **228** videos |
| Carpentry | Static positioning | Stable on workpiece | Sawing/hammering rhythm | **186** videos |
| Playing Instrument | Stationary | Stable on instrument | Musical rhythm | **158** videos |
| Desk Work | Stationary | Stable on screen | None | **150** videos |
| Gardening | Moderate walking + bending | Variable | Digging/planting cycles | **56** videos |

**Total**: 1,652 videos (8 Scenarios) with IMU sensor data

---

## 4. Data Processing Pipeline

### 4.1 High-Level Label Extraction

**Source**: Ego4D activity summaries (30-second annotations)
**Method**:
1. Parse `narration_pass_1.summaries` from `narration.json`
2. Classify summaries using semantic similarity (SentenceTransformer embeddings)
3. Match to 9 scenario templates with cosine similarity threshold > 0.7
4. Propagate labels to constituent 1-second windows

**Window Specifications**:
- Duration: 30 seconds (aligned with Ego4D annotation standard)
- Stride: 10 seconds (50% overlap for data augmentation)
- Sampling rate: 50 Hz (Nyquist-compliant for 25 Hz motion signals)

### 4.2 Low-Level Label Alignment

**Challenge**: Narrations are timestamped at irregular intervals (mean ~7 seconds apart), while model requires dense 1-second labels.

**Solution: Temporal Propagation**
```
For each narration at timestamp t with label L:
  - Assign label L to all 1-second windows in range [t, t+δ]
  - δ = min(next_narration_time - t, 5 seconds)
  - Overlapping windows use majority vote
```

**Alternative: Weak Supervision**
- Use keyword-matched labels as noisy initialization
- Rely on LLE's semi-supervised learning to denoise patterns
- Validate on manually annotated subset (10% of data, ~1,000 windows)

**Label Distribution** (v3 LLM-Validated):
- **Object Transfer**: 42.4% (150,726 windows) - *Logistics/Setup (pick up, put down)*
- **Essential Operation**: 27.9% (99,260 windows) - *Core task (cut, wash, mix)*
- **Stationary**: 12.9% (46,033 windows) - *Idle, waiting*
- **Locomotion**: 9.9% (35,060 windows) - *Moving body through space*
- **Search**: 6.1% (21,775 windows) - *Visual search or monitoring*
- **Error / Correction**: 0.7% (2,513 windows) - *Explicit failure, fumbling*

**Low-Level Taxonomy (6 Classes)**:
1.  **Locomotion**: High body acceleration, rhythmic (walking, climbing).
2.  **Essential Operation**: High hand acceleration, irregular/complex (cutting, mixing).
3.  **Object Transfer**: Short bursts of hand acceleration (picking up, putting down).
4.  **Search**: High head rotation (gyro), low hand acceleration (looking for item).
5.  **Error / Correction**: Jerky/sudden motion, breaks in rhythm (fumbling, dropping).
6.  **Stationary**: Low energy on all sensors (waiting, talking).

**Evolution of Labeling Strategy**:
1.  **v2 (Keyword-based)**: Relied on strict keyword matching. Resulted in high "Stationary" (41.8%) due to missing context.
2.  **v3 (LLM-based)**: Used Qwen-14B to infer actions from full sentences. Drastically reduced "Stationary" to 12.9% by correctly identifying subtle manual work.
3.  **Error Validation**: Specifically targeted "Error / Correction" labels.
    -   Input: `data/labels/action_labels_llm_clean_before_error.csv`
    -   Process: Re-verified 15,633 error labels with Qwen.
    -   Result: 84% reclassified as "Object Transfer" (e.g., "dropping" an object intentionally).
    -   Final Output: `data/labels/action_labels_llm_validated.csv`

---

## 5. Baseline Models

### 5.1 Comparison Targets

| Model | LLE Architecture | HLA Architecture | Parameters (L/H) | Reference |
|:------|:-----------------|:-----------------|:-----------------|:----------|
| MLP-MLP | Hand-picked features + MLP | MLP | 5k / 314k | EgoCHARM Table 2 |
| CNN-MLP | 1D-CNN | MLP | 18k / 36k | EgoCHARM Table 2 |
| IMU2CLIP | CNN-GRU (large) | MLP | 27k / 191k | Moon et al. 2022 |
| CNN-LSTM-GRU | CNN-LSTM | GRU | 26k / 225k | EgoCHARM Table 2 |
| **EgoCHARM (Target)** | CNN-GRU | GRU | 22k / 63k | Padmanabha et al. 2024 |

### 5.2 Additional Ablations

1. **Sampling Frequency**: 15 Hz, 25 Hz, 50 Hz, 75 Hz
2. **Window Size**: 5s, 10s, 15s, 20s, 25s, 30s
3. **Label Noise**: Train on keyword labels vs. LLM-refined labels
4. **Architecture Variants**: LSTM vs. GRU for HLA

---

## 6. Evaluation Metrics

### 6.1 High-Level Scenario Classification
- **Macro F1-score**: Average F1 across 9 classes (handles class imbalance)
- **Micro Accuracy**: Overall classification rate
- **Per-class Recall**: Identify weak scenario detection
- **Confusion Matrix**: Analyze cross-scenario errors

**Target**: F1 > 0.80 (EgoCHARM achieved 0.826)

### 6.2 Low-Level Action Classification (Probing)
- **Macro F1-score**: Average across 4 classes
- **4-fold Cross-Validation**: Account for participant variability
- **Per-class Precision/Recall**: Validate label quality

**Target**: F1 > 0.75 (EgoCHARM achieved 0.855 on 3 classes)

---

## 7. Implementations

**1. Data Preparation**
- Filter 4,141 videos for 9 scenarios
- Extract scenario labels from summaries
- Align low-level labels via temporal propagation
- Download IMU CSVs (requires AWS credentials)
- Process into model-ready tensors

**2. Model Training**
- Implement CNN-GRU LLE and GRU HLA
- Train concurrently on scenario task
- Hyperparameter search (learning rate, dropout, embedding dimension)
- Validate on held-out scenarios

**3. Probing & Baselines**
- Freeze LLE, train probing layer on low-level labels
- Implement 5 baseline models for comparison
- Ablation studies (sampling rate, window size, label noise)

**4. Analysis & Documentation**
- Confusion matrix analysis
- Error case visualization
- Performance vs. parameter trade-off curves
- Final research report

---

## 8. Open Research Questions

1. **Label Noise Tolerance**: How robust is semi-supervised LLE to noisy keyword labels (60-70% accuracy)?
2. **Cross-Domain Transfer**: Can LLE trained on 9 scenarios generalize to new unseen scenarios?

---

## 9. Expected Outputs

1. **Trained Models**:
   - LLE checkpoint (~22k params, <100 KB)
   - HLA checkpoint (~63k params, <300 KB)
   - Probing layer checkpoint (~400 params, <2 KB)

2. **Datasets**:
   - `scenario_labels.csv`: ~4,141 videos with 9-class labels
   - `action_labels_aligned.csv`: ~230k 1-second windows with 4-class labels
   - IMU tensors: Preprocessed numpy arrays (6 × 50 per window)

3. **Analysis**:
   - Baseline comparison table
   - Confusion matrices (HL and LL)
   - Ablation study results
   - Sample efficiency curves

4. **Code**:
   - Training scripts (LLE+HLA concurrent, probing)
   - Evaluation scripts (metrics, visualization)
   - Data processing pipeline (alignment, windowing)
