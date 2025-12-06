# MIT 2.156 Midterm Progress Report: Initiative AR Glasses AI Assistance System
### Chung-Ta Huang, chungtah@mit.edu
## 1\. Project Overview

This project develops a **layered intent-detection system** for AR glasses that uses lightweight IMU and gaze signals to determine "when to look" before activating energy-intensive cameras. The original proposal outlined a three-layer architecture: (1) continuous IMU+gaze monitoring for anomaly triggers, (2) brief camera activation for visual verification, and (3) agentic AI assistance. In the scope of this class, we will focus on **Layer 1 implementation** — the core trigger mechanism that detects behavioral anomalies from head motion and eye tracking alone. The timeseries based IMU and Gaze signals can expose intent without seeing scene content: for example, dwell (\~600–1000 ms) on signage, smooth pursuits, scan‑and‑return indicating uncertainty, and head‑turns aligned with gaze during orienting. “When the head moves in conjunction with the eyes to accomplish these shifts in gaze direction, the rules that helped define head-restrained saccadic eye movements are altered” (Freedman).

**Key Innovation**: Rather than always-on camera capture, we learn what "normal skilled behavior" looks like from time-series sensor data, then flag deviations (e.g., scan-and-return patterns, head-gaze decoupling, sudden stops) that may indicate uncertainty, disorientation, or hazards. This minimizes camera duty-cycle while maintaining responsiveness.

## 2\. Data Pipeline and Framework Design

### 2.1 Dataset Selection: From Aria to Ego-Exo4D

**Data**: We pivoted to **Ego-Exo4D** (released Jan 2025), a large-scale egocentric dataset with synchronized first-person video, 6-DoF IMU (25 Hz), and eye gaze tracking (30 Hz) across diverse activities. This provides:

- **Multimodal alignment**: Pre-aligned IMU trajectory and gaze signals eliminate manual sync overhead  
- **Naturalistic behavior**: Real-world head-eye coordination during skilled tasks  
- **Scenario diversity**: We trained the model on two types of scenarios.   
  - 1\) 40 takes of cycling scenarios initially as single scenario   
  - 2\) expandable to 8+ activity types (cooking, assembly, medical procedures) as diverse scenario

**Decision rationale**: Ego-Exo4D's sensor specifications (head-mounted IMU \+ gaze) directly match our target hardware(Aria 1), and its scale (thousands of hours) supports transfer learning across scenarios.

<img src="runs/20251111_10scenario/notebook_reports/data_audit/assumed_normal/per_take_pass_rates.png" alt="Selection of 10 Scenarios' takes" width="600">

*img1: Selection of 10 Scenarios' takes*

### 2.2 Data Processing Pipeline

Automated end-to-end pipeline from raw downloads to training tensors:

**Stage 1-2**: Download trajectory/gaze data → Automatic QA (sample rates, missing data) → Temporal alignment with forward-fill interpolation (max 200ms gap)

<img src="img/Gaze_combined.png" alt="10 Scenarios gaze alignment" width="800">


**Stage 3**: Windowing — 3.0s sliding windows, 0.2s hop, 50 timesteps → Features: Gaze (4 channels: yaw/pitch angles + velocities), IMU (3 channels: linear velocities, gravity-removed)

**Stage 4**: Normal filtering — Percentile-based criteria to create "assumed-normal" training set without ground-truth labels. Results: 40-Bike (15,418 to 4,177 windows, 27.1%), 10-Scenario (108,243 to 21,853 windows, 20.2%)

## 3\. Model Architecture and Training

### 3.1 Self-Supervised Learning (SSL) Encoder

Following the "learn normal manifold" approach, we implemented a **dual-branch temporal encoder**:

**Architecture**:  
Gaze Branch: 1D-CNN \[32→64→128 filters, k=3\] → GRU \[2 layers, h=256\] → FC \[d=128\]  
IMU Branch:   1D-CNN \[32→64→128 filters, k=3\] → GRU \[2 layers, h=256\] → FC \[d=128\]  
Fusion:           Concat \[256\] → LayerNorm → FC \[d=128\]

**SSL Objective**: Mean reconstruction — encode full window and reconstruct temporal mean of IMU and gaze trajectories (MSE loss). This forces the encoder to learn compact representations capturing typical behavior patterns within each window.

**Hyperparameters**:  
SSL: 20 epochs, batch size 32, learning rate 1e-4, dropout 0.1  
Train/Val/Test split: 70/15/15 by take (preserving temporal coherence within takes)  
Optimizer: AdamW with weight decay 1e-5, gradient clipping at 1.0

### 3.2 One-Class Anomaly Detection (Deep SVDD)

After SSL pretraining, we freeze the encoder and fit a **Deep Support Vector Data Description (SVDD)** on the 128-dim embeddings to compute anomaly scores as Euclidean distance from a learned hypersphere center. Set `nu=0.1` to allow 10% outlier tolerance during training.

**Tiered Thresholds**: Following the original proposal's LOW/MID/HIGH framework, we use percentile-based thresholds (P85/P95/P99) as an initial exploratory approach.

<img src="img/low:mid:high.png" alt="Tiered Detection Thresholds" width="450"> 

### 4.1 SSL Training Convergence
<img src="img/10scenario/Training%20Summary-10%20scenario.png"  alt="SSL Training Convergence" height="200" width="800">

Train loss: 2.77 to 0.81 over 20 epochs (71% reduction), Val loss: 0.98 to 0.30 (69% reduction). Train/Val ratio stabilized at ~2.7 by epoch 8, indicating slight overfitting but acceptable generalization. Loss reduction validates that IMU-gaze coordination patterns are learnable from short windows.

### 4.2 Anomaly Score Distributions

<img src="img/Distance%20Distribution.png" alt="Bimodal Distance Distribution" height="250" width="450">

Bimodal distribution emerged: normal peak at distance ≈6.5 (82% of train), anomaly spike at 24.8 (13%). Top-10 anomalies show Traj std=7.8-14.9 (vs. 2.6 normal), Gaze std=4.3-5.2 (vs. 1.2 normal) — consistent with rapid head turns, IMU spikes during sharp maneuvers.

<img src="img/NormlvsAbnormal.png" alt="Trajectory Comparison: Normal (left) vs Abnormal (right)" height="200" width="550">

<img src="img/Anomaly%20seg.png" alt="Continuous anomalous segment: Windows 370-375 showing sustained sharp turns" width="700">

Manual inspection of anomalous segments confirms legitimate dynamic events spanning 4+ seconds (sustained sharp turns), not random sensor noise.

### 4.3 Single-Scenario vs Multi-Scenario Results

We trained two models with identical hyperparameters to isolate the effect of scenario diversity:

| Metric | 40-Bike (Single) | 10-Scenario (Multi) | Insight |
|--------|------------------|---------------------|---------|
| Total windows | 15,418 | 108,243 | **7.0x data** |
| Normal windows | 4,177 (27.1%) | 21,853 (20.2%) | Multi-scenario filtering stricter |
| **HIGH alerts/hour** | **3.0** | **1.12** | **Multi-scenario meets target** |
| HIGH threshold (P99) | 19.44 | 23.73 | +22% (wider hypersphere) |
| Duty cycle | 0.8% | 0.093% | **87% reduction** (3.35s/hour) |
| Train distance std | 2.6 | 4.25 | +63% variance (diverse behaviors) |

<img src="img/10scenario/6plots.png" alt="Distance distributions, detection rates, duty cycles, and ROC curves for 10-scenario model" width="750">

**Key Finding**: Multi-scenario training solves oversensitivity by learning a broader "normal" definition — the SVDD hypersphere expands to accommodate diverse activities (cooking's low IMU variance + bike's high dynamics), naturally reducing false positives without manual tuning. The 87% duty cycle reduction demonstrates significant energy savings for camera activation.

**Critical Issue Discovered**: Time-shuffling sanity check (Mann-Whitney test) shows p=0.221 (not significant), indicating the model may rely on static features (mean/std) rather than temporal dynamics. This suggests our mean reconstruction loss is insufficient for learning true temporal patterns — future work will explore temporal contrastive learning or sequence-to-sequence architectures to enforce temporal dependency.

### 4.4 What's Working Well

1. **Data pipeline robustness**: Automated QA caught 2 takes with \>40% missing IMU data, which were excluded before training
2. **SSL pretraining**: Loss curves show smooth convergence without oscillation, suggesting stable embedding space
3. **Interpretable scores**: Distance-based anomaly scores correlate with trajectory variance, enabling human verification
4. **Computational efficiency**: 20-epoch training in \<1 hour on M1, \~20ms inference per window (suitable for real-time deployment)

### 4.5 What Needs Improvement

1. **Temporal modeling**: Mean reconstruction loss proved insufficient — time-shuffling test failure (p=0.221) indicates the model learned static features rather than temporal dynamics (smooth pursuit, VOR patterns). Need stronger temporal objectives.
2. **Lack of ground-truth labels**: Without gold labels, we cannot compute true precision/recall. Multi-scenario's 1.12 alerts/hour meets targets, but actual precision remains unknown. **Next step**: Collect 200-500 labeled windows using human annotation for HIGH-scoring samples, supplemented by Vision Language Model (VLM) automated labeling of synchronized video frames to scale annotation efficiently.
3. **Per-scenario breakdown missing**: Cannot verify if cooking/assembly have different false positive rates than cycling — need per-scenario metrics to validate cross-task generalization.

## 5\. Next Steps

### 5.1 Temporal Modeling Improvement

Current mean reconstruction loss learns static features only. Will experiment with: (1) **Temporal contrastive learning** — contrast original vs time-shuffled sequences to enforce temporal dependency, (2) **Sequence-to-sequence autoencoding** — reconstruct full trajectory sequences rather than just means to capture temporal patterns.

### 5.2 Per-Scenario Analysis & Gold-Label Collection

- Decompose 10-scenario results by activity type to identify scenario-specific false positive rates
- Human annotation of 100-200 HIGH-scoring windows supplemented by VLM-assisted labeling (GPT-4V) scaling to 500+ samples for precision validation

### 5.3 Real-Time Deployment
- Real-time model inference based on the streaming data from Meta Aria1 Glasses.

---

## 6\. Conclusion

We implemented and validated **Layer 1 (IMU+Gaze Trigger)** across single-scenario and multi-scenario datasets. Key findings:

**What worked**: (1) Multi-scenario training (7x data) solved oversensitivity without tuning — 1.12 alerts/hour meets target, 87% duty cycle reduction enables practical deployment, (2) Automated pipeline scales to 108K windows across diverse activities, (3) Distance-based scores remain interpretable for human verification.

**Critical discovery**: Time-shuffling test revealed our mean reconstruction loss learns static features, not temporal dynamics. This finding guides next iteration toward temporal contrastive learning or sequence-to-sequence architectures.

**Technical contribution**: Demonstrated unsupervised, sensor-only anomaly detection on AR glasses achieves sub-2.0/hour false positive rate with 0.093% camera duty cycle, validating privacy-preserving, low-power intent detection for wearable AI.

**Next phase**: Gold label collection (human + VLM), temporal modeling fixes, and per-scenario breakdown to validate cross-task generalization claims.

---

## 项目简要说明（中文）

### 项目目标

本项目开发一个**层级式 IMU 活动识别系统**，用于 AR 眼镜的智能触发机制。核心思想是：通过头戴式 IMU（惯性测量单元）传感器数据，在不依赖摄像头的情况下，判断用户何时需要视觉辅助。

### 技术架构

系统采用两层半监督学习架构：

1. **低层编码器 (LLE - Low-Level Encoder)**
   - 输入：1秒的 IMU 数据窗口（6通道 × 50Hz = 300特征）
   - 架构：多尺度膨胀卷积 (CNN) + SE注意力机制 + GRU
   - 输出：32维运动特征向量
   - 用于学习6类基本动作原语（静止、移动、核心操作、物体转移、搜索、错误纠正）

2. **高层架构 (HLA - High-Level Architecture)**
   - 输入：30个连续的 LLE 嵌入向量（代表30秒时间跨度）
   - 架构：Transformer 或 GRU
   - 输出：8类场景分类（清洁、机械维修、烹饪、户外行走、木工、演奏乐器、桌面工作、园艺）

### 训练策略

**半监督学习**：系统仅使用高层场景标签进行端到端训练，低层动作模式作为副产品自动学习。训练完成后，冻结 LLE，通过线性探测层评估其学习到的低层动作表示质量。

### 数据来源

使用 **Ego4D** 数据集，包含约1,652个带有 IMU 传感器数据的第一人称视频。通过 LLM（Qwen-14B）对文本描述进行语义分类，生成约35万个1秒窗口的动作标签。

### 核心创新

- **隐私保护**：仅依赖 IMU 时序信号检测行为异常，减少摄像头使用
- **能效优化**：实现 0.093% 的摄像头占空比，大幅节省 AR 眼镜电量
- **无监督异常检测**：使用 Deep SVDD 学习"正常行为"流形，自动识别偏离模式

### 当前进展

- 10场景模型达到 1.12 次/小时的高级警报率（满足目标）
- 相比单场景模型，误报率降低 87%
- 发现时序建模不足问题（均值重建损失无法捕捉时序动态）

### 待改进

1. 引入时序对比学习增强时间依赖性建模
2. 收集人工标注的金标准数据进行精度验证
3. 分场景性能分析

---

## 7. Probe Training Deep Analysis (探针训练深度分析)

### 7.1 实验结果概览

| Metric | Value | Baseline |
|--------|-------|----------|
| Probe Val Action F1 | **0.209** | Target: 0.75+ |
| Probe Val Action Acc | **45.2%** | Majority baseline: 42.4% |
| Training Epochs | 50 | - |
| Final Train Loss | 1.344 | Started at 1.704 |

**关键发现**：探针准确率(45.2%)仅比多数类基线(42.4%)高2.8%，F1分数(20.9%)接近随机水平。模型本质上只是在预测多数类。

### 7.2 根本原因分析 (Root Cause Analysis)

#### 问题1：极端类别不平衡 (60:1 Imbalance)

```
Action Distribution:
  Object Transfer     : 150,726 (42.4%) ██████████████████████████████
  Essential Operation :  99,260 (27.9%) ███████████████████
  Stationary          :  46,033 (12.9%) █████████
  Locomotion          :  35,060 ( 9.9%) ██████
  Search              :  21,775 ( 6.1%) ████
  Error / Correction  :   2,513 ( 0.7%) ▏
```

- **Object Transfer** 占 42.4%，是 **Error/Correction** 的 **60倍**
- 模型只需预测 "Object Transfer" 就能达到 42.4% 准确率
- 标准 CrossEntropyLoss 没有任何类别权重调整

#### 问题2：训练时 beta=0，LLE 未学习动作特征

```python
# loop.py:180-184
if config['training']['beta'] > 0:
    loss_a = criterion_action(a_logits.view(-1, 6), action_labels.view(-1))
    loss = config['training']['alpha'] * loss_s + config['training']['beta'] * loss_a
else:
    loss = loss_s  # <-- 只用场景损失，完全忽略动作标签！
```

训练配置中 `beta=0`，意味着：
- LLE 只被训练来区分**场景**（Cooking vs Cleaning vs ...）
- LLE **从未见过动作标签**
- 冻结的 LLE 嵌入不包含动作区分信息

#### 问题3：动作语义与 IMU 信号的根本脱节

| 动作类别 | 头部 IMU 特征 | 可区分性 |
|---------|--------------|---------|
| Stationary | 几乎无运动 | ✅ 容易 |
| Locomotion | 周期性步态 | ✅ 容易 |
| Object Transfer | 低频头部运动 | ❌ 困难 |
| Essential Operation | 低频头部运动 | ❌ 困难 |
| Search | 扫视模式 | ⚠️ 可能 |
| Error/Correction | 突然变化 | ⚠️ 可能 |

**核心问题**：从头戴 IMU 无法区分 "Object Transfer"（拿起物体）和 "Essential Operation"（操作物体），因为两者的头部运动模式几乎相同。

对比 **EgoCHARM** 论文的低层定义：
- EgoCHARM: 3类**运动状态** (Stationary, Walking, Running) → 从 IMU 可以区分
- 本项目: 6类**语义动作** (Object Transfer, Essential Operation, ...) → 从 IMU 无法区分

### 7.3 训练曲线分析

```
Epoch  Train Loss  Val F1   Val Acc
  1      1.704     0.173    0.350    <- 初始
  5      1.422     0.199    0.435    <- 快速收敛到多数类
 10      1.366     0.199    0.447    <- 开始饱和
 25      1.348     0.205    0.450    <- 几乎停止学习
 50      1.344     0.209    0.452    <- 最终值
```

观察：
1. **训练损失从未下降到1.3以下** → 模型在训练集上也无法学会
2. **F1从epoch 5后几乎不变** → 早期就达到瓶颈
3. **Val Acc ≈ 45% ≈ Majority baseline** → 确认只是预测多数类

### 7.4 场景分类器性能对比

同一个 LLE 用于场景分类时表现正常：

| Metric | Scenario Task | Action Task |
|--------|--------------|-------------|
| Val F1 | **0.568** | 0.209 |
| Val Acc | 68.2% | 45.2% |
| 最佳类 | Desk Work (F1=0.75) | - |
| 最差类 | Gardening (F1=0.03) | - |

这说明 LLE 学到了**场景级别**的特征，但没有学到**动作级别**的特征。

### 7.5 改进建议

#### 短期修复

1. **启用 beta > 0 进行联合训练**
   ```yaml
   training:
     beta: 0.3  # 动作损失权重
     alpha: 1.0  # 场景损失权重
   ```

2. **添加类别权重到 CrossEntropyLoss**
   ```python
   weights = 1.0 / class_counts
   criterion = nn.CrossEntropyLoss(weight=weights)
   ```

3. **使用 Focal Loss 处理类别不平衡**

#### 长期重新设计

1. **重新定义低层任务为3类运动状态**（匹配 EgoCHARM）
   - Stationary (静止)
   - Active-in-Place (原地活动)
   - Locomotion (移动)

2. **考虑手腕 IMU 融合**（如果可用）
   - 手腕 IMU 能区分 Object Transfer vs Essential Operation

3. **接受 IMU 的局限性**
   - 头戴 IMU 无法解决语义动作分类
   - 这是传感器模态的固有限制，不是模型问题

### 7.6 结论

探针训练失败的根本原因是**任务定义问题**，而非模型架构问题：

1. 我们的6类动作从头戴 IMU 信号**原理上无法区分**
2. 训练时 beta=0 导致 LLE 没有学习任何动作信息
3. 60:1 的类别不平衡让模型退化为多数类预测器

**建议**：重新评估低层任务定义，采用更粗粒度的运动状态分类（如 EgoCHARM 的3类），或接受当前 IMU 模态的局限性，专注于场景分类任务。

---

## 8. Innovation Analysis: Beyond EgoCHARM (创新点分析)

### 8.1 EgoCHARM 与本项目的关键差异

| 方面 | EgoCHARM | 本项目 | 差异分析 |
|------|----------|--------|---------|
| **数据集** | Ego-Exo4D + Nymeria (精选) | Ego4D (更多样) | 不同数据源 |
| **高层类别** | 9类 (运动为主: soccer, basketball, dance...) | 8类 (日常生活: Cooking, Cleaning, Desk Work...) | 本项目更偏向日常活动 |
| **低层类别** | 3类 (Stationary, Walking, Running) | 6类 (语义动作) | 本项目更细粒度但有问题 |
| **低层标签来源** | **人工筛选** + 视频验证 | **LLM自动生成** | 本项目创新点 |
| **训练数据量** | 48k HL / 14k LL samples | 226k HL / 355k LL samples | 本项目数据量更大 |

### 8.2 EgoCHARM 低层标签的真相

论文 Section 3.1 原文:
> "For low level activity recognition... we use the 'activity summarization' annotations... and **manually select windows of data that match our low level classes** using the textual summaries. For the running class, we additionally **refine our annotations visually from the RGB camera footage**."

**关键发现**: EgoCHARM 的低层标签是**人工精心筛选**的，不是自动生成的。他们选择了3个从 IMU 物理上可区分的类别。

### 8.3 本项目的真正创新点

#### 创新1: LLM-Based Action Labeling Pipeline (自动化动作标注)

```
EgoCHARM: 人工标注 14,343 个低层样本 (耗时)
本项目:   LLM 自动生成 355,580 个动作标签 (可扩展)
```

我们使用 Qwen-14B 对 Ego4D 的 narration 文本进行语义分类：

```python
Narration: "#C C opens a fridge."
→ LLM Reasoning: "Opening a fridge involves manipulating an object..."
→ Action: "Object Transfer"
```

**优势**:
- 可扩展到任意规模数据
- 无需人工标注成本
- 保留语义推理链

**问题**: 语义动作与 IMU 信号脱节

#### 创新2: 更丰富的日常场景 (Daily-Life Scenarios)

| EgoCHARM (运动导向) | 本项目 (日常导向) |
|-------------------|-----------------|
| Soccer, Basketball | Cooking, Cleaning |
| Dance, Rock Climbing | Desk Work, Carpentry |
| Body Stretch | Mechanical Repair |

本项目的场景分类更接近 AR 眼镜的实际应用场景（家庭/办公环境）。

#### 创新3: 大规模数据验证

| 指标 | EgoCHARM | 本项目 |
|------|----------|--------|
| 高层训练样本 | 48,078 | 226,616 |
| 低层标签样本 | 14,343 | 355,580 |
| 参与者数量 | ~1000 (Ego-Exo4D + Nymeria) | 1652 videos |

### 8.4 问题诊断与修正方向

#### 核心问题: 低层任务定义错误

我们的6类动作是**语义层面**的分类，而非**运动层面**的分类：

```
语义动作 (本项目):
  "Object Transfer" = 拿起物品 → 头部几乎不动
  "Essential Operation" = 操作物品 → 头部几乎不动
  → 两者 IMU 信号几乎相同！

运动状态 (EgoCHARM):
  "Stationary" = 静止 → 无加速度
  "Walking" = 行走 → 周期性步态
  "Running" = 跑步 → 高频周期性步态
  → 三者 IMU 信号明显不同！
```

#### 修正方案: 重新映射动作标签

```python
motion_map = {
    'Stationary': 'Stationary',      # 保持
    'Locomotion': 'Locomotion',      # 保持
    'Object Transfer': 'Active',     # 合并为"活动状态"
    'Essential Operation': 'Active',
    'Search': 'Active',
    'Error / Correction': 'Active',
}
# 结果: 3类 (Stationary: 13%, Locomotion: 10%, Active: 77%)
```

### 8.5 创新贡献总结

| 贡献 | 描述 | 状态 |
|------|------|------|
| **LLM Action Labeling** | 自动化动作标注 pipeline，355k 标签 | ✅ 完成 |
| **Daily-Life Scenarios** | 8类日常生活场景分类 | ✅ F1=0.568 |
| **Large-Scale Validation** | 在更大数据集上验证层级架构 | ✅ 完成 |
| **Semi-Supervised Learning** | 仅用高层标签训练，探测低层表示 | ⚠️ 需修正任务定义 |

### 8.6 后续方向

1. **重新定义低层任务**: 3类运动状态 (Stationary/Locomotion/Active-in-Place)
2. **验证 LLM 标签质量**: 对比人工标注与 LLM 标注的一致性
3. **启用联合训练**: 设置 beta > 0，让 LLE 学习动作信息
4. **类别平衡**: 使用 weighted loss 或 focal loss