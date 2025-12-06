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