# Round-1 合并验证结果与下一步策略（2026-02-23）

## 1. 本次交付与范围

目标：把 3 份 validated source 合并回 Round-1 的 `30,000` 主表（不覆盖原文件），并生成可用于深度分析的子集。

已新增（全部是新文件）：

- `HAR_Lab_Initiative_AI/data/annotation_rounds/r001_merged_validation/round1_assigned_only_30000_with_validations.csv`
- `HAR_Lab_Initiative_AI/data/annotation_rounds/r001_merged_validation/round1_validated_rows_all_matched.csv`
- `HAR_Lab_Initiative_AI/data/annotation_rounds/r001_merged_validation/round1_validated_rows_with_decisions.csv`
- `HAR_Lab_Initiative_AI/data/annotation_rounds/r001_merged_validation/round1_deep_analysis_subset_unique_narration_final_action.csv`
- `HAR_Lab_Initiative_AI/data/annotation_rounds/r001_merged_validation/round1_validation_conflict_log.csv`
- `HAR_Lab_Initiative_AI/data/annotation_rounds/r001_merged_validation/round1_validation_unmatched_rows.csv`
- `HAR_Lab_Initiative_AI/data/annotation_rounds/r001_merged_validation/round1_merge_summary.json`

对应合并脚本（新增）：

- `HAR_Lab_Initiative_AI/scripts/data_validation/merge_round1_validated_sources.py`

---

## 2. 合并方法（可复现）

### 2.1 输入

- Base 30k: `HAR_Lab_Initiative_AI/data/annotation_rounds/r001_numerical/round1_assigned_only.csv`
- Validated 3 源：
- `HAR_Lab_Initiative_AI/data/1000_validated/2400_project-7-at-2026-02-23-17-18-d8daa73b.csv`
- `HAR_Lab_Initiative_AI/data/1000_validated/Deduped_2500_project-8-at-2026-02-23-17-26-c66f8a5c.csv`
- `HAR_Lab_Initiative_AI/data/1000_validated/Batch12_Finished_project-1-at-2026-02-23-02-46-283f88ad.csv`

### 2.2 匹配 key

- 使用任务键：`(video_uid, timestamp_sec, narration_text, action)`  
- `timestamp_sec` 做 canonical normalization（避免字符串表示差异）。

### 2.3 重叠与冲突处理

- 先按 validated `id` 去重（若出现同 ID 多条，保留 `updated_at` 最新）。
- 再按任务键聚合；同一任务键多个候选时，保留 `updated_at` 最新（并记录到 conflict log）。
- 保持 base 行数不变：最终主表固定 `30,000` 行。

---

## 3. 合并后总体统计

来自 `round1_merge_summary.json`：

| 指标 | 数值 |
|---|---:|
| Base rows | 30,000 |
| Validated raw rows (3 源合计) | 6,217 |
| Matched rows in base | 6,117 |
| Decision rows (Gold/Bad/Skip/Delete) | 6,101 |
| Gold+Bad rows | 6,011 |
| Deep-analysis subset（unique normalized narration + final_action） | 5,278 |
| Conflict keys（同任务多候选） | 100 |
| Unmatched validated rows | 0 |

结论：你提到的“约 5,000 条深度分析”已落地（5,278 条）。

---

## 4. 关键发现（基于合并后决策数据）

分析文件：`HAR_Lab_Initiative_AI/data/annotation_rounds/r001_merged_validation/round1_validated_rows_with_decisions.csv`

### 4.1 质量分布

- Gold: `5,249`
- Bad: `762`
- Skip: `42`
- Delete Row: `48`
- Gold rate（Gold/(Gold+Bad)）: `87.32%`

### 4.2 最大问题不是随机错误，而是“边界冲突”

Top Bad transitions：

- `Stationary -> Essential Operation`: 240
- `Object Transfer -> Essential Operation`: 114
- `Stationary -> Object Transfer`: 98
- `Locomotion -> Object Transfer`: 52
- `Stationary -> Search`: 47

Per-class Bad rate：

- Stationary: `30.48%`（最高）
- Locomotion: `14.77%`
- Search: `11.59%`
- Object Transfer: `6.78%`
- Essential Operation: `3.29%`

解释：你们现在遇到的困惑（EO/OT/Locomotion/Stationary 边界不清）在数据上是主导矛盾。

### 4.3 标注规范本身存在“多类可接受”信号

- Gold 且 `corrected_action` 非空：`200` 条  
即有一批样本被标注为“Gold 但有 alternative class”，会影响后续监督学习的一致性。

### 4.4 具体模式（你提到的例子）

- `looks at person`：11 条，结果高度不稳定（Bad/Skip/Delete 都有），Gold/Bad 的 final action 里 `Stationary` 与 `Search` 各占一半（2 vs 2）。  
=> 直接硬规则改成 Stationary 风险高，建议进入 ambiguity queue。

- `looks around`：110 条，`Search` 精度高（107/110）。  
=> 可作为高置信 rule。

- `shoot(s) ... ball`：5 条里 4 条被 Delete。  
=> 当前 taxonomy 对 sports narration 不稳，不能直接喂 finetune。

---

## 5. “Finetune 前先修标签”可执行策略

你的想法是对的：先做 parser/rule + 冲突归一，再决定是否 finetune。

### 5.1 Rule-first（先高精度规则）

先只上线高 precision 规则（在 merged validated 上验证过）：

- `looks around -> Search`（support=110, precision≈97.3%）
- `adjusts the camera -> Essential Operation`（support=7, precision=100%）

暂缓上线（低 precision）：

- `looks at person -> Stationary`（support=4, precision=50%）
- `throws ... ball -> Object Transfer`（support=10, precision=30%）

### 5.2 Majority resolution（同 narration 多标签）

在 merged validated（Gold/Bad）中：

- 冲突 narration group：50 组
- 若用 `n>=3 且 majority>=0.67`，可自动修复 13 组，预计改动 15 行（保守、安全）。

在 full 355k（`action_labels_llm_clean_refined.csv`）中：

- 冲突 group：3,382（涉及 51,171 行）
- `n>=3 且 majority>=0.67`：可修 1,325 组，预计改动 4,354 行
- 更保守 `n>=5 且 majority>=0.8`：可修 835 组，预计改动 2,432 行

建议：先用保守阈值上线，剩余冲突进人工复核队列。

---

## 6. IMU separability 研究计划（传统 + 物理）

你提的“用传统和物理分析 study IMU 能否可分”建议直接做成 pre-finetune gate：

### 6.1 Feature families

- Traditional time-domain：mean/std/RMS/energy/zero-crossing/auto-corr
- Frequency-domain：dominant freq/bandpower/spectral entropy
- Physics-informed：cadence、jerk RMS、turn-rate variance、periodicity index、impact peak count

### 6.2 Pairwise separability 重点

- Stationary vs Locomotion
- EO vs OT
- Search vs Stationary
- Sports-related locomotion vs regular walking

### 6.3 基准模型与评估

- Classical ML: Logistic Regression / SVM / Random Forest / XGBoost
- GroupKFold by `video_uid`
- 指标：Macro F1、Balanced Accuracy、bootstrap CI、permutation test

---

## 7. Locomotion 拆分（Walking vs Running）建议

你提出“像 EgoCHARM 一样把 locomotion 拆成 walking/running”是合理的。

从 full labels 粗看（Locomotion 样本）：

- walk-like narration 占比约 `50.63%`
- run-like narration 占比约 `1.02%`
- sports-like narration 占比约 `4.85%`

从 merged validated：

- Locomotion 中 sports-like 样本 37 条，`Gold=16, Bad=12, Delete=9`，不确定性显著。

结论：先做 `walking/running` 二分 head（auxiliary head）比直接全量 taxonomy 重构更稳妥。

---

## 8. 与当前论文草稿/参考文献的对齐

### 8.1 你们自己的 paper 草稿已支持“taxonomy 要 sensor-observable”

`doc/report/Final_Paper/main.tex` 已明确：

- 6-class semantic 在 head IMU 上效果差（action F1=0.21），重映射为 4-class motion-based 后提升到 0.39（`doc/report/Final_Paper/main.tex:148` 到 `doc/report/Final_Paper/main.tex:158`）。
- 讨论部分也强调“task definition matters more than model capacity”（`doc/report/Final_Paper/main.tex:403`）。

### 8.2 EgoCHARM 参考方向可直接吸收

`doc/Padmanabha et al. - 2025 - EgoCHARM Resource-Efficient Hierarchical Activity Recognition using an Egocentric IMU Sensor.pdf` 报告：

- low-level 3 classes（Stationary/Walking/Running）效果高（low-level test F1 ≈ 0.855）
- 1s low-level window + 30s high-level window
- 50Hz 输入，且对较低采样率（15/25Hz）有可接受性能

这与当前“先把可观测 motion state 定义清楚，再做更高层语义”思路一致。

---

## 9. 建议的下一轮执行顺序（用于讨论会定案）

1. 固化 merge 产物为 round1 单一事实来源（使用新 30k merged CSV，不改原 CSV）。  
2. 先做 rule-first 清洗（仅高 precision 规则），输出 `rule_applied_candidate.csv`。  
3. 做 majority resolution（保守阈值），输出 `majority_resolved_map.csv`。  
4. 对剩余冲突与高不确定模式（sports / looks-at-person）建人工复核队列。  
5. 同步做 IMU separability benchmark（traditional + physics）验证“类是否物理可分”。  
6. 基于 separability 决定 finetune 策略：  
   - 若 label space 仍不稳定：优先 parser+rule+taxonomy 调整；  
   - 若稳定：再进入 finetune，且明确 train set 过滤规则（去掉 Delete/Skip/低置信冲突样本）。

---

## 10. 备注

- 本轮未覆盖或修改任何原始 CSV；所有产物均在 `r001_merged_validation` 新目录。
- `Batch12` 源文件存在异常换行（`\r\r\n`）风险，脚本已做容错读取，但建议后续统一导出格式。

