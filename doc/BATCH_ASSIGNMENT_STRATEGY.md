# Batch Assignment Strategy for Human Annotation

## Overview

This document describes the sampling and batch assignment strategy used to distribute annotation tasks across annotators in Label Studio. The strategy addresses **class imbalance** in the dataset using a scientifically-grounded power allocation method.

### Problem

The 355K LLM-generated action labels have severe class imbalance:

| Class | Count | Proportion |
|:------|------:|:----------:|
| Object Transfer | 150,700 | 42.4% |
| Stationary | 75,089 | 21.1% |
| Essential Operation | 71,023 | 20.0% |
| Locomotion | 42,275 | 11.9% |
| Search | 16,485 | 4.6% |

Purely proportional sampling would yield very few Search samples, making per-class accuracy estimates unreliable. Purely equal sampling (500/class) requires too many videos per batch due to Search scarcity.

---

## Sampling Method: Power Allocation (α = 0.5)

### Formula

Each class h is allocated samples proportional to `p_h^α`, where `p_h` is the class's population proportion:

```
n_h = N × (p_h^α) / Σ_j(p_j^α)
```

With α = 0.5 (square root allocation), this is:

```
n_h = N × √(p_h) / Σ_j(√(p_j))
```

where N = 2,500 (total tasks per batch).

### Why α = 0.5?

- **α = 1.0** (proportional): Preserves original distribution. Search gets only 115/batch — too few for reliable per-class accuracy.
- **α = 0.0** (equal): All classes get 500/batch. Search requires ~46 videos/batch — too many for annotator focus.
- **α = 0.5** (square root): A principled compromise. Search gets 255/batch from ~24 videos — practical and statistically sufficient.

### Per-Batch Targets (N = 2,500)

| Class | Population % | α=0.5 Target | Effective % | Boost Factor |
|:------|:-----------:|:------------:|:-----------:|:------------:|
| Object Transfer | 42.4% | 770 | 30.8% | 0.73× |
| Stationary | 21.1% | 540 | 21.6% | 1.02× |
| Essential Operation | 20.0% | 530 | 21.2% | 1.06× |
| Locomotion | 11.9% | 405 | 16.2% | 1.36× |
| Search | 4.6% | 255 | 10.2% | 2.22× |

### Comparison Across α Values

| α | OT | Stat | EO | Loco | Search | Search videos/batch |
|:-:|:--:|:----:|:--:|:----:|:------:|:-------------------:|
| 1.0 (proportional) | 1,060 | 528 | 500 | 298 | 115 | ~11 |
| 0.7 | 884 | 543 | 523 | 363 | 187 | ~17 |
| **0.5 (sqrt)** | **770** | **540** | **530** | **405** | **255** | **~24** |
| 0.3 | 657 | 533 | 524 | 449 | 337 | ~31 |
| 0.0 (equal) | 500 | 500 | 500 | 500 | 500 | ~46 |

---

## Statistical Justification

### Minimum Sample Size

For estimating per-class annotation accuracy with a 95% confidence interval (Cochran, 1977):

```
n = z² × p × (1-p) / E²
```

| Margin of Error | Worst case (p=0.5) | If accuracy ~85% |
|:---------------:|:------------------:|:----------------:|
| ±5% | 385 | 196 |
| ±7% | 196 | 100 |
| ±10% | 96 | 49 |

With α=0.5, Search gets 255 per batch:
- **Single batch**: ±6.1% margin (sufficient for initial estimates)
- **2 batches cumulative**: 510 samples → ±4.3% margin
- **All 12 batches**: 3,060 samples → ±1.8% margin

All classes exceed the 385 threshold within 2 batches.

### Inverse Probability Weighting

When computing unbiased population-level accuracy from oversampled data, apply inverse probability weights:

```
w_h = p_h / q_h
```

where `p_h` is the true population proportion and `q_h` is the sampling proportion.

| Class | p_h (true) | q_h (sampled) | Weight |
|:------|:----------:|:-------------:|:------:|
| Object Transfer | 0.424 | 0.308 | 1.38 |
| Stationary | 0.211 | 0.216 | 0.98 |
| Essential Operation | 0.200 | 0.212 | 0.94 |
| Locomotion | 0.119 | 0.162 | 0.73 |
| Search | 0.046 | 0.102 | 0.45 |

---

## Batch Assignment Algorithm

### Video-to-Batch Assignment

Videos are assigned exclusively to batches (no video appears in multiple batches) to allow annotators to review clips in temporal context.

1. Compute per-video class profile: count of each class per video
2. Sort videos by Search count descending (scarce class distributed first)
3. Greedy assignment: for each video, assign to the batch whose class deficits best match the video's class composition
4. Within each batch, randomly sample rows to hit per-class targets

### Round Design

| Field | Round 1 | Round 2 |
|:------|:--------|:--------|
| `round` | r001 | r002 |
| `batch` | 1–12 | 1–12 |
| Source labels | Original LLM labels | Fine-tuned LLM labels |
| Sampling pool | All unassigned rows | Complement of Round 1 |
| Focus | Baseline validation | Changed labels + weak classes |

The CSV retains all 355K rows. Assigned rows have `batch` and `round` populated; unassigned rows have these fields blank. Round 2 assigns from the unassigned pool.

### Label Studio Workflow

1. Import the full CSV into a Label Studio project
2. Each annotator creates a filter: `round = r001` AND `batch = 3`
3. They annotate only their filtered tasks
4. After all batches complete, export once → merge back

---

## References

1. Cochran, W.G. (1977). *Sampling Techniques*, 3rd ed. Wiley. — Square root allocation as compromise between proportional and equal allocation; minimum sample size formula.

2. Dalenius, T. & Hodges, J.L. (1959). Minimum Variance Stratification. *Journal of the American Statistical Association*, 54(285), 88–101. — Cumulative square root frequency method for optimal stratification.

3. Kish, L. (1965). *Survey Sampling*. Wiley. — Power allocation framework (p^α) for compromise between equal and proportional sampling.

4. Neyman, J. (1934). On the Two Different Aspects of the Representative Method. *Journal of the Royal Statistical Society*, 97(4), 558–625. — Optimal allocation minimizing variance given known within-stratum variances.

5. Lample, G. & Conneau, A. (2019). Cross-lingual Language Model Pretraining. *NeurIPS 2019*. arXiv:1901.07291. — Power-smoothed multinomial sampling with α for multilingual model training.

6. Conneau, A. et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *ACL 2020*. — XLM-R uses α=0.3 for 100-language sampling; demonstrates power allocation in large-scale ML.

7. Alharbi, F., Ouarbya, L. & Ward, J.A. (2022). Comparing Sampling Strategies for Tackling Imbalanced Data in Human Activity Recognition. *Sensors*, 22(4), 1373. — Direct comparison of sampling strategies for imbalanced HAR datasets.
