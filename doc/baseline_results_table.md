# Baseline results (comparison models)

Values below are in the order you specified: **Scenario F1**, **Scenario Acc**, **Action F1**, **Action Acc**.

## Table (same columns as paper Table 1)


| Model        | Scenario F1 | Scenario Acc | Action F1 | Action Acc | Params (M) | Time |
| ------------ | ----------- | ------------ | --------- | ---------- | ---------- | ---- |
| IMU2CLIP     | 0.5961      | 0.6113       | 0.3626    | 0.3953     | 4.0        | 8    |
| MLP-MLP      | 0.5803      | 0.6063       | 0.3482    | 0.4431     | 1.03       | 2    |
| CNN-MLP      | 0.5737      | 0.5858       | 0.3758    | 0.4552     | 1.07       | 3    |
| CNN-LSTM-GRU | 0.5535      | 0.5749       | 0.3884    | 0.4584     | 1.19       | 6    |



Fill **Params** and **Time** from your run logs / W&B when available.

---

## LaTeX snippet (optional)

```latex
\begin{table}[t]
\caption{Baseline comparison (multi-level HAR).}
\label{tab:baselines}
\centering
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Sc. F1} & \textbf{Sc. Acc} & \textbf{Act. F1} & \textbf{Act. Acc} & \textbf{Params (M)} & \textbf{Time} \\
\midrule
IMU2CLIP       & 0.5961 & 0.6113 & 0.3626 & 0.3953 & -- & -- \\
CNN-MLP        & 0.5737 & 0.5858 & 0.3758 & 0.4552 & -- & -- \\
CNN-LSTM-GRU   & 0.5535 & 0.5749 & 0.3884 & 0.4584 & -- & -- \\
MLP-MLP        & 0.5803 & 0.6063 & 0.3482 & 0.4431 & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Should you label these as “$\beta = 1$”?

**Only if** these runs were trained with the same `**training.beta`** you use for the hierarchical “$\beta = 1.0$” row—typically `configs/beta_1.0.yaml`.

In the **current** `train_baselines.py` implementation, the loss is:

$$\mathcal{L} = \ell_{\text{scenario}} + \beta \ell_{\text{action}}$$

(with `alpha` in YAML not applied in baselines). So `**beta = 1.0`** means **equal coefficients on the two cross-entropies** (1× scenario + 1× action), **not** “action only,” and **not** the convex mix $(1-\beta)\ell_{\text{action}} + \beta\ell_{\text{scenario}}$ unless you change the code to that form.

**Recommendation for the paper**

- In the **Comparison Models** block, you **do not have to** put a $\beta$ column if every baseline uses the same training recipe; say in caption/text: *“Baselines: joint training with $\mathcal{L} = \ell_{\text{sc}} + \beta \ell_{\text{act}}$, $\beta=1$”* **only if** that is literally what you ran.
- If baselines used a **different** $\beta$ (e.g. 0.5), report that value instead—**the label must match the config**, not the hierarchical row naming, unless you intentionally align recipes.

**Loss choice for baselines**

- **Fair vs hierarchical (joint rows):** use the **same** $(\ell_{\text{sc}}, \ell_{\text{act}})$ setup and the **same** $\beta$ (or same convex combination, if you switch both scripts) so the comparison is apples-to-apples.
- **Scenario-only baseline:** would be $\beta=0$ in the **current** code (scenario term only); that is a **different** row, not comparable to “$\beta=1$” hierarchical without explanation.

---

## Bold “best” (like Table 1)

Per column (among these four baselines only):


| Metric       | Best model   | Value      |
| ------------ | ------------ | ---------- |
| Scenario F1  | IMU2CLIP     | **0.5961** |
| Scenario Acc | IMU2CLIP     | **0.6113** |
| Action F1    | CNN-LSTM-GRU | **0.3884** |
| Action Acc   | CNN-LSTM-GRU | **0.4584** |


Use these for bolding if you mirror the paper style.