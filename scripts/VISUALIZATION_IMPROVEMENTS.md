# Publication-Quality Embedding Visualization - Improvements

## 🎨 Visual Enhancements

### 1. Unique Markers Per Class
**Before**: All centroids used 'X' marker
**Now**: Each class gets a unique marker from:
- **Point markers**: `o, s, ^, D, v, <, >, p, *, h`
- **Centroid markers**: `X, P, *, d, H, 8, p, s, ^, v`

**Why**: Makes it easier to distinguish overlapping classes

### 2. Adaptive Transparency & Size
**Dynamic based on data density**:
```python
alpha = min(0.7, max(0.3, 500 / n_points))  # More points = more transparent
point_size = min(25, max(5, 2000 / n_points))  # More points = smaller
```

**Benefits**:
- Small datasets (100 points): Larger, more visible points
- Large datasets (10k points): Smaller, transparent to show density
- Prevents cluttering

### 3. Class Labels on Plot
Added text annotations near each centroid:
- Label name in colored box
- Easier to identify classes without checking legend
- Bold font for readability

### 4. Better Color Scheme
- **≤10 classes**: `tab10` palette (distinct, colorblind-friendly)
- **>10 classes**: `husl` palette (evenly spaced hues)

### 5. Professional Styling
- Larger figure (14×10 inches) for detail
- High DPI (150 for display, 300 for save)
- Light gray background (#f8f9fa) for contrast
- Cleaner grid (dashed, low alpha)
- Styled legend with shadow and frame

## 📊 Both Scenario & Action Plots

The script now generates **4 plots total**:

### Scenario Embeddings
1. `beta_1.0_scenario_trainval_pca.png/pdf`
2. `beta_1.0_scenario_trainval_tsne.png/pdf`

### Action Embeddings
3. `beta_1.0_action_trainval_pca.png/pdf`
4. `beta_1.0_action_trainval_tsne.png/pdf`

## 🚀 Usage

### Quick Test (100 samples)
```bash
python scripts/visualize_embeddings_fast.py \
  --config configs/beta_1.0.yaml \
  --checkpoint checkpoints/experiments_20251206_224347/beta_1.0/best_model.pth \
  --gpu 7 \
  --subset 100 \
  --output-dir outputs/embeddings_viz_test
```

### Full Dataset with PCA (faster)
```bash
python scripts/visualize_embeddings_fast.py \
  --config configs/beta_1.0.yaml \
  --checkpoint checkpoints/experiments_20251206_224347/beta_1.0/best_model.pth \
  --gpu 7 \
  --output-dir outputs/embeddings_viz
```

### Full Dataset with t-SNE (better separation, slower)
```bash
python scripts/visualize_embeddings_fast.py \
  --config configs/beta_1.0.yaml \
  --checkpoint checkpoints/experiments_20251206_224347/beta_1.0/best_model.pth \
  --gpu 7 \
  --use-tsne \
  --output-dir outputs/embeddings_viz
```

## 📈 What to Expect

### Scenario Plot (7 classes)
- **Well-separated**: Walking Outdoors, Playing Instrument, Mechanical Repair
- **Some overlap**: Cooking, Cleaning, Desk Work (similar IMU patterns)
- **Centroids** show class centers clearly

### Action Plot (4 classes)
- **Stationary**: Tight cluster (low movement)
- **Locomotion**: Periodic patterns (walking/running)
- **Manipulation**: Variable (depends on object)
- **Search_Interrupt**: Transitional (mixed patterns)

## 💡 Interpretation Guide

### Good Separation
- Distinct, non-overlapping clusters
- Centroids far apart
- Tight within-class clustering

### Overlap/Confusion
- Classes sharing same space
- Close centroids
- Mixed boundaries

### Quality Indicators
1. **PCA variance explained**: >40% is good for 2D
2. **t-SNE perplexity**: Auto-adjusted based on data size
3. **Cluster tightness**: Look at std around centroids

## 🎯 For Your Paper

### Figure Caption Example
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.48\textwidth]{beta_1.0_scenario_trainval_tsne.pdf}
  \includegraphics[width=0.48\textwidth]{beta_1.0_action_trainval_tsne.pdf}
  \caption{
    t-SNE visualization of learned embeddings on train+val data (β=1.0 model). 
    \textbf{Left}: High-level scenario embeddings show clear separation between 
    activity contexts. \textbf{Right}: Low-level action embeddings reveal 
    distinct motion primitives. Markers denote different classes; large markers 
    indicate class centers. The hierarchical model learns discriminative 
    representations at both temporal levels.
  }
  \label{fig:embeddings}
\end{figure}
```

### Key Points to Highlight
1. **Hierarchical representations**: Show both scenario and action plots side-by-side
2. **Class separation**: Quantify using inter-cluster distance
3. **Learned structure**: Compare to random embeddings or baseline
4. **Task-specific**: β=1.0 for action, β=0.5 for balanced

## 🔧 Customization Options

If you want to further customize:

### Change Colors
```python
# In plot_embedding()
colors = sns.color_palette('Set2', n_colors=n_classes)  # Pastel
colors = sns.color_palette('Dark2', n_colors=n_classes)  # Dark
```

### Adjust Point Density
```python
# Make points even smaller for very large datasets
point_size = min(15, max(3, 1500 / len(class_data)))
alpha = min(0.6, max(0.2, 300 / len(class_data)))
```

### Remove Class Labels (cleaner)
```python
# Comment out the ax.annotate() block if you want no text labels
```

## 📁 Output Files

Each run produces:
- **PNG**: High-res for presentations (300 DPI)
- **PDF**: Vector graphics for LaTeX (scalable, no pixelation)

**File sizes**:
- PNG: ~2-5 MB per plot
- PDF: ~500 KB - 2 MB per plot

Use PDF for final paper submission!
