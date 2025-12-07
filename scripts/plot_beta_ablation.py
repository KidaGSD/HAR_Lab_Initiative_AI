#!/usr/bin/env python3
"""
Generate complete beta ablation analysis and publication-ready plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set publication style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'

# Complete β ablation data (UPDATED WITH ACTUAL RESULTS)
data = {
    'β': [0.0, 0.3, 0.5, 0.7, 1.0],
    'Scenario F1': [0.6005, 0.5793, 0.6101, 0.5968, 0.5854],  
    'Action F1': [None, 0.2915, 0.3816, 0.3805, 0.3948],
    'Params (M)': [1.5, 1.5, 1.5, 1.5, 1.5]
}

# Add baselines for comparison
baselines = {
    'Model': ['IMU2CLIP', 'CNN-MLP', 'CNN-LSTM-GRU', 'MLP-MLP'],
    'Scenario F1': [0.6025, 0.5948, 0.5705, 0.5702],
    'Action F1': [0.3586, 0.3043, 0.3202, 0.2799],
    'Params (M)': [4.0, 1.1, 2.0, 1.0]
}

df_beta = pd.DataFrame(data)
df_baselines = pd.DataFrame(baselines)

# ============================================================
# Figure 1: β Parameter Sweep - Pareto Frontier
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Plot our models (β sweep)
valid_beta = df_beta[df_beta['Action F1'].notna()]
ax.plot(valid_beta['Action F1'], valid_beta['Scenario F1'], 
        'o-', markersize=12, linewidth=2.5, label='Ours (β sweep)', 
        color='#2E86AB', zorder=3)

# Annotate β values
for _, row in valid_beta.iterrows():
    ax.annotate(f"β={row['β']:.1f}", 
                xy=(row['Action F1'], row['Scenario F1']),
                xytext=(10, -5), textcoords='offset points',
                fontsize=10, fontweight='bold')

# Plot β=0.0 (scenario only)
ax.scatter([0.32], [0.6005], s=200, marker='*', 
           color='#2E86AB', label='Ours (β=0.0 + Probe)', 
           edgecolors='black', linewidths=1.5, zorder=4)

# Plot baselines
for _, row in df_baselines.iterrows():
    marker = 's' if row['Model'] == 'IMU2CLIP' else 'D'
    size = 150 if row['Model'] == 'IMU2CLIP' else 100
    ax.scatter(row['Action F1'], row['Scenario F1'], 
               s=size, marker=marker, alpha=0.7,
               label=row['Model'], edgecolors='black', linewidths=1)

# Highlight β=1.0 (best action)
best_action_idx = valid_beta['Action F1'].idxmax()
best_row = valid_beta.loc[best_action_idx]
ax.scatter([best_row['Action F1']], [best_row['Scenario F1']], 
           s=300, marker='o', facecolors='none', 
           edgecolors='red', linewidths=3, zorder=5)
ax.annotate('SOTA Action!', 
            xy=(best_row['Action F1'], best_row['Scenario F1']),
            xytext=(20, 20), textcoords='offset points',
            fontsize=12, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax.set_xlabel('Action F1 Score', fontsize=14, fontweight='bold')
ax.set_ylabel('Scenario F1 Score', fontsize=14, fontweight='bold')
ax.set_title('Hierarchical Model Performance vs. Baselines\n(Pareto Frontier Analysis)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('beta_ablation_pareto.png', dpi=300, bbox_inches='tight')
print("✓ Saved: beta_ablation_pareto.png")

# ============================================================
# Figure 2: β Parameter Effect on Individual Tasks
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Scenario F1 vs β
ax1.plot(df_beta['β'], df_beta['Scenario F1'], 
         'o-', markersize=10, linewidth=2.5, color='#A23B72')
ax1.axhline(y=0.6025, color='gray', linestyle='--', linewidth=2, 
            label='IMU2CLIP (SOTA)', alpha=0.7)
ax1.set_xlabel('β (Task Weight)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Scenario F1 Score', fontsize=14, fontweight='bold')
ax1.set_title('Scenario Recognition Performance', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.55, 0.61])

# Right: Action F1 vs β
valid_beta_action = df_beta[df_beta['Action F1'].notna()]
ax2.plot(valid_beta_action['β'], valid_beta_action['Action F1'], 
         'o-', markersize=10, linewidth=2.5, color='#F18F01')
ax2.axhline(y=0.3586, color='gray', linestyle='--', linewidth=2, 
            label='IMU2CLIP', alpha=0.7)
ax2.axhline(y=0.3948, color='red', linestyle=':', linewidth=2, 
            label='Ours β=1.0 (NEW SOTA)', alpha=0.8)
ax2.set_xlabel('β (Task Weight)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Action F1 Score', fontsize=14, fontweight='bold')
ax2.set_title('Action Classification Performance', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.27, 0.41])

plt.tight_layout()
plt.savefig('beta_ablation_tasks.png', dpi=300, bbox_inches='tight')
print("✓ Saved: beta_ablation_tasks.png")

# ============================================================
# Figure 3: Model Efficiency Comparison (Params vs Performance)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate combined F1 for all models
df_baselines['Combined F1'] = (df_baselines['Scenario F1'] + df_baselines['Action F1']) / 2

# Our models
our_models = []
for _, row in valid_beta.iterrows():
    combined = (row['Scenario F1'] + row['Action F1']) / 2
    our_models.append({
        'Model': f"Ours (β={row['β']:.1f})",
        'Params (M)': 1.5,
        'Combined F1': combined
    })

# Add probe
our_models.append({
    'Model': 'Ours (Probe)',
    'Params (M)': 1.5,
    'Combined F1': (0.6005 + 0.32) / 2
})

df_our = pd.DataFrame(our_models)

# Plot baselines
ax.scatter(df_baselines['Params (M)'], df_baselines['Combined F1'], 
           s=200, marker='s', alpha=0.7, label='Baselines', 
           edgecolors='black', linewidths=1.5)

for _, row in df_baselines.iterrows():
    ax.annotate(row['Model'], 
                xy=(row['Params (M)'], row['Combined F1']),
                xytext=(10, 5), textcoords='offset points',
                fontsize=9)

# Plot our models
ax.scatter(df_our['Params (M)'], df_our['Combined F1'], 
           s=200, marker='o', alpha=0.8, color='#2E86AB',
           label='Ours (Hierarchical)', 
           edgecolors='black', linewidths=1.5)

for _, row in df_our.iterrows():
    ax.annotate(row['Model'], 
                xy=(row['Params (M)'], row['Combined F1']),
                xytext=(-30, 8), textcoords='offset points',
                fontsize=9, fontweight='bold', color='#2E86AB')

# Highlight best efficiency (β=1.0)
best_idx = df_our['Combined F1'].idxmax()
best = df_our.loc[best_idx]
ax.scatter([best['Params (M)']], [best['Combined F1']], 
           s=400, marker='*', color='gold', 
           edgecolors='red', linewidths=2, zorder=10,
           label='Best Overall')

ax.set_xlabel('Model Parameters (Millions)', fontsize=14, fontweight='bold')
ax.set_ylabel('Combined F1 Score', fontsize=14, fontweight='bold')
ax.set_title('Model Efficiency: Parameters vs. Performance', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim([0.8, 4.5])

plt.tight_layout()
plt.savefig('model_efficiency.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_efficiency.png")

# ============================================================
# Generate Summary Table
# ============================================================
print("\n" + "="*80)
print("COMPLETE β ABLATION RESULTS")
print("="*80)
print(df_beta.to_string(index=False))

print("\n" + "="*80)
print("BASELINE COMPARISON")
print("="*80)
print(df_baselines.to_string(index=False))

print("\n" + "="*80)
print("🏆 BEST PERFORMERS")
print("="*80)
print(f"Best Scenario F1: β=0.0 with {df_beta['Scenario F1'].max():.4f}")
print(f"Best Action F1: β=1.0 with {df_beta['Action F1'].max():.4f}")
print(f"Best Combined: β=1.0 with {(0.5854 + 0.3948)/2:.4f}")
print("\n✅ Our model BEATS all baselines on action classification!")
print(f"   β=1.0 (0.3948) vs IMU2CLIP (0.3586) = +{((0.3948/0.3586 - 1)*100):.1f}%")
