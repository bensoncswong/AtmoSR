import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. Parameter Configuration
# ==========================================
Y_MIN = 0       # Y-axis minimum value
Y_MAX = 80      # Y-axis maximum value
Y_STEP = 10     # Y-axis tick interval
y_ticks = np.arange(Y_MIN, Y_MAX + 1, Y_STEP)

# ==========================================
# 2. Data Input & Processing
# ==========================================
# Extracted RMSE values for Base and Physical models
data = {
    'Variable': ['Φ', 'q', 'T', 'u', 'v'],
    'AtmoSR_Base': [3.677317, 5.31e-05, 0.044975, 0.127764, 0.109565],
    'AtmoSR_Phys': [1.407202, 4.94e-05, 0.039958, 0.101241, 0.091759],
    'VDSR_Base': [4.989890, 5.53e-05, 0.053213, 0.141924, 0.115379],
    'VDSR_Phys': [2.410880, 5.19e-05, 0.048067, 0.114493, 0.101712],
    'EDSR_Base': [4.506973, 5.31e-05, 0.048664, 0.143885, 0.115685],
    'EDSR_Phys': [2.159496, 5.17e-05, 0.047814, 0.114431, 0.102679],
    'SwinIR_Base': [3.420090, 5.33e-05, 0.041255, 0.132407, 0.113349],
    'SwinIR_Phys': [2.822272, 5.24e-05, 0.039777, 0.120442, 0.106407]
}

df = pd.DataFrame(data)

# Calculate Relative Error Reduction (%) = (Base - Phys) / Base * 100
models = ['AtmoSR', 'VDSR', 'EDSR', 'SwinIR']
for model in models:
    df[f'{model}_Reduction'] = ((df[f'{model}_Base'] - df[f'{model}_Phys']) / df[f'{model}_Base']) * 100

# ==========================================
# 3. Visualization Setup (Publication Quality)
# ==========================================
# Set font family (sans-serif)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

fig, ax = plt.subplots(figsize=(10, 6), dpi=600)

# X-axis setup
x = np.arange(len(df['Variable']))
width = 0.2  # Bar width

# Color scheme configuration
colors = {
    'AtmoSR': '#d62728',  # Red accent for proposed method
    'VDSR': '#2ca02c',    # Green
    'EDSR': '#ff7f0e',    # Orange
    'SwinIR': '#8c564b'   # Brown
}

# Plot grouped bar chart
rects1 = ax.bar(x - 1.5 * width, df['AtmoSR_Reduction'], width, label='AtmoSR (Ours)', color=colors['AtmoSR'], edgecolor='black', linewidth=0.5)
rects2 = ax.bar(x - 0.5 * width, df['VDSR_Reduction'], width, label='VDSR', color=colors['VDSR'], edgecolor='black', linewidth=0.5)
rects3 = ax.bar(x + 0.5 * width, df['EDSR_Reduction'], width, label='EDSR', color=colors['EDSR'], edgecolor='black', linewidth=0.5)
rects4 = ax.bar(x + 1.5 * width, df['SwinIR_Reduction'], width, label='SwinIR', color=colors['SwinIR'], edgecolor='black', linewidth=0.5)

# ==========================================
# 4. Axes & Style Refinements
# ==========================================
# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Y-axis formatting
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks, fontsize=12)
ax.set_ylabel('RMSE Reduction (%)\n[Base vs. Phys]', fontsize=14, fontweight='bold', labelpad=15)

# X-axis formatting
ax.set_xticks(x)
ax.set_xticklabels(['Φ', 'q', 'T', 'u', 'v'], fontsize=16, fontweight='bold')

# Reference baseline and grid
ax.axhline(0, color='black', linewidth=1.2)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#d3d3d3')
ax.set_axisbelow(True)

# Legend setup
ax.legend(loc='upper right', frameon=False, fontsize=14, ncol=2)

# ==========================================
# 5. Export Figure
# ==========================================
plt.tight_layout()

# Save high-resolution PDF and PNG
plt.savefig('cross_architecture_generalization.pdf', format='pdf', bbox_inches='tight')
plt.savefig('cross_architecture_generalization.png', format='png', dpi=600, bbox_inches='tight')

plt.show()