import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 1. Global Font and Style Configuration (Journal Standard)
# ==============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['pdf.fonttype'] = 42  # Ensure text remains vectorized in exported PDF

# ==============================================================================
# 2. Data Preparation and Preprocessing
# ==============================================================================
variables = ['Φ', 'q', 'T', 'u', 'v']

# 2x Downscaling RMSE Data
cs_2x = np.array([6.2013560, 0.0000816, 0.0700747, 0.2550877, 0.2220427])
base_2x = np.array([3.6773170, 0.0000531, 0.0449754, 0.1277642, 0.1095648])
atmo_2x = np.array([1.4072020, 0.0000494, 0.0399576, 0.1012408, 0.0917588])

# 4x Downscaling RMSE Data
cs_4x = np.array([19.746800, 0.000180, 0.183536, 0.668933, 0.565885])
base_4x = np.array([6.556644, 0.000130, 0.120026, 0.449925, 0.3923801])
atmo_4x = np.array([3.878338, 0.000123, 0.112427, 0.400055, 0.360883])

# Calculate percentage RMSE improvement relative to Cubic Spline (%)
imp_base_2x = (cs_2x - base_2x) / cs_2x * 100
imp_atmo_2x = (cs_2x - atmo_2x) / cs_2x * 100

imp_base_4x = (cs_4x - base_4x) / cs_4x * 100
imp_atmo_4x = (cs_4x - atmo_4x) / cs_4x * 100

# ==============================================================================
# 3. Plot Layout and Parameter Setup
# ==============================================================================
color_base = '#1f77b4'  # AtmoSR-Base (Blue)
color_atmo = '#d62728'  # AtmoSR (Red)

x = np.arange(len(variables))
width = 0.35  # Bar width

# Create 1x2 subplot layout
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=600)

# Y-axis bounds and tick spacing
y_min, y_max = 0, 100
y_ticks = np.arange(y_min, y_max + 1, 20)

# ==============================================================================
# 4. Plot 2x Downscaling Subplot (Left Panel)
# ==============================================================================
ax1 = axes[0]
rects1_base = ax1.bar(x - width / 2, imp_base_2x, width, label='AtmoSR-Base', color=color_base, edgecolor='black', linewidth=1)
rects1_atmo = ax1.bar(x + width / 2, imp_atmo_2x, width, label='AtmoSR', color=color_atmo, edgecolor='black', linewidth=1)

ax1.set_title('(b) 2X Downscaling', fontsize=14, fontweight='bold', loc='left', pad=12)
ax1.set_ylabel('RMSE Improvement vs. Cubic Spline (%)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(variables, fontsize=16, fontweight='bold')
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_ticks, fontsize=12)
ax1.set_ylim([y_min, y_max])

# ==============================================================================
# 5. Plot 4x Downscaling Subplot (Right Panel)
# ==============================================================================
ax2 = axes[1]
rects2_base = ax2.bar(x - width / 2, imp_base_4x, width, label='AtmoSR-Base', color=color_base, edgecolor='black', linewidth=1)
rects2_atmo = ax2.bar(x + width / 2, imp_atmo_4x, width, label='AtmoSR', color=color_atmo, edgecolor='black', linewidth=1)

ax2.set_title('(c) 4X Downscaling', fontsize=14, fontweight='bold', loc='left', pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(variables, fontsize=16, fontweight='bold')
ax2.set_yticks(y_ticks)
ax2.set_yticklabels(y_ticks, fontsize=12)
ax2.set_ylim([y_min, y_max])

# Place legend in top right of the right panel without a frame
ax2.legend(loc='upper right', frameon=False, fontsize=14)

# ==============================================================================
# 6. Fine-Tuning and Spines/Grid Styling
# ==============================================================================
for ax in axes:
    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add subtle horizontal gridlines for easier visual alignment
    ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='gray', zorder=0)

    # Ensure bars are drawn above gridlines
    ax.set_axisbelow(True)

plt.tight_layout()

# ==============================================================================
# 7. Output and Figure Saving
# ==============================================================================
plt.savefig('AtmoSR_RMSE_Improvement.pdf', format='pdf', bbox_inches='tight')
plt.savefig('AtmoSR_RMSE_Improvement.png', format='png', bbox_inches='tight', dpi=600)

plt.show()