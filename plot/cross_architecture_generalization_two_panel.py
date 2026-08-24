"""
Figure Generation Script: Cross-Architecture Evaluation of GRE
--------------------------------------------------------------
This script generates a publication-ready two-panel figure comparing baseline
and physics-constrained (Phys) deep learning models across atmospheric variables.

Panel (a): Accuracy-Efficiency Trade-off (Training Time vs. Phi RMSE)
Panel (b): Cross-Architecture Generalization of GRE (RMSE Reduction %)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# =============================================================================
# Global Styling & Figure Setup (Publication Standards)
# =============================================================================
FIG_W, FIG_H = 17.2, 6.8
FIG_DPI = 600
SAVE_PREFIX = "two_panel_evaluation"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "mathtext.fontset": "stix",    # Stable mathematical font rendering
    "pdf.fonttype": 42,            # Embed TrueType fonts for vector outputs
    "ps.fonttype": 42,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})

FS_PANEL = 18
FS_TITLE = 14
FS_LABEL = 14
FS_TICK = 14
FS_LEGEND = 12
FS_NOTE = 12

COLORS = {
    "AtmoSR": "#d62728",
    "VDSR":   "#2ca02c",
    "EDSR":   "#ff7f0e",
    "SwinIR": "#8c564b",
}
BASELINE_COLOR = "#666666"

# =============================================================================
# Data Initialization & Processing
# =============================================================================
data = {
    "Variable": ["Φ", "q", "T", "u", "v"],
    "Cubic_Spline": [6.201356, 0.000082, 0.070075, 0.255088, 0.222043],
    "AtmoSR_Base": [3.677317, 5.31e-05, 0.044975, 0.127764, 0.109565],
    "AtmoSR_Phys": [1.407202, 4.94e-05, 0.039958, 0.101241, 0.091759],
    "VDSR_Base": [4.989890, 5.53e-05, 0.053213, 0.141924, 0.115379],
    "VDSR_Phys": [2.410880, 5.19e-05, 0.048067, 0.114493, 0.101712],
    "EDSR_Base": [4.506973, 5.31e-05, 0.048664, 0.143885, 0.115685],
    "EDSR_Phys": [2.159496, 5.17e-05, 0.047814, 0.114431, 0.102679],
    "SwinIR_Base": [3.420090, 5.33e-05, 0.041255, 0.132407, 0.113349],
    "SwinIR_Phys": [2.822272, 5.24e-05, 0.039777, 0.120442, 0.106407]
}
df = pd.DataFrame(data)
models = ["AtmoSR", "VDSR", "EDSR", "SwinIR"]

# Calculate induced RMSE reduction (%)
for m in models:
    df[f"{m}_Reduction"] = (df[f"{m}_Base"] - df[f"{m}_Phys"]) / df[f"{m}_Base"] * 100.0

# Define training time metrics (seconds per epoch)
PHYS_TIME_SEC = {
    "AtmoSR": 21.0,
    "VDSR": 36.0,
    "EDSR": 20.0,
    "SwinIR": 94.0
}
FIXED_PHYS_COST = 4.0
TRAIN_TIME_SEC = {}
for m in models:
    TRAIN_TIME_SEC[f"{m}_Phys"] = PHYS_TIME_SEC[m]
    TRAIN_TIME_SEC[f"{m}_Base"] = PHYS_TIME_SEC[m] - FIXED_PHYS_COST

# =============================================================================
# Canvas & Layout Initialization
# =============================================================================
fig, (ax_left, ax_right) = plt.subplots(
    1, 2,
    figsize=(FIG_W, FIG_H),
    dpi=FIG_DPI,
    gridspec_kw={"width_ratios": [1.0, 1.25]}
)
fig.subplots_adjust(wspace=0.22)

# =============================================================================
# Panel (a): Accuracy-Efficiency Trade-off (Scatter Plot + Arrows)
# =============================================================================
phi_row = df.loc[df["Variable"] == "Φ"].iloc[0]
phi_cubic = float(phi_row["Cubic_Spline"])
marker_map = {"AtmoSR": "o", "VDSR": "s", "EDSR": "^", "SwinIR": "D"}

for m in models:
    color = COLORS[m]
    mk = marker_map[m]

    xb = float(TRAIN_TIME_SEC[f"{m}_Base"])
    yb = float(phi_row[f"{m}_Base"])
    xp = float(TRAIN_TIME_SEC[f"{m}_Phys"])
    yp = float(phi_row[f"{m}_Phys"])

    # Base model: Hollow marker
    ax_left.scatter(
        xb, yb, s=88, marker=mk,
        facecolors="white", edgecolors=color, linewidths=1.5, zorder=4
    )
    # Phys model: Filled marker
    ax_left.scatter(
        xp, yp, s=88, marker=mk,
        facecolors=color, edgecolors=color, linewidths=1.0, zorder=5
    )

    # Transition arrow (Base -> Phys)
    ax_left.annotate(
        "", xy=(xp, yp), xytext=(xb, yb),
        arrowprops=dict(
            arrowstyle="-|>", lw=1.25, color=color, alpha=0.95,
            shrinkA=11, shrinkB=11
        ),
        zorder=3
    )

# Baseline reference (Cubic Spline)
ax_left.axhline(phi_cubic, linestyle=(0, (4, 2)), color=BASELINE_COLOR, linewidth=1.3, zorder=1)
ax_left.text(
    10.4, phi_cubic + 0.08,
    rf"Cubic Spline baseline ($\mathit{{\Phi}}$ RMSE = {phi_cubic:.2f})",
    fontsize=FS_NOTE + 1, color=BASELINE_COLOR
)

# Highlight best-performing point
x_best = float(TRAIN_TIME_SEC["AtmoSR_Phys"])
y_best = float(phi_row["AtmoSR_Phys"])
ax_left.scatter(
    x_best, y_best, s=195, marker="o",
    facecolors="none", edgecolors=COLORS["AtmoSR"], linewidths=1.9, zorder=6
)
ax_left.annotate(
    "AtmoSR-Phys\n(best accuracy)",
    xy=(x_best + 0.15, y_best - 0.35), xytext=(10, 10), textcoords="offset points",
    fontsize=FS_NOTE + 1, color=COLORS["AtmoSR"], ha="left", va="bottom"
)

# Axis configuration for Panel (a)
ax_left.set_xscale("log")
ax_left.set_xlim(10, 100)
xticks = np.arange(10, 101, 10)
ax_left.xaxis.set_major_locator(FixedLocator(xticks))
ax_left.xaxis.set_major_formatter(
    FuncFormatter(lambda val, pos: f"{int(round(val))}" if np.any(np.isclose(val, xticks)) else "")
)
ax_left.xaxis.set_minor_locator(NullLocator())

ax_left.set_ylim(0.0, 7.0)
yticks = np.arange(0.0, 7.1, 1.0)
ax_left.set_yticks(yticks)
ax_left.set_yticklabels([f"{v:.1f}" for v in yticks], fontsize=FS_TICK)

# Fine-tune bottom spine and labels
ax_left.spines["bottom"].set_position(("outward", -2))
ax_left.tick_params(axis="x", labelsize=FS_TICK, pad=2)

ax_left.set_xlabel("Training time per epoch (s, log scale)", fontsize=FS_LABEL + 2, fontweight="bold", labelpad=3)
ax_left.set_ylabel(r"$\mathit{\Phi}$ RMSE (m² s⁻²)", fontsize=FS_LABEL + 2, fontweight="bold")
ax_left.set_title("(a) Accuracy-Efficiency Trade-off", fontsize=FS_TITLE + 3, fontweight="bold", pad=10)

ax_left.grid(which="major", linestyle="--", alpha=0.33, color="#bdbdbd")
ax_left.set_axisbelow(True)
ax_left.spines["top"].set_visible(False)
ax_left.spines["right"].set_visible(False)

# Construct combined legend for Panel (a)
arch_handles = []
for m in models:
    face = COLORS[m] if m == "AtmoSR" else "white"  # AtmoSR is filled in the architecture legend
    arch_handles.append(
        Line2D([0], [0], marker=marker_map[m], linestyle="None",
               markerfacecolor=face, markeredgecolor=COLORS[m], markeredgewidth=1.5,
               markersize=8.0, label=("AtmoSR" if m == "AtmoSR" else m))
    )

style_handles = [
    Line2D([0], [0], marker="o", linestyle="None",
           markerfacecolor="white", markeredgecolor="black", markersize=8.0,
           label="Base (hollow)"),
    Line2D([0], [0], marker="o", linestyle="None",
           markerfacecolor="black", markeredgecolor="black", markersize=8.0,
           label="Phys (filled)"),
    Line2D([0], [0], color=BASELINE_COLOR, linestyle=(0, (4, 2)), linewidth=1.3,
           label="Cubic Spline baseline"),
]
legend_handles = arch_handles + style_handles

ax_left.legend(
    handles=legend_handles,
    loc="upper right",
    bbox_to_anchor=(0.99, 0.99),
    frameon=True, framealpha=1.00, facecolor="white", edgecolor="#999999",
    fontsize=FS_LEGEND, title="Architecture + Style", title_fontsize=FS_LEGEND + 0.8
)

# =============================================================================
# Panel (b): Cross-Architecture Generalization (Bar Chart)
# =============================================================================
x = np.arange(len(df["Variable"]))
bar_w = 0.19

ax_right.bar(x - 1.5*bar_w, df["AtmoSR_Reduction"], bar_w, label="AtmoSR",
             color=COLORS["AtmoSR"], edgecolor="black", linewidth=0.55)
ax_right.bar(x - 0.5*bar_w, df["VDSR_Reduction"], bar_w, label="VDSR",
             color=COLORS["VDSR"], edgecolor="black", linewidth=0.55)
ax_right.bar(x + 0.5*bar_w, df["EDSR_Reduction"], bar_w, label="EDSR",
             color=COLORS["EDSR"], edgecolor="black", linewidth=0.55)
ax_right.bar(x + 1.5*bar_w, df["SwinIR_Reduction"], bar_w, label="SwinIR",
             color=COLORS["SwinIR"], edgecolor="black", linewidth=0.55)

# Format variable names using italic mathtext
var_labels_math = [r"$\mathit{\Phi}$", r"$\mathit{q}$", r"$\mathit{T}$", r"$\mathit{u}$", r"$\mathit{v}$"]
ax_right.set_xticks(x)
ax_right.set_xticklabels(var_labels_math, fontsize=FS_TICK + 8)

ax_right.set_ylim(0, 80)
ax_right.set_yticks(np.arange(0, 81, 10))
ax_right.tick_params(axis="y", labelsize=FS_TICK)

ax_right.set_ylabel("GRE-Induced RMSE Reduction (%)", fontsize=FS_LABEL + 1, fontweight="bold")
ax_right.set_title("(b) Cross-Architecture Generalization of GRE", fontsize=FS_TITLE + 2, fontweight="bold", pad=10)

ax_right.grid(axis="y", linestyle="--", alpha=0.33, color="#bdbdbd")
ax_right.set_axisbelow(True)
ax_right.spines["top"].set_visible(False)
ax_right.spines["right"].set_visible(False)

# Legend placement for Panel (b)
ax_right.legend(loc="upper right", frameon=False, fontsize=FS_LEGEND + 3, ncol=2)

# =============================================================================
# Save & Display Options
# =============================================================================
plt.tight_layout()
fig.savefig(f"{SAVE_PREFIX}.pdf", format="pdf", bbox_inches="tight")
fig.savefig(f"{SAVE_PREFIX}.png", format="png", dpi=FIG_DPI, bbox_inches="tight")
plt.show()