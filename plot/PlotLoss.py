# =========================================================
# Physical & L1 Loss Comparison Plot (Train + Valid Overlay)
# 2x3 Subplots (Top 3, Bottom 2 + 1 hidden):
# - (a) L1 Loss
# - (b) Divergence Loss
# - (c) Vorticity Loss
# - (d) Geostrophic Balance Loss
# - (e) Moist Static Energy Gradient Loss
# =========================================================

import warnings
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# 1) Hyperparameters & Configurations
# =========================================================
# Input file paths
CSV_NO_PHY = "/content/training_history_noPhy.csv"
CSV_PHY = "/content/training_history_AllPhy.csv"

# Output export settings
SAVE_FIG = True
SAVE_PATH_PNG = "/content/physics_losses_train_valid_overlay_X2.png"
SAVE_PATH_PDF = "/content/physics_losses_train_valid_overlay_X2.pdf"
SAVE_DPI = 600

# Canvas layout configurations
FIG_SIZE = (21, 11)
PLOT_DPI = 600

# Model styling configurations
MODEL_STYLES: Dict[str, Dict[str, Any]] = {
    "no_phy": {
        "label": "AtmoSR-Base",
        "color": "#1f77b4",  # Classic Blue
        "marker": "o",
    },
    "phy": {
        "label": "AtmoSR",
        "color": "#d62728",  # Classic Red
        "marker": "s",
    },
}

# Data split styling configurations
SPLIT_STYLES: Dict[str, Dict[str, Any]] = {
    "train": {
        "label": "Train",
        "linestyle": "-",
        "linewidth": 2.0,
        "alpha": 0.97,
        "markevery": 15,
    },
    "valid": {
        "label": "Valid",
        "linestyle": "--",
        "linewidth": 2.0,
        "alpha": 0.95,
        "markevery": (7, 15),  # Offset marker sampling for visibility
    },
}

# Axis range configurations (Set to None for auto-scaling)
AXIS_CONFIG: Dict[str, Dict[str, Any]] = {
    "l1": {"x_min": 1, "x_max": 200, "x_step": 50, "y_min": None, "y_max": None, "y_step": None},
    "div": {"x_min": 1, "x_max": 200, "x_step": 50, "y_min": 0.05, "y_max": 0.40, "y_step": 0.05},
    "vor": {"x_min": 1, "x_max": 200, "x_step": 50, "y_min": 0.05, "y_max": 0.25, "y_step": 0.05},
    "geo": {"x_min": 1, "x_max": 200, "x_step": 50, "y_min": 0.10, "y_max": 0.40, "y_step": 0.05},
    "mse": {"x_min": 1, "x_max": 200, "x_step": 50, "y_min": 0.10, "y_max": 0.35, "y_step": 0.05},
}

# Metrics metadata: (metric_key, subplot_title, y_label)
METRICS = [
    ("l1", "(a) L1 Loss", "L1 loss (dimensionless)"),
    ("div", "(b) Divergence Loss", "GRE loss (dimensionless)"),
    ("vor", "(c) Vorticity Loss", "GRE loss (dimensionless)"),
    ("geo", "(d) Geostrophic Balance Loss", "GRE loss (dimensionless)"),
    ("mse", "(e) Moist Static Energy Gradient Loss", "GRE loss (dimensionless)"),
]

# Required column headers in raw CSV files
REQUIRED_COLS = [
    "Epoch",
    "Train_L1", "Valid_L1",
    "Train_P_div", "Valid_P_div",
    "Train_P_vor", "Valid_P_vor",
    "Train_P_geo", "Valid_P_geo",
    "Train_P_mse", "Valid_P_mse",
]


# =========================================================
# 2) Utility Functions
# =========================================================
def read_history(csv_path: str) -> pd.DataFrame:
    """Read training history CSV file and perform data cleansing.

    Args:
        csv_path (str): Path to the target CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame sorted by Epoch.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    # Retain required columns and parse to numeric type
    df = df[REQUIRED_COLS].copy()
    for c in REQUIRED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop NaN values, sort by Epoch, and remove duplicates
    df = (
        df.dropna(subset=REQUIRED_COLS)
          .sort_values("Epoch")
          .drop_duplicates(subset="Epoch", keep="last")
          .reset_index(drop=True)
    )
    return df


def apply_axis_settings(ax: plt.Axes, cfg: dict) -> None:
    """Apply customized axis limits and tick locators to a subplot.

    Args:
        ax (plt.Axes): Target Matplotlib Axes object.
        cfg (dict): Configuration dictionary containing axis limits and steps.
    """
    x_min = cfg.get("x_min")
    x_max = cfg.get("x_max")
    x_step = cfg.get("x_step")

    if x_min is not None and x_max is not None:
        ax.set_xlim(x_min, x_max)
    if x_step is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_step))

    y_min = cfg.get("y_min")
    y_max = cfg.get("y_max")
    y_step = cfg.get("y_step")

    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    if y_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_step))

    ax.tick_params(axis="both", which="major", length=6, width=1.15)
    ax.tick_params(axis="both", which="minor", length=3, width=0.8)


def get_merged_col_name(split_key: str, metric_key: str, model_key: str) -> str:
    """Construct merged DataFrame column name.

    (e.g., Train_L1_phy or Train_P_div_no_phy)
    """
    prefix = "Train" if split_key == "train" else "Valid"
    if metric_key == "l1":
        base_col = f"{prefix}_L1"
    else:
        base_col = f"{prefix}_P_{metric_key}"
    return f"{base_col}_{model_key}"


# =========================================================
# 3) Main Plotting Execution
# =========================================================
def main() -> None:
    # Set global plotting style
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "axes.titleweight": "bold",
        "axes.labelweight": "normal",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    # 1. Load data from CSV files
    df_no_phy = read_history(CSV_NO_PHY)
    df_phy = read_history(CSV_PHY)

    # 2. Perform inner merge on Epoch
    df = pd.merge(
        df_no_phy,
        df_phy,
        on="Epoch",
        how="inner",
        suffixes=("_no_phy", "_phy"),
    ).sort_values("Epoch").reset_index(drop=True)

    if df.empty:
        raise ValueError("No overlapping Epoch values were found between the two CSV files.")

    # 3. Initialize 2x3 grid canvas
    fig, axes = plt.subplots(2, 3, figsize=FIG_SIZE, dpi=PLOT_DPI)
    axes = axes.flatten()
    fig.patch.set_facecolor("white")

    # 4. Plot the first 5 subplots
    for idx, (metric_key, title, y_label) in enumerate(METRICS):
        ax = axes[idx]

        for model_key, mstyle in MODEL_STYLES.items():
            for split_key, sstyle in SPLIT_STYLES.items():
                col = get_merged_col_name(split_key, metric_key, model_key)

                if col not in df.columns:
                    raise ValueError(f"Column not found: {col}")

                label = (
                    f"{mstyle['label']} ({sstyle['label']})"
                    if idx == 0 else "_nolegend_"
                )

                ax.plot(
                    df["Epoch"],
                    df[col],
                    label=label,
                    color=mstyle["color"],
                    linestyle=sstyle["linestyle"],
                    linewidth=sstyle["linewidth"],
                    alpha=sstyle["alpha"],
                    marker=mstyle["marker"],
                    markersize=5.5,
                    markevery=sstyle["markevery"],
                    markeredgewidth=1.1,
                    markeredgecolor=mstyle["color"],
                    markerfacecolor=("white" if split_key == "valid" else mstyle["color"]),
                )

        ax.set_title(title, pad=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(y_label)

        apply_axis_settings(ax, AXIS_CONFIG[metric_key])

        # Gridlines and subplot background styling
        ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.42)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.55, alpha=0.22)
        ax.minorticks_on()
        ax.set_facecolor("white")

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("#b9bec5")

    # 5. Hide the unused 6th subplot (bottom-right)
    axes[5].set_visible(False)

    # 6. Place unified legend inside subplot (a)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles, labels,
        loc="upper right",
        ncol=2,
        handlelength=2.2,
        handletextpad=0.5,
        labelspacing=0.3,
        columnspacing=0.8,
        frameon=False,
    )

    plt.tight_layout()

    # 7. Export figures
    if SAVE_FIG:
        Path(SAVE_PATH_PNG).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(SAVE_PATH_PNG, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
        fig.savefig(SAVE_PATH_PDF, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
        print(f"[Saved] Figure (PNG): {SAVE_PATH_PNG}")
        print(f"[Saved] Figure (PDF): {SAVE_PATH_PDF}")

    plt.show()


if __name__ == "__main__":
    main()