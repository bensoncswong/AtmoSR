#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=================================================================================
Extreme Tail Diagnostics: Deficit Curves + Absolute Quantile Curves + Table 3
(AGU / JAMES Publication Style)
=================================================================================
Inputs:
    Fig_tail_return_curves.csv        [Required] Quantile values (ERA5 / cubic / noPhy / AllPhy)
    extremes_statements.txt           [Optional] Bootstrap 95% CI and 99.9th paired significance
    <EXTERNAL_CI_CSV>                 [Optional] External long CSV with CIs (takes priority over statements)

Outputs:
    Fig_tail_return_curves_1.png/.pdf Top: Deficit curves, Bottom: Absolute quantile curves
    Fig_tail_return_curves_2.png/.pdf All 8 panels: Deficit curves
    Table3_tail_deficit_maintext.csv  Main text Table 3 (Formatted wide table)
    Table3_tail_deficit_maintext.md   Main text Table 3 (Markdown format)
    TableS7_tail_deficit_full_long.csv SI full long table (Full precision, raw quantiles)
    TableS7_tail_deficit_full_wide.csv SI full wide table
    QC_report_extremes.txt            Statistical QC report (CI inversion / p-value conflicts / BH deduplication)

Deficit Definitions:
    Relative Deficit  D_rel  = 100 * (1 - Q_model / Q_ERA5)   [%]              Positive = Underestimation
    Physical Deficit  D_phys = Q_ERA5 - Q_model                [Physical unit]  Positive = Underestimation
    Signed deficits are preserved (no absolute value applied).
=================================================================================
"""

from __future__ import annotations

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, ScalarFormatter

warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# ==============================================================================
#                      1. USER CONFIGURATION SECTION
#                      ── Modify parameters in this section ──
# ==============================================================================
# ==============================================================================

# ------------------------------------------------------------------------------
# 1.1 File Paths
# ------------------------------------------------------------------------------
CSV_PATH        = "Fig_tail_return_curves.csv"     # [Required]
STATEMENTS_PATH = "extremes_statements.txt"        # [Optional] Set None to skip CI & QC
OUT_DIR         = "."

# [Optional] External long CSV containing bootstrap CIs (e.g., extremes_quantile_long.csv).
# Set to None if disabled. Takes priority over STATEMENTS_PATH.
EXTERNAL_CI_CSV  = None
EXTERNAL_CI_COLS = {
    "evar":       "EVar",            # Variable code (e.g., PHI / DIV)
    "percentile": "Percentile",      # Percentile value (e.g., 99.9 or "99.9th")
    "model":      "Model_Display",   # Model name (e.g., Cubic spline / AtmoSR-Base / AtmoSR)
    "deficit":    "Deficit_pct",     # Signed relative deficit (%)
    "ci_low":     "CI_Low",
    "ci_high":    "CI_High",
}

BASENAME_FIG1 = "Fig_tail_return_curves_1"
BASENAME_FIG2 = "Fig_tail_return_curves_2"
TABLE3_CSV    = "Table3_tail_deficit_maintext.csv"
TABLE3_MD     = "Table3_tail_deficit_maintext.md"
TABLES7_LONG  = "TableS7_tail_deficit_full_long.csv"
TABLES7_WIDE  = "TableS7_tail_deficit_full_wide.csv"
QC_REPORT     = "QC_report_extremes.txt"

SAVE_PDF = True          # Save vector PDF format (recommended for publication)
DPI_SAVE = 600

# ------------------------------------------------------------------------------
# 1.2 Figure Canvas and Fonts
# ------------------------------------------------------------------------------
FIG_WIDTH  = 16.0        # Inches
FIG_HEIGHT = 7.6
NROW, NCOL = 2, 4

FONT_SIZES = {
    "title":  14.0,      # Subplot titles (bold)
    "label":  13.0,      # Axis labels
    "tick":   13.0,      # Tick labels
    "legend": 11.0,      # Legend text
}
PREFER_FONTS = ["Helvetica", "Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]

# ------------------------------------------------------------------------------
# 1.3 X-Axis: Exceedance Probability -> Percentile
#     1e-1 = 90th, 1e-2 = 99th, 1e-3 = 99.9th, 1e-4 = 99.99th, 1e-5 = 99.999th
# ------------------------------------------------------------------------------
X_LIMITS = (1.0e-1, 1.0e-5)                               # Left -> Right (reversed log)
X_TICKS  = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
X_LABEL  = "Percentile (%)"
X_MINOR_TICKS = True                                       # Display minor log ticks

# ------------------------------------------------------------------------------
# 1.4 Subplot Layout and Plotting Modes
#     "deficit"  = Deficit curves (% or physical units)
#     "absolute" = Absolute quantile curves
# ------------------------------------------------------------------------------
PANEL_ORDER = ["PHI", "Q", "T", "WSPD", "DIV", "VOR", "GEO", "MSE_GRAD"]

# Figure 1: Top row (a)-(d) Deficit, Bottom row (e)-(h) Absolute
FIG1_MODE = {
    "PHI": "deficit",  "Q":   "deficit",  "T":   "deficit",  "WSPD":     "deficit",
    "DIV": "absolute", "VOR": "absolute", "GEO": "absolute", "MSE_GRAD": "absolute",
}
# Figure 2: All 8 panels display Deficit curves
FIG2_MODE = {ev: "deficit" for ev in PANEL_ORDER}

# ------------------------------------------------------------------------------
# 1.5 Deficit Metrics & Units
#     Variables in PHYS_DEFICIT_UNITS -> Use "Physical Deficit" (prevents large-offset compression)
#     Other variables                  -> Use "Relative Deficit (%)"
# ------------------------------------------------------------------------------
PHYS_DEFICIT_UNITS = {
    "PHI": r"$\mathrm{m^{2}\ s^{-2}}$",
    "T":   r"$\mathrm{K}$",
}
PHYS_DECIMALS = {"PHI": 3, "T": 4}       # Decimal places for physical unit deficits
PCT_DECIMALS  = 2                        # Decimal places for relative deficits (%)

DEFICIT_SIGN  = +1.0     # +1: Positive = Underestimation; -1: Reversed
SMOOTH_WINDOW = 1        # 1 = No smoothing; 3 or 5 = Rolling median along log(exceedance prob)
DRAW_ZERO_LINE = True    # Draw zero reference line on deficit plots

# ------------------------------------------------------------------------------
# 1.6 Panel Titles and Table Labels
# ------------------------------------------------------------------------------
PANEL_TITLES_ABS = {
    "PHI":      r"Geopotential ($\mathrm{m^{2}\ s^{-2}}$)",
    "Q":        r"Specific humidity ($\mathrm{kg\ kg^{-1}}$)",
    "T":        r"Temperature ($\mathrm{K}$)",
    "WSPD":     r"Wind speed ($\mathrm{m\ s^{-1}}$)",
    "DIV":      r"Divergence ($\mathrm{s^{-1}}$)",
    "VOR":      r"Relative vorticity ($\mathrm{s^{-1}}$)",
    "GEO":      r"Ageostrophic wind ($\mathrm{m\ s^{-1}}$)",
    "MSE_GRAD": r"$|\nabla \mathrm{MSE}|$ ($\mathrm{J\ kg^{-1}\ m^{-1}}$)",
}
PANEL_TITLES_DEF = {
    "PHI":      r"Geopotential deficit ($\mathrm{m^{2}\ s^{-2}}$)",
    "Q":        r"Specific humidity deficit (%)",
    "T":        r"Temperature deficit ($\mathrm{K}$)",
    "WSPD":     r"Wind speed deficit (%)",
    "DIV":      r"Divergence deficit (%)",
    "VOR":      r"Relative vorticity deficit (%)",
    "GEO":      r"Ageostrophic wind deficit (%)",
    "MSE_GRAD": r"$|\nabla \mathrm{MSE}|$ deficit (%)",
}
# Plain text names and units for table exports
TABLE_NAMES = {
    "PHI": "Geopotential",  "Q": "Specific humidity", "T": "Temperature",
    "WSPD": "Wind speed",   "DIV": "Divergence",      "VOR": "Relative vorticity",
    "GEO": "Ageostrophic wind", "MSE_GRAD": "|grad MSE|",
}
TABLE_PHYS_UNIT_TEXT = {"PHI": "m^2 s^-2", "T": "K"}

# ------------------------------------------------------------------------------
# 1.7 Y-Axis Configuration (min / max / step); Set None for automatic scaling
#     LOGY_OVERRIDE forces log-scale on specific variables
# ------------------------------------------------------------------------------
LOGY_OVERRIDE = {}          # e.g., {"GEO": True}

# Absolute quantile curves (extended to 90th percentile)
Y_AXIS_CONFIG_ABS = {
    "PHI":      {"min": 57790,   "max": 58440,   "step": None},
    "Q":        {"min": 4.7e-3,  "max": 8.6e-3,  "step": None},
    "T":        {"min": 270.4,   "max": 277.9,   "step": None},
    "WSPD":     {"min": 25.5,    "max": 63.5,    "step": None},
    "DIV":      {"min": 1.9e-5,  "max": 3.10e-4, "step": None},
    "VOR":      {"min": 3.8e-5,  "max": 8.30e-4, "step": None},
    "GEO":      {"min": 7.5,     "max": 245.0,   "step": None},
    "MSE_GRAD": {"min": 3.3e-2,  "max": 2.35e-1, "step": None},
}

# Deficit curves
Y_AXIS_CONFIG_DEF = {
    "PHI":      {"min": -0.10,  "max":  1.10, "step": 0.2},    # m^2 s^-2
    "Q":        {"min": -0.25,  "max":  2.60, "step": 0.5},    # %
    "T":        {"min": -0.010, "max":  0.105,"step": 0.02},   # K
    "WSPD":     {"min": -0.05,  "max":  0.50, "step": 0.1},    # %
    "DIV":      {"min":  -3.0,   "max": 36.0,  "step": 5.0},    # %
    "VOR":      {"min": -1.0,   "max": 12.0,  "step": 2.0},    # %
    "GEO":      {"min": -1.1,   "max": 14.0,  "step": 2.0},    # %
    "MSE_GRAD": {"min":  -2.5,   "max": 30.0,  "step": 5.0},    # %
}

# ------------------------------------------------------------------------------
# 1.8 Legend Configurations
# ------------------------------------------------------------------------------
LEGEND_LOC_ABS = "lower right"
LEGEND_LOC_DEF = "upper left"
LEGEND_ONE_PER_MODE = True     # True: Place one legend per panel mode
LEGEND_LABELS = {              # Custom legend display names
    "ERA5":         "ERA5",
    "Cubic spline": "Cubic spline",
    "AtmoSR-Base":  "AtmoSR-Base",
    "AtmoSR":       "AtmoSR",
}
ZERO_LINE_LABEL = "Ground Truth (ERA5)"

# ------------------------------------------------------------------------------
# 1.9 Table Export Configurations
# ------------------------------------------------------------------------------
REF_DISPLAY  = "ERA5"
MODEL_ORDER  = ["Cubic spline", "AtmoSR-Base", "AtmoSR"]

# Main text Table 3 target variables and percentiles
TABLE3_QUANTITIES  = ["PHI", "T", "Q", "WSPD", "DIV", "VOR", "GEO", "MSE_GRAD"]
TABLE3_PERCENTILES = [99.9, 99.99]

# SI Table S7 complete setup
TABLES7_QUANTITIES  = PANEL_ORDER
TABLES7_PERCENTILES = [90.0, 95.0, 99.0, 99.9, 99.99, 99.999]

BOLD_BEST_IN_MD = True      # Bold minimum |deficit| per row in Markdown table
ATTACH_CI       = True      # Attempt to attach 95% CIs

# ------------------------------------------------------------------------------
# 1.10 Canonical Series Name Mapping (CSV Series column -> Standard display)
# ------------------------------------------------------------------------------
SERIES_CANON = {
    "era5": "ERA5", "ground truth": "ERA5", "truth": "ERA5", "ref": "ERA5", "target": "ERA5",
    "cubic spline": "Cubic spline", "cubic": "Cubic spline", "bicubic": "Cubic spline",
    "spline": "Cubic spline", "interp": "Cubic spline",
    "nophy": "AtmoSR-Base", "atmosr-base": "AtmoSR-Base", "atmosr_base": "AtmoSR-Base",
    "base": "AtmoSR-Base", "baseline": "AtmoSR-Base", "w/o phy": "AtmoSR-Base",
    "allphy": "AtmoSR", "atmosr": "AtmoSR", "ours": "AtmoSR", "proposed": "AtmoSR",
    "full": "AtmoSR", "w/ phy": "AtmoSR",
}

STYLE_MAP = {
    "ERA5":         {"color": "#000000", "ls": "solid",  "lw": 1.6, "zorder": 30},
    "Cubic spline": {"color": "#6e6e6e", "ls": "--", "lw": 1.4, "zorder": 25},
    "AtmoSR-Base":  {"color": "#1f77b4", "ls": "-",  "lw": 1.5, "zorder": 28},
    "AtmoSR":       {"color": "#d62728", "ls": "-",  "lw": 1.6, "zorder": 29},
}
FALLBACK_COLORS = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]


# ==============================================================================
# ==============================================================================
#                      2. CORE IMPLEMENTATION
# ==============================================================================
# ==============================================================================


# ------------------------------------------------------------------------------
# 2.1 Publication Styles
# ------------------------------------------------------------------------------
def set_publication_style() -> str:
    have = {f.name for f in font_manager.fontManager.ttflist}
    sel = [f for f in PREFER_FONTS if f in have] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": sel,
        "mathtext.fontset": "dejavusans",
        "font.size": FONT_SIZES["tick"],
        "axes.labelsize": FONT_SIZES["label"],
        "axes.titlesize": FONT_SIZES["title"],
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
        "legend.fontsize": FONT_SIZES["legend"],
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.size": 3.5,  "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,  "ytick.minor.size": 2.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False,
        "legend.handlelength": 1.9,
        "legend.handletextpad": 0.6, "legend.labelspacing": 0.35,
        "legend.borderpad": 0.45,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })
    return sel[0]


def get_style(canon: str) -> dict:
    if canon in STYLE_MAP:
        return dict(STYLE_MAP[canon])
    idx = abs(hash(canon)) % len(FALLBACK_COLORS)
    return {"color": FALLBACK_COLORS[idx], "ls": "-", "lw": 1.3, "zorder": 20}


# ------------------------------------------------------------------------------
# 2.2 Axis Utilities
# ------------------------------------------------------------------------------
def percentile_label(q: float) -> str:
    """Format float percentile: 0.999 -> '99.9'"""
    s = f"{q * 100.0:.10f}".rstrip("0").rstrip(".")
    return s if s else "0"


def add_percentile_axis(ax, show_xlabel: bool):
    ax.set_xscale("log")
    ax.set_xlim(X_LIMITS[0], X_LIMITS[1])
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels([percentile_label(1.0 - t) for t in X_TICKS])
    if not X_MINOR_TICKS:
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if show_xlabel:
        ax.set_xlabel(X_LABEL, fontsize=FONT_SIZES["label"], labelpad=5)


def apply_y_config(ax, cfg: dict | None, logy: bool):
    if not cfg:
        return
    ymin, ymax, step = cfg.get("min"), cfg.get("max"), cfg.get("step")
    if ymin is not None or ymax is not None:
        cur = ax.get_ylim()
        ax.set_ylim(ymin if ymin is not None else cur[0],
                    ymax if ymax is not None else cur[1])
    if (step is not None) and (step > 0) and (not logy):
        ax.yaxis.set_major_locator(MultipleLocator(step))
    if not logy:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-3, 4))
        ax.yaxis.set_major_formatter(fmt)


def _to_bool(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in ("1", "true", "t", "yes", "y")


# ------------------------------------------------------------------------------
# 2.3 Data Loading & Deficit Computation
# ------------------------------------------------------------------------------
def load_curve_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"[FATAL] Input CSV file not found: {path}")
    df = pd.read_csv(path)
    need = ["EVar", "Series", "ExceedProb", "Value"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"[FATAL] Input CSV missing required columns: {miss}")

    # Standardize canonical display names
    if "Display" in df.columns:
        df["Canon"] = df["Display"].astype(str).str.strip()
    else:
        df["Canon"] = df["Series"].astype(str).str.strip().str.lower().map(
            lambda s: SERIES_CANON.get(s, s)
        )
    df["ExceedProb"] = pd.to_numeric(df["ExceedProb"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df[np.isfinite(df["ExceedProb"]) & (df["ExceedProb"] > 0)]

    # Handle LogY attribute
    if "LogY" in df.columns:
        logy = df.groupby("EVar")["LogY"].first().map(_to_bool).to_dict()
    else:
        logy = {ev: False for ev in df["EVar"].unique()}
    logy.update({k: bool(v) for k, v in LOGY_OVERRIDE.items()})
    df.attrs["logy"] = logy
    return df


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return y
    return pd.Series(y).rolling(window, center=True, min_periods=1).median().to_numpy()


def build_deficit(df: pd.DataFrame) -> pd.DataFrame:
    """Construct tidy deficit table containing:
    EVar / Model / ExceedProb / Percentile / Q_ERA5 / Q_model / Deficit_pct / Deficit_phys
    """
    out = []
    for ev, g in df.groupby("EVar", sort=False):
        piv = g.pivot_table(index="ExceedProb", columns="Canon",
                            values="Value", aggfunc="mean")
        if REF_DISPLAY not in piv.columns:
            print(f"[WARN] {ev}: Reference series '{REF_DISPLAY}' not found. Skipping deficit calculation.")
            continue
        ref = piv[REF_DISPLAY].to_numpy(float)
        prob = piv.index.to_numpy(float)
        for m in MODEL_ORDER:
            if m not in piv.columns:
                continue
            mod = piv[m].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = 100.0 * (1.0 - mod / ref) * DEFICIT_SIGN
            rel = np.where(np.isfinite(rel) & (np.abs(ref) > 0), rel, np.nan)
            phys = (ref - mod) * DEFICIT_SIGN
            out.append(pd.DataFrame({
                "EVar": ev, "Model": m,
                "ExceedProb": prob,
                "Percentile": 100.0 * (1.0 - prob),
                "Q_ERA5": ref, "Q_model": mod,
                "Deficit_pct": rel, "Deficit_phys": phys,
            }))
    if not out:
        sys.exit("[FATAL] Could not calculate deficits. Check Series / Display column mappings.")
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------------------
# 2.4 Figure Rendering Functions
# ------------------------------------------------------------------------------
def _draw_absolute(ax, g_ev: pd.DataFrame, logy: bool):
    order = [REF_DISPLAY] + MODEL_ORDER
    for canon in order:
        d = g_ev[g_ev["Canon"] == canon].sort_values("ExceedProb", ascending=False)
        if d.empty:
            continue
        x = d["ExceedProb"].to_numpy(float)
        y = smooth(d["Value"].to_numpy(float), SMOOTH_WINDOW)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        if logy:
            mask &= y > 0
        if not mask.any():
            continue
        st = get_style(canon)
        ax.plot(x[mask], y[mask], color=st["color"], ls=st["ls"], lw=st["lw"],
                zorder=st["zorder"], label=LEGEND_LABELS.get(canon, canon))
    if logy:
        ax.set_yscale("log")


def _draw_deficit(ax, d_ev: pd.DataFrame, ev: str):
    col = "Deficit_phys" if ev in PHYS_DEFICIT_UNITS else "Deficit_pct"
    if DRAW_ZERO_LINE:
        ax.axhline(
            0.0,
            color="#000000",
            ls="-",
            lw=1.1,
            zorder=15,
            label=ZERO_LINE_LABEL,
        )
    for m in MODEL_ORDER:
        d = d_ev[d_ev["Model"] == m].sort_values("ExceedProb", ascending=False)
        if d.empty:
            continue
        x = d["ExceedProb"].to_numpy(float)
        y = smooth(d[col].to_numpy(float), SMOOTH_WINDOW)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        if not mask.any():
            continue
        st = get_style(m)
        ax.plot(x[mask], y[mask], color=st["color"], ls=st["ls"], lw=st["lw"],
                zorder=st["zorder"], label=LEGEND_LABELS.get(m, m))


def make_figure(df: pd.DataFrame, dfd: pd.DataFrame, mode_map: dict,
                basename: str, title_note: str = "") -> None:
    logy_map = df.attrs.get("logy", {})
    fig, axes = plt.subplots(NROW, NCOL, figsize=(FIG_WIDTH, FIG_HEIGHT))
    axes = np.atleast_1d(axes).ravel()

    modes_with_legend = set()
    n_panels = NROW * NCOL

    for i in range(n_panels):
        ax = axes[i]
        if i >= len(PANEL_ORDER):
            ax.axis("off")
            continue

        ev = PANEL_ORDER[i]
        mode = mode_map.get(ev, "absolute")
        g_ev = df[df["EVar"] == ev]
        d_ev = dfd[dfd["EVar"] == ev]
        if g_ev.empty:
            ax.axis("off")
            continue

        if mode == "absolute":
            logy = bool(logy_map.get(ev, False))
            _draw_absolute(ax, g_ev, logy)
            apply_y_config(ax, Y_AXIS_CONFIG_ABS.get(ev), logy)
            title = PANEL_TITLES_ABS.get(ev, ev)
            leg_loc = LEGEND_LOC_ABS
        else:
            _draw_deficit(ax, d_ev, ev)
            apply_y_config(ax, Y_AXIS_CONFIG_DEF.get(ev), False)
            title = PANEL_TITLES_DEF.get(ev, ev + " deficit")
            leg_loc = LEGEND_LOC_DEF

        add_percentile_axis(ax, show_xlabel=(i >= (NROW - 1) * NCOL))
        ax.set_title(f"({chr(97 + i)}) {title}", fontsize=FONT_SIZES["title"],
                     fontweight="bold", pad=7)
        ax.grid(True, which="major", ls="--", lw=0.5, alpha=0.35, color="#bcbcbc")

        need_legend = (not LEGEND_ONE_PER_MODE) or (mode not in modes_with_legend)
        if need_legend:
            h, l = ax.get_legend_handles_labels()
            if h:
                ax.legend(h, l, loc=leg_loc, frameon=False)
                modes_with_legend.add(mode)

    plt.tight_layout(pad=1.2, w_pad=1.2, h_pad=1.7)
    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, basename + ".png")
    fig.savefig(png, dpi=DPI_SAVE)
    print(f"  [OK] Saved image: {png}")
    if SAVE_PDF:
        pdf = os.path.join(OUT_DIR, basename + ".pdf")
        fig.savefig(pdf)
        print(f"  [OK] Saved vector PDF: {pdf}")
    plt.close(fig)


# ------------------------------------------------------------------------------
# 2.5 Quantile Interpolation (Linear interpolation in log-exceedance space)
# ------------------------------------------------------------------------------
def interp_at(prob: np.ndarray, val: np.ndarray, target_pct: float):
    """Returns tuple: (interpolated_value, is_within_grid_bounds)"""
    tprob = 1.0 - target_pct / 100.0
    if tprob <= 0:
        return np.nan, False
    m = np.isfinite(prob) & np.isfinite(val) & (prob > 0)
    if m.sum() < 2:
        return np.nan, False
    lx, ly = np.log10(prob[m]), val[m]
    o = np.argsort(lx)
    lx, ly = lx[o], ly[o]
    lt = np.log10(tprob)
    inside = bool((lt >= lx.min() - 1e-12) and (lt <= lx.max() + 1e-12))
    return float(np.interp(lt, lx, ly)), inside


def build_point_table(dfd: pd.DataFrame, evars: list, pcts: list) -> pd.DataFrame:
    rows = []
    for ev in evars:
        sub = dfd[dfd["EVar"] == ev]
        if sub.empty:
            continue
        use_phys = ev in PHYS_DEFICIT_UNITS
        for p in pcts:
            for m in MODEL_ORDER:
                d = sub[sub["Model"] == m].sort_values("ExceedProb")
                if d.empty:
                    continue
                prob = d["ExceedProb"].to_numpy(float)
                q_ref, ins1 = interp_at(prob, d["Q_ERA5"].to_numpy(float), p)
                q_mod, _    = interp_at(prob, d["Q_model"].to_numpy(float), p)
                d_pct, _    = interp_at(prob, d["Deficit_pct"].to_numpy(float), p)
                d_phy, _    = interp_at(prob, d["Deficit_phys"].to_numpy(float), p)
                rows.append({
                    "EVar": ev,
                    "Quantity": TABLE_NAMES.get(ev, ev),
                    "Percentile": p,
                    "Percentile_Label": f"{percentile_label(p / 100.0)}th",
                    "Model": m,
                    "Metric": "physical" if use_phys else "relative",
                    "Unit": TABLE_PHYS_UNIT_TEXT.get(ev, "%"),
                    "Q_ERA5": q_ref, "Q_model": q_mod,
                    "Deficit_pct": d_pct, "Deficit_phys": d_phy,
                    "Deficit_reported": d_phy if use_phys else d_pct,
                    "In_grid": ins1,
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# 2.6 Parser for statements file (CI + Paired Significance) & Quality Control
# ------------------------------------------------------------------------------
_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_ORD = r"(?:st|nd|rd|th)"
_SEP = "@@DASH@@"

_STMT_RE = re.compile(
    r"\[with\s+95%\s+CI\]\s*"
    r"Cubic spline interpolation underestimates the\s+(\d+(?:\.\d+)?)" + _ORD +
    r"\s+percentile\s+.+?\s+by\s+(" + _NUM + r")%\s*"
    r"\(\s*95%\s*CI:\s*(" + _NUM + r")\s*" + re.escape(_SEP) + r"\s*(" + _NUM + r")%\s*\)\s*"
    r"and AtmoSR-Base by\s+(" + _NUM + r")%\s*"
    r"\(\s*(" + _NUM + r")\s*" + re.escape(_SEP) + r"\s*(" + _NUM + r")%\s*\)\s*,\s*"
    r"whereas AtmoSR reduces the deficit to\s+(" + _NUM + r")%\s*"
    r"\(\s*(" + _NUM + r")\s*" + re.escape(_SEP) + r"\s*(" + _NUM + r")%\s*\)",
    re.IGNORECASE,
)

_PAIR_RE = re.compile(
    r"^\s*(?P<EVar>[A-Za-z_0-9]+)\s+"
    r"(?P<Pct>\d+(?:\.\d+)?" + _ORD + r")\s+"
    r"(?P<Base>Cubic spline|AtmoSR-Base|AtmoSR)\s+"
    r"(?P<Model>Cubic spline|AtmoSR-Base|AtmoSR)\s+"
    r"(?P<Rest>\S.*)$"
)


def _norm(text: str) -> str:
    t = text.replace("\u2013", _SEP).replace("\u2014", _SEP)
    t = re.sub(r"\s+", " ", t)
    return t


def parse_statements(path: str):
    """Returns tuple: (ci_df, pair_df)"""
    if (path is None) or (not os.path.exists(path)):
        print(f"[INFO] Statement file {path} not provided or missing. Skipping 95% CIs and QC report.")
        return pd.DataFrame(), pd.DataFrame()

    raw = open(path, "r", encoding="utf-8", errors="replace").read()

    # ---- (a) Parse CI Statements ----
    ci_rows = []
    chunks = re.split(r"(?m)^###\s+", raw)
    for ch in chunks[1:]:
        head, _, body = ch.partition("\n")
        ev = re.split(r"[\u2014\u2013\-]", head, maxsplit=1)[0].strip()
        for m in _STMT_RE.finditer(_norm(body)):
            g = m.groups()
            p = float(g[0])
            trio = [("Cubic spline", g[1], g[2], g[3]),
                    ("AtmoSR-Base",  g[4], g[5], g[6]),
                    ("AtmoSR",       g[7], g[8], g[9])]
            for name, v, lo, hi in trio:
                ci_rows.append({"EVar": ev, "Percentile": p, "Model": name,
                                "Deficit_stmt_pct": float(v),
                                "CI_Low": float(lo), "CI_High": float(hi)})
    ci_df = pd.DataFrame(ci_rows)

    # ---- (b) Parse Paired Significance Table ----
    pair_rows = []
    for line in raw.splitlines():
        mm = _PAIR_RE.match(line)
        if not mm:
            continue
        toks = mm.group("Rest").split()
        if len(toks) < 8:
            continue
        try:
            nums = [float(t) for t in toks[:7]]
        except ValueError:
            continue
        pair_rows.append({
            "EVar": mm.group("EVar"),
            "Percentile_Label": mm.group("Pct"),
            "Baseline": mm.group("Base"), "Model": mm.group("Model"),
            "AbsDef_Baseline_pct": nums[0], "AbsDef_Model_pct": nums[1],
            "Delta_AbsDef_pct": nums[2],
            "Delta_CI_Low": nums[3], "Delta_CI_High": nums[4],
            "p_value_boot": nums[5], "q_value_BH": nums[6],
            "Significant_at_95": _to_bool(toks[7]),
        })
    pair_df = pd.DataFrame(pair_rows)

    print(f"[INFO] Statements parsed successfully: {len(ci_df)} CI entries, {len(pair_df)} paired significance entries.")
    return ci_df, pair_df


def qc_report(ci_df: pd.DataFrame, pair_df: pd.DataFrame, path: str):
    L = []
    add = L.append
    add("=" * 78)
    add(" Extreme Value Statistical Quality Control Report (QC Report)")
    add("=" * 78)

    # --- CI Verification ---
    add("\n[1] 95% CI Bounds Order / Point Estimate Enclosure")
    if ci_df.empty:
        add("    (No CI data provided)")
    else:
        bad_ord = ci_df[ci_df["CI_Low"] > ci_df["CI_High"]]
        bad_in = ci_df[(ci_df["Deficit_stmt_pct"] < ci_df[["CI_Low", "CI_High"]].min(axis=1) - 1e-9) |
                       (ci_df["Deficit_stmt_pct"] > ci_df[["CI_Low", "CI_High"]].max(axis=1) + 1e-9)]
        add(f"    Bound Inversion (CI_Low > CI_High)           : {len(bad_ord)} rows")
        for _, r in bad_ord.iterrows():
            add(f"      - {r.EVar:9s} {r.Percentile:<7} {r.Model:13s} "
                f"{r.Deficit_stmt_pct:.4f}  CI=({r.CI_Low:.4f}, {r.CI_High:.4f})")
        add(f"    Point estimate outside CI bounds             : {len(bad_in)} rows")
        for _, r in bad_in.iterrows():
            add(f"      - {r.EVar:9s} {r.Percentile:<7} {r.Model:13s} "
                f"{r.Deficit_stmt_pct:.4f}  CI=({r.CI_Low:.4f}, {r.CI_High:.4f})")
        add("    >> Root cause: Taking absolute values before calculating CIs stripped negative signs and inverted bounds.")
        add("    >> Solution: Recompute BCa CIs on signed deficits directly without prior absolute value conversion.")

        zero_ci = ci_df[(ci_df["CI_Low"].abs() < 1e-9) & (ci_df["CI_High"].abs() < 1e-9)]
        add(f"\n    CI rounded to 0.00-0.00 (uninformative)     : {len(zero_ci)} rows")
        if len(zero_ci):
            evs = sorted(zero_ci["EVar"].unique())
            add(f"      Affected variables: {evs}")
            add("      >> Large mean offset variables (PHI, T) compress relative percentage deficits below decimal precision.")
            add("      >> Solution: Report deficits and CIs in physical units (m^2 s^-2, K).")

    # --- Paired Test Consistency ---
    add("\n[2] Paired Significance Consistency Check")
    if pair_df.empty:
        add("    (No paired table data provided)")
    else:
        d = pair_df.copy()
        bad_flag = d[(d["Significant_at_95"]) & (d["p_value_boot"] > 0.05)]
        add(f"    Significant_at_95=True but p > 0.05          : {len(bad_flag)} rows")
        for _, r in bad_flag.iterrows():
            add(f"      - {r.EVar:9s} {r.Baseline:13s} vs {r.Model:13s} "
                f"p={r.p_value_boot:.4f}  q={r.q_value_BH:.4f}")
        d["ci_covers_zero"] = (d["Delta_CI_Low"] <= 0) & (d["Delta_CI_High"] >= 0)
        bad_zero = d[(d["Significant_at_95"]) & (d["ci_covers_zero"])]
        add(f"    Delta CI covers zero but marked significant  : {len(bad_zero)} rows")
        for _, r in bad_zero.iterrows():
            add(f"      - {r.EVar:9s} {r.Baseline:13s} vs {r.Model:13s} "
                f"CI=({r.Delta_CI_Low:.4g}, {r.Delta_CI_High:.4g})")

        pmin = d["p_value_boot"].min()
        if pmin > 0:
            b_implied = round(1.0 / pmin - 1.0)
            add(f"\n    Minimum p = {pmin:.6g} => Implied bootstrap replicates B ≈ {b_implied:.0f}")
            add("    >> Inconsistent bootstrap counts between CI (e.g. B=2000) and p-value (e.g. B=1000) should be unified.")

        d["pair_key"] = d.apply(
            lambda r: (r["EVar"], r["Percentile_Label"],
                       tuple(sorted([r["Baseline"], r["Model"]]))), axis=1)
        dup = d["pair_key"].duplicated(keep=False).sum()
        add(f"\n    Duplicate A-vs-B / B-vs-A comparison rows   : {dup} rows "
            f"(Unique comparisons: {d['pair_key'].nunique()})")
        add("    >> Deduplicate prior to Benjamini-Hochberg (BH) adjustment to avoid inflating family size.")

    add("\n[3] Recommendations")
    add("    (i)   Recompute all BCa CIs using signed deficits with B = 2000 replicates.")
    add("    (ii)  Define significance strictly by whether the paired difference CI excludes zero.")
    add("    (iii) Report tail deficits for PHI and T in physical units (m^2 s^-2, K).")
    add("    (iv)  Quantile aggregations lack area weighting (~13% grid-area variation across domain); note in Methods.")
    add("=" * 78)

    txt = "\n".join(L)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, path), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    print(f"  [OK] Saved QC report: {os.path.join(OUT_DIR, path)}")


# ------------------------------------------------------------------------------
# 2.7 Confidence Interval Attachment
# ------------------------------------------------------------------------------
def attach_ci(pt: pd.DataFrame, ci_df: pd.DataFrame) -> pd.DataFrame:
    pt = pt.copy()
    pt["CI_Low"] = np.nan
    pt["CI_High"] = np.nan
    pt["CI_Source"] = ""
    pt["CI_Usable"] = False

    # (a) External CI CSV Priority
    if EXTERNAL_CI_CSV and os.path.exists(EXTERNAL_CI_CSV):
        try:
            ex = pd.read_csv(EXTERNAL_CI_CSV)
            c = EXTERNAL_CI_COLS
            ex = ex.rename(columns={c["evar"]: "EVar", c["percentile"]: "Percentile",
                                    c["model"]: "Model", c["ci_low"]: "CI_Low_x",
                                    c["ci_high"]: "CI_High_x"})
            ex["Percentile"] = (ex["Percentile"].astype(str)
                                .str.replace(r"(st|nd|rd|th)$", "", regex=True)
                                .astype(float))
            pt = pt.merge(ex[["EVar", "Percentile", "Model", "CI_Low_x", "CI_High_x"]],
                          on=["EVar", "Percentile", "Model"], how="left")
            hit = pt["CI_Low_x"].notna()
            pt.loc[hit, "CI_Low"] = pt.loc[hit, "CI_Low_x"]
            pt.loc[hit, "CI_High"] = pt.loc[hit, "CI_High_x"]
            pt.loc[hit, "CI_Source"] = "external"
            pt.loc[hit, "CI_Usable"] = True
            pt = pt.drop(columns=["CI_Low_x", "CI_High_x"])
            print(f"[INFO] Attached {int(hit.sum())} CIs from external file: {EXTERNAL_CI_CSV}")
        except Exception as e:
            print(f"[WARN] Failed to read external CI file: {e}")

    # (b) Fallback to statements parsing (valid for relative deficits)
    if (not ci_df.empty):
        tmp = ci_df.rename(columns={"CI_Low": "CI_Low_s", "CI_High": "CI_High_s"})
        pt = pt.merge(tmp[["EVar", "Percentile", "Model", "CI_Low_s", "CI_High_s"]],
                      on=["EVar", "Percentile", "Model"], how="left")
        need = pt["CI_Low"].isna() & pt["CI_Low_s"].notna()
        pt.loc[need, "CI_Low"] = pt.loc[need, "CI_Low_s"]
        pt.loc[need, "CI_High"] = pt.loc[need, "CI_High_s"]
        pt.loc[need, "CI_Source"] = "statements(rounded)"
        # Validity condition: relative metric + un-inverted bounds + non-zero bounds
        ok = (need
              & (pt["Metric"] == "relative")
              & (pt["CI_Low"] <= pt["CI_High"])
              & ~((pt["CI_Low"].abs() < 1e-9) & (pt["CI_High"].abs() < 1e-9)))
        pt.loc[ok, "CI_Usable"] = True
        pt = pt.drop(columns=["CI_Low_s", "CI_High_s"])
        print(f"[INFO] Attached {int(need.sum())} CIs from statements ({int(ok.sum())} marked usable).")
    return pt


# ------------------------------------------------------------------------------
# 2.8 Table Export Routines
# ------------------------------------------------------------------------------
def fmt_def(v: float, ev: str) -> str:
    if not np.isfinite(v):
        return "--"
    if ev in PHYS_DEFICIT_UNITS:
        return f"{v:.{PHYS_DECIMALS.get(ev, 3)}f}"
    return f"{v:.{PCT_DECIMALS}f}"


def fmt_cell(row) -> str:
    ev = row["EVar"]
    s = fmt_def(row["Deficit_reported"], ev)
    if ATTACH_CI and bool(row.get("CI_Usable", False)):
        lo, hi = row["CI_Low"], row["CI_High"]
        s += f" [{lo:.{PCT_DECIMALS}f}, {hi:.{PCT_DECIMALS}f}]"
    return s


def export_tables(pt: pd.DataFrame, evars: list, pcts: list,
                  csv_name: str, md_name: str | None,
                  long_name: str | None = None):
    sub = pt[pt["EVar"].isin(evars) & pt["Percentile"].isin(pcts)].copy()
    if sub.empty:
        print(f"[WARN] {csv_name}: No data available for export.")
        return

    sub["Cell"] = sub.apply(fmt_cell, axis=1)
    sub["absdef"] = sub["Deficit_reported"].abs()

    # Determine best model per row (minimum absolute deficit)
    best = (sub.groupby(["EVar", "Percentile"])["absdef"]
              .idxmin().dropna())
    best_model = sub.loc[best, ["EVar", "Percentile", "Model"]] \
                    .rename(columns={"Model": "Best"})

    wide = sub.pivot_table(index=["EVar", "Quantity", "Percentile",
                                  "Percentile_Label", "Metric", "Unit"],
                           columns="Model", values="Cell", aggfunc="first")
    wide = wide.reindex(columns=[m for m in MODEL_ORDER if m in wide.columns])
    wide = wide.reset_index().merge(best_model, on=["EVar", "Percentile"], how="left")

    # Custom sorting based on defined evars / pcts order
    wide["_o1"] = wide["EVar"].map({e: i for i, e in enumerate(evars)})
    wide["_o2"] = wide["Percentile"].map({p: i for i, p in enumerate(pcts)})
    wide = wide.sort_values(["_o1", "_o2"]).drop(columns=["_o1", "_o2"])

    note = []
    for _, r in wide.iterrows():
        n = []
        if r["Metric"] == "physical":
            n.append("physical-unit deficit")
        g = sub[(sub["EVar"] == r["EVar"]) & (sub["Percentile"] == r["Percentile"])]
        if not bool(g["In_grid"].all()):
            n.append("clamped to grid edge")
        if ATTACH_CI and not bool(g["CI_Usable"].any()):
            n.append("CI unavailable")
        note.append("; ".join(n))
    wide["Note"] = note

    os.makedirs(OUT_DIR, exist_ok=True)
    p_csv = os.path.join(OUT_DIR, csv_name)
    wide.to_csv(p_csv, index=False, encoding="utf-8-sig")
    print(f"  [OK] Exported wide table CSV: {p_csv}")

    if long_name:
        p_long = os.path.join(OUT_DIR, long_name)
        cols = ["EVar", "Quantity", "Percentile", "Percentile_Label", "Model",
                "Metric", "Unit", "Q_ERA5", "Q_model",
                "Deficit_pct", "Deficit_phys", "Deficit_reported",
                "CI_Low", "CI_High", "CI_Source", "CI_Usable", "In_grid"]
        sub[[c for c in cols if c in sub.columns]].to_csv(
            p_long, index=False, encoding="utf-8-sig")
        print(f"  [OK] Exported long table CSV: {p_long}")

    if md_name:
        lines = []
        hdr = ["Quantity", "Percentile", "Unit"] + \
              [m for m in MODEL_ORDER if m in wide.columns]
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
        for _, r in wide.iterrows():
            cells = [str(r["Quantity"]), str(r["Percentile_Label"]), str(r["Unit"])]
            for m in MODEL_ORDER:
                if m not in wide.columns:
                    continue
                v = "--" if pd.isna(r[m]) else str(r[m])
                if BOLD_BEST_IN_MD and (r.get("Best") == m) and v != "--":
                    v = f"**{v}**"
                cells.append(v)
            lines.append("| " + " | ".join(cells) + " |")
        p_md = os.path.join(OUT_DIR, md_name)
        with open(p_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  [OK] Exported Markdown table: {p_md}")


# ------------------------------------------------------------------------------
# 2.9 Console Summary Output
# ------------------------------------------------------------------------------
def print_summary(pt: pd.DataFrame):
    print("\n" + "=" * 78)
    print(" Tail Deficit Summary (For Main Text Verification)")
    print("=" * 78)
    for p in [99.9, 99.99]:
        print(f"\n--- {percentile_label(p/100.0)}th percentile ---")
        print(f"{'Quantity':<20s}{'Unit':<11s}"
              f"{'Cubic spline':>14s}{'AtmoSR-Base':>14s}{'AtmoSR':>12s}")
        for ev in PANEL_ORDER:
            g = pt[(pt["EVar"] == ev) & (pt["Percentile"] == p)]
            if g.empty:
                continue
            vals = []
            for m in MODEL_ORDER:
                r = g[g["Model"] == m]
                vals.append(fmt_def(float(r["Deficit_reported"].iloc[0]), ev)
                            if len(r) else "--")
            unit = TABLE_PHYS_UNIT_TEXT.get(ev, "%")
            print(f"{TABLE_NAMES.get(ev, ev):<20s}{unit:<11s}"
                  f"{vals[0]:>14s}{vals[1]:>14s}{vals[2]:>12s}")
    print("=" * 78 + "\n")


# ------------------------------------------------------------------------------
# 2.10 Main Execution Flow
# ------------------------------------------------------------------------------
def main():
    print("=" * 78)
    print(" Extreme Tail Diagnostics: Deficit Curves + Absolute Curves + Table 3")
    print("=" * 78)
    font = set_publication_style()
    print(f"[Style] Active font: {font}")

    df = load_curve_csv(CSV_PATH)
    evs = [e for e in PANEL_ORDER if e in set(df["EVar"].unique())]
    print(f"[Data ] Variables found: {evs}")
    print(f"[Data ] Series found: {sorted(df['Canon'].unique())}")
    print(f"[Data ] Exceedance probability range: {df['ExceedProb'].max():.5g} -> "
          f"{df['ExceedProb'].min():.5g}  ("
          f"{100*(1-df['ExceedProb'].max()):.4g}th -> "
          f"{100*(1-df['ExceedProb'].min()):.5g}th percentile)")

    dfd = build_deficit(df)

    print("\n[Fig 1] Top row = Deficit curves, Bottom row = Absolute quantile curves")
    make_figure(df, dfd, FIG1_MODE, BASENAME_FIG1)

    print("\n[Fig 2] All 8 panels = Deficit curves")
    make_figure(df, dfd, FIG2_MODE, BASENAME_FIG2)

    ci_df, pair_df = parse_statements(STATEMENTS_PATH) if ATTACH_CI \
        else (pd.DataFrame(), pd.DataFrame())

    print("\n[Table] Constructing point quantile estimate tables")
    pt3 = build_point_table(dfd, TABLE3_QUANTITIES, TABLE3_PERCENTILES)
    pt7 = build_point_table(dfd, TABLES7_QUANTITIES, TABLES7_PERCENTILES)
    pt3 = attach_ci(pt3, ci_df)
    pt7 = attach_ci(pt7, ci_df)

    print("\n[Table 3] Main text table export")
    export_tables(pt3, TABLE3_QUANTITIES, TABLE3_PERCENTILES,
                  TABLE3_CSV, TABLE3_MD)

    print("\n[Table S7] SI full table export")
    export_tables(pt7, TABLES7_QUANTITIES, TABLES7_PERCENTILES,
                  TABLES7_WIDE, None, long_name=TABLES7_LONG)

    print_summary(pt7)

    if (not ci_df.empty) or (not pair_df.empty):
        print("[QC] Running statistical quality control check...")
        qc_report(ci_df, pair_df, QC_REPORT)

    print("[DONE] Execution completed successfully.\n")


if __name__ == "__main__":
    main()