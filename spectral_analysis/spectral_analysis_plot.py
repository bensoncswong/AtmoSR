#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
=============================================================================
AtmoSR Figure: One-dimensional spatial spectra and spectral coherence
=============================================================================

Layout: 2 x 4 panels
  - Top row    (a)-(d): 1D spatial spectral density for Phi, q, T, and KE.
  - Bottom row (e)-(h): Spectral coherence magnitude against ERA5 for Phi, q, T,
                        and horizontal wind vector (u, v combined).

-----------------------------------------------------------------------------
WHY THIS SCRIPT CONVERTS THE SPECTRA
-----------------------------------------------------------------------------
The upstream CSV producer stores a radially AVERAGED two-dimensional
periodogram:

    Ebar(k_b) = (1 / N_b) * sum_{k in bin b} |F(k)|^2 * fft_scale        (A)

which represents "variance per Fourier mode". It is NOT a spectral density in k.
For two-dimensional white noise, (A) is flat, whereas the two-dimensional
spectral density E(k_h) that satisfies

    integral E(k_h) dk_h = spatial variance                              (B)

rises linearly with k. The two therefore differ by exactly one power of k,
so plotting (A) directly alongside canonical k^-3 / k^-5/3 reference lines is
inconsistent.

Durran, Weyn & Menchaca (2017, Mon. Wea. Rev., 145, 3901-3910) provide the
correct formulation. This script restores the true 1D spectral density using:

  - "band_integrate"  (Parseval-exact):
        E(k_b) = N_b * Ebar(k_b) / dk
  - "analytic_2pik"   (Durran et al. 2017, default):
        Removes lattice-count noise by replacing the discrete mode count N_b
        with the ideal annulus population 2*pi*k_b*dk / (dfx*dfy):
        E(k_b) = 2*pi*k_b * Ebar(k_b) / (dfx * dfy)

Both conversions are proportional to k * Ebar, steepening every spectral slope
by exactly one power of k and making -3 and -5/3 reference slopes directly
comparable. Durran et al. (2017, Section 5) further demonstrate that for an
isotropic field, 1D and 2D spectral densities share the SAME power-law exponent.

Because identical factors scale both the reconstructed fields and ERA5, all ratio
diagnostics and spectral coherence values remain numerically UNCHANGED.
=============================================================================
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter


# =============================================================================
# 0) USER HYPERPARAMETERS
# =============================================================================

# -----------------------------------------------------------------------------
# 0.1 Input / Output paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./spectral_outputs"
OUTPUT_DIR = "./spectral_figures"

KE_CSV = "ke_spectrum.csv"
SCALAR_CSV = "scalar_psd_spectrum.csv"
COH_CSV = "spectral_coherence.csv"
META_JSON = "spectral_run_metadata.json"

COMBINED_FIG_NAME = "Fig5_spectra1D_coherence_ZQT_WIND_2x4"

OUT_SPECTRA_CSV = "figure5_spectra_1d_curves.csv"
OUT_COH_CSV = "figure5_coherence_curves.csv"
OUT_COH_CUT_CSV = "figure5_coherence_cutoffs.csv"        # Table 2
OUT_RATIO_CUT_CSV = "figure5_spectral_ratio_cutoffs.csv"  # Section 3.5.1
OUT_SUMMARY_CSV = "figure5_summary_metrics.csv"

# -----------------------------------------------------------------------------
# 0.2 2D -> 1D Spectral density conversion mode
# -----------------------------------------------------------------------------
# "analytic_2pik"  : E = 2*pi*k * Ebar / (dfx*dfy)      (Durran et al. 2017)
# "band_integrate" : E = n_modes * Ebar / dk            (Parseval-exact)
# "none"           : Keep raw annulus mean from upstream producer
ONE_D_CONVERSION = "analytic_2pik"

# Fallback grid metadata (used only when META_JSON is absent or incomplete)
# Default values correspond to a 100 x 100 grid at ~0.25 deg spacing
FALLBACK_GRID = {
    "dx_m": 24685.7,        # Zonal grid spacing at domain mean latitude
    "dy_m": 27798.9,        # Meridional grid spacing
    "H": 100,               # Number of latitude grid points (HR)
    "W": 100,               # Number of longitude grid points (HR)
    "dk_cpkm": None,        # None -> min(dfx, dfy) * BIN_WIDTH_FACTOR
    "nyq_cpkm": None,       # None -> min(0.5/dx, 0.5/dy)
    "kmax_eff_cpkm": None,  # None -> use nyq_cpkm
}
BIN_WIDTH_FACTOR_ASSUMED = 1.0  # Must match producer BIN_WIDTH_FACTOR

# -----------------------------------------------------------------------------
# 0.3 X-axis configuration
# -----------------------------------------------------------------------------
X_AXIS_MODE = "k"               # "k" (cycles/km) or "wavelength" (km)

PLOT_KMIN_CPKM = None          # None -> use data minimum
PLOT_KMAX_CPKM = None          # None -> depend on CLIP_KMAX_TO_NYQUIST
CLIP_KMAX_TO_NYQUIST = True    # Cap plotted range at min(0.5/dx, 0.5/dy)
XLIM_PAD_DEX = 0.015           # Small logarithmic margin on bounds

SHOW_WAVELENGTH_AXIS = True    # Secondary wavelength axis on top row
WAVELENGTH_TICKS_KM = [2000, 1000, 500, 200, 100, 60]
WAVELENGTH_AXIS_LABEL = "Wavelength [km]"
WAVELENGTH_LABEL_ON_ALL_TOP_PANELS = True

SHOW_NYQUIST_LINE = False      # Vertical marker at target Nyquist wavenumber
NYQUIST_LINE_KW = dict(color="#888888", linestyle=(0, (6, 4)), linewidth=1.2)

# -----------------------------------------------------------------------------
# 0.4 Figure geometry and export settings
# -----------------------------------------------------------------------------
COMBINED_FIGSIZE = (24.0, 11.6)
DPI = 600
SAVE_FORMATS = ("png", "pdf")
FIG_FACE_COLOR = "white"

LAYOUT_W_PAD = 3.0 / 72
LAYOUT_H_PAD = 3.0 / 72
LAYOUT_WSPACE = 0.045
LAYOUT_HSPACE = 0.055

# -----------------------------------------------------------------------------
# 0.5 Method ordering, labels, colors, and line styles
# -----------------------------------------------------------------------------
METHOD_ORDER = ["cubic", "noPhy", "AllPhy"]
METHOD_LABEL = {
    "cubic": "Cubic Spline",
    "noPhy": "AtmoSR-Base",
    "AllPhy": "AtmoSR",
}
METHOD_COLOR = {
    "cubic": "#7f7f7f",
    "noPhy": "#1f77b4",
    "AllPhy": "#d62728",
}
METHOD_LINESTYLE = {
    "cubic": "--",
    "noPhy": "-",
    "AllPhy": "-",
}

TRUTH_LABEL = "Ground Truth (ERA5)"
TRUTH_COLOR = "k"

# -----------------------------------------------------------------------------
# 0.6 Panel variables and titles
# -----------------------------------------------------------------------------
ABS_ORDER = ["Z", "Q", "T", "KE"]        # Top row
COH_ORDER = ["Z", "Q", "T", "WIND"]      # Bottom row
COH_WIND_KEY = "WIND"                    # Internal key for u+v composite wind

ABS_TITLES = {
    "Z": "(a) \u03A6 Spectrum",
    "Q": "(b) q Spectrum",
    "T": "(c) T Spectrum",
    "KE": "(d) KE Spectrum (u, v)",
}
COH_TITLES = {
    "Z": "(e) \u03A6 Coherence",
    "Q": "(f) q Coherence",
    "T": "(g) T Coherence",
    "WIND": "(h) Wind Coherence (u, v)",
}

ABS_YLABELS = {
    "Z": r"$E_\Phi(k)$ [{u}]",
    "Q": r"$E_q(k)$ [{u}]",
    "T": r"$E_T(k)$ [{u}]",
    "KE": r"$E(k)$ [{u}]",
}
COH_YLABEL = "Coherence magnitude"

# -----------------------------------------------------------------------------
# 0.7 Spectral units
# -----------------------------------------------------------------------------
# "compact"  -> m$^4$ s$^{-4}$ km
# "explicit" -> m$^4$ s$^{-4}$ (cycles km$^{-1}$)$^{-1}$
UNIT_STYLE = "compact"

UNITS_COMPACT = {
    "Z": r"m$^{4}$ s$^{-4}$ km",
    "Q": r"(kg kg$^{-1}$)$^{2}$ km",
    "T": r"K$^{2}$ km",
    "KE": r"m$^{2}$ s$^{-2}$ km",
}
UNITS_EXPLICIT = {
    "Z": r"m$^{4}$ s$^{-4}$ (cycles km$^{-1}$)$^{-1}$",
    "Q": r"(kg kg$^{-1}$)$^{2}$ (cycles km$^{-1}$)$^{-1}$",
    "T": r"K$^{2}$ (cycles km$^{-1}$)$^{-1}$",
    "KE": r"m$^{2}$ s$^{-2}$ (cycles km$^{-1}$)$^{-1}$",
}
UNITS_ANNULUS_MEAN = {  # Used when ONE_D_CONVERSION == "none"
    "Z": r"m$^{4}$ s$^{-4}$ mode$^{-1}$",
    "Q": r"(kg kg$^{-1}$)$^{2}$ mode$^{-1}$",
    "T": r"K$^{2}$ mode$^{-1}$",
    "KE": r"m$^{2}$ s$^{-2}$ mode$^{-1}$",
}

# -----------------------------------------------------------------------------
# 0.8 Reference power-law slopes (KE panel)
# -----------------------------------------------------------------------------
SHOW_KE_REFERENCE_SLOPES = False
KE_REFERENCE_SLOPES = [
    {
        "slope": -3.0,
        "label": r"$k^{-3}$",
        "anchor_k_cpkm": 0.0016,
        "span_dex": 0.40,
        "offset_factor": 0.16,
    },
    {
        "slope": -5.0 / 3.0,
        "label": r"$k^{-5/3}$",
        "anchor_k_cpkm": 0.0085,
        "span_dex": 0.26,
        "offset_factor": 0.16,
    },
]
REF_SLOPE_KW = dict(color="#444444", linestyle="-.", linewidth=2.0)
REF_SLOPE_LABEL_SIZE = 18
REF_SLOPE_LABEL_DX_DEX = 0.03

# -----------------------------------------------------------------------------
# 0.9 Coherence panel decorations
# -----------------------------------------------------------------------------
COHERENCE_THRESHOLD = 0.5
SHOW_COH_THRESHOLD_LINE = True
COH_THRESHOLD_LINE_KW = dict(color="#555555", linestyle="--", linewidth=1.6)
ANNOTATE_COH_THRESHOLD = True
COH_THRESHOLD_TEXT = ""
COH_THRESHOLD_TEXT_SIZE = 17

SHOW_COH_UNITY_LINE = True
COH_UNITY_LINE_KW = dict(color="#aaaaaa", linestyle=":", linewidth=1.0)

SHOW_COH_CUTOFF_MARKERS = False
COH_CUTOFF_MARKER_KW = dict(
    marker="o",
    markersize=9,
    markeredgecolor="white",
    markeredgewidth=1.4,
)

COH_YLIM = (0.0, 1.02)

# -----------------------------------------------------------------------------
# 0.10 Quantitative cutoff options
# -----------------------------------------------------------------------------
COH_CUTOFF_LEVELS = [0.5, 0.7, 0.9]
COH_CROSSING_MODE = "last"       # "first" | "last"

RATIO_CUTOFF_LEVELS = [0.9, 0.5]
RATIO_CROSSING_MODE = "first"    # Conservative departure scale

# -----------------------------------------------------------------------------
# 0.11 Curve smoothing options
# -----------------------------------------------------------------------------
ENABLE_SMOOTH = False
SMOOTH_WINDOW = 9
SMOOTH_POLYORDER = 2

# -----------------------------------------------------------------------------
# 0.12 Numerical bounds and tolerances
# -----------------------------------------------------------------------------
MIN_POSITIVE = 1e-300
COHERENCE_EPS = 1e-20
RATIO_EPS = 1e-300

# -----------------------------------------------------------------------------
# 0.13 Typography and visual elements
# -----------------------------------------------------------------------------
FONT_FAMILY = "DejaVu Sans"
TITLE_SIZE = 23
LABEL_SIZE = 20
TICK_SIZE = 20
LEGEND_SIZE = 18
SECONDARY_TICK_SIZE = 17

LINE_WIDTH_TRUTH = 2.8
LINE_WIDTH_METHOD = 2.4

TITLE_PAD_TOP_ROW = 54        # Clears secondary wavelength axis
TITLE_PAD_BOTTOM_ROW = 10

LEGEND_FRAMEON = False
LEGEND_LOC_ABS = "lower left"
LEGEND_LOC_COH = "lower left"
LEGEND_NCOL = 1
LEGEND_TOP_PANEL_INDEX = 0
LEGEND_BOTTOM_PANEL_INDEX = 0


# =============================================================================
# 1) UTILITY FUNCTIONS
# =============================================================================

def ensure_dir(path):
    """Ensure that the target output directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_figure(fig, out_dir, name):
    """Save matplotlib figure in specified formats."""
    out_dir = Path(out_dir)
    for ext in SAVE_FORMATS:
        fig.savefig(
            out_dir / f"{name}.{ext}",
            dpi=DPI,
            bbox_inches="tight",
            facecolor=FIG_FACE_COLOR,
        )
    plt.close(fig)


def safe_inverse(v):
    """Compute 1/v while safely suppressing division-by-zero warnings."""
    v = np.asarray(v, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 / v
    return out


def pick_methods_multi(*dfs):
    """Extract and order unique method names across input DataFrames."""
    present = []
    for df in dfs:
        if df is not None and len(df) and "method" in df.columns:
            present.extend(list(pd.unique(df["method"])))
    present = list(dict.fromkeys(present))
    ordered = [m for m in METHOD_ORDER if m in present]
    ordered += [m for m in present if m not in ordered]
    return ordered


def method_label(m):
    return METHOD_LABEL.get(m, m)


def method_color(m, fallback):
    return METHOD_COLOR.get(m, fallback)


def method_ls(m):
    return METHOD_LINESTYLE.get(m, "-")


def get_xcol_xlabel():
    if X_AXIS_MODE == "k":
        return "k_cyc_per_km", r"Wavenumber $k$ [cycles km$^{-1}$]", False
    if X_AXIS_MODE == "wavelength":
        return "wavelength_km", "Wavelength [km]", True
    raise ValueError(f"Unknown X_AXIS_MODE: {X_AXIS_MODE}")


def unit_string(var):
    if ONE_D_CONVERSION == "none":
        return UNITS_ANNULUS_MEAN[var]
    if UNIT_STYLE == "compact":
        return UNITS_COMPACT[var]
    if UNIT_STYLE == "explicit":
        return UNITS_EXPLICIT[var]
    raise ValueError(f"Unknown UNIT_STYLE: {UNIT_STYLE}")


def finite_positive_mask(x, y):
    return np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > MIN_POSITIVE)


def finite_nonneg_mask(x, y):
    return np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y >= 0.0)


def maybe_smooth(y):
    """Optionally smooth spectral curve using Savitzky-Golay filter in log space."""
    yy = np.asarray(y, dtype=float)
    if not ENABLE_SMOOTH:
        return yy
    good = np.isfinite(yy) & (yy > MIN_POSITIVE)
    n = int(np.sum(good))
    if n < max(SMOOTH_WINDOW, SMOOTH_POLYORDER + 3):
        return yy
    win = int(SMOOTH_WINDOW)
    if win % 2 == 0:
        win += 1
    if win > n:
        win = n if n % 2 == 1 else n - 1
    if win < SMOOTH_POLYORDER + 2:
        return yy
    ylog = np.log10(np.clip(yy[good], MIN_POSITIVE, None))
    ylog_s = savgol_filter(
        ylog,
        window_length=win,
        polyorder=SMOOTH_POLYORDER,
        mode="interp",
    )
    out = yy.copy()
    out[good] = 10**ylog_s
    return out


def loglog_interp(x, y, x0):
    """Interpolate y at x0 linearly in (log10 x, log10 y) space."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = finite_positive_mask(x, y)
    if np.sum(m) < 2:
        return np.nan
    xs, ys = x[m], y[m]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    lx0 = np.log10(np.clip(x0, MIN_POSITIVE, None))
    ly = np.interp(lx0, np.log10(xs), np.log10(ys))
    return float(10**ly)


# =============================================================================
# 2) GRID METADATA AND CONVERSION FACTORS
# =============================================================================

def resolve_grid_info(meta):
    """
    Resolve grid dimensions and wavenumber parameters in cycles km^-1.
    """
    g = dict(FALLBACK_GRID)

    if meta:
        gm = meta.get("grid", {}) or {}
        rb = meta.get("radial_bins", {}) or {}

        if gm.get("dx") is not None:
            g["dx_m"] = float(gm["dx"])
        if gm.get("dy") is not None:
            g["dy_m"] = float(gm["dy"])
        if gm.get("H_hr") is not None:
            g["H"] = int(gm["H_hr"])
        if gm.get("W_hr") is not None:
            g["W"] = int(gm["W_hr"])
        if gm.get("nyquist_cpm") is not None:
            g["nyq_cpkm"] = float(gm["nyquist_cpm"]) * 1000.0
        if rb.get("dk_cyc_per_m") is not None:
            g["dk_cpkm"] = float(rb["dk_cyc_per_m"]) * 1000.0
        if rb.get("kmax_cpm_effective") is not None:
            g["kmax_eff_cpkm"] = float(rb["kmax_cpm_effective"]) * 1000.0

        if gm.get("use_physical_wavenumber") is False:
            warnings.warn(
                "Metadata indicates use_physical_wavenumber=False. "
                "Wavenumbers are in grid units; physical scale labels may be incorrect."
            )

    # Fundamental frequency increments (cycles / km)
    g["dfx_cpkm"] = 1000.0 / (float(g["W"]) * float(g["dx_m"]))
    g["dfy_cpkm"] = 1000.0 / (float(g["H"]) * float(g["dy_m"]))

    if g.get("dk_cpkm") is None:
        g["dk_cpkm"] = min(g["dfx_cpkm"], g["dfy_cpkm"]) * float(
            BIN_WIDTH_FACTOR_ASSUMED
        )
    if g.get("nyq_cpkm") is None:
        g["nyq_cpkm"] = min(500.0 / float(g["dx_m"]), 500.0 / float(g["dy_m"]))
    if g.get("kmax_eff_cpkm") is None:
        g["kmax_eff_cpkm"] = g["nyq_cpkm"]

    g["nyq_wavelength_km"] = 1.0 / g["nyq_cpkm"]
    g["longest_wavelength_km"] = 1.0 / min(g["dfx_cpkm"], g["dfy_cpkm"])
    g["domain_x_km"] = float(g["W"]) * float(g["dx_m"]) / 1000.0
    g["domain_y_km"] = float(g["H"]) * float(g["dy_m"]) / 1000.0
    return g


def one_d_factor(n_modes, k_cpkm, grid, conversion):
    """
    Multiplicative factor converting radially averaged 2D periodogram into
    1D spectral density (cycles km^-1).
    """
    n_modes = np.asarray(n_modes, dtype=float)
    k_cpkm = np.asarray(k_cpkm, dtype=float)

    if conversion == "none":
        return np.ones_like(k_cpkm)
    if conversion == "band_integrate":
        return n_modes / float(grid["dk_cpkm"])
    if conversion == "analytic_2pik":
        return (2.0 * np.pi * k_cpkm) / (
            float(grid["dfx_cpkm"]) * float(grid["dfy_cpkm"])
        )
    raise ValueError(f"Unknown ONE_D_CONVERSION: {conversion}")


def add_one_d_columns(df, value_cols, grid, conversion):
    out = df.copy()
    if len(out) == 0:
        for c in value_cols:
            out[c + "_1d"] = np.nan
        out["one_d_factor"] = np.nan
        return out
    fac = one_d_factor(
        out["n_modes"].to_numpy(float),
        out["k_cyc_per_km"].to_numpy(float),
        grid,
        conversion,
    )
    out["one_d_factor"] = fac
    for c in value_cols:
        out[c + "_1d"] = out[c].to_numpy(float) * fac
    return out


# =============================================================================
# 3) DATA PREPARATION
# =============================================================================

def ensure_ratio_columns(ke_df, scalar_df):
    ke, sc = ke_df.copy(), scalar_df.copy()
    if {"ke_pred_1d", "ke_true_1d"}.issubset(ke.columns):
        ke["ke_ratio"] = ke["ke_pred_1d"] / np.clip(
            ke["ke_true_1d"], RATIO_EPS, None
        )
    if {"psd_pred_1d", "psd_true_1d"}.issubset(sc.columns):
        sc["psd_ratio"] = sc["psd_pred_1d"] / np.clip(
            sc["psd_true_1d"], RATIO_EPS, None
        )
    return ke, sc


def weighted_truth_curve(df, true_col, xcol):
    """
    Compute sample-weighted ERA5 reference curve across available methods.
    """
    cols = ["k_bin", "k_cyc_per_km", "wavelength_km", true_col, "n_samples"]
    tmp = df[cols].copy()
    tmp["w"] = np.clip(tmp["n_samples"].astype(float), 1.0, None)
    tmp["vw"] = tmp[true_col].astype(float) * tmp["w"]
    out = (
        tmp.groupby("k_bin", as_index=False).agg(
            k_cyc_per_km=("k_cyc_per_km", "mean"),
            wavelength_km=("wavelength_km", "mean"),
            vw=("vw", "sum"),
            w=("w", "sum"),
        )
    )
    out[true_col] = out["vw"] / np.clip(out["w"], 1e-12, None)
    return out.sort_values("k_bin").reset_index(drop=True)


def build_wind_coherence_from_uv(coh_df, eps=COHERENCE_EPS):
    """
    Form vector-wind coherence by summing u and v cross- and auto-spectra over
    annuli before taking the ratio.
    """
    need = {
        "method", "variable", "n_samples", "k_bin", "k_cyc_per_m",
        "k_cyc_per_km", "k_rad_per_m", "wavelength_km", "n_modes",
        "cross_real", "cross_imag", "auto_pred", "auto_true",
    }
    miss = need - set(coh_df.columns)
    if miss:
        raise ValueError(f"Coherence CSV is missing columns: {sorted(miss)}")

    uv = coh_df[coh_df["variable"].isin(["U", "V"])].copy()
    if len(uv) == 0:
        return coh_df.iloc[0:0].copy()

    g = (
        uv.groupby(["method", "k_bin"], as_index=False).agg(
            k_cyc_per_m=("k_cyc_per_m", "mean"),
            k_cyc_per_km=("k_cyc_per_km", "mean"),
            k_rad_per_m=("k_rad_per_m", "mean"),
            wavelength_km=("wavelength_km", "mean"),
            n_modes=("n_modes", "max"),
            n_samples=("n_samples", "max"),
            cross_real=("cross_real", "sum"),
            cross_imag=("cross_imag", "sum"),
            auto_pred=("auto_pred", "sum"),
            auto_true=("auto_true", "sum"),
        )
    )

    cross = g["cross_real"].to_numpy(float) + 1j * g["cross_imag"].to_numpy(
        float
    )
    ap = np.maximum(g["auto_pred"].to_numpy(float), 0.0)
    at = np.maximum(g["auto_true"].to_numpy(float), 0.0)
    coh = np.clip(np.abs(cross) / (np.sqrt(ap * at) + eps), 0.0, 1.0)

    g["cross_abs"] = np.abs(cross)
    g["cross_phase_rad"] = np.angle(cross)
    g["coherence_mag"] = coh
    g["coherence_mag_squared"] = coh**2
    g["variable"] = COH_WIND_KEY

    return g[
        [
            "method", "variable", "n_samples", "k_bin",
            "k_cyc_per_m", "k_cyc_per_km", "k_rad_per_m", "wavelength_km",
            "n_modes", "cross_real", "cross_imag", "cross_abs",
            "cross_phase_rad", "auto_pred", "auto_true",
            "coherence_mag", "coherence_mag_squared",
        ]
    ]


def build_coherence_frames(coh_raw):
    """
    Prepare coherence dataframes for visualization and quantitative export.
    """
    keep = coh_raw[coh_raw["variable"].isin(["Z", "Q", "T", "U", "V"])].copy()
    wind = build_wind_coherence_from_uv(coh_raw)
    coh_all = pd.concat([keep, wind], axis=0, ignore_index=True)
    coh_all = coh_all.sort_values(["method", "variable", "k_bin"]).reset_index(
        drop=True
    )
    coh_plot = coh_all[coh_all["variable"].isin(COH_ORDER)].copy()
    return coh_plot, coh_all


def apply_k_filter(df, kmin, kmax):
    if len(df) == 0:
        return df
    out = df.copy()
    if kmin is not None:
        out = out[out["k_cyc_per_km"] >= float(kmin)]
    if kmax is not None:
        out = out[out["k_cyc_per_km"] <= float(kmax)]
    return out


# =============================================================================
# 4) THRESHOLD CROSSING ANALYSIS
# =============================================================================

def crossing_wavenumber(x, y, threshold, mode="first"):
    """
    Locate where curve y crosses threshold via linear log10(k) interpolation.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
    x, y = x[good], y[good]
    if x.size < 2:
        return np.nan, "insufficient"

    order = np.argsort(x)
    x, y = x[order], y[order]

    above = y >= float(threshold)
    if not above[0]:
        return np.nan, "below_at_kmin"

    idx = np.where(above[:-1] & (~above[1:]))[0]
    if idx.size == 0:
        return float(x[-1]), "no_crossing_within_range"

    i = idx[0] if mode == "first" else idx[-1]
    lx0, lx1 = np.log10(x[i]), np.log10(x[i + 1])
    y0, y1 = y[i], y[i + 1]
    t = 0.0 if (y1 == y0) else (float(threshold) - y0) / (y1 - y0)
    t = float(np.clip(t, 0.0, 1.0))
    return float(10 ** (lx0 + t * (lx1 - lx0))), "ok"


def coherence_cutoff_table(coh_all, grid, levels, mode):
    rows = []
    for (m, v), d in coh_all.groupby(["method", "variable"]):
        d = d.sort_values("k_cyc_per_km")
        kk = d["k_cyc_per_km"].to_numpy(float)
        cc = np.clip(d["coherence_mag"].to_numpy(float), 0.0, 1.0)
        ww = np.clip(d["n_modes"].to_numpy(float), 1.0, None)
        for lev in levels:
            k_pri, st_pri = crossing_wavenumber(kk, cc, lev, mode=mode)
            k_first, _ = crossing_wavenumber(kk, cc, lev, mode="first")
            k_last, _ = crossing_wavenumber(kk, cc, lev, mode="last")
            rows.append(
                {
                    "method": m,
                    "variable": v,
                    "threshold_on": "coherence_magnitude",
                    "threshold": float(lev),
                    "equivalent_squared_coherence": float(lev) ** 2,
                    "crossing_mode": mode,
                    "k_cutoff_cyc_per_km": k_pri,
                    "wavelength_cutoff_km": (
                        np.nan if not np.isfinite(k_pri) else 1.0 / k_pri
                    ),
                    "fraction_of_nyquist": (
                        np.nan
                        if not np.isfinite(k_pri)
                        else k_pri / grid["nyq_cpkm"]
                    ),
                    "k_cutoff_first_crossing": k_first,
                    "k_cutoff_last_crossing": k_last,
                    "status": st_pri,
                    "coherence_at_kmin": float(cc[0]) if cc.size else np.nan,
                    "coherence_at_kmax": float(cc[-1]) if cc.size else np.nan,
                    "mode_weighted_mean_coherence": (
                        float(np.average(cc, weights=ww)) if cc.size else np.nan
                    ),
                    "n_bins": int(len(d)),
                    "n_samples": int(d["n_samples"].max()) if len(d) else 0,
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["threshold", "variable", "method"]).reset_index(
            drop=True
        )
    return out


def ratio_cutoff_table(ke_df, scalar_df, grid, levels, mode):
    rows = []
    blocks = []
    if len(ke_df) and "ke_ratio" in ke_df.columns:
        blocks.append(("KE", ke_df.assign(variable="KE"), "ke_ratio"))
    if len(scalar_df) and "psd_ratio" in scalar_df.columns:
        for v, d in scalar_df.groupby("variable"):
            blocks.append((v, d, "psd_ratio"))

    for v, dv, rcol in blocks:
        for m, d in dv.groupby("method"):
            d = d.sort_values("k_cyc_per_km")
            kk = d["k_cyc_per_km"].to_numpy(float)
            rr = d[rcol].to_numpy(float)
            ww = np.clip(d["n_modes"].to_numpy(float), 1.0, None)
            for lev in levels:
                k_pri, st = crossing_wavenumber(kk, rr, lev, mode=mode)
                k_first, _ = crossing_wavenumber(kk, rr, lev, mode="first")
                k_last, _ = crossing_wavenumber(kk, rr, lev, mode="last")
                ok = np.isfinite(rr) & (rr > 0)
                lrr = np.log10(rr[ok]) if np.any(ok) else np.array([np.nan])
                rows.append(
                    {
                        "method": m,
                        "variable": v,
                        "threshold_on": "spectrum_pred_over_true",
                        "threshold": float(lev),
                        "crossing_mode": mode,
                        "k_departure_cyc_per_km": k_pri,
                        "wavelength_departure_km": (
                            np.nan if not np.isfinite(k_pri) else 1.0 / k_pri
                        ),
                        "fraction_of_nyquist": (
                            np.nan
                            if not np.isfinite(k_pri)
                            else k_pri / grid["nyq_cpkm"]
                        ),
                        "k_first_crossing": k_first,
                        "k_last_crossing": k_last,
                        "status": st,
                        "ratio_at_kmax": float(rr[-1]) if rr.size else np.nan,
                        "log10_ratio_rmse_mode_weighted": (
                            float(
                                np.sqrt(
                                    np.average(lrr**2, weights=ww[ok])
                                )
                            )
                            if np.any(ok)
                            else np.nan
                        ),
                        "n_bins": int(len(d)),
                    }
                )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["threshold", "variable", "method"]).reset_index(
            drop=True
        )
    return out


def summary_metrics(ke_df, scalar_df, coh_all):
    rows = []
    if len(ke_df) and "ke_ratio" in ke_df.columns:
        for m, d in ke_df.groupby("method"):
            rr = d["ke_ratio"].to_numpy(float)
            ww = np.clip(d["n_modes"].to_numpy(float), 1.0, None)
            k = np.isfinite(rr) & (rr > 0)
            if np.any(k):
                rows.append(
                    {
                        "category": "KE",
                        "method": m,
                        "variable": "UV",
                        "metric": "log10_ratio_RMSE",
                        "value": float(
                            np.sqrt(
                                np.average(
                                    np.log10(rr[k]) ** 2, weights=ww[k]
                                )
                            )
                        ),
                    }
                )
    if len(scalar_df) and "psd_ratio" in scalar_df.columns:
        for (m, v), d in scalar_df.groupby(["method", "variable"]):
            rr = d["psd_ratio"].to_numpy(float)
            ww = np.clip(d["n_modes"].to_numpy(float), 1.0, None)
            k = np.isfinite(rr) & (rr > 0)
            if np.any(k):
                rows.append(
                    {
                        "category": "ScalarPSD",
                        "method": m,
                        "variable": v,
                        "metric": "log10_ratio_RMSE",
                        "value": float(
                            np.sqrt(
                                np.average(
                                    np.log10(rr[k]) ** 2, weights=ww[k]
                                )
                            )
                        ),
                    }
                )
    for (m, v), d in coh_all.groupby(["method", "variable"]):
        cc = np.clip(d["coherence_mag"].to_numpy(float), 0.0, 1.0)
        ww = np.clip(d["n_modes"].to_numpy(float), 1.0, None)
        if cc.size:
            rows.append(
                {
                    "category": "Coherence",
                    "method": m,
                    "variable": v,
                    "metric": "mode_weighted_mean_magnitude",
                    "value": float(np.average(cc, weights=ww)),
                }
            )
            rows.append(
                {
                    "category": "Coherence",
                    "method": m,
                    "variable": v,
                    "metric": "mode_weighted_mean_squared",
                    "value": float(np.average(cc**2, weights=ww)),
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# 5) AXIS STYLING HELPERS
# =============================================================================

def style_axis(ax, x_log=True, y_log=False):
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", alpha=0.35, linewidth=0.75)
    ax.grid(True, which="minor", linestyle=":", alpha=0.22, linewidth=0.55)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_SIZE,
        width=0.95,
        length=5,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        labelsize=max(TICK_SIZE - 2, 8),
        width=0.75,
        length=3,
    )
    for s in ax.spines.values():
        s.set_linewidth(1.1)


def resolve_xlim(kmin_data, kmax_data):
    left = PLOT_KMIN_CPKM if PLOT_KMIN_CPKM is not None else kmin_data
    right = PLOT_KMAX_CPKM if PLOT_KMAX_CPKM is not None else kmax_data
    if XLIM_PAD_DEX and XLIM_PAD_DEX > 0:
        left = 10 ** (np.log10(left) - XLIM_PAD_DEX)
        right = 10 ** (np.log10(right) + XLIM_PAD_DEX)
    return float(left), float(right)


def add_wavelength_axis(ax, xlim, show_label):
    if not SHOW_WAVELENGTH_AXIS or X_AXIS_MODE != "k":
        return None
    secax = ax.secondary_xaxis("top", functions=(safe_inverse, safe_inverse))
    lo, hi = xlim
    ticks = [w for w in WAVELENGTH_TICKS_KM if lo <= (1.0 / w) <= hi]
    if ticks:
        secax.set_xticks(ticks)
        secax.set_xticklabels([f"{w:g}" for w in ticks])
    secax.xaxis.set_minor_locator(mticker.NullLocator())
    secax.tick_params(
        axis="x", labelsize=SECONDARY_TICK_SIZE, width=0.95, length=5
    )
    if show_label:
        secax.set_xlabel(WAVELENGTH_AXIS_LABEL, fontsize=LABEL_SIZE, labelpad=6)
    return secax


def add_legend(ax, loc):
    leg = ax.legend(
        loc=loc,
        fontsize=LEGEND_SIZE,
        frameon=LEGEND_FRAMEON,
        ncol=LEGEND_NCOL,
        handlelength=2.4,
        borderaxespad=0.6,
        labelspacing=0.35,
    )
    if LEGEND_FRAMEON:
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("#cccccc")
        leg.get_frame().set_alpha(0.95)
    return leg


def draw_reference_slope(ax, x_ref_curve, y_ref_curve, spec):
    k0 = float(spec["anchor_k_cpkm"])
    y0 = loglog_interp(x_ref_curve, y_ref_curve, k0)
    if not np.isfinite(y0) or y0 <= 0:
        return
    y0 = y0 * float(spec["offset_factor"])
    span = float(spec["span_dex"])
    xs = np.array([10 ** (np.log10(k0) - span), 10 ** (np.log10(k0) + span)])
    ys = y0 * (xs / k0) ** float(spec["slope"])
    ax.plot(xs, ys, zorder=3, **REF_SLOPE_KW)
    xlab = 10 ** (np.log10(xs[1]) + REF_SLOPE_LABEL_DX_DEX)
    ylab = y0 * (xlab / k0) ** float(spec["slope"])
    ax.text(
        xlab,
        ylab,
        spec["label"],
        fontsize=REF_SLOPE_LABEL_SIZE,
        color=REF_SLOPE_KW.get("color", "#444444"),
        ha="left",
        va="center",
        zorder=6,
    )


# =============================================================================
# 6) PANEL RENDERERS
# =============================================================================

def draw_spectrum_panel(
    ax,
    df,
    methods,
    pred_col,
    true_col,
    y_label,
    title,
    xcol,
    xlabel,
    invert_x,
    palette,
    xlim,
    grid,
    show_legend=False,
    show_ref_slopes=False,
    show_wl_axis=False,
    show_wl_label=False,
):
    d0 = df.copy().sort_values("k_bin")

    if len(d0) == 0:
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=LABEL_SIZE,
        )
        ax.set_title(
            title, fontsize=TITLE_SIZE, weight="bold", pad=TITLE_PAD_TOP_ROW
        )
        ax.set_ylabel(y_label, fontsize=LABEL_SIZE)
        return

    truth = weighted_truth_curve(d0, true_col, xcol)
    x_t = truth[xcol].to_numpy(float)
    y_t = maybe_smooth(truth[true_col].to_numpy(float))
    m_t = finite_positive_mask(x_t, y_t)
    if np.any(m_t):
        ax.plot(
            x_t[m_t],
            y_t[m_t],
            color=TRUTH_COLOR,
            linewidth=LINE_WIDTH_TRUTH,
            label=TRUTH_LABEL,
            zorder=5,
        )

    for i, method in enumerate(methods):
        d = d0[d0["method"] == method].sort_values("k_bin")
        if len(d) == 0:
            continue
        x = d[xcol].to_numpy(float)
        y = maybe_smooth(d[pred_col].to_numpy(float))
        m = finite_positive_mask(x, y)
        if np.any(m):
            ax.plot(
                x[m],
                y[m],
                color=method_color(method, palette[i % len(palette)]),
                linestyle=method_ls(method),
                linewidth=LINE_WIDTH_METHOD,
                label=method_label(method),
                zorder=4,
            )

    style_axis(ax, x_log=True, y_log=True)
    ax.set_xlim(*xlim)

    if show_ref_slopes and SHOW_KE_REFERENCE_SLOPES and np.any(m_t):
        for spec in KE_REFERENCE_SLOPES:
            draw_reference_slope(ax, x_t[m_t], y_t[m_t], spec)

    if SHOW_NYQUIST_LINE:
        ax.axvline(grid["nyq_cpkm"], zorder=2, **NYQUIST_LINE_KW)

    ax.set_title(
        title,
        fontsize=TITLE_SIZE,
        weight="bold",
        pad=TITLE_PAD_TOP_ROW if show_wl_axis else TITLE_PAD_BOTTOM_ROW,
    )
    ax.set_ylabel(y_label, fontsize=LABEL_SIZE)
    ax.set_xlabel("")

    if invert_x:
        ax.invert_xaxis()
    if show_wl_axis:
        add_wavelength_axis(ax, xlim, show_wl_label)
    if show_legend:
        add_legend(ax, LEGEND_LOC_ABS)


def draw_coherence_panel(
    ax,
    df,
    methods,
    title,
    xcol,
    xlabel,
    invert_x,
    palette,
    xlim,
    grid,
    cutoff_lookup=None,
    show_legend=False,
    annotate_threshold=False,
):
    d0 = df.copy().sort_values("k_bin")

    if len(d0) == 0:
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=LABEL_SIZE,
        )
        ax.set_title(
            title, fontsize=TITLE_SIZE, weight="bold", pad=TITLE_PAD_BOTTOM_ROW
        )
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
        ax.set_ylabel(COH_YLABEL, fontsize=LABEL_SIZE)
        return

    for i, method in enumerate(methods):
        d = d0[d0["method"] == method].sort_values("k_bin")
        if len(d) == 0:
            continue
        x = d[xcol].to_numpy(float)
        y = np.clip(d["coherence_mag"].to_numpy(float), 0.0, 1.0)
        m = finite_nonneg_mask(x, y)
        if not np.any(m):
            continue
        col = method_color(method, palette[i % len(palette)])
        ax.plot(
            x[m],
            y[m],
            color=col,
            linestyle=method_ls(method),
            linewidth=LINE_WIDTH_METHOD,
            label=method_label(method),
            zorder=4,
        )

        if SHOW_COH_CUTOFF_MARKERS and cutoff_lookup is not None:
            kc = cutoff_lookup.get(method, np.nan)
            if np.isfinite(kc):
                ax.plot(
                    [kc],
                    [COHERENCE_THRESHOLD],
                    color=col,
                    zorder=7,
                    **COH_CUTOFF_MARKER_KW,
                )

    style_axis(ax, x_log=True, y_log=False)
    ax.set_xlim(*xlim)
    ax.set_ylim(*COH_YLIM)

    if SHOW_COH_UNITY_LINE:
        ax.axhline(1.0, zorder=2, **COH_UNITY_LINE_KW)
    if SHOW_COH_THRESHOLD_LINE:
        ax.axhline(COHERENCE_THRESHOLD, zorder=3, **COH_THRESHOLD_LINE_KW)
        if ANNOTATE_COH_THRESHOLD and annotate_threshold:
            ax.text(
                0.085,
                COHERENCE_THRESHOLD + 0.025,
                COH_THRESHOLD_TEXT,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=COH_THRESHOLD_TEXT_SIZE,
                color=COH_THRESHOLD_LINE_KW.get("color", "#555555"),
            )
    if SHOW_NYQUIST_LINE:
        ax.axvline(grid["nyq_cpkm"], zorder=2, **NYQUIST_LINE_KW)

    ax.set_title(
        title, fontsize=TITLE_SIZE, weight="bold", pad=TITLE_PAD_BOTTOM_ROW
    )
    ax.set_ylabel(COH_YLABEL, fontsize=LABEL_SIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)

    if invert_x:
        ax.invert_xaxis()
    if show_legend:
        add_legend(ax, LEGEND_LOC_COH)


# =============================================================================
# 7) 2 x 4 COMBINED FIGURE ASSEMBLY
# =============================================================================

def plot_combined_2x4(ke_df, scalar_df, coh_plot_df, coh_cut_df, grid, out_dir):
    xcol, xlabel, invert_x = get_xcol_xlabel()
    methods = pick_methods_multi(ke_df, scalar_df, coh_plot_df)
    palette = sns.color_palette("colorblind", n_colors=max(len(methods), 3))

    k_all = np.concatenate(
        [
            ke_df["k_cyc_per_km"].to_numpy(float)
            if len(ke_df)
            else np.array([]),
            scalar_df["k_cyc_per_km"].to_numpy(float)
            if len(scalar_df)
            else np.array([]),
            coh_plot_df["k_cyc_per_km"].to_numpy(float)
            if len(coh_plot_df)
            else np.array([]),
        ]
    )
    k_all = k_all[np.isfinite(k_all) & (k_all > 0)]
    if k_all.size == 0:
        raise RuntimeError("No finite positive wavenumbers found to plot.")
    xlim = resolve_xlim(k_all.min(), k_all.max())

    fig, axes = plt.subplots(
        2,
        4,
        figsize=COMBINED_FIGSIZE,
        constrained_layout=True,
        sharex="col",
    )
    try:
        fig.get_layout_engine().set(
            w_pad=LAYOUT_W_PAD,
            h_pad=LAYOUT_H_PAD,
            wspace=LAYOUT_WSPACE,
            hspace=LAYOUT_HSPACE,
        )
    except Exception:
        try:
            fig.set_constrained_layout_pads(
                w_pad=LAYOUT_W_PAD,
                h_pad=LAYOUT_H_PAD,
                wspace=LAYOUT_WSPACE,
                hspace=LAYOUT_HSPACE,
            )
        except Exception:
            pass

    # Top row: 1D Spectra
    for j, var in enumerate(ABS_ORDER):
        if var == "KE":
            dvar = ke_df.copy()
            pred_col, true_col = "ke_pred_1d", "ke_true_1d"
        else:
            dvar = scalar_df[scalar_df["variable"] == var].copy()
            pred_col, true_col = "psd_pred_1d", "psd_true_1d"

        draw_spectrum_panel(
            ax=axes[0, j],
            df=dvar,
            methods=methods,
            pred_col=pred_col,
            true_col=true_col,
            y_label=ABS_YLABELS[var].format(u=unit_string(var)),
            title=ABS_TITLES[var],
            xcol=xcol,
            xlabel=xlabel,
            invert_x=invert_x,
            palette=palette,
            xlim=xlim,
            grid=grid,
            show_legend=(j == LEGEND_TOP_PANEL_INDEX),
            show_ref_slopes=(var == "KE"),
            show_wl_axis=SHOW_WAVELENGTH_AXIS,
            show_wl_label=(WAVELENGTH_LABEL_ON_ALL_TOP_PANELS or j == 0),
        )

    # Bottom row: Coherence Magnitude
    for j, var in enumerate(COH_ORDER):
        dvar = coh_plot_df[coh_plot_df["variable"] == var].copy()

        lookup = {}
        if len(coh_cut_df):
            sub = coh_cut_df[
                (coh_cut_df["variable"] == var)
                & (np.isclose(coh_cut_df["threshold"], COHERENCE_THRESHOLD))
            ]
            lookup = dict(zip(sub["method"], sub["k_cutoff_cyc_per_km"]))

        draw_coherence_panel(
            ax=axes[1, j],
            df=dvar,
            methods=methods,
            title=COH_TITLES[var],
            xcol=xcol,
            xlabel=xlabel,
            invert_x=invert_x,
            palette=palette,
            xlim=xlim,
            grid=grid,
            cutoff_lookup=lookup,
            show_legend=(j == LEGEND_BOTTOM_PANEL_INDEX),
            annotate_threshold=(j == len(COH_ORDER) - 1),
        )

    save_figure(fig, out_dir, COMBINED_FIG_NAME)


# =============================================================================
# 8) REPORTING AND CONSOLE SUMMARY
# =============================================================================

def print_grid_report(grid):
    print("=" * 96)
    print("GRID / SPECTRAL CONSTANTS")
    print("-" * 96)
    print(f"  dx                   = {grid['dx_m']:.1f} m = {grid['dx_m']/1000:.2f} km")
    print(f"  dy                   = {grid['dy_m']:.1f} m = {grid['dy_m']/1000:.2f} km")
    print(f"  grid size            = {grid['H']} x {grid['W']}")
    print(
        f"  domain extent        = {grid['domain_x_km']:.0f} km (x) "
        f"x {grid['domain_y_km']:.0f} km (y)"
    )
    print(f"  dfx                  = {grid['dfx_cpkm']:.6e} cycles/km")
    print(f"  dfy                  = {grid['dfy_cpkm']:.6e} cycles/km")
    print(f"  radial bin width dk  = {grid['dk_cpkm']:.6e} cycles/km")
    print(f"  longest wavelength   = {grid['longest_wavelength_km']:.0f} km")
    print(f"  Nyquist k            = {grid['nyq_cpkm']:.6f} cycles/km")
    print(f"  Nyquist wavelength   = {grid['nyq_wavelength_km']:.2f} km")
    print(f"  kmax effective       = {grid['kmax_eff_cpkm']:.6f} cycles/km")
    print(f"  1D conversion mode   = {ONE_D_CONVERSION}")
    print("=" * 96)


def print_coherence_cutoffs(df, level):
    if len(df) == 0:
        return
    sub = df[np.isclose(df["threshold"], level)]
    if len(sub) == 0:
        return
    print()
    print("=" * 96)
    print(
        f"COHERENCE CUTOFF  |C| = {level:g}   "
        f"(equivalent squared coherence: {level**2:g}) -> Table 2"
    )
    print("-" * 96)
    print(
        f"{'variable':>9s} {'method':>9s} "
        f"{'k_0.5 [cyc/km]':>16s} {'lambda_0.5 [km]':>17s} "
        f"{'k/k_Nyq':>9s}  status"
    )
    order_v = ["Z", "Q", "T", "U", "V", COH_WIND_KEY]
    for v in order_v:
        for m in METHOD_ORDER:
            r = sub[(sub["variable"] == v) & (sub["method"] == m)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            print(
                f"{v:>9s} {m:>9s} "
                f"{r['k_cutoff_cyc_per_km']:>16.5f} "
                f"{r['wavelength_cutoff_km']:>17.1f} "
                f"{r['fraction_of_nyquist']:>9.3f}  {r['status']}"
            )
    print("=" * 96)


def print_ratio_cutoffs(df, level):
    if len(df) == 0:
        return
    sub = df[np.isclose(df["threshold"], level)]
    if len(sub) == 0:
        return
    print()
    print("=" * 96)
    print(
        f"SPECTRAL DEPARTURE  E_pred / E_true = {level:g}  "
        f"({RATIO_CROSSING_MODE} downward crossing) -> Section 3.5.1"
    )
    print("-" * 96)
    print(
        f"{'variable':>9s} {'method':>9s} "
        f"{'k_dep [cyc/km]':>16s} {'lambda_dep [km]':>17s} "
        f"{'k/k_Nyq':>9s}  status"
    )
    for v in ["Z", "Q", "T", "KE"]:
        for m in METHOD_ORDER:
            r = sub[(sub["variable"] == v) & (sub["method"] == m)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            print(
                f"{v:>9s} {m:>9s} "
                f"{r['k_departure_cyc_per_km']:>16.5f} "
                f"{r['wavelength_departure_km']:>17.1f} "
                f"{r['fraction_of_nyquist']:>9.3f}  {r['status']}"
            )
    print("=" * 96)


# =============================================================================
# 9) MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "figure.facecolor": FIG_FACE_COLOR,
            "axes.facecolor": FIG_FACE_COLOR,
            "axes.titleweight": "bold",
            "axes.labelweight": "normal",
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.antialiased": True,
            "axes.linewidth": 1.1,
        }
    )

    in_dir = Path(INPUT_DIR)
    ke_path, sc_path = in_dir / KE_CSV, in_dir / SCALAR_CSV
    coh_path, meta_path = in_dir / COH_CSV, in_dir / META_JSON

    for p in (ke_path, sc_path, coh_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input file: {p}")

    ke_raw = pd.read_csv(ke_path)
    sc_raw = pd.read_csv(sc_path)
    coh_raw = pd.read_csv(coh_path)
    for d in (ke_raw, sc_raw, coh_raw):
        d.replace([np.inf, -np.inf], np.nan, inplace=True)

    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        warnings.warn(f"{meta_path} not found. Using FALLBACK_GRID defaults.")

    grid = resolve_grid_info(meta)
    print_grid_report(grid)

    # 2D annulus mean -> 1D spectral density conversion
    ke_df = add_one_d_columns(
        ke_raw, ["ke_pred", "ke_true"], grid, ONE_D_CONVERSION
    )
    sc_df = add_one_d_columns(
        sc_raw, ["psd_pred", "psd_true"], grid, ONE_D_CONVERSION
    )
    ke_df, sc_df = ensure_ratio_columns(ke_df, sc_df)

    # Prepare coherence dataframes
    coh_plot_df, coh_all_df = build_coherence_frames(coh_raw)

    # Apply wavenumber filtering
    kmax = PLOT_KMAX_CPKM
    if kmax is None and CLIP_KMAX_TO_NYQUIST:
        kmax = min(grid["nyq_cpkm"], grid["kmax_eff_cpkm"])
    ke_df = apply_k_filter(ke_df, PLOT_KMIN_CPKM, kmax)
    sc_df = apply_k_filter(sc_df, PLOT_KMIN_CPKM, kmax)
    coh_plot_df = apply_k_filter(coh_plot_df, PLOT_KMIN_CPKM, kmax)
    coh_all_df = apply_k_filter(coh_all_df, PLOT_KMIN_CPKM, kmax)

    if len(ke_df) == 0 or len(sc_df) == 0 or len(coh_plot_df) == 0:
        raise RuntimeError(
            "Dataset empty after wavenumber filtering. "
            "Please relax PLOT_KMIN_CPKM / PLOT_KMAX_CPKM limits."
        )

    print()
    print("=== DATA LOADED ===")
    print(f"  KE rows            : {len(ke_df)}")
    print(f"  Scalar rows        : {len(sc_df)}")
    print(
        f"  Coherence rows     : {len(coh_all_df)} "
        f"(plotted: {len(coh_plot_df)})"
    )
    print(
        f"  Methods present    : {pick_methods_multi(ke_df, sc_df, coh_plot_df)}"
    )
    if meta:
        print(
            f"  n_time_used_target : {meta.get('n_time_used_target', 'N/A')}"
        )
        sp = meta.get("spectral_preprocess", {}) or {}
        print(
            f"  Window / Detrend   : {sp.get('window_type', 'N/A')} "
            f"(alpha={sp.get('tukey_alpha', 'N/A')}), "
            f"plane detrend={sp.get('detrend_2d_plane', 'N/A')}"
        )
        print(f"  Coherence order    : {meta.get('coherence_order', 'N/A')}")

    # Calculate quantitative threshold cutoffs
    coh_cut_df = coherence_cutoff_table(
        coh_all_df, grid, COH_CUTOFF_LEVELS, COH_CROSSING_MODE
    )
    ratio_cut_df = ratio_cutoff_table(
        ke_df, sc_df, grid, RATIO_CUTOFF_LEVELS, RATIO_CROSSING_MODE
    )

    for lev in COH_CUTOFF_LEVELS:
        print_coherence_cutoffs(coh_cut_df, lev)
    for lev in RATIO_CUTOFF_LEVELS:
        print_ratio_cutoffs(ratio_cut_df, lev)

    # Generate Figure
    plot_combined_2x4(ke_df, sc_df, coh_plot_df, coh_cut_df, grid, OUTPUT_DIR)

    # Export processed tabular data for reproducibility
    out = Path(OUTPUT_DIR)

    spec_cols = [
        "method", "variable", "n_samples", "k_bin", "k_cyc_per_km",
        "wavelength_km", "n_modes", "one_d_factor",
    ]
    ke_out = ke_df.assign(variable="KE").rename(
        columns={
            "ke_pred": "spectrum_pred_2dmean",
            "ke_true": "spectrum_true_2dmean",
            "ke_pred_1d": "spectrum_pred_1d",
            "ke_true_1d": "spectrum_true_1d",
            "ke_ratio": "ratio_pred_over_true",
        }
    )
    sc_out = sc_df.rename(
        columns={
            "psd_pred": "spectrum_pred_2dmean",
            "psd_true": "spectrum_true_2dmean",
            "psd_pred_1d": "spectrum_pred_1d",
            "psd_true_1d": "spectrum_true_1d",
            "psd_ratio": "ratio_pred_over_true",
        }
    )
    val_cols = [
        "spectrum_pred_2dmean", "spectrum_true_2dmean",
        "spectrum_pred_1d", "spectrum_true_1d", "ratio_pred_over_true",
    ]
    spectra_out = pd.concat(
        [sc_out[spec_cols + val_cols], ke_out[spec_cols + val_cols]],
        axis=0,
        ignore_index=True,
    )
    spectra_out["one_d_conversion"] = ONE_D_CONVERSION
    spectra_out = spectra_out.sort_values(["variable", "method", "k_bin"])
    spectra_out.to_csv(out / OUT_SPECTRA_CSV, index=False, float_format="%.8e")

    coh_all_df.to_csv(out / OUT_COH_CSV, index=False, float_format="%.8e")
    coh_cut_df.to_csv(out / OUT_COH_CUT_CSV, index=False, float_format="%.8e")
    ratio_cut_df.to_csv(out / OUT_RATIO_CUT_CSV, index=False, float_format="%.8e")
    summary_metrics(ke_df, sc_df, coh_all_df).to_csv(
        out / OUT_SUMMARY_CSV, index=False, float_format="%.6e"
    )

    print()
    print("=== PROCESSING COMPLETE ===")
    print(f"Output directory: {OUTPUT_DIR}")
    for p in sorted(out.glob("*")):
        print(" -", p.name)


if __name__ == "__main__":
    main()