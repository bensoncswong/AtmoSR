#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
report_coherence_half_point.py
--------------------------------------------------------------------------------
Purpose:
    Reads spectral evaluation outputs in a strictly READ-ONLY manner without
    modifying or overwriting any CSV files:
        - spectral_coherence.csv       (Required)
        - ke_spectrum.csv              (Optional; supplementary statistics)
        - scalar_psd_spectrum.csv      (Optional; supplementary statistics)
    Outputs all calculated metrics required to populate manuscript text:

    "AtmoSR exceeds AtmoSR-Base for all five variables across the mid to high
     wavenumber range. Taking the wavenumber at which squared coherence falls to
     0.5 as a single summary, AtmoSR reaches [k = xx cycles km-1, wavelength xx km]
     for the horizontal wind against [xx cycles km-1, xx km] for AtmoSR-Base and
     [xx cycles km-1, xx km] for cubic spline interpolation. In that range
     AtmoSR-Base falls below cubic spline interpolation."

Method Mapping:
    cubic  -> cubic spline interpolation
    noPhy  -> AtmoSR-Base
    AllPhy -> AtmoSR
Each method evaluates 5 state variables (Z, Q, T, U, V) plus derived horizontal wind.

Output Sections:
    [0] Environment / Files / Configuration
    [1] Data Integrity Check (3 methods x 5 variables = 15 groups)
    [2] Coherence Threshold Crossover Table (gamma^2 = 0.5)
    [3] Mid-to-High Band Statistics & Comparisons
    [4] Manuscript Claim Verification (PASS/FAIL with Exact Binomial Sign Test)
    [5] Full Coherence Curves Table
    [6] Supplementary: KE / PSD Predicted-to-True Spectral Ratios
    [7] Pre-populated Manuscript Text
    [8] Publication-Ready Tables (LaTeX / Markdown)
================================================================================
"""

import os
import sys
import glob
import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", category=RuntimeWarning)


# =============================================================================
# 0) Configuration
# =============================================================================

# ---------- Input CSV Paths (Auto-searches if not found) ----------
COH_CSV = "./spectral_outputs/spectral_coherence.csv"
KE_CSV  = "./spectral_outputs/ke_spectrum.csv"
PSD_CSV = "./spectral_outputs/scalar_psd_spectrum.csv"

SEARCH_DIRS = [
    ".", "./spectral_outputs",
    "/content", "/content/spectral_outputs",
    "/content/drive/MyDrive", "/content/drive/MyDrive/spectral_outputs",
]

# ---------- Thresholds and Crossing Solvers ----------
THRESHOLD       = 0.5         # Squared coherence threshold
INTERP_SPACE    = "log"       # "log" | "linear" (Interpolate on log10(k) or linear k)
CROSSING_PICK   = "first"     # "first" | "last" | "persistent"
PERSIST_BINS    = 2           # Persistence definition: N consecutive bins below threshold
SMOOTH_BINS     = 1           # Centered moving average on gamma^2; 1 = no smoothing
EXTRAP_FIT_BINS = 6           # Bins used for log-k linear extrapolation if censored

# ---------- Horizontal Wind Synthesis Mode ----------
WIND_PRIMARY = "vector"       # "vector" (Recommended) | "meanUV"

# ---------- Mid-to-High Wavenumber Band ----------
PRIMARY_BAND_KEY = "mid+high(log-thirds)"

# ---------- Output Toggles ----------
PRINT_FULL_CURVES = True
PRINT_SUPPLEMENT  = True
PRINT_LATEX       = True
CURVE_MAX_ROWS    = 60        # Maximum rows printed for full curve tables

# ---------- Formatting & Aliases ----------
METHOD_ORDER   = ["cubic", "noPhy", "AllPhy"]
METHOD_PAPER   = {
    "cubic": "cubic spline interpolation",
    "noPhy": "AtmoSR-Base",
    "AllPhy": "AtmoSR"
}
METHOD_SHORT   = {"cubic": "cubic", "noPhy": "AtmoSR-Base", "AllPhy": "AtmoSR"}
SCALAR_VARS    = ["Z", "Q", "T"]
WIND_VARS      = ["U", "V"]
FIVE_VARS      = ["Z", "Q", "T", "U", "V"]
WIND_VEC_KEY   = "WIND(vector U+V)"
WIND_AVG_KEY   = "WIND(mean of U,V)"

EPS = 1e-30


# =============================================================================
# 1) Utility Functions
# =============================================================================
def hr(char="=", n=100):
    """Prints a horizontal dividing line."""
    print(char * n)


def sec(title: str):
    """Prints a section header block."""
    hr("=")
    print(title)
    hr("=")


def resolve_path(path: str, filename_hint: Optional[str] = None) -> Optional[str]:
    """Locates target file in explicit path, common search dirs, or recursive glob."""
    if path and os.path.isfile(path):
        return path
    fname = filename_hint or os.path.basename(path)
    for d in SEARCH_DIRS:
        cand = os.path.join(d, fname)
        if os.path.isfile(cand):
            return cand
    for root in [".", "/content"]:
        if not os.path.isdir(root):
            continue
        hits = sorted(glob.glob(os.path.join(root, "**", fname), recursive=True))
        if hits:
            return hits[0]
    return None


def fmt(x, nd=6, none="  n/a  ") -> str:
    """Formats float numbers with fixed precision."""
    if x is None:
        return none
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(xf):
        return none
    return f"{xf:.{nd}f}"


def fmt_g(x, nd=4, none="n/a") -> str:
    """Formats float numbers using general representation."""
    if x is None:
        return none
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(xf):
        return none
    return f"{xf:.{nd}g}"


def print_table(headers: List[str], rows: List[List[str]], indent: int = 0):
    """Renders plain-text formatted tables for console logs."""
    if not rows:
        print(" " * indent + "(no rows)")
        return
    ncol = len(headers)
    widths = [len(str(h)) for h in headers]
    for r in rows:
        for j in range(ncol):
            widths[j] = max(widths[j], len(str(r[j])) if j < len(r) else 0)
    pad = " " * indent
    line = pad + " | ".join(str(headers[j]).ljust(widths[j]) for j in range(ncol))
    print(line)
    print(pad + "-+-".join("-" * widths[j] for j in range(ncol)))
    for r in rows:
        cells = [str(r[j]) if j < len(r) else "" for j in range(ncol)]
        print(pad + " | ".join(cells[j].ljust(widths[j]) for j in range(ncol)))


def moving_average(y: np.ndarray, w: int) -> np.ndarray:
    """Applies centered moving average smoothing."""
    if w is None or w <= 1:
        return np.asarray(y, dtype=float).copy()
    w = int(w)
    if w % 2 == 0:
        w += 1
    pad = w // 2
    yp = np.pad(np.asarray(y, dtype=float), (pad, pad), mode="edge")
    ker = np.ones(w, dtype=float) / float(w)
    return np.convolve(yp, ker, mode="valid")


def exact_binom_two_sided(k: int, n: int) -> float:
    """Computes exact two-sided binomial sign test p-value (p=0.5) without scipy dependency."""
    if n <= 0:
        return float("nan")
    k = int(min(max(k, 0), n))
    total = float(2 ** n)
    lo = sum(math.comb(n, i) for i in range(0, k + 1)) / total
    hi = sum(math.comb(n, i) for i in range(k, n + 1)) / total
    return float(min(1.0, 2.0 * min(lo, hi)))


# =============================================================================
# 2) Data Loading & Method Normalization
# =============================================================================
def normalize_method(raw: str) -> str:
    """Standardizes method string identifiers."""
    s = str(raw).strip().lower()
    if ("cubic" in s) or ("spline" in s) or ("bicubic" in s):
        return "cubic"
    if ("allphy" in s) or ("all_phy" in s) or ("all-phy" in s) or (s == "atmosr") or (s == "phy"):
        return "AllPhy"
    if ("nophy" in s) or ("no_phy" in s) or ("no-phy" in s) or ("base" in s):
        return "noPhy"
    return str(raw).strip()


def load_coherence(path: str) -> pd.DataFrame:
    """Loads and validates spectral coherence CSV data."""
    df = pd.read_csv(path)
    need = ["method", "variable", "k_bin", "k_cyc_per_km", "wavelength_km",
            "coherence_mag_squared", "cross_real", "cross_imag",
            "auto_pred", "auto_true", "n_samples", "n_modes"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"{os.path.basename(path)} missing required columns: {miss}\nColumns present: {list(df.columns)}")
    df = df.copy()
    df["_m"] = df["method"].map(normalize_method)
    df["variable"] = df["variable"].astype(str).str.upper().str.strip()
    for c in ["k_cyc_per_km", "wavelength_km", "coherence_mag_squared",
              "cross_real", "cross_imag", "auto_pred", "auto_true"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_wind_variable(df: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    """Derives horizontal wind coherence in-memory without modifying disk CSVs."""
    extra = []
    for m in methods:
        u = df[(df["_m"] == m) & (df["variable"] == "U")].set_index("k_bin").sort_index()
        v = df[(df["_m"] == m) & (df["variable"] == "V")].set_index("k_bin").sort_index()
        if len(u) == 0 or len(v) == 0:
            continue
        idx = u.index.intersection(v.index)
        if len(idx) == 0:
            continue
        u, v = u.loc[idx], v.loc[idx]

        cu = u["cross_real"].to_numpy(float) + 1j * u["cross_imag"].to_numpy(float)
        cv = v["cross_real"].to_numpy(float) + 1j * v["cross_imag"].to_numpy(float)
        c_sum = cu + cv
        ap = u["auto_pred"].to_numpy(float) + v["auto_pred"].to_numpy(float)
        at = u["auto_true"].to_numpy(float) + v["auto_true"].to_numpy(float)

        g2_vec = np.clip(np.abs(c_sum) ** 2 / np.maximum(ap * at, EPS), 0.0, 1.0)
        g2_avg = 0.5 * (u["coherence_mag_squared"].to_numpy(float) +
                        v["coherence_mag_squared"].to_numpy(float))

        base = pd.DataFrame({
            "method": m, "_m": m,
            "k_bin": idx.to_numpy(),
            "k_cyc_per_km": u["k_cyc_per_km"].to_numpy(float),
            "wavelength_km": u["wavelength_km"].to_numpy(float),
            "n_modes": u["n_modes"].to_numpy(),
            "n_samples": u["n_samples"].to_numpy(),
        })

        d1 = base.copy()
        d1["variable"] = WIND_VEC_KEY
        d1["coherence_mag_squared"] = g2_vec
        d1["cross_real"] = np.real(c_sum)
        d1["cross_imag"] = np.imag(c_sum)
        d1["auto_pred"] = ap
        d1["auto_true"] = at

        d2 = base.copy()
        d2["variable"] = WIND_AVG_KEY
        d2["coherence_mag_squared"] = g2_avg
        d2["cross_real"] = np.nan
        d2["cross_imag"] = np.nan
        d2["auto_pred"] = np.nan
        d2["auto_true"] = np.nan

        extra += [d1, d2]

    return pd.concat([df] + extra, ignore_index=True) if extra else df


def get_curve(df: pd.DataFrame, method: str, variable: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Extracts sorted (wavenumber, squared coherence) series for a method-variable pair."""
    sub = df[(df["_m"] == method) & (df["variable"] == variable)].copy()
    sub = sub.dropna(subset=["k_cyc_per_km", "coherence_mag_squared"])
    sub = sub[sub["k_cyc_per_km"] > 0].sort_values("k_cyc_per_km")
    k = sub["k_cyc_per_km"].to_numpy(float)
    g2 = np.clip(sub["coherence_mag_squared"].to_numpy(float), 0.0, 1.0)
    return k, g2, sub


# =============================================================================
# 3) Threshold Crossover Solvers (gamma^2 = threshold)
# =============================================================================
def _interp_cross(k0, k1, y0, y1, thr, space="log") -> float:
    """Interpolates threshold crossover in log or linear wavenumber space."""
    if y1 == y0:
        return float(k1)
    f = (thr - y0) / (y1 - y0)
    f = float(min(max(f, 0.0), 1.0))
    if space == "log" and k0 > 0 and k1 > 0:
        return float(10.0 ** (np.log10(k0) + f * (np.log10(k1) - np.log10(k0))))
    return float(k0 + f * (k1 - k0))


def _loglin_extrap(k: np.ndarray, y: np.ndarray, thr: float, nfit: int, side: str) -> Optional[float]:
    """Linear log-k extrapolation for censored curves."""
    n = len(k)
    if n < 2:
        return None
    nfit = int(min(max(nfit, 2), n))
    idx = np.arange(n - nfit, n) if side == "right" else np.arange(0, nfit)
    lk = np.log10(k[idx])
    yy = y[idx]
    try:
        b, a = np.polyfit(lk, yy, 1)  # y = a + b*lk
    except Exception:
        return None
    if abs(b) < 1e-15:
        return None
    lk_c = (thr - a) / b
    kc = float(10.0 ** lk_c)
    if not np.isfinite(kc) or kc <= 0:
        return None
    return kc


def analyze_curve(k: np.ndarray, y_raw: np.ndarray,
                  thr: float = THRESHOLD,
                  space: str = INTERP_SPACE,
                  smooth: int = SMOOTH_BINS,
                  persist: int = PERSIST_BINS,
                  extrap_bins: int = EXTRAP_FIT_BINS) -> Dict:
    """Analyzes downward threshold crossings of a coherence curve."""
    out = {
        "n_bins": int(len(k)), "k_min": None, "k_max": None,
        "g2_at_kmin": None, "g2_at_kmax": None,
        "n_down": 0, "n_up": 0,
        "k_first": None, "k_last": None, "k_persist": None,
        "bracket": None, "status": "empty",
        "k_extrap": None, "extrap_side": None,
    }
    if len(k) < 2:
        return out

    y = moving_average(y_raw, smooth)
    out["k_min"], out["k_max"] = float(k[0]), float(k[-1])
    out["g2_at_kmin"], out["g2_at_kmax"] = float(y[0]), float(y[-1])

    downs, ups = [], []
    for i in range(len(k) - 1):
        y0, y1 = y[i], y[i + 1]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if (y0 >= thr) and (y1 < thr):
            downs.append((i, _interp_cross(k[i], k[i + 1], y0, y1, thr, space)))
        if (y0 < thr) and (y1 >= thr):
            ups.append(i)

    out["n_down"], out["n_up"] = len(downs), len(ups)

    if downs:
        out["k_first"] = float(downs[0][1])
        out["k_last"] = float(downs[-1][1])
        for i, kc in downs:
            j2 = min(i + 1 + persist, len(y))
            if np.all(y[i + 1:j2] < thr):
                out["k_persist"] = float(kc)
                break
        i0 = downs[0][0]
        out["bracket"] = (float(k[i0]), float(y[i0]), float(k[i0 + 1]), float(y[i0 + 1]))
        out["status"] = "ok" if (len(downs) == 1 and len(ups) == 0) else "ok(multi-crossing)"
    else:
        if np.all(y >= thr):
            out["status"] = "right_censored(never<0.5 within k_max)"
            out["k_extrap"] = _loglin_extrap(k, y, thr, extrap_bins, "right")
            out["extrap_side"] = "right"
        elif np.all(y < thr):
            out["status"] = "left_censored(already<0.5 at k_min)"
            out["k_extrap"] = _loglin_extrap(k, y, thr, extrap_bins, "left")
            out["extrap_side"] = "left"
        else:
            out["status"] = "no_downward_crossing"
    return out


def pick_k(res: Dict, pick: str = CROSSING_PICK) -> Tuple[Optional[float], str]:
    """Selects target wavenumber k according to priority policy."""
    order = {"first": ["k_first", "k_persist", "k_last"],
             "last": ["k_last", "k_persist", "k_first"],
             "persistent": ["k_persist", "k_first", "k_last"]}[pick]
    for key in order:
        if res.get(key) is not None:
            return float(res[key]), key
    if res.get("k_extrap") is not None:
        return float(res["k_extrap"]), f"k_extrap({res.get('extrap_side')})"
    return None, "none"


def lam_of(k: Optional[float]) -> Optional[float]:
    """Converts wavenumber k (cycles/km) to spatial wavelength lambda (km)."""
    if k is None or not np.isfinite(k) or k <= 0:
        return None
    return 1.0 / k


# =============================================================================
# 4) Band Definitions and Statistical Evaluation
# =============================================================================
def make_bands(k: np.ndarray) -> Dict[str, Dict]:
    """Constructs predefined wavenumber spectral bands."""
    kmin, kmax = float(k[0]), float(k[-1])
    lk0, lk1 = np.log10(kmin), np.log10(kmax)
    k13 = 10 ** (lk0 + (lk1 - lk0) / 3.0)
    k23 = 10 ** (lk0 + 2.0 * (lk1 - lk0) / 3.0)
    bands = {
        "mid+high(log-thirds)": {"mask": (k >= k13), "desc": f"k >= {k13:.6f} (upper 2/3 log-thirds)"},
        "mid(log-thirds)":      {"mask": (k >= k13) & (k < k23), "desc": f"{k13:.6f} <= k < {k23:.6f}"},
        "high(log-thirds)":     {"mask": (k >= k23), "desc": f"k >= {k23:.6f}"},
        "k>=0.25*kmax":         {"mask": (k >= 0.25 * kmax), "desc": f"k >= {0.25*kmax:.6f}"},
        "k>=0.50*kmax":         {"mask": (k >= 0.50 * kmax), "desc": f"k >= {0.50*kmax:.6f}"},
        "lambda<=500km":        {"mask": (k >= 1.0 / 500.0), "desc": "wavelength <= 500 km"},
        "lambda<=300km":        {"mask": (k >= 1.0 / 300.0), "desc": "wavelength <= 300 km"},
        "full":                 {"mask": np.ones_like(k, dtype=bool), "desc": f"{kmin:.6f} <= k <= {kmax:.6f}"},
    }
    for b in bands.values():
        m = b["mask"]
        b["n"] = int(m.sum())
        b["k_lo"] = float(k[m][0]) if b["n"] > 0 else None
        b["k_hi"] = float(k[m][-1]) if b["n"] > 0 else None
    return bands


def compare_in_band(kk: np.ndarray, a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> Dict:
    """Computes comparative band statistics between two method series."""
    ka, aa, bb = kk[mask], a[mask], b[mask]
    ok = np.isfinite(aa) & np.isfinite(bb)
    ka, aa, bb = ka[ok], aa[ok], bb[ok]
    n = len(ka)
    if n == 0:
        return {"n": 0}
    d = aa - bb
    wins = int(np.sum(d > 0))
    return {
        "n": n,
        "mean_a": float(np.mean(aa)), "mean_b": float(np.mean(bb)),
        "mean_diff": float(np.mean(d)), "median_diff": float(np.median(d)),
        "min_diff": float(np.min(d)), "max_diff": float(np.max(d)),
        "win_bins": wins, "win_rate": float(wins) / n,
        "all_bins_win": bool(np.all(d > 0)),
        "p_sign": exact_binom_two_sided(wins, n),
    }


def leading_suffix_start(kk: np.ndarray, a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Finds smallest wavenumber k where series a > b continuously up to k_max."""
    ok = np.isfinite(a) & np.isfinite(b)
    kk, a, b = kk[ok], a[ok], b[ok]
    if len(kk) == 0:
        return None
    g = a > b
    if not g[-1]:
        return None
    i = len(kk) - 1
    while i - 1 >= 0 and g[i - 1]:
        i -= 1
    return float(kk[i])


# =============================================================================
# 5) Main Execution Flow
# =============================================================================
def main():
    # ---------------- [0] Configuration & Setup ----------------
    sec("[0] Environment / Configuration (READ-ONLY: No CSV files modified)")

    coh_path = resolve_path(COH_CSV, "spectral_coherence.csv")
    ke_path = resolve_path(KE_CSV, "ke_spectrum.csv")
    psd_path = resolve_path(PSD_CSV, "scalar_psd_spectrum.csv")

    if coh_path is None:
        print("!! ERROR: spectral_coherence.csv not found. Check paths or run evaluation script first.")
        print("   Searched directories:", SEARCH_DIRS)
        sys.exit(1)

    print(f"Coherence CSV           : {coh_path}")
    print(f"KE CSV                  : {ke_path if ke_path else '(not found, skipping KE supplement)'}")
    print(f"PSD CSV                 : {psd_path if psd_path else '(not found, skipping PSD supplement)'}")
    print()
    print(f"Threshold               : squared coherence = {THRESHOLD}")
    print(f"Crossing interpolation  : linear in {'log10(k)' if INTERP_SPACE=='log' else 'k'}")
    print(f"Crossing pick (primary) : {CROSSING_PICK} (persist={PERSIST_BINS} bins)")
    print(f"Coherence smoothing     : moving average over {SMOOTH_BINS} bin(s)"
          f"{' (= no smoothing)' if SMOOTH_BINS<=1 else ''}")
    print(f"Horizontal wind mode    : {WIND_PRIMARY} "
          f"[vector = |Cu+Cv|^2 / ((Pu^p+Pv^p)(Pu^t+Pv^t)); meanUV = (g2_U+g2_V)/2]")
    print(f"Primary band key        : {PRIMARY_BAND_KEY}")
    print("Wavelength formula      : lambda = 1 / k (k in cycles/km -> lambda in km)")

    # ---------------- [1] Data Integrity Check ----------------
    df = load_coherence(coh_path)
    methods_present = [m for m in METHOD_ORDER if m in set(df["_m"])]
    unknown = sorted(set(df["_m"]) - set(METHOD_ORDER))

    sec("[1] Data Integrity Check")
    print(f"Normalized methods found : {sorted(set(df['_m']))}")
    if unknown:
        print(f"Unrecognized methods ignored: {unknown}")
    print(f"Evaluation method order  : {[METHOD_SHORT.get(m, m) for m in methods_present]}")
    missing_methods = [m for m in METHOD_ORDER if m not in methods_present]
    if missing_methods:
        print(f"!! Missing methods: {missing_methods}")

    df = add_wind_variable(df, methods_present)

    rows = []
    for m in methods_present:
        for v in FIVE_VARS + [WIND_VEC_KEY, WIND_AVG_KEY]:
            k, g2, sub = get_curve(df, m, v)
            ns = int(sub["n_samples"].iloc[0]) if len(sub) else 0
            rows.append([METHOD_SHORT.get(m, m), v, len(k), ns,
                         fmt(k[0], 6) if len(k) else "n/a",
                         fmt(k[-1], 6) if len(k) else "n/a",
                         fmt(lam_of(k[-1]) if len(k) else None, 1),
                         fmt(lam_of(k[0]) if len(k) else None, 1)])
    print()
    print_table(["method", "variable", "n_k_bins", "n_samples",
                 "k_min[cyc/km]", "k_max[cyc/km]", "lambda_min[km]", "lambda_max[km]"], rows)
    print()
    print(f"Variable combinations (excluding wind derivatives) = {len(methods_present)} methods x 5 vars = "
          f"{len(methods_present)*5} (Expected: 15)")

    ref_k = None
    for m in methods_present:
        k, _, _ = get_curve(df, m, "Z")
        if ref_k is None:
            ref_k = k
        elif len(k) != len(ref_k) or not np.allclose(k, ref_k, rtol=1e-9, atol=0):
            print("!! WARNING: Inconsistent k-grids across methods. Aligning via index intersection.")
    if ref_k is None or len(ref_k) < 2:
        print("!! ERROR: Insufficient k-grid points for analysis.")
        sys.exit(1)

    bands = make_bands(ref_k)
    print()
    print("Wavenumber band definitions (Log-thirds division):")
    brows = [[name, b["desc"], b["n"], fmt(b["k_lo"], 6), fmt(b["k_hi"], 6),
              fmt(lam_of(b["k_hi"]), 1), fmt(lam_of(b["k_lo"]), 1)]
             for name, b in bands.items()]
    print_table(["band", "definition", "n_bins", "k_lo", "k_hi", "lambda_min[km]", "lambda_max[km]"], brows)

    # ---------------- [2] gamma^2 = 0.5 Crossover Table ----------------
    sec(f"[2] Squared Coherence Threshold Crossings (gamma^2 = {THRESHOLD})")

    all_vars = FIVE_VARS + [WIND_VEC_KEY, WIND_AVG_KEY]
    results: Dict[Tuple[str, str], Dict] = {}
    curves: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}

    trows = []
    for v in all_vars:
        for m in methods_present:
            k, g2, _ = get_curve(df, m, v)
            curves[(m, v)] = (k, g2)
            res = analyze_curve(k, g2)
            results[(m, v)] = res
            k_used, src = pick_k(res)
            lam = lam_of(k_used)
            br = res["bracket"]
            br_s = (f"[{br[0]:.6f}->{br[2]:.6f}] g2:[{br[1]:.3f}->{br[3]:.3f}]" if br else "-")
            trows.append([
                v, METHOD_SHORT.get(m, m),
                fmt(k_used, 6), fmt(lam, 1),
                fmt(res["k_first"], 6), fmt(res["k_last"], 6), fmt(res["k_persist"], 6),
                res["n_down"], res["n_up"],
                fmt(res["g2_at_kmin"], 3), fmt(res["g2_at_kmax"], 3),
                src, res["status"], br_s
            ])
        trows.append(["-" * 10] + ["-" * 6] * 13)

    print_table(["variable", "method", f"k(g2={THRESHOLD})[cyc/km]", "lambda[km]",
                 "k_first", "k_last", "k_persist", "#down", "#up",
                 "g2@k_min", "g2@k_max", "picked", "status", "bracket(bin_lo->bin_hi)"], trows)

    print()
    print("Status Notes:")
    print("  * status=ok            -> Clean single crossover.")
    print("  * ok(multi-crossing)   -> Oscillation around threshold; check k_first vs k_persist.")
    print("  * right_censored       -> Coherence remains > 0.5 up to k_max.")
    print("  * left_censored        -> Coherence falls below 0.5 prior to k_min.")

    print()
    print("Effective Resolution Gain (lambda ratio; < 1.0 indicates AtmoSR resolves smaller scales):")
    irows = []
    for v in all_vars:
        kA, _ = pick_k(results.get(("AllPhy", v), {})) if ("AllPhy", v) in results else (None, "")
        kB, _ = pick_k(results.get(("noPhy", v), {})) if ("noPhy", v) in results else (None, "")
        kC, _ = pick_k(results.get(("cubic", v), {})) if ("cubic", v) in results else (None, "")
        lA, lB, lC = lam_of(kA), lam_of(kB), lam_of(kC)
        r_AB = (lA / lB) if (lA and lB) else None
        r_AC = (lA / lC) if (lA and lC) else None
        r_BC = (lB / lC) if (lB and lC) else None
        irows.append([v, fmt(lA, 1), fmt(lB, 1), fmt(lC, 1),
                      fmt_g(r_AB, 4), fmt_g(r_AC, 4), fmt_g(r_BC, 4),
                      fmt_g((kA / kB) if (kA and kB) else None, 4)])
    print_table(["variable", "lam_AtmoSR[km]", "lam_Base[km]", "lam_cubic[km]",
                 "lam_A/lam_B", "lam_A/lam_cubic", "lam_B/lam_cubic", "k_A/k_B"], irows)

    # ---------------- [3] In-Band Statistics ----------------
    sec("[3] Mid-to-High Wavenumber Band Coherence Statistics")

    def band_block(band_key: str):
        band = bands[band_key]
        print()
        print(f"--- Band = {band_key} ({band['desc']}; n_bins={band['n']}, "
              f"k=[{fmt(band['k_lo'],6)}, {fmt(band['k_hi'],6)}] cyc/km, "
              f"lambda=[{fmt(lam_of(band['k_hi']),1)}, {fmt(lam_of(band['k_lo']),1)}] km) ---")

        rows_mean = []
        for v in all_vars:
            cells = [v]
            for m in METHOD_ORDER:
                if (m, v) not in curves:
                    cells.append("n/a")
                    continue
                k, g2 = curves[(m, v)]
                mask = np.interp(k, ref_k, band["mask"].astype(float)) > 0.5 \
                    if len(k) != len(ref_k) else band["mask"]
                gg = g2[mask]
                cells.append(fmt(np.mean(gg) if len(gg) else None, 4))
            rows_mean.append(cells)
        print()
        print("In-band mean(gamma^2):")
        print_table(["variable"] + [METHOD_SHORT[m] for m in METHOD_ORDER], rows_mean, indent=2)

        for (ma, mb, label) in [("AllPhy", "noPhy", "AtmoSR vs AtmoSR-Base"),
                                ("noPhy", "cubic", "AtmoSR-Base vs cubic"),
                                ("AllPhy", "cubic", "AtmoSR vs cubic")]:
            if ma not in methods_present or mb not in methods_present:
                continue
            print()
            print(f"  [{label}] (diff = {METHOD_SHORT[ma]} - {METHOD_SHORT[mb]})")
            crows = []
            for v in all_vars:
                ka, ga = curves[(ma, v)]
                kb, gb = curves[(mb, v)]
                n = min(len(ka), len(kb))
                kk, ga, gb = ka[:n], ga[:n], gb[:n]
                mask = band["mask"][:n]
                st = compare_in_band(kk, ga, gb, mask)
                if st["n"] == 0:
                    crows.append([v] + ["n/a"] * 8)
                    continue
                ks = leading_suffix_start(kk, ga, gb)
                crows.append([
                    v, st["n"], fmt(st["mean_a"], 4), fmt(st["mean_b"], 4),
                    ("+" if st["mean_diff"] >= 0 else "") + fmt(st["mean_diff"], 4),
                    ("+" if st["min_diff"] >= 0 else "") + fmt(st["min_diff"], 4),
                    f"{st['win_bins']}/{st['n']} ({100*st['win_rate']:.0f}%)",
                    fmt_g(st["p_sign"], 3),
                    (f"{ks:.6f} (lam<={lam_of(ks):.0f}km)" if ks else "not to k_max"),
                ])
            print_table(["variable", "n_bins", f"mean_{METHOD_SHORT[ma]}", f"mean_{METHOD_SHORT[mb]}",
                         "mean_diff", "min_diff", "bins_with_diff>0", "p(sign,2-sided)",
                         "leads continuously from k"], crows, indent=2)

    band_block(PRIMARY_BAND_KEY)
    for bk in ["mid(log-thirds)", "high(log-thirds)", "k>=0.25*kmax", "k>=0.50*kmax", "lambda<=500km"]:
        if bk != PRIMARY_BAND_KEY:
            band_block(bk)

    # ---------------- [4] Manuscript Claim Verification ----------------
    sec("[4] Manuscript Claim Verification (Primary Band = " + PRIMARY_BAND_KEY + ")")

    band = bands[PRIMARY_BAND_KEY]
    claim1_detail, claim2_detail = [], []
    claim1_ok_mean = claim1_ok_all = True
    claim2_ok_mean = claim2_ok_all = True

    for v in FIVE_VARS:
        # Claim 1: AllPhy > noPhy
        if ("AllPhy", v) in curves and ("noPhy", v) in curves:
            ka, ga = curves[("AllPhy", v)]
            kb, gb = curves[("noPhy", v)]
            n = min(len(ka), len(kb))
            st = compare_in_band(ka[:n], ga[:n], gb[:n], band["mask"][:n])
            ok_mean = st.get("mean_diff", -1) > 0
            ok_all = st.get("all_bins_win", False)
            claim1_ok_mean &= ok_mean
            claim1_ok_all &= ok_all
            claim1_detail.append([v, fmt(st.get("mean_a"), 4), fmt(st.get("mean_b"), 4),
                                  fmt(st.get("mean_diff"), 4),
                                  f"{st.get('win_bins')}/{st.get('n')}",
                                  "YES" if ok_mean else "NO", "YES" if ok_all else "NO"])

        # Claim 2: noPhy < cubic
        if ("noPhy", v) in curves and ("cubic", v) in curves:
            kb, gb = curves[("noPhy", v)]
            kc, gc = curves[("cubic", v)]
            n = min(len(kb), len(kc))
            st = compare_in_band(kb[:n], gc[:n], gb[:n], band["mask"][:n])  # cubic - noPhy
            ok_mean = st.get("mean_diff", -1) > 0
            ok_all = st.get("all_bins_win", False)
            claim2_ok_mean &= ok_mean
            claim2_ok_all &= ok_all
            claim2_detail.append([v, fmt(st.get("mean_b"), 4), fmt(st.get("mean_a"), 4),
                                  fmt(st.get("mean_diff"), 4),
                                  f"{st.get('win_bins')}/{st.get('n')}",
                                  "YES" if ok_mean else "NO", "YES" if ok_all else "NO"])

    print("Claim 1: 'AtmoSR exceeds AtmoSR-Base for all five variables across the mid to high wavenumber range'")
    print_table(["variable", "mean_AtmoSR", "mean_Base", "mean_diff",
                 "bins AtmoSR>Base", "mean-level PASS", "every-bin PASS"], claim1_detail, indent=2)
    print(f"  => Mean-level evaluation : {'PASS (5/5 variables verified)' if claim1_ok_mean else 'FAIL'}")
    print(f"  => Strict bin-level evaluation: {'PASS' if claim1_ok_all else 'FAIL (Consider phrasing as \"on average\")'}")

    print()
    print("Claim 2: 'In that range AtmoSR-Base falls below cubic spline interpolation'")
    print_table(["variable", "mean_Base", "mean_cubic", "mean(cubic-Base)",
                 "bins cubic>Base", "mean-level PASS", "every-bin PASS"], claim2_detail, indent=2)
    print(f"  => Mean-level evaluation : {'PASS (5/5 variables verified)' if claim2_ok_mean else 'FAIL'}")
    print(f"  => Strict bin-level evaluation: {'PASS' if claim2_ok_all else 'FAIL'}")

    # ---------------- [5] Full Curves Output ----------------
    if PRINT_FULL_CURVES:
        sec("[5] Full gamma^2(k) Curves (All methods side-by-side)")
        for v in all_vars:
            print()
            print(f"--- Variable = {v} ---")
            k0, _ = curves[(methods_present[0], v)]
            rows_c = []
            step = max(1, int(np.ceil(len(k0) / CURVE_MAX_ROWS)))
            for i in range(0, len(k0), step):
                row = [i, fmt(k0[i], 6), fmt(lam_of(k0[i]), 1)]
                for m in METHOD_ORDER:
                    if (m, v) in curves and i < len(curves[(m, v)][1]):
                        row.append(fmt(curves[(m, v)][1][i], 4))
                    else:
                        row.append("n/a")
                inb = "*" if (i < len(band["mask"]) and band["mask"][i]) else ""
                row.append(inb)
                rows_c.append(row)
            print_table(["idx", "k[cyc/km]", "lambda[km]"] + [METHOD_SHORT[m] for m in METHOD_ORDER],
                        rows_c, indent=2)
            print("  ('*' indicates bin resides within primary mid-to-high band)")

    # ---------------- [6] Supplementary KE / PSD Ratios ----------------
    if PRINT_SUPPLEMENT:
        sec("[6] Supplementary: Spectral Energy Ratio (Predicted / True; 1.0 = Ideal)")
        if ke_path:
            ke = pd.read_csv(ke_path)
            ke["_m"] = ke["method"].map(normalize_method)
            rows_k = []
            for m in METHOD_ORDER:
                sub = ke[ke["_m"] == m].sort_values("k_cyc_per_km")
                if len(sub) == 0:
                    continue
                kk = sub["k_cyc_per_km"].to_numpy(float)
                rr = sub["ke_pred_over_true"].to_numpy(float)
                mask = kk >= (band["k_lo"] if band["k_lo"] else kk[0])
                if mask.sum() == 0:
                    continue
                rows_k.append([METHOD_SHORT.get(m, m), int(mask.sum()),
                               fmt(np.mean(rr[mask]), 4), fmt(np.median(rr[mask]), 4),
                               fmt(np.min(rr[mask]), 4), fmt(rr[-1], 4)])
            print("KE spectrum (0.5*(Suu+Svv)) ratio, band = " + PRIMARY_BAND_KEY + ":")
            print_table(["method", "n_bins", "mean_ratio", "median_ratio", "min_ratio", "ratio@k_max"],
                        rows_k, indent=2)
        if psd_path:
            psd = pd.read_csv(psd_path)
            psd["_m"] = psd["method"].map(normalize_method)
            psd["variable"] = psd["variable"].astype(str).str.upper()
            rows_p = []
            for v in SCALAR_VARS:
                for m in METHOD_ORDER:
                    sub = psd[(psd["_m"] == m) & (psd["variable"] == v)].sort_values("k_cyc_per_km")
                    if len(sub) == 0:
                        continue
                    kk = sub["k_cyc_per_km"].to_numpy(float)
                    rr = sub["psd_pred_over_true"].to_numpy(float)
                    mask = kk >= (band["k_lo"] if band["k_lo"] else kk[0])
                    if mask.sum() == 0:
                        continue
                    rows_p.append([v, METHOD_SHORT.get(m, m), int(mask.sum()),
                                   fmt(np.mean(rr[mask]), 4), fmt(np.median(rr[mask]), 4),
                                   fmt(np.min(rr[mask]), 4), fmt(rr[-1], 4)])
            print()
            print("Scalar PSD ratio (Z/Q/T), band = " + PRIMARY_BAND_KEY + ":")
            print_table(["variable", "method", "n_bins", "mean_ratio", "median_ratio",
                         "min_ratio", "ratio@k_max"], rows_p, indent=2)

    # ---------------- [7] Pre-populated Manuscript Sentence ----------------
    sec("[7] Pre-populated Manuscript Sentences")

    wind_key = WIND_VEC_KEY if WIND_PRIMARY == "vector" else WIND_AVG_KEY

    def trio(vkey: str):
        out = {}
        for m in METHOD_ORDER:
            if (m, vkey) in results:
                ku, src = pick_k(results[(m, vkey)])
                out[m] = (ku, lam_of(ku), src, results[(m, vkey)]["status"])
            else:
                out[m] = (None, None, "none", "missing")
        return out

    print("[BLANKS] Horizontal wind, primary mode = " + wind_key)
    tw = trio(wind_key)
    brows = []
    for m in METHOD_ORDER:
        ku, lu, src, st = tw[m]
        brows.append([METHOD_PAPER.get(m, m), fmt(ku, 6), fmt_g(ku, 3),
                      fmt(lu, 1), (f"{lu:.0f}" if lu else "n/a"), src, st])
    print_table(["method (paper name)", "k [cycles/km]", "k (3 s.f.)",
                 "lambda [km]", "lambda (rounded)", "picked", "status"], brows, indent=2)

    kA, lA = tw["AllPhy"][0], tw["AllPhy"][1]
    kB, lB = tw["noPhy"][0], tw["noPhy"][1]
    kC, lC = tw["cubic"][0], tw["cubic"][1]

    def s_k(x):
        return f"{x:.5f}" if (x and np.isfinite(x)) else "xx"

    def s_l(x):
        return f"{x:.0f}" if (x and np.isfinite(x)) else "xx"

    sentence = (
        "AtmoSR exceeds AtmoSR-Base for all five variables across the mid to high wavenumber range. "
        f"Taking the wavenumber at which squared coherence falls to {THRESHOLD} as a single summary, "
        f"AtmoSR reaches k = {s_k(kA)} cycles km\u207b\u00b9 (wavelength {s_l(lA)} km) for the horizontal wind "
        f"against {s_k(kB)} cycles km\u207b\u00b9 ({s_l(lB)} km) for AtmoSR-Base and "
        f"{s_k(kC)} cycles km\u207b\u00b9 ({s_l(lC)} km) for cubic spline interpolation. "
        "In that range AtmoSR-Base falls below cubic spline interpolation."
    )
    print()
    print(">>> FILLED MANUSCRIPT SENTENCE <<<")
    print(sentence)

    # Verification variants
    print()
    print(">>> Alternative Synthesis Formulations (Robustness Check) <<<")
    for vkey in [WIND_VEC_KEY, WIND_AVG_KEY, "U", "V"]:
        t = trio(vkey)
        ka, la = t["AllPhy"][0], t["AllPhy"][1]
        kb, lb = t["noPhy"][0], t["noPhy"][1]
        kc, lc = t["cubic"][0], t["cubic"][1]
        print(f"  [{vkey:<18}] AtmoSR: k={s_k(ka)} (lam={s_l(la)} km) | "
              f"AtmoSR-Base: k={s_k(kb)} (lam={s_l(lb)} km) | "
              f"cubic: k={s_k(kc)} (lam={s_l(lc)} km)")
    print()
    print("  [Five Variables k(g2=0.5) Summary]")
    vrows = []
    for v in FIVE_VARS:
        t = trio(v)
        vrows.append([v] + [f"{s_k(t[m][0])} / {s_l(t[m][1])}" for m in METHOD_ORDER])
    print_table(["variable"] + [f"{METHOD_SHORT[m]} k / lambda" for m in METHOD_ORDER], vrows, indent=2)

    print()
    ordering_ok = (kA is not None and kB is not None and kC is not None
                   and kA > kC > kB)
    print("Ordering check (AtmoSR > cubic > AtmoSR-Base for k(g2=0.5)): "
          + ("PASS" if ordering_ok else "CHECK! (Ordering differs from hypothesis)"))
    if not ordering_ok:
        trio_list = [(METHOD_SHORT[m], tw[m][0]) for m in METHOD_ORDER if tw[m][0]]
        trio_list.sort(key=lambda x: -x[1])
        print("  Observed descending order: " + " > ".join(f"{n}({v:.5f})" for n, v in trio_list))

    # ---------------- [8] Publication Tables ----------------
    if PRINT_LATEX:
        sec("[8] Publication-Ready Tables")
        print("--- Markdown Table ---")
        print("| Variable | " + " | ".join(
            f"{METHOD_PAPER[m]} k (cycles km^-1) / λ (km)" for m in METHOD_ORDER) + " |")
        print("|---" * (1 + len(METHOD_ORDER)) + "|")
        for v in FIVE_VARS + [WIND_VEC_KEY]:
            t = trio(v)
            cells = [f"{s_k(t[m][0])} / {s_l(t[m][1])}" for m in METHOD_ORDER]
            print(f"| {v} | " + " | ".join(cells) + " |")

        print()
        print("--- LaTeX Table ---")
        print(r"\begin{tabular}{l" + "c" * len(METHOD_ORDER) + "}")
        print(r"\hline")
        print("Variable & " + " & ".join(METHOD_PAPER[m] for m in METHOD_ORDER) + r" \\")
        print(r"\hline")
        for v in FIVE_VARS + [WIND_VEC_KEY]:
            t = trio(v)
            cells = [f"{s_k(t[m][0])} / {s_l(t[m][1])}" for m in METHOD_ORDER]
            print(f"{v} & " + " & ".join(cells) + r" \\")
        print(r"\hline")
        print(r"\end{tabular}")
        print(r"% Note: Cells formatted as k(gamma^2=0.5) in cycles km$^{-1}$ / wavelength in km")

    hr("=")
    print("Completed execution successfully (READ-ONLY mode ensured).")
    hr("=")


if __name__ == "__main__":
    main()