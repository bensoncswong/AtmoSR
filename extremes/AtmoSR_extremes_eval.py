#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA5 Super-Resolution Model: Extreme Value & Tail Distribution Evaluation
including JAMES-style Quantile-Quantile (Q-Q) Plots (v1.2)

Version 1.2 Updates:
1) Exports data for Fig_tail_return_curves to:
   extremes_out/Fig_tail_return_curves.csv
2) Supports standalone figure re-rendering from the exported CSV.
"""

import os
import math
import json
import random
import warnings
import textwrap
from collections import OrderedDict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm as _norm
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# 0) User Configuration
# =============================================================================
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EVAL_MAX_SAMPLES = None

X_TEST_PATH = "X2_test.nc"
Y_TEST_PATH = "Y_test.nc"
SCALER_PATH = "scalers.joblib"

MODEL_PATHS = OrderedDict([
    ("noPhy",  "best_model_noPhy.pth"),
    ("AllPhy", "best_model_AllPhy.pth"),
])

DISPLAY_NAME_MAP = OrderedDict([
    ("cubic spline", "Cubic spline"),
    ("noPhy",        "AtmoSR-Base"),
    ("AllPhy",       "AtmoSR"),
])

SENTENCE_NAME_MAP = OrderedDict([
    ("cubic spline", "Cubic spline interpolation"),
    ("noPhy",        "AtmoSR-Base"),
    ("AllPhy",       "AtmoSR"),
])

VARIABLES = ["z", "q", "t", "u", "v"]

CUBIC_SPLINE_KX = 3
CUBIC_SPLINE_KY = 3

CP_AIR = 1004.0
LV_WATER = 2.501e6
EARTH_RADIUS = 6371000.0
OMEGA = 7.2921e-5
F_CORIOLIS_ABS_MIN = 2.5e-5
TAN_CLIP = 50.0

EXTREME_SPECS = OrderedDict([
    ("PHI", dict(kind="var:z", label="Geopotential", panel="Φ", tag="Φ",
                 short=r"\(\Phi\)", units=r"m\(^{2}\) s\(^{-2}\)", log=False)),
    ("Q", dict(kind="var:q", label="Specific humidity", panel="q", tag="q",
               short=r"$q$", units=r"kg kg\(^{-1}\)", log=True)),
    ("T", dict(kind="var:t", label="Temperature", panel="T", tag="T",
               short=r"$T$", units="K", log=False)),
    ("U", dict(kind="var:u", label="Zonal wind", panel="u", tag="u",
               short=r"$u$", units=r"m s\(^{-1}\)", log=False)),
    ("V", dict(kind="var:v", label="Meridional wind", panel="v", tag="v",
               short=r"$v$", units=r"m s\(^{-1}\)", log=False)),
    ("WSPD", dict(kind="wspd", label="Wind speed", panel="Wind speed", tag="WSPD",
                  short=r"\(|\mathbf{V}|\)", units=r"m s\(^{-1}\)", log=False)),
    ("DIV", dict(kind="div", label="Divergence", panel="DIV", tag="DIV",
                 short=r"\(\delta\)", units=r"s\(^{-1}\)", log=False)),
    ("VOR", dict(kind="vor", label="Relative vorticity", panel="VOR", tag="VOR",
                 short=r"\(\zeta\)", units=r"s\(^{-1}\)", log=False)),
    ("GEO", dict(kind="ageo", label="Geostrophic imbalance magnitude", panel="GEO", tag="GEO",
                 short="GEO", units=r"m s\(^{-1}\)", log=False)),
    ("MSE_DX", dict(kind="mse_dx", label="MSE x-gradient", panel=r"\(\partial\mathrm{MSE}/\partial x\)",
                    tag="∂MSE/∂x", short=r"\(\partial h/\partial x\)",
                    units=r"J kg\(^{-1}\) m\(^{-1}\)", log=False)),
    ("MSE_DY", dict(kind="mse_dy", label="MSE y-gradient", panel=r"\(\partial\mathrm{MSE}/\partial y\)",
                    tag="∂MSE/∂y", short=r"\(\partial h/\partial y\)",
                    units=r"J kg\(^{-1}\) m\(^{-1}\)", log=False)),
    ("MSE_GRAD", dict(kind="mse_grad", label="MSE gradient magnitude", panel="MSE gradient",
                      tag="MSE gradient", short=r"\(|\nabla h|\)",
                      units=r"J kg\(^{-1}\) m\(^{-1}\)", log=True)),
])

ACTIVE_EXTREME_VARS = [
    "PHI", "Q", "T", "U", "V", "WSPD", "DIV", "VOR", "GEO", "MSE_DX", "MSE_DY", "MSE_GRAD"
]

SUMMARY_PRINT_EVARS = [
    "WSPD", "PHI", "Q", "T", "U", "V", "DIV", "VOR", "GEO", "MSE_DX", "MSE_DY"
]

RETURN_CURVE_EVARS = ["PHI", "Q", "T", "WSPD", "DIV", "VOR", "GEO", "MSE_GRAD"]
EVENT_FIG_EVARS = ["PHI", "Q", "T", "WSPD", "DIV", "VOR", "GEO", "MSE_GRAD"]

AREA_WEIGHTED = False

CI_QUANTILE_LEVELS = [0.90, 0.95, 0.98, 0.99, 0.995, 0.998,
                      0.999, 0.9995, 0.9999, 0.99995, 0.99999]
QQ_BODY_N = 40
QQ_TAIL_N = 140
KEY_QUANTILE_LEVELS = [0.90, 0.99, 0.999, 0.9999]
EVENT_QUANTILE_LEVELS = [0.99, 0.999, 0.9999]

NBINS_BODY = 200_000
BODY_RANGE_PAD = 0.20
TAIL_GRID_M = 600
TAIL_P_MAX = 0.25
TAIL_P_MIN = 1.0e-5
TAIL_GRID_EXT = 10
MIN_TAIL_EFF_SAMPLES = 20.0

ENABLE_BOOTSTRAP = True
BOOTSTRAP_N_RESAMPLES = 2000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_CI_METHOD = "BCa"
BOOTSTRAP_SCHEME = "circular_block"
BOOTSTRAP_BLOCK_LENGTH = "auto"
BOOTSTRAP_JACKKNIFE = "auto"
BOOTSTRAP_SEED = 12345
BOOTSTRAP_CHUNK = 256

BASELINES_FOR_DIFF = ["cubic spline", "noPhy"]

COLLECT_PAIRED_TAIL_SAMPLES = True
PAIRED_SCATTER_EVAR = "WSPD"
PAIRED_SCATTER_TRUTH_Q = 0.99
PAIRED_SCATTER_MAX_POINTS = 1_500_000

COMPUTE_PIXELWISE_MAPS = True
PIXELWISE_QUANTILES = [0.99]
PIXELWISE_MAP_EVARS = ["WSPD"]

STATEMENT_EVAR = "WSPD"
STATEMENT_LEVEL = 0.999
STATEMENT_ORDER = ["cubic spline", "noPhy", "AllPhy"]

OUT_DIR = "extremes_out"
FIG_DIR = os.path.join(OUT_DIR, "figs")
CSV_QUANTILE_LONG = os.path.join(OUT_DIR, "extremes_quantile_long.csv")
CSV_QUANTILE_WIDE = os.path.join(OUT_DIR, "extremes_quantile_bias_wide.csv")
CSV_PAIRED_DIFF = os.path.join(OUT_DIR, "extremes_paired_deficit_test.csv")
CSV_EVENT_SCORES = os.path.join(OUT_DIR, "extremes_event_scores.csv")
CSV_CONDITIONAL = os.path.join(OUT_DIR, "extremes_conditional_error.csv")
CSV_FIELDMAX = os.path.join(OUT_DIR, "extremes_field_maxima.csv")
CSV_QQ_CURVE_DATA = os.path.join(OUT_DIR, "qq_curve_data.csv")
CSV_SUMMARY_ALLVARS = os.path.join(OUT_DIR, "extremes_bias_summary_all_vars.csv")
CSV_FIG_RETURN_CURVES = os.path.join(OUT_DIR, "Fig_tail_return_curves.csv")
TXT_STATEMENTS = os.path.join(OUT_DIR, "extremes_statements.txt")
NPZ_BOOT = os.path.join(OUT_DIR, "extremes_bootstrap_distributions.npz")

TRUTH_KEY = "ERA5"


# =============================================================================
# 1) Utilities / Scaler / Checkpoint Loading
# =============================================================================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class ChannelScaler:
    def __init__(self):
        self.mean = None
        self.std = None


def robust_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_scaler_stats(path):
    obj = joblib.load(path)
    if not (isinstance(obj, dict) and "scaler_X" in obj and "scaler_Y" in obj):
        raise ValueError("Invalid scalers.joblib format: expected scaler_X and scaler_Y")
    sx, sy = obj["scaler_X"], obj["scaler_Y"]
    xm = np.asarray(sx.mean, np.float32); xs = np.asarray(sx.std, np.float32)
    ym = np.asarray(sy.mean, np.float32); ys = np.asarray(sy.std, np.float32)
    xs = np.where(np.abs(xs) < 1e-12, 1.0, xs)
    ys = np.where(np.abs(ys) < 1e-12, 1.0, ys)
    return {"mean": xm, "std": xs}, {"mean": ym, "std": ys}


def standardize_batch(x, mean, std):
    return ((x - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)).astype(np.float32)


def inverse_standardize_batch(y, mean, std):
    return (y * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)).astype(np.float32)


def read_batch_from_ds(ds, variables, s, e):
    return np.stack([ds[v].isel(time=slice(s, e)).values.astype(np.float32)
                     for v in variables], axis=1)


# =============================================================================
# 2) Deep Learning Models
# =============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.drop(self.relu(self.conv1(x)))) + x


class ResidualCorrectionBackbone(nn.Module):
    def __init__(self, in_channels=5, out_channels=5, num_features=64, num_res_blocks=8,
                 upscale_factor=2, dropout_rate=0.0, residual_scale=1.0, zero_init_last=False):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.shallow_feature = nn.Conv2d(in_channels, num_features, 9, padding=4)
        self.deep_features = nn.Sequential(
            *[ResidualBlock(num_features, dropout_rate) for _ in range(num_res_blocks)])
        self.transition = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (upscale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(upscale_factor))
        self.post_shuffle_fuse = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.post_shuffle_act = nn.ReLU(inplace=True)
        self.output_conv = nn.Conv2d(num_features, out_channels, 3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if zero_init_last:
            nn.init.zeros_(self.output_conv.weight)
            if self.output_conv.bias is not None:
                nn.init.zeros_(self.output_conv.bias)

    def forward(self, x):
        s = self.shallow_feature(x)
        d = self.transition(self.deep_features(s))
        up = self.post_shuffle_act(self.post_shuffle_fuse(self.upsample(d + s)))
        return self.output_conv(up) * self.residual_scale


class BicubicResidualSRModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_res_blocks, upscale_factor,
                 dropout_rate, scaler_X_mean, scaler_X_std, scaler_Y_mean, scaler_Y_std,
                 align_corners=False, residual_scale=1.0, zero_init_last=False):
        super().__init__()
        self.upscale_factor = int(upscale_factor)
        self.align_corners = bool(align_corners)
        self.backbone = ResidualCorrectionBackbone(
            in_channels, out_channels, num_features, num_res_blocks,
            upscale_factor, dropout_rate, residual_scale, zero_init_last)
        xm = np.asarray(scaler_X_mean, np.float32); xs = np.asarray(scaler_X_std, np.float32)
        ym = np.asarray(scaler_Y_mean, np.float32); ys = np.asarray(scaler_Y_std, np.float32)
        self.register_buffer("x2y_alpha", torch.tensor(xs / ys, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("x2y_beta", torch.tensor((xm - ym) / ys, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x_std):
        base = F.interpolate(x_std, scale_factor=self.upscale_factor,
                             mode="bicubic", align_corners=self.align_corners)
        base = base * self.x2y_alpha + self.x2y_beta
        return base + self.backbone(x_std)


def infer_model_hparams_from_state_dict(sd):
    in_ch = sd["backbone.shallow_feature.weight"].shape[1]
    nf = sd["backbone.shallow_feature.weight"].shape[0]
    out_ch = sd["backbone.output_conv.weight"].shape[0]
    idx = [int(k.split(".")[2]) for k in sd
           if k.startswith("backbone.deep_features.") and k.endswith(".conv1.weight")]
    nrb = (max(idx) + 1) if idx else 0
    up = sd["backbone.upsample.0.weight"].shape[0]
    return dict(in_channels=in_ch, out_channels=out_ch, num_features=nf,
                num_res_blocks=nrb, upscale_factor=int(round(math.sqrt(up / float(nf)))))


def load_one_model(ckpt_path, sx, sy, device):
    ckpt = robust_torch_load(ckpt_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    hp = infer_model_hparams_from_state_dict(sd)
    meta = ckpt if isinstance(ckpt, dict) else {}
    model = BicubicResidualSRModel(
        in_channels=hp["in_channels"], out_channels=hp["out_channels"],
        num_features=hp["num_features"], num_res_blocks=hp["num_res_blocks"],
        upscale_factor=hp["upscale_factor"], dropout_rate=0.0,
        scaler_X_mean=np.asarray(meta.get("scaler_X_mean", sx["mean"]), np.float32),
        scaler_X_std=np.asarray(meta.get("scaler_X_std", sx["std"]), np.float32),
        scaler_Y_mean=np.asarray(meta.get("scaler_Y_mean", sy["mean"]), np.float32),
        scaler_Y_std=np.asarray(meta.get("scaler_Y_std", sy["std"]), np.float32),
        align_corners=bool(meta.get("bicubic_align_corners", False)),
        residual_scale=float(meta.get("residual_scale", 1.0)),
        zero_init_last=bool(meta.get("zero_init_last", False)))
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model


# =============================================================================
# 3) Cubic Spline Interpolation & Spherical Physics Operators
# =============================================================================
def to_increasing_axis(a):
    a = np.asarray(a, np.float64)
    return (a, False) if a[1] > a[0] else (a[::-1].copy(), True)


class CubicSplineInterpolator2D:
    def __init__(self, lat_lr, lon_lr, lat_hr, lon_hr, kx=3, ky=3):
        self.lat_lr_inc, self.f_lat_lr = to_increasing_axis(lat_lr)
        self.lon_lr_inc, self.f_lon_lr = to_increasing_axis(lon_lr)
        self.lat_hr_inc, self.f_lat_hr = to_increasing_axis(lat_hr)
        self.lon_hr_inc, self.f_lon_hr = to_increasing_axis(lon_hr)
        self.kx, self.ky = int(kx), int(ky)
        self.h_hr, self.w_hr = len(lat_hr), len(lon_hr)

    def __call__(self, x):
        b, c = x.shape[0], x.shape[1]
        out = np.empty((b, c, self.h_hr, self.w_hr), np.float32)
        for i in range(b):
            for j in range(c):
                f = x[i, j]
                if self.f_lat_lr: f = f[::-1, :]
                if self.f_lon_lr: f = f[:, ::-1]
                sp = RectBivariateSpline(self.lat_lr_inc, self.lon_lr_inc,
                                         np.ascontiguousarray(f, np.float64),
                                         kx=self.kx, ky=self.ky, s=0.0)
                p = sp(self.lat_hr_inc, self.lon_hr_inc)
                if self.f_lat_hr: p = p[::-1, :]
                if self.f_lon_hr: p = p[:, ::-1]
                out[i, j] = p.astype(np.float32)
        return out


class SphericalPhysicsOps:
    def __init__(self, latitudes, longitudes):
        lat = np.asarray(latitudes, np.float64); lon = np.asarray(longitudes, np.float64)
        lat_rad, lon_rad = np.deg2rad(lat), np.deg2rad(lon)
        dlat = np.abs(np.mean(np.diff(lat_rad))); dlon = np.abs(np.mean(np.diff(lon_rad)))
        self.lat_decreasing = bool(lat[1] < lat[0])
        self.inv_dy = 1.0 / (EARTH_RADIUS * dlat)
        self.inv_dx = 1.0 / np.maximum(EARTH_RADIUS * np.cos(lat_rad) * dlon, 100.0)
        self.tan_over_r = np.clip(np.tan(lat_rad), -TAN_CLIP, TAN_CLIP) / EARTH_RADIUS
        f = 2.0 * OMEGA * np.sin(lat_rad)
        m = np.abs(f) < F_CORIOLIS_ABS_MIN
        f[m] = np.where(f[m] >= 0, F_CORIOLIS_ABS_MIN, -F_CORIOLIS_ABS_MIN)
        self.inv_f = 1.0 / f

    def ddx(self, a):
        l = np.concatenate([a[:, :, :1], a[:, :, :-1]], 2)
        r = np.concatenate([a[:, :, 1:], a[:, :, -1:]], 2)
        return 0.5 * (r - l) * self.inv_dx[None, :, None]

    def ddy(self, a):
        t = np.concatenate([a[:, :1, :], a[:, :-1, :]], 1)
        b = np.concatenate([a[:, 1:, :], a[:, -1:, :]], 1)
        return 0.5 * ((t - b) if self.lat_decreasing else (b - t)) * self.inv_dy

    def divergence(self, u, v):
        return self.ddx(u) + self.ddy(v) - v * self.tan_over_r[None, :, None]

    def vorticity(self, u, v):
        return self.ddx(v) - self.ddy(u) + u * self.tan_over_r[None, :, None]

    def geostrophic_wind(self, z):
        return (-self.inv_f[None, :, None] * self.ddy(z),
                self.inv_f[None, :, None] * self.ddx(z))

    def moist_static_energy_gradients(self, t, z, q):
        h = CP_AIR * t + z + LV_WATER * q
        return self.ddx(h), self.ddy(h)


class ExtremeFieldFactory:
    def __init__(self, ops, active_names):
        self.ops = ops
        self.active = list(active_names)
        kinds = {EXTREME_SPECS[n]["kind"] for n in self.active}
        self.need_wspd = "wspd" in kinds
        self.need_div = any(k in kinds for k in ("div", "abs_div"))
        self.need_vor = any(k in kinds for k in ("vor", "abs_vor"))
        self.need_ageo = "ageo" in kinds
        self.need_geo_mag = "geo_mag" in kinds
        self.need_mse = any(k in kinds for k in ("mse_grad", "mse_dx", "mse_dy"))

    def __call__(self, batch5):
        z, q, t, u, v = (batch5[:, i] for i in range(5))
        out = {}
        wspd = np.sqrt(u * u + v * v) if (self.need_wspd or self.need_ageo or self.need_geo_mag) else None
        div = self.ops.divergence(u, v) if self.need_div else None
        vor = self.ops.vorticity(u, v) if self.need_vor else None
        ug = vg = None
        if self.need_ageo or self.need_geo_mag:
            ug, vg = self.ops.geostrophic_wind(z)
        gx = gy = None
        if self.need_mse:
            gx, gy = self.ops.moist_static_energy_gradients(t, z, q)

        for n in self.active:
            k = EXTREME_SPECS[n]["kind"]
            if k == "wspd":
                out[n] = wspd
            elif k.startswith("var:"):
                out[n] = {"z": z, "q": q, "t": t, "u": u, "v": v}[k.split(":")[1]]
            elif k == "div":
                out[n] = div
            elif k == "abs_div":
                out[n] = np.abs(div)
            elif k == "vor":
                out[n] = vor
            elif k == "abs_vor":
                out[n] = np.abs(vor)
            elif k == "ageo":
                out[n] = np.sqrt((u - ug) ** 2 + (v - vg) ** 2)
            elif k == "geo_mag":
                out[n] = np.sqrt(ug * ug + vg * vg)
            elif k == "mse_dx":
                out[n] = gx
            elif k == "mse_dy":
                out[n] = gy
            elif k == "mse_grad":
                out[n] = np.sqrt(gx * gx + gy * gy)
            else:
                raise ValueError(k)
            out[n] = np.ascontiguousarray(out[n], dtype=np.float32)
        return out


# =============================================================================
# 4) Distribution Accumulators
# =============================================================================
class BodyHistogram:
    def __init__(self, lo, hi, nbins):
        self.lo = float(lo); self.hi = float(hi); self.nb = int(nbins)
        self.bw = (self.hi - self.lo) / self.nb
        self.hist = np.zeros(self.nb, np.float64)
        self.under = 0.0
        self.over = 0.0
        self.vmin = np.inf
        self.vmax = -np.inf
        self.cum = None

    def update(self, x_flat, w_flat):
        self.vmin = min(self.vmin, float(x_flat.min()))
        self.vmax = max(self.vmax, float(x_flat.max()))
        k = np.floor((x_flat - self.lo) / self.bw).astype(np.int64)
        mu = k < 0
        mo = k >= self.nb
        if mu.any():
            self.under += float(w_flat[mu].sum()) if w_flat is not None else float(mu.sum())
        if mo.any():
            self.over += float(w_flat[mo].sum()) if w_flat is not None else float(mo.sum())
        ok = ~(mu | mo)
        if ok.all():
            self.hist += np.bincount(k, weights=w_flat, minlength=self.nb)[:self.nb]
        else:
            self.hist += np.bincount(k[ok],
                                     weights=(w_flat[ok] if w_flat is not None else None),
                                     minlength=self.nb)[:self.nb]

    def finalize(self):
        self.cum = np.cumsum(self.hist) + self.under
        self.total = float(self.cum[-1] + self.over)
        return self

    def quantile(self, q):
        if self.cum is None:
            self.finalize()
        if self.total <= 0:
            return np.nan
        target = float(q) * self.total
        if target <= self.under:
            return self.vmin
        if target >= self.total - self.over:
            return self.vmax
        j = int(np.searchsorted(self.cum, target, side="left"))
        j = min(max(j, 0), self.nb - 1)
        c_lo = self.under if j == 0 else self.cum[j - 1]
        wj = self.hist[j]
        frac = 0.5 if wj <= 0 else (target - c_lo) / wj
        v = self.lo + (j + float(np.clip(frac, 0.0, 1.0))) * self.bw
        return float(min(max(v, self.vmin), self.vmax))


class TailExceedance:
    def __init__(self, theta, n_time, store_per_time=True):
        self.theta = np.asarray(theta, np.float64)
        self.M = self.theta.size
        self.n_time = int(n_time)
        self.store_per_time = bool(store_per_time)
        if store_per_time:
            self.hist = np.zeros((self.n_time, self.M + 1), np.float32)
        else:
            self.hist = np.zeros((1, self.M + 1), np.float64)

    def update(self, x_bhw, t0, w_flat_1d, hw):
        b = x_bhw.shape[0]
        k = np.searchsorted(self.theta, x_bhw.reshape(-1), side="left")
        if self.store_per_time:
            offs = np.repeat(np.arange(b, dtype=np.int64) * (self.M + 1), hw)
            cnt = np.bincount(offs + k, weights=w_flat_1d,
                             minlength=b * (self.M + 1))[: b * (self.M + 1)]
            self.hist[t0:t0 + b] += cnt.reshape(b, self.M + 1).astype(np.float32)
        else:
            self.hist[0] += np.bincount(k, weights=w_flat_1d, minlength=self.M + 1)[: self.M + 1]

    def total_hist(self):
        return self.hist.sum(axis=0).astype(np.float64)


def exceedance_from_hist(hist_2d):
    h = np.asarray(hist_2d, np.float64)
    tot = h.sum(axis=-1, keepdims=True)
    return (tot - np.cumsum(h, axis=-1))[..., :-1], tot[..., 0]


def tail_quantile_from_exceedance(theta, exceed, total, p_target):
    theta = np.asarray(theta, np.float64)
    exceed = np.atleast_2d(np.asarray(exceed, np.float64))
    total = np.asarray(total, np.float64).reshape(-1, 1)
    M = theta.size
    p = exceed / np.maximum(total, 1e-300)
    pt = float(p_target)

    idx = (p >= pt).sum(axis=-1) - 1
    valid = (idx >= 0) & (idx <= M - 2)
    i = np.clip(idx, 0, M - 2)
    j = i + 1
    p_i = np.take_along_axis(p, i[:, None], -1)[:, 0]
    p_j = np.take_along_axis(p, j[:, None], -1)[:, 0]
    t_i, t_j = theta[i], theta[j]

    with np.errstate(divide="ignore", invalid="ignore"):
        f_log = (np.log(pt) - np.log(p_i)) / (np.log(p_j) - np.log(p_i))
        f_lin = (p_i - pt) / (p_i - p_j)
    use_log = (p_j > 0) & (p_i > p_j)
    f = np.where(use_log, f_log, np.where(p_i > p_j, f_lin, 0.0))
    f = np.clip(np.nan_to_num(f, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    out = t_i + f * (t_j - t_i)
    out = np.where(valid, out, np.nan)
    return out


# =============================================================================
# 5) Bootstrap Engine
# =============================================================================
def resolve_block_length(n, spec):
    if isinstance(spec, str):
        s = spec.lower()
        L = int(max(1, round(n ** (1 / 3)))) if s == "auto" else \
            int(max(1, round(math.sqrt(n)))) if s == "sqrt" else None
        if L is None:
            raise ValueError(spec)
    else:
        L = int(spec)
    return int(np.clip(L, 1, max(1, n)))


def gen_bootstrap_indices(n, n_rep, scheme, L, rng):
    if scheme == "iid":
        return rng.integers(0, n, size=(n_rep, n), dtype=np.int64)
    if scheme in ("moving_block", "circular_block"):
        nb = int(math.ceil(n / L))
        offs = np.arange(L, dtype=np.int64)[None, None, :]
        if scheme == "moving_block":
            st = rng.integers(0, max(1, n - L + 1), size=(n_rep, nb)).astype(np.int64)
            return np.minimum((st[:, :, None] + offs).reshape(n_rep, -1)[:, :n], n - 1)
        st = rng.integers(0, n, size=(n_rep, nb)).astype(np.int64)
        return ((st[:, :, None] + offs) % n).reshape(n_rep, -1)[:, :n]
    if scheme == "stationary":
        t = np.arange(n, dtype=np.int64)[None, :]
        nb = rng.random((n_rep, n)) < (1.0 / L)
        nb[:, 0] = True
        st = rng.integers(0, n, size=(n_rep, n)).astype(np.int64)
        last = np.maximum.accumulate(np.where(nb, np.broadcast_to(t, nb.shape), -1), axis=1)
        off = np.broadcast_to(t, nb.shape) - last
        return (np.take_along_axis(st, last, axis=1) + off) % n
    raise ValueError(scheme)


def indices_to_counts(idx, n):
    r = idx.shape[0]
    c = np.empty((r, n), np.float64)
    for i in range(r):
        c[i] = np.bincount(idx[i], minlength=n).astype(np.float64)
    return c


class BootstrapEngine:
    def __init__(self, n_samples, n_rep, scheme, block_length, jackknife_mode,
                 seed, chunk, confidence, ci_method):
        self.n = int(n_samples); self.n_rep = int(n_rep); self.scheme = scheme
        self.L = resolve_block_length(self.n, block_length)
        self.jk_mode = ("delete1" if scheme == "iid" else "block") \
            if jackknife_mode == "auto" else jackknife_mode
        self.jk_starts = None if self.jk_mode == "delete1" else \
            np.arange(0, self.n, self.L, dtype=np.int64)
        self.seed = int(seed); self.chunk = int(chunk)
        self.confidence = float(confidence); self.alpha = 1.0 - float(confidence)
        self.ci_method = ci_method

    def iter_count_chunks(self):
        rng = np.random.default_rng(self.seed)
        pos = 0
        while pos < self.n_rep:
            r = min(self.chunk, self.n_rep - pos)
            idx = gen_bootstrap_indices(self.n, r, self.scheme, self.L, rng)
            yield pos, indices_to_counts(idx, self.n)
            pos += r

    def jackknife_counts(self):
        if self.jk_mode == "delete1":
            J = self.n
            W = np.ones((J, self.n), np.float64)
            W[np.arange(J), np.arange(J)] = 0.0
            return W
        starts = self.jk_starts
        J = starts.size
        W = np.ones((J, self.n), np.float64)
        for j, s in enumerate(starts):
            e = min(s + self.L, self.n)
            W[j, s:e] = 0.0
        return W

    def ci(self, theta_hat, boot, jack=None, method=None):
        method = method or self.ci_method
        a = self.alpha
        boot = np.asarray(boot, np.float64)
        good = boot[np.isfinite(boot)]
        res = dict(CI_Low=np.nan, CI_High=np.nan, CI_Method=method,
                   Boot_Mean=np.nan, Boot_SE=np.nan, Boot_Bias=np.nan,
                   N_Valid=int(good.size), Note="")
        if good.size < 20 or not np.isfinite(theta_hat):
            res["Note"] = "insufficient_valid_resamples"
            return res
        res["Boot_Mean"] = float(good.mean())
        res["Boot_SE"] = float(good.std(ddof=1))
        res["Boot_Bias"] = float(good.mean() - theta_hat)
        lo_p = float(np.percentile(good, 100 * a / 2))
        hi_p = float(np.percentile(good, 100 * (1 - a / 2)))
        if method == "percentile":
            res["CI_Low"], res["CI_High"] = lo_p, hi_p; return res
        if method == "basic":
            res["CI_Low"], res["CI_High"] = 2 * theta_hat - hi_p, 2 * theta_hat - lo_p
            return res
        prop = (np.sum(good < theta_hat) + 0.5 * np.sum(good == theta_hat)) / good.size
        prop = float(np.clip(prop, 1 / (2 * good.size), 1 - 1 / (2 * good.size)))
        z0 = _norm.ppf(prop)
        acc = 0.0
        if jack is not None:
            jk = np.asarray(jack, np.float64); jk = jk[np.isfinite(jk)]
            if jk.size >= 3:
                d = jk.mean() - jk
                s2 = np.sum(d ** 2)
                if s2 > 0:
                    acc = float(np.sum(d ** 3) / (6.0 * s2 ** 1.5))
        if not np.isfinite(acc):
            acc = 0.0
        zl, zh = _norm.ppf(a / 2), _norm.ppf(1 - a / 2)

        def adj(z):
            dd = 1.0 - acc * (z0 + z)
            return np.nan if abs(dd) < 1e-12 else _norm.cdf(z0 + (z0 + z) / dd)

        a1, a2 = adj(zl), adj(zh)
        if not (np.isfinite(a1) and np.isfinite(a2)) or a1 >= a2:
            res["CI_Low"], res["CI_High"] = lo_p, hi_p
            res["CI_Method"] = "percentile(BCa_fallback)"; res["Note"] = "BCa_degenerate"
            return res
        lo = float(np.percentile(good, 100 * float(np.clip(a1, 1e-6, 1 - 1e-6))))
        hi = float(np.percentile(good, 100 * float(np.clip(a2, 1e-6, 1 - 1e-6))))
        if lo > hi:
            lo, hi = hi, lo
        res["CI_Low"], res["CI_High"] = lo, hi
        res["BCa_z0"], res["BCa_a"] = float(z0), float(acc)
        return res


def bootstrap_pvalue(diff_boot):
    d = np.asarray(diff_boot, np.float64); d = d[np.isfinite(d)]
    if d.size < 20:
        return np.nan
    b = d.size
    return float(min(1.0, 2 * min((1 + np.sum(d <= 0)) / (b + 1),
                                  (1 + np.sum(d >= 0)) / (b + 1))))


def bh_fdr(p):
    p = np.asarray(p, np.float64)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p); pv = p[ok]; m = pv.size
    if m == 0:
        return q
    o = np.argsort(pv)
    r = pv[o] * m / (np.arange(m) + 1.0)
    r = np.minimum.accumulate(r[::-1])[::-1]
    qq = np.empty(m); qq[o] = np.clip(r, 0, 1); q[ok] = qq
    return q


# =============================================================================
# 6) Statistical Containers
# =============================================================================
CONT_KEYS = ["hit", "miss", "fa", "cn", "w_cond", "err_cond",
             "abs_cond", "sq_cond", "true_cond", "pred_cond"]


class SeriesExtremeStats:
    def __init__(self, name, evars, n_time, body_ranges, theta_map,
                 n_event_levels, store_per_time, is_truth):
        self.name = name
        self.is_truth = bool(is_truth)
        self.body = {ev: BodyHistogram(*body_ranges[ev], NBINS_BODY) for ev in evars}
        self.tail = {ev: TailExceedance(theta_map[ev], n_time, store_per_time) for ev in evars}
        self.blockmax = {ev: np.full(n_time, -np.inf, np.float64) for ev in evars}
        self.cont = None if is_truth else {
            ev: np.zeros((n_time, n_event_levels, len(CONT_KEYS)), np.float64) for ev in evars}
        self.paired = {}


# =============================================================================
# 7) Data Scanning & Feature Extraction
# =============================================================================
def build_weights(lat_hr, n_lon, area_weighted):
    if area_weighted:
        w = np.cos(np.deg2rad(np.asarray(lat_hr, np.float64)))
        w = np.maximum(w, 1e-6)
    else:
        w = np.ones(len(lat_hr), np.float64)
    return w, float(w.sum() * n_lon)


def pass1_truth_range(ds_y, factory, n_total, batch_size, evars):
    rng = {ev: [np.inf, -np.inf] for ev in evars}
    for st in tqdm(range(0, n_total, batch_size), desc="Pass 1a: Truth range estimation"):
        y = read_batch_from_ds(ds_y, VARIABLES, st, min(st + batch_size, n_total))
        for ev, f in factory(y).items():
            rng[ev][0] = min(rng[ev][0], float(f.min()))
            rng[ev][1] = max(rng[ev][1], float(f.max()))
    out = {}
    for ev in evars:
        lo, hi = rng[ev]
        span = max(hi - lo, 1e-12)
        out[ev] = (lo - BODY_RANGE_PAD * span, hi + BODY_RANGE_PAD * span)
    return out


def pass1_truth_hist(ds_y, factory, n_total, batch_size, evars, body_ranges,
                     w_row, area_weighted):
    H = {ev: BodyHistogram(*body_ranges[ev], NBINS_BODY) for ev in evars}
    for st in tqdm(range(0, n_total, batch_size), desc="Pass 1b: Truth histogram construction"):
        ed = min(st + batch_size, n_total)
        y = read_batch_from_ds(ds_y, VARIABLES, st, ed)
        fields = factory(y)
        wflat = None
        if area_weighted:
            any_f = next(iter(fields.values()))
            wflat = np.broadcast_to(w_row[None, :, None], any_f.shape).reshape(-1).astype(np.float64)
        for ev, f in fields.items():
            H[ev].update(f.reshape(-1).astype(np.float64), wflat)
    for ev in evars:
        H[ev].finalize()
    return H


def make_theta_grid(truth_hist, evars):
    theta_map, pgrid_map = {}, {}
    p_grid = np.geomspace(TAIL_P_MAX, TAIL_P_MIN, TAIL_GRID_M)
    for ev in evars:
        th = np.array([truth_hist[ev].quantile(1.0 - p) for p in p_grid], np.float64)
        th = np.maximum.accumulate(th)
        scale = max(abs(th[-1]), abs(th[0]), 1e-30)
        eps = 1e-9 * scale
        for i in range(1, th.size):
            if th[i] <= th[i - 1]:
                th[i] = th[i - 1] + eps
        d = max(th[-1] - th[-2], eps)
        ext = th[-1] + d * np.cumsum(1.7 ** np.arange(1, TAIL_GRID_EXT + 1))
        theta_map[ev] = np.concatenate([th, ext])
        pgrid_map[ev] = p_grid
    return theta_map, pgrid_map


def pass2_all_series(ds_x, ds_y, models, cubic, factory, n_total, batch_size,
                     evars, body_ranges, theta_map, event_thresholds,
                     sx, sy, w_row, area_weighted, device, store_per_time,
                     paired_cfg, pixelwise_cfg):
    series_names = [TRUTH_KEY, "cubic spline"] + list(models.keys())
    n_ev_lv = len(EVENT_QUANTILE_LEVELS)
    S = OrderedDict()
    for nm in series_names:
        S[nm] = SeriesExtremeStats(nm, evars, n_total, body_ranges, theta_map,
                                   n_ev_lv, store_per_time, is_truth=(nm == TRUTH_KEY))

    rng_pair = np.random.default_rng(20240501)
    p_keep = paired_cfg.get("p_keep", 1.0)
    pev = paired_cfg.get("evar", None)

    pw = {}
    if pixelwise_cfg["enable"]:
        m_keep = pixelwise_cfg["m_keep"]
        for nm in series_names:
            pw[nm] = {ev: np.full((pixelwise_cfg["hw"], m_keep), -np.inf, np.float32)
                      for ev in pixelwise_cfg["evars"]}

    xm, xs_ = sx["mean"], sx["std"]
    ym, ys_ = sy["mean"], sy["std"]

    for st in tqdm(range(0, n_total, batch_size), desc="Pass 2: Full-series extreme value analysis"):
        ed = min(st + batch_size, n_total)
        b = ed - st
        x_raw = read_batch_from_ds(ds_x, VARIABLES, st, ed)
        y_raw = read_batch_from_ds(ds_y, VARIABLES, st, ed)

        preds = OrderedDict()
        preds[TRUTH_KEY] = y_raw
        preds["cubic spline"] = cubic(x_raw)
        if models:
            xt = torch.from_numpy(standardize_batch(x_raw, xm, xs_)).to(device)
            with torch.no_grad():
                for nm, mdl in models.items():
                    p = mdl(xt).detach().cpu().numpy().astype(np.float32)
                    preds[nm] = inverse_standardize_batch(p, ym, ys_)

        fields = {nm: factory(arr) for nm, arr in preds.items()}
        shp = fields[TRUTH_KEY][evars[0]].shape
        hw = shp[1] * shp[2]
        if area_weighted:
            wflat = np.broadcast_to(w_row[None, :, None], shp).reshape(-1).astype(np.float64)
            w2d = wflat.reshape(b, hw)
        else:
            wflat = np.ones(b * hw, np.float64)
            w2d = wflat.reshape(b, hw)

        for nm in series_names:
            for ev in evars:
                f = fields[nm][ev]
                S[nm].body[ev].update(f.reshape(-1).astype(np.float64), wflat)
                S[nm].tail[ev].update(f, st, wflat, hw)
                S[nm].blockmax[ev][st:ed] = f.reshape(b, -1).max(axis=1)

                if pixelwise_cfg["enable"] and ev in pixelwise_cfg["evars"]:
                    m_keep = pixelwise_cfg["m_keep"]
                    cat = np.concatenate([pw[nm][ev], f.reshape(b, hw).T], axis=1)
                    kth = cat.shape[1] - m_keep
                    pw[nm][ev] = np.partition(cat, kth, axis=1)[:, kth:].astype(np.float32)

        for nm in series_names:
            if nm == TRUTH_KEY:
                continue
            for ev in evars:
                tf = fields[TRUTH_KEY][ev].reshape(b, hw)
                pf = fields[nm][ev].reshape(b, hw)
                err = pf - tf
                for li, q in enumerate(EVENT_QUANTILE_LEVELS):
                    u = event_thresholds[ev][li]
                    te = tf > u
                    pe = pf > u
                    C = S[nm].cont[ev]
                    C[st:ed, li, 0] += (w2d * (te & pe)).sum(1)
                    C[st:ed, li, 1] += (w2d * (te & ~pe)).sum(1)
                    C[st:ed, li, 2] += (w2d * (~te & pe)).sum(1)
                    C[st:ed, li, 3] += (w2d * (~te & ~pe)).sum(1)
                    wc = w2d * te
                    C[st:ed, li, 4] += wc.sum(1)
                    C[st:ed, li, 5] += (wc * err).sum(1)
                    C[st:ed, li, 6] += (wc * np.abs(err)).sum(1)
                    C[st:ed, li, 7] += (wc * err ** 2).sum(1)
                    C[st:ed, li, 8] += (wc * tf).sum(1)
                    C[st:ed, li, 9] += (wc * pf).sum(1)

                if paired_cfg["enable"] and ev == pev:
                    u = paired_cfg["threshold"]
                    m = tf > u
                    if m.any():
                        tv = tf[m]; pv = pf[m]
                        if p_keep < 1.0:
                            sel = rng_pair.random(tv.size) < p_keep
                            tv, pv = tv[sel], pv[sel]
                        if tv.size:
                            S[nm].paired.setdefault(ev, []).append(
                                np.stack([tv, pv], 1).astype(np.float32))

    for nm in series_names:
        for ev in evars:
            S[nm].body[ev].finalize()
        for ev, lst in list(S[nm].paired.items()):
            S[nm].paired[ev] = np.concatenate(lst, 0) if lst else np.zeros((0, 2), np.float32)

    return S, pw


# =============================================================================
# 8) Quantile Tables, CIs, and Paired Significance Tests
# =============================================================================
def resolvable_levels(levels, total_weight):
    return [q for q in levels if (1.0 - q) * total_weight >= MIN_TAIL_EFF_SAMPLES
            and (1.0 - q) <= TAIL_P_MAX]


def percentile_label(q):
    p = q * 100.0
    s = f"{p:.10f}".rstrip("0").rstrip(".")
    suf = "th"
    if s.endswith("1") and not s.endswith("11"): suf = "st"
    elif s.endswith("2") and not s.endswith("12"): suf = "nd"
    elif s.endswith("3") and not s.endswith("13"): suf = "rd"
    return f"{s}{suf}"


def compute_quantile_tables(S, evars, engine, total_weight, enable_boot):
    series = list(S.keys())
    models = [s for s in series if s != TRUTH_KEY]
    rows, diff_rows = [], []
    boot_store = {}

    for ev in evars:
        levels = resolvable_levels(CI_QUANTILE_LEVELS, total_weight)
        if not levels:
            continue

        v_hist = {s: np.array([S[s].body[ev].quantile(q) for q in levels]) for s in series}

        theta = S[TRUTH_KEY].tail[ev].theta
        v_theta = {}
        for s in series:
            h = S[s].tail[ev].total_hist()[None, :]
            e, t = exceedance_from_hist(h)
            v_theta[s] = np.array([tail_quantile_from_exceedance(theta, e, t, 1 - q)[0]
                                   for q in levels])

        boot_v = {s: np.full((engine.n_rep, len(levels)), np.nan) for s in series}
        jack_v = {}
        if enable_boot:
            Hm = {s: S[s].tail[ev].hist.astype(np.float64) for s in series}
            for pos, cnt in engine.iter_count_chunks():
                r = cnt.shape[0]
                for s in series:
                    pooled = cnt @ Hm[s]
                    e, t = exceedance_from_hist(pooled)
                    for li, q in enumerate(levels):
                        boot_v[s][pos:pos + r, li] = tail_quantile_from_exceedance(theta, e, t, 1 - q)
            Wjk = engine.jackknife_counts()
            for s in series:
                pooled = Wjk @ Hm[s]
                e, t = exceedance_from_hist(pooled)
                jack_v[s] = np.column_stack(
                    [tail_quantile_from_exceedance(theta, e, t, 1 - q) for q in levels]
                )
            del Hm

        for li, q in enumerate(levels):
            vt = v_hist[TRUTH_KEY][li]
            for s in series:
                shift = v_hist[s][li] - v_theta[s][li]
                r = engine.ci(v_theta[s][li] + shift,
                              boot_v[s][:, li] + shift,
                              (jack_v[s][:, li] + shift) if s in jack_v else None) \
                    if enable_boot else dict(CI_Low=np.nan, CI_High=np.nan,
                                             CI_Method="none", Boot_SE=np.nan, Note="")
                relb = (v_hist[s][li] - vt) / abs(vt) * 100.0 if np.isfinite(vt) and vt != 0 else np.nan
                rows.append(dict(
                    EVar=ev, Level=q, Percentile=percentile_label(q), Series=s,
                    Display=DISPLAY_NAME_MAP.get(s, s),
                    Value=v_hist[s][li], Value_ERA5=vt,
                    Value_CI_Low=r["CI_Low"], Value_CI_High=r["CI_High"],
                    RelBias_pct=relb, CI_Method=r["CI_Method"], Note=r["Note"]))

            for s in models:
                shift_m = v_hist[s][li] - v_theta[s][li]
                shift_t = v_hist[TRUTH_KEY][li] - v_theta[TRUTH_KEY][li]
                bm = boot_v[s][:, li] + shift_m
                bt = boot_v[TRUTH_KEY][:, li] + shift_t
                rel_boot = (bm - bt) / np.abs(bt) * 100.0
                rel_hat = (v_hist[s][li] - vt) / abs(vt) * 100.0
                if enable_boot:
                    jm = jack_v[s][:, li] + shift_m
                    jt = jack_v[TRUTH_KEY][:, li] + shift_t
                    rel_jack = (jm - jt) / np.abs(jt) * 100.0
                    rr = engine.ci(rel_hat, rel_boot, rel_jack)
                    pv = bootstrap_pvalue(rel_boot)
                else:
                    rr = dict(CI_Low=np.nan, CI_High=np.nan, CI_Method="none", Boot_SE=np.nan, Note="")
                    pv = np.nan
                idx = next(i for i, rw in enumerate(rows[::-1])
                           if rw["EVar"] == ev and rw["Level"] == q and rw["Series"] == s)
                rows[len(rows) - 1 - idx].update(
                    RelBias_CI_Low=rr["CI_Low"], RelBias_CI_High=rr["CI_High"],
                    RelBias_pvalue=pv, RelBias_SE=rr["Boot_SE"])
                boot_store[(ev, q, s)] = rel_boot

            if enable_boot:
                for base in BASELINES_FOR_DIFF:
                    if base not in models:
                        continue
                    for s in models:
                        if s == base:
                            continue
                        db = np.abs(boot_store[(ev, q, s)]) - np.abs(boot_store[(ev, q, base)])
                        jm_s = (jack_v[s][:, li] + (v_hist[s][li] - v_theta[s][li]))
                        jm_b = (jack_v[base][:, li] + (v_hist[base][li] - v_theta[base][li]))
                        jt = jack_v[TRUTH_KEY][:, li] + (v_hist[TRUTH_KEY][li] - v_theta[TRUTH_KEY][li])
                        dj = np.abs((jm_s - jt) / np.abs(jt) * 100.0) - np.abs((jm_b - jt) / np.abs(jt) * 100.0)
                        d_hat = abs((v_hist[s][li] - vt) / abs(vt) * 100.0) - abs((v_hist[base][li] - vt) / abs(vt) * 100.0)
                        rr = engine.ci(d_hat, db, dj)
                        pv = bootstrap_pvalue(db)
                        diff_rows.append(dict(
                            EVar=ev, Level=q, Percentile=percentile_label(q),
                            Baseline=base, Baseline_Display=DISPLAY_NAME_MAP.get(base, base),
                            Model=s, Model_Display=DISPLAY_NAME_MAP.get(s, s),
                            AbsDeficit_Model_pct=abs((v_hist[s][li] - vt) / abs(vt) * 100.0),
                            AbsDeficit_Baseline_pct=abs((v_hist[base][li] - vt) / abs(vt) * 100.0),
                            Delta_AbsDeficit_pct=d_hat,
                            Delta_CI_Low=rr["CI_Low"], Delta_CI_High=rr["CI_High"],
                            p_value_boot=pv,
                            Significant_at_95=bool(np.isfinite(rr["CI_Low"]) and np.isfinite(rr["CI_High"]) and
                                                   (rr["CI_Low"] > 0 or rr["CI_High"] < 0)),
                            CI_Method=rr["CI_Method"]))

    df_long = pd.DataFrame(rows)
    df_diff = pd.DataFrame(diff_rows)
    if len(df_diff):
        df_diff["q_value_BH"] = np.nan
        for _, sub in df_diff.groupby(["EVar", "Level", "Baseline"], sort=False):
            df_diff.loc[sub.index, "q_value_BH"] = bh_fdr(sub["p_value_boot"].values)
    return df_long, df_diff, boot_store


# =============================================================================
# 9) Event Skill Scores & Field Maxima Evaluation
# =============================================================================
def _scores_from_cont(C):
    hit, miss, fa, cn = C[..., 0], C[..., 1], C[..., 2], C[..., 3]
    wc, se, sa, ss, stt, spp = (C[..., i] for i in range(4, 10))
    eps = 1e-30
    H = hit / np.maximum(hit + miss, eps)
    Fa = fa / np.maximum(fa + cn, eps)
    out = dict(
        POD=H, FAR=fa / np.maximum(hit + fa, eps),
        CSI=hit / np.maximum(hit + miss + fa, eps),
        FreqBias=(hit + fa) / np.maximum(hit + miss, eps))
    Hc = np.clip(H, 1e-12, 1 - 1e-12); Fc = np.clip(Fa, 1e-12, 1 - 1e-12)
    num = np.log(Fc) - np.log(Hc) - np.log(1 - Fc) + np.log(1 - Hc)
    den = np.log(Fc) + np.log(Hc) + np.log(1 - Fc) + np.log(1 - Hc)
    out["SEDI"] = num / np.where(np.abs(den) < 1e-12, np.nan, den)
    out["Cond_MeanBias"] = se / np.maximum(wc, eps)
    out["Cond_MAE"] = sa / np.maximum(wc, eps)
    out["Cond_RMSE"] = np.sqrt(np.maximum(ss / np.maximum(wc, eps), 0.0))
    out["Cond_RelBias_pct"] = se / np.where(np.abs(stt) < eps, np.nan, stt) * 100.0
    return out


EVENT_SCORE_KEYS = ["POD", "FAR", "CSI", "FreqBias", "SEDI",
                    "Cond_MeanBias", "Cond_MAE", "Cond_RMSE", "Cond_RelBias_pct"]


def compute_event_tables(S, evars, engine, enable_boot):
    rows = []
    models = [s for s in S if s != TRUTH_KEY]
    for ev in evars:
        for li, q in enumerate(EVENT_QUANTILE_LEVELS):
            for s in models:
                C = S[s].cont[ev][:, li, :]
                tot = C.sum(0)
                pt = _scores_from_cont(tot)
                boot = {k: np.full(engine.n_rep, np.nan) for k in EVENT_SCORE_KEYS}
                jack = {}
                if enable_boot:
                    for pos, cnt in engine.iter_count_chunks():
                        sc = _scores_from_cont(cnt @ C)
                        for k in EVENT_SCORE_KEYS:
                            boot[k][pos:pos + cnt.shape[0]] = sc[k]
                    jk = _scores_from_cont(engine.jackknife_counts() @ C)
                    jack = {k: jk[k] for k in EVENT_SCORE_KEYS}
                for k in EVENT_SCORE_KEYS:
                    r = engine.ci(float(pt[k]), boot[k], jack.get(k)) if enable_boot \
                        else dict(CI_Low=np.nan, CI_High=np.nan, CI_Method="none")
                    rows.append(dict(EVar=ev, Level=q, Percentile=percentile_label(q),
                                     Series=s, Display=DISPLAY_NAME_MAP.get(s, s),
                                     Score=k, Value=float(pt[k]),
                                     CI_Low=r["CI_Low"], CI_High=r["CI_High"],
                                     CI_Method=r["CI_Method"]))
    return pd.DataFrame(rows)


def compute_fieldmax_table(S, evars, engine, enable_boot):
    rows = []
    series = list(S.keys())
    for ev in evars:
        bm = {s: S[s].blockmax[ev] for s in series}
        for s in series:
            m_hat = float(np.mean(bm[s]))
            ratio_hat = float(np.mean(bm[s]) / np.mean(bm[TRUTH_KEY]))
            boot_m = np.full(engine.n_rep, np.nan)
            boot_r = np.full(engine.n_rep, np.nan)
            jack_m = jack_r = None
            if enable_boot:
                n = engine.n
                for pos, cnt in engine.iter_count_chunks():
                    mm = (cnt @ bm[s]) / n
                    mt = (cnt @ bm[TRUTH_KEY]) / n
                    boot_m[pos:pos + cnt.shape[0]] = mm
                    boot_r[pos:pos + cnt.shape[0]] = mm / mt
                Wjk = engine.jackknife_counts()
                nj = Wjk.sum(1)
                jm = (Wjk @ bm[s]) / nj
                jt = (Wjk @ bm[TRUTH_KEY]) / nj
                jack_m, jack_r = jm, jm / jt
            r1 = engine.ci(m_hat, boot_m, jack_m) if enable_boot else dict(CI_Low=np.nan, CI_High=np.nan)
            r2 = engine.ci(ratio_hat, boot_r, jack_r) if enable_boot else dict(CI_Low=np.nan, CI_High=np.nan)
            rows.append(dict(EVar=ev, Series=s, Display=DISPLAY_NAME_MAP.get(s, s),
                             Mean_FieldMax=m_hat, CI_Low=r1["CI_Low"], CI_High=r1["CI_High"],
                             Ratio_to_ERA5=ratio_hat,
                             Ratio_CI_Low=r2["CI_Low"], Ratio_CI_High=r2["CI_High"],
                             RelBias_pct=(ratio_hat - 1.0) * 100.0,
                             Global_Max=float(bm[s].max())))
    return pd.DataFrame(rows)


# =============================================================================
# 10) Visualization
# =============================================================================
W_SINGLE = 8.4 / 2.54
W_ONEHALF = 11.4 / 2.54
W_FULL = 16.9 / 2.54
H_MAX = 22.9 / 2.54

OKABE_ITO = ["#000000", "#0072B2", "#E69F00", "#D55E00",
             "#009E73", "#CC79A7", "#56B4E9", "#F0E442"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", (0, (5, 1.6)), (0, (1.6, 1.4)), (0, (4, 1.4, 1, 1.4)),
              (0, (7, 1.6, 1, 1.6)), (0, (2.5, 1.2))]


def set_james_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    prefer = ["Helvetica", "Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    fam = [f for f in prefer if f in have] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": fam,
        "mathtext.fontset": "dejavusans",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "grid.linewidth": 0.4,
        "lines.linewidth": 1.0, "lines.markersize": 3.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "legend.handlelength": 2.0,
        "legend.handletextpad": 0.5, "legend.labelspacing": 0.25,
        "axes.grid": False, "figure.dpi": 140,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })
    print(f"[style] Using font: {fam[0]}")
    return plt


def preferred_series_order(series_names):
    pref = [TRUTH_KEY, "cubic spline", "noPhy", "AllPhy"]
    out = [s for s in pref if s in series_names]
    out.extend([s for s in series_names if s not in out])
    return out


def series_label(s):
    return "Ground Truth" if s == TRUTH_KEY else DISPLAY_NAME_MAP.get(s, s)


def style_for(series_names):
    fixed = {
        TRUTH_KEY:     dict(color="k", marker="", ls="-",  lw=1.20, zorder=30),
        "cubic spline": dict(color="k", marker="s", ls="--", lw=1.00, zorder=29),
        "noPhy":        dict(color="#1f77b4", marker="o", ls="-", lw=1.05, zorder=28),
        "AllPhy":       dict(color="#d62728", marker="^", ls="-", lw=1.05, zorder=28),
    }
    st = {}
    k = 0
    for s in series_names:
        if s in fixed:
            st[s] = fixed[s].copy()
        else:
            st[s] = dict(
                color=OKABE_ITO[(k + 2) % len(OKABE_ITO)],
                marker=MARKERS[k % len(MARKERS)],
                ls=LINESTYLES[(k + 1) % len(LINESTYLES)],
                lw=1.0,
                zorder=20 + k
            )
            k += 1
    return st


def save_fig(fig, path_noext):
    fig.savefig(path_noext + ".pdf")
    fig.savefig(path_noext + ".png", dpi=600)
    print(f"    -> {path_noext}.pdf / .png")


def panel_letter(ax, i):
    ax.text(0.035, 0.965, f"({chr(97 + i)})", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, fontweight="bold")


def add_percentile_axis(ax, xlabel=True):
    ax.set_xscale("log")
    ax.invert_xaxis()
    ticks = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    ax.set_xticks(ticks)
    ax.set_xticklabels([percentile_label(1 - t).replace("th", "").replace("st", "") for t in ticks])
    if xlabel:
        ax.set_xlabel("Percentile")


def fig_qq_panels(plt, curves, evars, series, out_base, ncol=3,
                  key_levels=KEY_QUANTILE_LEVELS, with_inset=True):
    st = style_for(series)
    models = [s for s in preferred_series_order(series) if s != TRUTH_KEY]
    n = len(evars)
    ncol = min(ncol, max(1, n))
    nrow = int(math.ceil(n / ncol))
    pw = W_FULL if ncol >= 3 else (W_ONEHALF if ncol == 2 else W_SINGLE)
    fig, axes = plt.subplots(nrow, ncol, figsize=(pw, min(pw / ncol * nrow * 1.06, H_MAX)))
    axes = np.atleast_1d(axes).ravel()

    for i, ev in enumerate(evars):
        ax = axes[i]
        sp = EXTREME_SPECS[ev]
        q = curves[ev]["levels"]
        vt = curves[ev][TRUTH_KEY]
        ok = np.isfinite(vt)
        if sp["log"]:
            ok &= vt > 0
        lo = np.nanmin(vt[ok]); hi = np.nanmax(vt[ok])
        for s in models:
            vm = curves[ev][s]
            m = ok & np.isfinite(vm)
            if sp["log"]:
                m &= vm > 0
            lo = min(lo, np.nanmin(vm[m])); hi = max(hi, np.nanmax(vm[m]))
        pad = 0.03 * (hi - lo) if not sp["log"] else 0.0
        lim = (lo * 0.85, hi * 1.15) if sp["log"] else (lo - pad, hi + pad)

        ax.plot(lim, lim, color="0.55", ls=(0, (3.5, 2)), lw=0.7, zorder=1)
        for s in models:
            vm = curves[ev][s]
            m = ok & np.isfinite(vm)
            ax.plot(vt[m], vm[m], color=st[s]["color"], ls=st[s]["ls"], lw=st[s]["lw"],
                    marker="", zorder=st[s]["zorder"], label=DISPLAY_NAME_MAP.get(s, s))
            kk = [np.argmin(np.abs(q - kl)) for kl in key_levels]
            ax.plot(vt[kk], vm[kk], color=st[s]["color"], ls="none", marker=st[s]["marker"],
                    ms=3.2, mew=0.5, mfc="white", zorder=st[s]["zorder"] + 10)

        if sp["log"]:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_box_aspect(1.0)
        ax.set_xlabel(f"ERA5 {sp['short']} ({sp['units']})")
        ax.set_ylabel(f"Downscaled {sp['short']} ({sp['units']})")
        ax.set_title(sp["label"], pad=3)
        panel_letter(ax, i)

        if i == 0:
            for kl in key_levels:
                j = int(np.argmin(np.abs(q - kl)))
                ax.annotate(percentile_label(kl).replace("th", ""),
                            xy=(vt[j], vt[j]), xytext=(2.5, -6.5),
                            textcoords="offset points", fontsize=8, color="0.35")

        if with_inset:
            j0 = int(np.searchsorted(q, 0.99))
            if j0 < q.size - 3:
                sub = slice(j0, q.size)
                v0 = np.nanmin(vt[sub]); v1 = np.nanmax(vt[sub])
                axi = ax.inset_axes([0.56, 0.10, 0.40, 0.40])
                axi.plot([v0, v1], [v0, v1], color="0.55", ls=(0, (3, 2)), lw=0.6)
                for s in models:
                    vm = curves[ev][s]
                    axi.plot(vt[sub], vm[sub], color=st[s]["color"], ls=st[s]["ls"], lw=0.85)
                if sp["log"]:
                    axi.set_xscale("log"); axi.set_yscale("log")
                axi.set_xlim(v0, v1); axi.set_ylim(v0, v1)
                axi.tick_params(labelsize=6, pad=1, length=1.6)
                axi.set_title(r"\(\geq\)" + percentile_label(0.99).replace("th", "") + "th", fontsize=6, pad=1.5)
                for sp_ in axi.spines.values():
                    sp_.set_linewidth(0.45)

    for j in range(len(evars), axes.size):
        axes[j].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    save_fig(fig, out_base)
    plt.close(fig)


def fig_quantile_bias_profile(plt, df_long, evars, series, out_base, ncol=3):
    st = style_for(series)
    models = [s for s in preferred_series_order(series) if s != TRUTH_KEY]
    n = len(evars); ncol = min(ncol, max(1, n)); nrow = int(math.ceil(n / ncol))
    pw = W_FULL if ncol >= 3 else W_ONEHALF
    fig, axes = plt.subplots(nrow, ncol, figsize=(pw, min(pw / ncol * nrow * 0.95, H_MAX)), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for i, ev in enumerate(evars):
        ax = axes[i]
        sub = df_long[df_long["EVar"] == ev]
        ax.axhline(0.0, color="0.55", ls=(0, (3.5, 2)), lw=0.7, zorder=1)
        for s in models:
            d = sub[sub["Series"] == s].sort_values("Level")
            if d.empty:
                continue
            p = 1.0 - d["Level"].values
            y = d["RelBias_pct"].values
            lo = d.get("RelBias_CI_Low", pd.Series(np.nan, index=d.index)).values
            hi = d.get("RelBias_CI_High", pd.Series(np.nan, index=d.index)).values
            m = np.isfinite(lo) & np.isfinite(hi)
            if m.any():
                ax.fill_between(p[m], lo[m], hi[m], color=st[s]["color"], alpha=0.16, lw=0, zorder=2)
            ax.plot(p, y, color=st[s]["color"], ls=st[s]["ls"], lw=st[s]["lw"],
                    marker=st[s]["marker"], ms=2.8, mew=0.5, mfc="white", zorder=st[s]["zorder"],
                    label=DISPLAY_NAME_MAP.get(s, s))
        add_percentile_axis(ax)
        ax.set_ylabel("Quantile bias (%)")
        ax.set_title(EXTREME_SPECS[ev]["label"], pad=3)
        ax.grid(True, which="major", axis="both", alpha=0.25, lw=0.4)
        panel_letter(ax, i)
    for j in range(len(evars), axes.size):
        axes[j].axis("off")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 4), bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, out_base)
    plt.close(fig)


def fig_return_curves(plt, curves, panel_evars, series, out_base):
    panel_evars = [ev for ev in panel_evars if ev in curves]
    if len(panel_evars) == 0:
        print("    [Skip] Fig_tail_return_curves: No available variables")
        return

    st = style_for(series)
    draw_series = preferred_series_order(series)
    nrow, ncol = 2, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(W_FULL, min(W_FULL * 0.68, H_MAX)), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for i, ev in enumerate(panel_evars[:8]):
        ax = axes[i]
        sp = EXTREME_SPECS[ev]
        q = curves[ev]["levels"]
        sel = q >= 0.90
        p = 1.0 - q[sel]

        for s in draw_series:
            v = curves[ev][s][sel]
            ax.plot(p, v, color=st[s]["color"], ls=st[s]["ls"], lw=st[s]["lw"],
                    zorder=st[s]["zorder"], label=series_label(s) if i == 0 else None)

        add_percentile_axis(ax, xlabel=(i // ncol == nrow - 1))
        if sp["log"]:
            ax.set_yscale("log")
        ax.set_ylabel(f"{sp['short']} ({sp['units']})")
        ax.set_title(sp.get("panel", sp["label"]), pad=3)
        ax.grid(True, alpha=0.25, lw=0.4)
        panel_letter(ax, i)

    for j in range(min(len(panel_evars), 8), nrow * ncol):
        axes[j].axis("off")

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(4, len(l)), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    save_fig(fig, out_base)
    plt.close(fig)


def fig_headline(plt, curves, df_long, ev, series, out_base):
    st = style_for(series)
    models = [s for s in preferred_series_order(series) if s != TRUTH_KEY]
    sp = EXTREME_SPECS[ev]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W_SINGLE, W_SINGLE * 1.72))

    q = curves[ev]["levels"]; vt = curves[ev][TRUTH_KEY]
    lo, hi = np.nanmin(vt), np.nanmax(vt)
    for s in models:
        lo = min(lo, np.nanmin(curves[ev][s])); hi = max(hi, np.nanmax(curves[ev][s]))
    pad = 0.03 * (hi - lo); lim = (lo - pad, hi + pad)

    ax1.plot(lim, lim, color="0.55", ls=(0, (3.5, 2)), lw=0.7)
    for s in models:
        ax1.plot(vt, curves[ev][s], color=st[s]["color"], ls=st[s]["ls"], lw=st[s]["lw"],
                 label=DISPLAY_NAME_MAP.get(s, s))
        kk = [int(np.argmin(np.abs(q - kl))) for kl in KEY_QUANTILE_LEVELS]
        ax1.plot(vt[kk], curves[ev][s][kk], ls="none", marker=st[s]["marker"],
                 ms=3.2, mew=0.5, mfc="white", color=st[s]["color"])
    for kl in KEY_QUANTILE_LEVELS:
        j = int(np.argmin(np.abs(q - kl)))
        ax1.annotate(percentile_label(kl).replace("th", ""), xy=(vt[j], vt[j]),
                     xytext=(2.5, -6.5), textcoords="offset points", fontsize=8, color="0.35")
    ax1.set_xlim(lim); ax1.set_ylim(lim); ax1.set_box_aspect(1.0)
    ax1.set_xlabel(f"ERA5 {sp['short']} ({sp['units']})")
    ax1.set_ylabel(f"Downscaled {sp['short']} ({sp['units']})")
    ax1.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    panel_letter(ax1, 0)

    sub = df_long[df_long["EVar"] == ev]
    ax2.axhline(0.0, color="0.55", ls=(0, (3.5, 2)), lw=0.7)
    for s in models:
        d = sub[sub["Series"] == s].sort_values("Level")
        p = 1 - d["Level"].values
        y = d["RelBias_pct"].values
        lo_, hi_ = d["RelBias_CI_Low"].values, d["RelBias_CI_High"].values
        m = np.isfinite(lo_) & np.isfinite(hi_)
        if m.any():
            ax2.fill_between(p[m], lo_[m], hi_[m], color=st[s]["color"], alpha=0.16, lw=0)
        ax2.plot(p, y, color=st[s]["color"], ls=st[s]["ls"], lw=st[s]["lw"],
                 marker=st[s]["marker"], ms=2.8, mew=0.5, mfc="white")
    add_percentile_axis(ax2)
    ax2.set_ylabel("Quantile bias (%)")
    ax2.grid(True, alpha=0.25, lw=0.4)
    panel_letter(ax2, 1)

    fig.tight_layout()
    save_fig(fig, out_base)
    plt.close(fig)


def fig_fieldmax_and_scatter(plt, S, ev, series, out_base):
    st = style_for(series)
    models = [s for s in preferred_series_order(series) if s != TRUTH_KEY]
    has_pair = any(ev in S[s].paired and S[s].paired[ev].size for s in models)
    ncol = 2 if has_pair else 1
    fig, axes = plt.subplots(1, ncol, figsize=(W_ONEHALF if ncol == 2 else W_SINGLE, W_SINGLE * 1.02))
    axes = np.atleast_1d(axes)
    sp = EXTREME_SPECS[ev]

    ax = axes[0]
    bt = np.sort(S[TRUTH_KEY].blockmax[ev])
    lo, hi = bt[0], bt[-1]
    for s in models:
        bm = np.sort(S[s].blockmax[ev]); lo = min(lo, bm[0]); hi = max(hi, bm[-1])
    pad = 0.03 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.55", ls=(0, (3.5, 2)), lw=0.7)
    for s in models:
        bm = np.sort(S[s].blockmax[ev])
        ax.plot(bt, bm, color=st[s]["color"], ls="none", marker=st[s]["marker"],
                ms=2.2, mew=0.35, alpha=0.75, mfc="none", label=DISPLAY_NAME_MAP.get(s, s))
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad); ax.set_box_aspect(1.0)
    ax.set_xlabel(f"ERA5 field maximum ({sp['units']})")
    ax.set_ylabel(f"Downscaled field maximum ({sp['units']})")
    ax.legend(loc="upper left")
    panel_letter(ax, 0)

    if has_pair:
        ax = axes[1]
        s = models[-1]
        arr = S[s].paired[ev]
        lo2, hi2 = float(arr.min()), float(arr.max())
        hb = ax.hexbin(arr[:, 0], arr[:, 1], gridsize=55, bins="log", cmap="magma_r",
                       mincnt=1, linewidths=0.0)
        ax.plot([lo2, hi2], [lo2, hi2], color="0.2", ls=(0, (3.5, 2)), lw=0.7)
        nb = 25
        edges = np.quantile(arr[:, 0], np.linspace(0, 1, nb + 1))
        cx, cy = [], []
        for k in range(nb):
            m = (arr[:, 0] >= edges[k]) & (arr[:, 0] < edges[k + 1])
            if m.sum() > 30:
                cx.append(np.median(arr[m, 0])); cy.append(np.median(arr[m, 1]))
        ax.plot(cx, cy, color=OKABE_ITO[4], lw=1.2, label="Conditional median")
        ax.set_xlim(lo2, hi2); ax.set_ylim(lo2, hi2); ax.set_box_aspect(1.0)
        ax.set_xlabel(f"ERA5 {sp['short']} ({sp['units']})")
        ax.set_ylabel(f"{DISPLAY_NAME_MAP.get(s, s)} {sp['short']} ({sp['units']})")
        cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label("Count", labelpad=2)
        cb.ax.tick_params(labelsize=7, width=0.5, length=1.8)
        ax.legend(loc="upper left")
        panel_letter(ax, 1)
    fig.tight_layout()
    save_fig(fig, out_base)
    plt.close(fig)


def fig_event_scores(plt, df_ev, evars, series, out_base, scores=("POD", "CSI", "SEDI", "FreqBias")):
    if len(evars) == 0:
        print("    [Skip] Fig_event_scores: No available variables")
        return

    st = style_for(series)
    models = [s for s in preferred_series_order(series) if s != TRUTH_KEY]
    levels = EVENT_QUANTILE_LEVELS
    nrow, ncol = len(scores), len(levels)
    fig, axes = plt.subplots(nrow, ncol, figsize=(W_FULL, min(1.55 * nrow + 0.6, H_MAX)), sharey="row")
    axes = np.atleast_1d(axes)
    y = np.arange(len(evars))
    hgt = 0.8 / max(len(models), 1)
    for r, sc in enumerate(scores):
        for c, lv in enumerate(levels):
            ax = axes[r, c]
            for k, s in enumerate(models):
                d = df_ev[(df_ev["Score"] == sc) & (df_ev["Level"] == lv) &
                          (df_ev["Series"] == s)].set_index("EVar").reindex(evars)
                val = d["Value"].values.astype(float)
                lo = d["CI_Low"].values.astype(float)
                hi = d["CI_High"].values.astype(float)
                err = np.vstack([np.clip(val - lo, 0, None), np.clip(hi - val, 0, None)])
                ax.barh(y + (k - (len(models) - 1) / 2) * hgt, val, height=hgt * 0.9,
                        color=st[s]["color"], alpha=0.85, lw=0.3, edgecolor="white",
                        label=DISPLAY_NAME_MAP.get(s, s) if (r == 0 and c == 0) else None)
                ax.errorbar(val, y + (k - (len(models) - 1) / 2) * hgt, xerr=err,
                            fmt="none", ecolor="0.15", elinewidth=0.6, capsize=1.4)
            if sc == "FreqBias":
                ax.axvline(1.0, color="0.4", ls=(0, (3, 2)), lw=0.6)
            ax.set_yticks(y)
            ax.set_yticklabels([EXTREME_SPECS[e]["short"] for e in evars] if c == 0 else [])
            ax.invert_yaxis()
            ax.grid(True, axis="x", alpha=0.25, lw=0.4)
            if r == 0:
                ax.set_title(f"> {percentile_label(lv)} pct", pad=3)
            if c == 0:
                ax.set_ylabel(sc)
            if r == nrow - 1:
                ax.set_xlabel("Score")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 4), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    save_fig(fig, out_base)
    plt.close(fig)


def fig_pixelwise_maps(plt, pw_q, evars, series, lat, lon, q, out_base):
    models = [s for s in preferred_series_order(series) if s != TRUTH_KEY]
    n = len(models)
    if len(evars) == 0 or n == 0:
        print("    [Skip] Fig_pixelwise_percentile_bias: No available variables or models")
        return

    fig, axes = plt.subplots(len(evars), n,
                             figsize=(W_FULL, min(W_FULL / n * len(evars) * 0.62 + 0.7, H_MAX)),
                             squeeze=False)
    LON, LAT = np.meshgrid(lon, lat)
    idx = 0
    for r, ev in enumerate(evars):
        ref = pw_q[TRUTH_KEY][ev]
        vmax = float(np.nanpercentile(np.abs(np.concatenate(
            [(pw_q[s][ev] - ref) / np.maximum(np.abs(ref), 1e-30) * 100 for s in models]
        ).ravel()), 98))
        vmax = max(vmax, 1.0)
        for c, s in enumerate(models):
            ax = axes[r][c]
            rel = (pw_q[s][ev] - ref) / np.maximum(np.abs(ref), 1e-30) * 100.0
            pc = ax.pcolormesh(LON, LAT, rel, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                               shading="auto", rasterized=True)
            ax.set_title(f"{DISPLAY_NAME_MAP.get(s, s)} · {EXTREME_SPECS[ev]['short']}", pad=2.5)
            if c == 0:
                ax.set_ylabel("Latitude (°)")
            if r == len(evars) - 1:
                ax.set_xlabel("Longitude (°)")
            ax.tick_params(labelsize=7)
            panel_letter(ax, idx); idx += 1
            if c == n - 1:
                cb = fig.colorbar(pc, ax=axes[r], pad=0.015, fraction=0.03)
                cb.set_label(f"{percentile_label(q)} pct bias (%)", labelpad=2)
                cb.ax.tick_params(labelsize=7, width=0.5, length=1.8)
    save_fig(fig, out_base)
    plt.close(fig)


# =============================================================================
# 11) Text Statement Generation & Summary Export
# =============================================================================
def _verb(rel):
    return "underestimates" if rel < 0 else "overestimates"


def make_statements(df_long, evar, level, order):
    sub = df_long[(df_long["EVar"] == evar) & (np.isclose(df_long["Level"], level))].set_index("Series")
    order = [s for s in order if s in sub.index]
    if len(order) < 1:
        return ["(Insufficient data to generate statements)"]
    lab = EXTREME_SPECS[evar]["label"].lower()
    pl = percentile_label(level)
    out = []

    def g(s):
        r = sub.loc[s]
        return (float(r["RelBias_pct"]), float(r.get("RelBias_CI_Low", np.nan)),
                float(r.get("RelBias_CI_High", np.nan)), float(r.get("RelBias_pvalue", np.nan)))

    if len(order) >= 3:
        a, b, c = order[0], order[1], order[2]
        ra, la, ha, _ = g(a); rb, lb, hb, _ = g(b); rc, lc, hc, _ = g(c)
        out.append(
            f"{SENTENCE_NAME_MAP.get(a, a)} {_verb(ra)} the {pl} percentile {lab} by {abs(ra):.2f}% "
            f"and {SENTENCE_NAME_MAP.get(b, b)} by {abs(rb):.2f}%, whereas "
            f"{SENTENCE_NAME_MAP.get(c, c)} reduces the deficit to {abs(rc):.2f}%."
        )
        out.append(
            f"[with 95% CI] {SENTENCE_NAME_MAP.get(a, a)} {_verb(ra)} the {pl} percentile {lab} "
            f"by {abs(ra):.2f}% (95% CI: {abs(ha):.2f}–{abs(la):.2f}%) and "
            f"{SENTENCE_NAME_MAP.get(b, b)} by {abs(rb):.2f}% ({abs(hb):.2f}–{abs(lb):.2f}%), "
            f"whereas {SENTENCE_NAME_MAP.get(c, c)} reduces the deficit to {abs(rc):.2f}% "
            f"({abs(hc):.2f}–{abs(lc):.2f}%)."
        )
    else:
        for s in order:
            r, lo, hi, _ = g(s)
            out.append(f"{SENTENCE_NAME_MAP.get(s, s)} {_verb(r)} the {pl} percentile {lab} "
                       f"by {abs(r):.2f}% (95% CI: {abs(hi):.2f}–{abs(lo):.2f}%).")
    return out


def write_all_statements(path, df_long, df_diff, evars, engine, meta):
    lines = []
    lines.append("=" * 90)
    lines.append("Extreme Value Statements (Formatted for manuscript; values from extremes_quantile_long.csv)")
    lines.append("=" * 90)
    lines.append(json.dumps(meta, indent=2, ensure_ascii=False))
    lines.append("")
    for ev in evars:
        lines.append("-" * 90)
        lines.append(f"### {ev} -- {EXTREME_SPECS[ev]['label']}")
        lines.append("-" * 90)
        for lv in KEY_QUANTILE_LEVELS:
            if not np.any(np.isclose(df_long[df_long["EVar"] == ev]["Level"].values, lv)):
                continue
            for s in make_statements(df_long, ev, lv, STATEMENT_ORDER):
                lines.append(textwrap.fill(s, 110))
            lines.append("")
    if len(df_diff):
        lines.append("=" * 90)
        lines.append("Paired Significance (|Deficit| difference < 0 indicates significantly lower tail bias for the model)")
        lines.append("=" * 90)
        d = df_diff[np.isclose(df_diff["Level"], STATEMENT_LEVEL)]
        if len(d):
            lines.append(d[["EVar", "Percentile", "Baseline_Display", "Model_Display",
                            "AbsDeficit_Baseline_pct", "AbsDeficit_Model_pct",
                            "Delta_AbsDeficit_pct", "Delta_CI_Low", "Delta_CI_High",
                            "p_value_boot", "q_value_BH", "Significant_at_95"]]
                         .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"    -> {path}")
    return lines


def build_allvar_bias_summary(df_long, target_evars, series_names):
    if df_long.empty or ("Level" not in df_long.columns):
        return pd.DataFrame(), []

    methods = [s for s in preferred_series_order(series_names) if s != TRUTH_KEY]
    levels = sorted(df_long["Level"].dropna().unique().tolist())
    pcols = [percentile_label(q) for q in levels]

    rows = []
    for ev in target_evars:
        d = df_long[(df_long["EVar"] == ev) & (df_long["Series"] != TRUTH_KEY)].copy()
        if d.empty:
            continue
        d["PLabel"] = d["Level"].map(percentile_label)
        piv = d.pivot_table(index="Series", columns="PLabel", values="RelBias_pct", aggfunc="first")
        for s in methods:
            if s not in piv.index:
                continue
            row = OrderedDict()
            row["EVar"] = ev
            row["Variable"] = EXTREME_SPECS[ev]["label"]
            row["Tag"] = EXTREME_SPECS[ev]["tag"]
            row["Method"] = s
            row["Display"] = DISPLAY_NAME_MAP.get(s, s)
            for c in pcols:
                row[c] = float(piv.loc[s, c]) if c in piv.columns else np.nan
            rows.append(row)

    return pd.DataFrame(rows), pcols


def print_allvar_bias_summary(df_summary, pcols):
    if df_summary.empty:
        print("[Warning] Multi-variable bias summary is empty.")
        return
    for ev in df_summary["EVar"].drop_duplicates():
        sub = df_summary[df_summary["EVar"] == ev].copy()
        use_cols = [c for c in pcols if c in sub.columns and np.isfinite(sub[c].values).any()]
        if len(use_cols) == 0:
            continue
        tag = EXTREME_SPECS.get(ev, {}).get("tag", ev)
        print("\n" + "=" * 96)
        print(f"Summary: Relative quantile bias for {tag} (%) -- negative values indicate underestimation")
        print("=" * 96)
        tbl = sub.set_index("Display")[use_cols]
        print(tbl.to_string(float_format=lambda x: f"{x:+.3f}"))


def export_fig_tail_return_curves_csv(curves, panel_evars, series, out_csv):
    draw_series = preferred_series_order(series)
    panel_vars = [ev for ev in panel_evars if ev in curves][:8]
    rows = []
    for pidx, ev in enumerate(panel_vars):
        sp = EXTREME_SPECS[ev]
        q = curves[ev]["levels"]
        sel = q >= 0.90
        q_sel = q[sel]
        p_sel = 1.0 - q_sel
        for sidx, s in enumerate(draw_series):
            v = curves[ev][s][sel]
            for k in range(q_sel.size):
                rows.append(dict(
                    PanelIndex=pidx + 1,
                    EVar=ev,
                    PanelTitle=sp.get("panel", sp["label"]),
                    Label=sp["label"],
                    Short=sp["short"],
                    Units=sp["units"],
                    LogY=int(bool(sp["log"])),
                    Series=s,
                    Display=DISPLAY_NAME_MAP.get(s, s),
                    Legend=series_label(s),
                    SeriesIndex=sidx + 1,
                    Level=float(q_sel[k]),
                    ExceedProb=float(p_sel[k]),
                    Percentile=float(q_sel[k] * 100.0),
                    PercentileLabel=percentile_label(float(q_sel[k])),
                    Value=float(v[k]),
                ))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, float_format="%.10g")
    print(f"    -> {out_csv}")
    return df


# =============================================================================
# 12) Main Execution Pipeline
# =============================================================================
def main():
    set_seed(RANDOM_SEED)
    device = torch.device(DEVICE)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 96)
    print("ERA5 Super-Resolution -- Extreme Value / Tail Distribution Evaluation + JAMES Q-Q Plots v1.2")
    print(f"Device: {device}")
    print("=" * 96)

    for p in [X_TEST_PATH, Y_TEST_PATH, SCALER_PATH] + list(MODEL_PATHS.values()):
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    ds_x = xr.open_dataset(X_TEST_PATH, decode_times=False)
    ds_y = xr.open_dataset(Y_TEST_PATH, decode_times=False)
    for v in VARIABLES:
        for ds, tag in ((ds_x, X_TEST_PATH), (ds_y, Y_TEST_PATH)):
            if v not in ds.variables:
                raise KeyError(f"{tag} missing variable {v}")
    if int(ds_x.sizes["time"]) != int(ds_y.sizes["time"]):
        raise ValueError("Time dimension length mismatch between X and Y")

    lat_lr = ds_x["latitude"].values.astype(np.float64)
    lon_lr = ds_x["longitude"].values.astype(np.float64)
    lat_hr = ds_y["latitude"].values.astype(np.float64)
    lon_hr = ds_y["longitude"].values.astype(np.float64)
    H, W = len(lat_hr), len(lon_hr)

    n_total = int(ds_y.sizes["time"])
    if EVAL_MAX_SAMPLES is not None:
        n_total = min(n_total, int(EVAL_MAX_SAMPLES))
    print(f"Time samples N = {n_total} | HR Grid = {H} x {W} = {H*W} points")

    sx, sy = load_scaler_stats(SCALER_PATH)
    models = OrderedDict()
    for nm, p in MODEL_PATHS.items():
        models[nm] = load_one_model(p, sx, sy, device)
        print(f"  Loaded model {nm:8s} params={sum(q.numel() for q in models[nm].parameters()):,}")

    cubic = CubicSplineInterpolator2D(lat_lr, lon_lr, lat_hr, lon_hr, CUBIC_SPLINE_KX, CUBIC_SPLINE_KY)
    ops = SphericalPhysicsOps(lat_hr, lon_hr)
    evars = [e for e in ACTIVE_EXTREME_VARS if e in EXTREME_SPECS]
    factory = ExtremeFieldFactory(ops, evars)
    w_row, W_FIELD = build_weights(lat_hr, W, AREA_WEIGHTED)
    total_weight = n_total * W_FIELD
    n_series = 2 + len(models)

    print(f"\nExtreme variables ({len(evars)}): {', '.join(evars)}")
    print(f"Area weighting (cos phi): {AREA_WEIGHTED} | Total pooled weight = {total_weight:.4g}")
    mem = n_series * len(evars) * n_total * (TAIL_GRID_M + TAIL_GRID_EXT + 1) * 4 / 1e6
    print(f"Estimated theta tail matrix memory approx {mem:.0f} MB")
    if mem > 6000:
        print("  [Warning] Memory usage is high: Consider reducing TAIL_GRID_M / ACTIVE_EXTREME_VARS, or setting ENABLE_BOOTSTRAP=False")

    body_ranges = pass1_truth_range(ds_y, factory, n_total, BATCH_SIZE, evars)
    truth_hist = pass1_truth_hist(ds_y, factory, n_total, BATCH_SIZE, evars, body_ranges, w_row, AREA_WEIGHTED)
    theta_map, _ = make_theta_grid(truth_hist, evars)
    event_thr = {ev: [truth_hist[ev].quantile(q) for q in EVENT_QUANTILE_LEVELS] for ev in evars}

    paired_cfg = dict(enable=COLLECT_PAIRED_TAIL_SAMPLES and PAIRED_SCATTER_EVAR in evars,
                      evar=PAIRED_SCATTER_EVAR,
                      threshold=truth_hist[PAIRED_SCATTER_EVAR].quantile(PAIRED_SCATTER_TRUTH_Q)
                      if PAIRED_SCATTER_EVAR in evars else np.inf)
    if paired_cfg["enable"]:
        n_exp = (1 - PAIRED_SCATTER_TRUTH_Q) * n_total * H * W
        paired_cfg["p_keep"] = float(min(1.0, PAIRED_SCATTER_MAX_POINTS / max(n_exp, 1)))

    pw_ok = (COMPUTE_PIXELWISE_MAPS and
             all((1 - q) * n_total >= 5 for q in PIXELWISE_QUANTILES) and
             all(e in evars for e in PIXELWISE_MAP_EVARS))
    m_keep = int(max(5, math.ceil((1 - min(PIXELWISE_QUANTILES)) * n_total * 2))) if pw_ok else 0
    pixelwise_cfg = dict(enable=pw_ok, m_keep=min(m_keep, n_total), hw=H * W,
                         evars=[e for e in PIXELWISE_MAP_EVARS if e in evars])

    S, pw = pass2_all_series(ds_x, ds_y, models, cubic, factory, n_total, BATCH_SIZE,
                             evars, body_ranges, theta_map, event_thr, sx, sy,
                             w_row, AREA_WEIGHTED, device, store_per_time=ENABLE_BOOTSTRAP,
                             paired_cfg=paired_cfg, pixelwise_cfg=pixelwise_cfg)
    series = list(S.keys())

    engine = BootstrapEngine(n_total, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SCHEME,
                             BOOTSTRAP_BLOCK_LENGTH, BOOTSTRAP_JACKKNIFE,
                             BOOTSTRAP_SEED, BOOTSTRAP_CHUNK,
                             BOOTSTRAP_CONFIDENCE, BOOTSTRAP_CI_METHOD)

    print("\nComputing quantiles, deficits, and CIs ...")
    df_long, df_diff, boot_store = compute_quantile_tables(S, evars, engine, total_weight, ENABLE_BOOTSTRAP)
    df_long.to_csv(CSV_QUANTILE_LONG, index=False, float_format="%.8g")
    if len(df_diff):
        df_diff.to_csv(CSV_PAIRED_DIFF, index=False, float_format="%.8g")

    def _fmt(r):
        if not np.isfinite(r["RelBias_pct"]):
            return ""
        if np.isfinite(r.get("RelBias_CI_Low", np.nan)):
            return f"{r['RelBias_pct']:+.2f} [{r['RelBias_CI_Low']:+.2f}, {r['RelBias_CI_High']:+.2f}]"
        return f"{r['RelBias_pct']:+.2f}"

    wide = []
    for ev in evars:
        for s in [x for x in series if x != TRUTH_KEY]:
            row = OrderedDict(EVar=ev, Variable=EXTREME_SPECS[ev]["label"], Method=DISPLAY_NAME_MAP.get(s, s))
            for lv in KEY_QUANTILE_LEVELS:
                d = df_long[(df_long["EVar"] == ev) & (df_long["Series"] == s) & (np.isclose(df_long["Level"], lv))]
                row[f"p{lv*100:g}_bias%"] = _fmt(d.iloc[0]) if len(d) else ""
            for lv in KEY_QUANTILE_LEVELS:
                d = df_long[(df_long["EVar"] == ev) & (df_long["Series"] == TRUTH_KEY) &
                            (np.isclose(df_long["Level"], lv))]
                row[f"p{lv*100:g}_ERA5"] = f"{d.iloc[0]['Value']:.6g}" if len(d) else ""
            wide.append(row)
    pd.DataFrame(wide).to_csv(CSV_QUANTILE_WIDE, index=False)

    df_summary, summary_pcols = build_allvar_bias_summary(df_long, [ev for ev in SUMMARY_PRINT_EVARS if ev in evars], series)
    df_summary.to_csv(CSV_SUMMARY_ALLVARS, index=False, float_format="%.8g")

    df_ev = compute_event_tables(S, evars, engine, ENABLE_BOOTSTRAP)
    df_ev.to_csv(CSV_EVENT_SCORES, index=False, float_format="%.8g")
    df_ev[df_ev["Score"].str.startswith("Cond_")].to_csv(CSV_CONDITIONAL, index=False, float_format="%.8g")

    df_fm = compute_fieldmax_table(S, evars, engine, ENABLE_BOOTSTRAP)
    df_fm.to_csv(CSV_FIELDMAX, index=False, float_format="%.8g")

    q_body = 1.0 - np.geomspace(0.5, 0.05, QQ_BODY_N)
    p_end = max(TAIL_P_MIN, MIN_TAIL_EFF_SAMPLES / total_weight)
    q_tail = 1.0 - np.geomspace(0.05, p_end, QQ_TAIL_N)
    q_all = np.unique(np.concatenate([q_body, q_tail]))
    curves = {}
    rows = []
    for ev in evars:
        curves[ev] = {"levels": q_all}
        for s in series:
            v = np.array([S[s].body[ev].quantile(q) for q in q_all])
            curves[ev][s] = v
            for q, vv in zip(q_all, v):
                rows.append(dict(EVar=ev, Series=s, Display=DISPLAY_NAME_MAP.get(s, s),
                                 Level=q, ExceedProb=1 - q, Value=vv))
    pd.DataFrame(rows).to_csv(CSV_QQ_CURVE_DATA, index=False, float_format="%.8g")

    _ = export_fig_tail_return_curves_csv(curves, RETURN_CURVE_EVARS, series, CSV_FIG_RETURN_CURVES)

    pw_q = None
    if pixelwise_cfg["enable"]:
        pw_q = {}
        for s in series:
            pw_q[s] = {}
            for ev in pixelwise_cfg["evars"]:
                arr = pw[s][ev]
                m = arr.shape[1]
                for q in PIXELWISE_QUANTILES:
                    k = int(round((1 - q) * n_total))
                    k = int(np.clip(k, 1, m))
                    part = np.partition(arr, m - k, axis=1)[:, m - k:]
                    pw_q[s][ev] = np.sort(part, axis=1)[:, 0].reshape(H, W)

    print("\nPlotting figures ...")
    plt = set_james_style()
    fig_qq_panels(plt, curves, evars, series, os.path.join(FIG_DIR, "Fig_QQ_tail_panels"))
    fig_quantile_bias_profile(plt, df_long, evars, series, os.path.join(FIG_DIR, "Fig_quantile_bias_profile"))
    fig_return_curves(plt, curves, RETURN_CURVE_EVARS, series, os.path.join(FIG_DIR, "Fig_tail_return_curves"))

    if STATEMENT_EVAR in evars:
        fig_headline(plt, curves, df_long, STATEMENT_EVAR, series, os.path.join(FIG_DIR, f"Fig_headline_QQ_{STATEMENT_EVAR}"))
        fig_fieldmax_and_scatter(plt, S, STATEMENT_EVAR, series, os.path.join(FIG_DIR, f"Fig_fieldmax_scatter_{STATEMENT_EVAR}"))
    fig_event_scores(plt, df_ev, [ev for ev in EVENT_FIG_EVARS if ev in evars], series, os.path.join(FIG_DIR, "Fig_event_scores"))
    if pw_q is not None:
        fig_pixelwise_maps(plt, pw_q, pixelwise_cfg["evars"], series, lat_hr, lon_hr, PIXELWISE_QUANTILES[0],
                           os.path.join(FIG_DIR, "Fig_pixelwise_percentile_bias"))

    meta = dict(N_time=n_total, grid=f"{H}x{W}", area_weighted=AREA_WEIGHTED,
                bootstrap=dict(scheme=engine.scheme, block_length=engine.L, n_resamples=engine.n_rep,
                               ci=engine.ci_method, confidence=engine.confidence),
                nbins_body=NBINS_BODY, tail_grid_M=TAIL_GRID_M,
                statement_evar=STATEMENT_EVAR, statement_level=STATEMENT_LEVEL)
    _ = write_all_statements(TXT_STATEMENTS, df_long, df_diff, evars, engine, meta)

    if ENABLE_BOOTSTRAP and boot_store:
        np.savez_compressed(NPZ_BOOT, **{f"{ev}|{q}|{s}": a.astype(np.float32) for (ev, q, s), a in boot_store.items()})

    print_allvar_bias_summary(df_summary, summary_pcols)

    print("\n" + "=" * 96)
    print("Output Files")
    print("=" * 96)
    for p in [CSV_QUANTILE_LONG, CSV_QUANTILE_WIDE, CSV_SUMMARY_ALLVARS, CSV_FIG_RETURN_CURVES,
              CSV_PAIRED_DIFF, CSV_EVENT_SCORES, CSV_CONDITIONAL, CSV_FIELDMAX,
              CSV_QQ_CURVE_DATA, TXT_STATEMENTS]:
        if os.path.exists(p):
            print(f"  {p}")
    print(f"  {FIG_DIR}/*.pdf and *.png")
    print("=" * 96)

    ds_x.close(); ds_y.close()


if __name__ == "__main__":
    main()