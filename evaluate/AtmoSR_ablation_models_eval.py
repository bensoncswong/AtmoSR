#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
ERA5 Super-Resolution Model Evaluation Pipeline v2.0
  = Baseline & Model Evaluation + Bootstrap 95% CI + Paired Significance Tests
================================================================================

Evaluation Targets:
  1) Cubic spline baseline
  2) Ablation models: noPhy / AllPhy / DIV / VOR / GEO / MSE

Metrics:
  A. Raw Variables (Z, Q, T, U, V):
     RMSE, MSE, MAE, MRE, Correlation, PSNR, SSIM
  B. Physics-Derived Variables (DIV, VOR, AGEO_MAG, MSE_GRAD_X, MSE_GRAD_Y):
     RMSE, MSE, MAE, MRE, Correlation

Features (v2.0):
  - Exact bootstrap calculation based on per-sample sufficient statistics (no model re-runs required)
  - Resampling schemes: iid / moving_block / circular_block / stationary
  - CI Methods: percentile / basic / BCa (BCa supports delete-1 or delete-one-block jackknife)
  - Shared resampling indices across all models for valid paired difference CIs
  - Paired comparisons: Delta, Relative Improvement %, bootstrap p-values, BH-FDR q-values
  - Paper-ready wide tables (Est [Lo, Hi]), long tables, bootstrap distribution NPZ, and error bar plots
================================================================================
"""

import os
import math
import random
import warnings
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
from skimage.metrics import structural_similarity as ssim_2d
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# 0) User Hyperparameters & Configurations
# =============================================================================
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EVAL_MAX_SAMPLES = None          # Set an integer (e.g., 512) for quick test; None for full evaluation

CUBIC_SPLINE_KX = 3
CUBIC_SPLINE_KY = 3

MRE_EPS_REL = 1e-3
MRE_EPS_ABS = 1e-8

SSIM_WIN_SIZE = 7

# ---- Physical Constants ----
CP_AIR = 1004.0
LV_WATER = 2.501e6
EARTH_RADIUS = 6371000.0
OMEGA = 7.2921e-5
F_CORIOLIS_ABS_MIN = 2.5e-5
TAN_CLIP = 50.0

# ---- File Paths ----
X_TEST_PATH = "X2_test.nc"
Y_TEST_PATH = "Y_test.nc"
SCALER_PATH = "scalers.joblib"

MODEL_PATHS = OrderedDict([
    ("noPhy",  "best_model_noPhy.pth"),
    ("AllPhy", "best_model_AllPhy.pth"),
    ("DIV",    "best_model_DIV.pth"),
    ("VOR",    "best_model_VOR.pth"),
    ("GEO",    "best_model_GEO.pth"),
    ("MSE",    "best_model_MSE.pth"),
])

# ---- Primary CSV Outputs ----
CSV_RAW_METRICS_PATH  = "test_metrics_raw.csv"
CSV_PHYS_METRICS_PATH = "test_metrics_physics.csv"
CSV_ALL_IN_ONE_PATH   = "test_metrics_all_in_one.csv"

# ---- Bootstrap & Confidence Interval Outputs ----
CSV_CI_LONG_PATH        = "test_metrics_ci_long.csv"
CSV_CI_RAW_WIDE_PATH    = "test_metrics_raw_with_ci.csv"
CSV_CI_PHYS_WIDE_PATH   = "test_metrics_physics_with_ci.csv"
CSV_PAIRED_DIFF_PATH    = "test_metrics_paired_diff_ci.csv"
NPZ_BOOT_DIST_PATH      = "bootstrap_distributions.npz"
FIG_DIR                 = "figs_ci"

# ---- Variable Definitions ----
VARIABLES = ["z", "q", "t", "u", "v"]
VAR_DISPLAY = ["Z", "Q", "T", "U", "V"]
PHYS_DERIVED_NAMES = ["DIV", "VOR", "AGEO_MAG", "MSE_GRAD_X", "MSE_GRAD_Y"]

RAW_METRIC_ORDER  = ["RMSE", "MSE", "MAE", "MRE", "Correlation", "PSNR", "SSIM"]
PHYS_METRIC_ORDER = ["RMSE", "MSE", "MAE", "MRE", "Correlation"]
HIGHER_IS_BETTER  = {"Correlation", "PSNR", "SSIM"}


# =============================================================================
# 0b) Bootstrap Configurations
# =============================================================================
ENABLE_BOOTSTRAP        = True
BOOTSTRAP_N_RESAMPLES   = 2000          # Recommended >= 2000 for BCa
BOOTSTRAP_CONFIDENCE    = 0.95
BOOTSTRAP_CI_METHOD     = "BCa"         # Options: {"percentile", "basic", "BCa"}
BOOTSTRAP_SCHEME        = "circular_block"
# Options for BOOTSTRAP_SCHEME:
#   "iid"            : Independent and identically distributed
#   "moving_block"   : Moving Block Bootstrap (Künsch 1989)
#   "circular_block" : Circular Block Bootstrap (Politis & Romano 1992)
#   "stationary"     : Stationary Bootstrap with block lengths ~ Geometric(1/L)
BOOTSTRAP_BLOCK_LENGTH  = "auto"        # int / "auto" (N^(1/3)) / "sqrt" (N^(1/2))
BOOTSTRAP_JACKKNIFE     = "auto"        # Options: "auto" / "delete1" / "block"
BOOTSTRAP_SEED          = 12345
BOOTSTRAP_CHUNK         = 256           # Number of resamples processed per batch (memory control)

# Baseline models for paired difference comparisons
BASELINES_FOR_DIFF      = ["cubic spline", "noPhy"]

SAVE_BOOT_DISTRIBUTIONS = True
MAKE_PLOTS              = True
PLOT_RAW_METRICS        = ["RMSE", "Correlation"]
PLOT_PHYS_METRICS       = ["RMSE"]


# =============================================================================
# 1) Basic Utilities
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ChannelScaler:
    """Class definition kept for compatibility with joblib deserialization."""
    def __init__(self):
        self.mean = None
        self.std = None


def robust_torch_load(path, map_location):
    """
    Load PyTorch checkpoint safely across versions.
    In PyTorch >= 2.6, weights_only defaults to True, which may fail on numpy objects.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_scaler_stats(scaler_path: str):
    obj = joblib.load(scaler_path)

    if isinstance(obj, dict) and ("scaler_X" in obj) and ("scaler_Y" in obj):
        sx = obj["scaler_X"]
        sy = obj["scaler_Y"]
        x_mean = np.asarray(sx.mean, dtype=np.float32)
        x_std = np.asarray(sx.std, dtype=np.float32)
        y_mean = np.asarray(sy.mean, dtype=np.float32)
        y_std = np.asarray(sy.std, dtype=np.float32)
    else:
        raise ValueError("Invalid scalers.joblib format. Expected 'scaler_X' and 'scaler_Y' keys.")

    x_std = np.where(np.abs(x_std) < 1e-12, 1.0, x_std)
    y_std = np.where(np.abs(y_std) < 1e-12, 1.0, y_std)
    return {"mean": x_mean, "std": x_std}, {"mean": y_mean, "std": y_std}


def standardize_batch(x_raw, mean, std):
    return ((x_raw - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)).astype(np.float32)


def inverse_standardize_batch(y_std, mean, std):
    return (y_std * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)).astype(np.float32)


def read_batch_from_ds(ds, variables, start, end):
    arr = np.stack(
        [ds[v].isel(time=slice(start, end)).values.astype(np.float32) for v in variables],
        axis=1
    )  # Shape: (B, C, H, W)
    return arr


# =============================================================================
# 2) Model Architectures
# =============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        return out + identity


class ResidualCorrectionBackbone(nn.Module):
    def __init__(self, in_channels=5, out_channels=5, num_features=64, num_res_blocks=8,
                 upscale_factor=2, dropout_rate=0.0, residual_scale=1.0, zero_init_last=False):
        super().__init__()
        self.residual_scale = float(residual_scale)

        self.shallow_feature = nn.Conv2d(in_channels, num_features, kernel_size=9, padding=4)
        self.deep_features = nn.Sequential(
            *[ResidualBlock(num_features, dropout_rate) for _ in range(num_res_blocks)]
        )
        self.transition = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )

        self.post_shuffle_fuse = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.post_shuffle_act = nn.ReLU(inplace=True)
        self.output_conv = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

        self._initialize_weights(zero_init_last=zero_init_last)

    def _initialize_weights(self, zero_init_last=False):
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
        shallow = self.shallow_feature(x)
        deep = self.deep_features(shallow)
        deep = self.transition(deep)
        fused = deep + shallow

        up = self.upsample(fused)
        up = self.post_shuffle_fuse(up)
        up = self.post_shuffle_act(up)

        residual = self.output_conv(up) * self.residual_scale
        return residual


class BicubicResidualSRModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_res_blocks, upscale_factor,
                 dropout_rate, scaler_X_mean, scaler_X_std, scaler_Y_mean, scaler_Y_std,
                 align_corners=False, residual_scale=1.0, zero_init_last=False):
        super().__init__()
        self.upscale_factor = int(upscale_factor)
        self.align_corners = bool(align_corners)

        self.backbone = ResidualCorrectionBackbone(
            in_channels=in_channels, out_channels=out_channels,
            num_features=num_features, num_res_blocks=num_res_blocks,
            upscale_factor=upscale_factor, dropout_rate=dropout_rate,
            residual_scale=residual_scale, zero_init_last=zero_init_last
        )

        x_mean = np.asarray(scaler_X_mean, dtype=np.float32)
        x_std = np.asarray(scaler_X_std, dtype=np.float32)
        y_mean = np.asarray(scaler_Y_mean, dtype=np.float32)
        y_std = np.asarray(scaler_Y_std, dtype=np.float32)

        alpha = x_std / y_std
        beta = (x_mean - y_mean) / y_std

        self.register_buffer("x2y_alpha", torch.tensor(alpha, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("x2y_beta", torch.tensor(beta, dtype=torch.float32).view(1, -1, 1, 1))

    def bicubic_baseline_ystd(self, x_std):
        base_x_std = F.interpolate(x_std, scale_factor=self.upscale_factor,
                                   mode="bicubic", align_corners=self.align_corners)
        return base_x_std * self.x2y_alpha + self.x2y_beta

    def forward(self, x_std):
        base_y_std = self.bicubic_baseline_ystd(x_std)
        residual_y_std = self.backbone(x_std)
        return base_y_std + residual_y_std


def infer_model_hparams_from_state_dict(state_dict):
    in_channels = state_dict["backbone.shallow_feature.weight"].shape[1]
    num_features = state_dict["backbone.shallow_feature.weight"].shape[0]
    out_channels = state_dict["backbone.output_conv.weight"].shape[0]

    rb_idx = []
    for k in state_dict.keys():
        if k.startswith("backbone.deep_features.") and k.endswith(".conv1.weight"):
            rb_idx.append(int(k.split(".")[2]))
    num_res_blocks = (max(rb_idx) + 1) if rb_idx else 0

    up_out = state_dict["backbone.upsample.0.weight"].shape[0]
    upscale_factor = int(round(math.sqrt(up_out / float(num_features))))

    return {"in_channels": in_channels, "out_channels": out_channels,
            "num_features": num_features, "num_res_blocks": num_res_blocks,
            "upscale_factor": upscale_factor}


def load_one_model(ckpt_path, scaler_x_stats, scaler_y_stats, device):
    ckpt = robust_torch_load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    hp = infer_model_hparams_from_state_dict(state_dict)

    meta = ckpt if isinstance(ckpt, dict) else {}
    x_mean = np.asarray(meta.get("scaler_X_mean", scaler_x_stats["mean"]), dtype=np.float32)
    x_std  = np.asarray(meta.get("scaler_X_std",  scaler_x_stats["std"]),  dtype=np.float32)
    y_mean = np.asarray(meta.get("scaler_Y_mean", scaler_y_stats["mean"]), dtype=np.float32)
    y_std  = np.asarray(meta.get("scaler_Y_std",  scaler_y_stats["std"]),  dtype=np.float32)

    align_corners  = bool(meta.get("bicubic_align_corners", False))
    residual_scale = float(meta.get("residual_scale", 1.0))
    zero_init_last = bool(meta.get("zero_init_last", False))

    model = BicubicResidualSRModel(
        in_channels=hp["in_channels"], out_channels=hp["out_channels"],
        num_features=hp["num_features"], num_res_blocks=hp["num_res_blocks"],
        upscale_factor=hp["upscale_factor"], dropout_rate=0.0,
        scaler_X_mean=x_mean, scaler_X_std=x_std,
        scaler_Y_mean=y_mean, scaler_Y_std=y_std,
        align_corners=align_corners, residual_scale=residual_scale,
        zero_init_last=zero_init_last
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# =============================================================================
# 3) Cubic Spline & Spherical Physics Operators
# =============================================================================
def to_increasing_axis(axis_1d):
    axis_1d = np.asarray(axis_1d, dtype=np.float64)
    if axis_1d[1] > axis_1d[0]:
        return axis_1d, False
    return axis_1d[::-1].copy(), True


class CubicSplineInterpolator2D:
    def __init__(self, lat_lr, lon_lr, lat_hr, lon_hr, kx=3, ky=3):
        self.lat_lr_inc, self.flip_lat_lr = to_increasing_axis(lat_lr)
        self.lon_lr_inc, self.flip_lon_lr = to_increasing_axis(lon_lr)
        self.lat_hr_inc, self.flip_lat_hr = to_increasing_axis(lat_hr)
        self.lon_hr_inc, self.flip_lon_hr = to_increasing_axis(lon_hr)

        self.kx = int(kx)
        self.ky = int(ky)
        self.h_hr = len(lat_hr)
        self.w_hr = len(lon_hr)

    def __call__(self, x_batch):
        bsz, ch, _, _ = x_batch.shape
        out = np.empty((bsz, ch, self.h_hr, self.w_hr), dtype=np.float32)

        for b in range(bsz):
            for c in range(ch):
                field = x_batch[b, c]
                if self.flip_lat_lr:
                    field = field[::-1, :]
                if self.flip_lon_lr:
                    field = field[:, ::-1]
                field = np.ascontiguousarray(field, dtype=np.float64)

                spline = RectBivariateSpline(self.lat_lr_inc, self.lon_lr_inc, field,
                                            kx=self.kx, ky=self.ky, s=0.0)
                pred = spline(self.lat_hr_inc, self.lon_hr_inc)

                if self.flip_lat_hr:
                    pred = pred[::-1, :]
                if self.flip_lon_hr:
                    pred = pred[:, ::-1]
                out[b, c] = pred.astype(np.float32)
        return out


class SphericalPhysicsOps:
    def __init__(self, latitudes, longitudes):
        lat = np.asarray(latitudes, dtype=np.float64)
        lon = np.asarray(longitudes, dtype=np.float64)

        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        dlat = np.abs(np.mean(np.diff(lat_rad)))
        dlon = np.abs(np.mean(np.diff(lon_rad)))

        self.lat_decreasing = bool(lat[1] < lat[0])

        dy = EARTH_RADIUS * dlat
        dx = EARTH_RADIUS * np.cos(lat_rad) * dlon
        dx = np.maximum(dx, 100.0)

        self.inv_dy = 1.0 / dy
        self.inv_dx = 1.0 / dx

        tan_lat = np.clip(np.tan(lat_rad), -TAN_CLIP, TAN_CLIP)
        self.tan_over_r = tan_lat / EARTH_RADIUS

        f_raw = 2.0 * OMEGA * np.sin(lat_rad)
        f_clip = f_raw.copy()
        mask = np.abs(f_clip) < F_CORIOLIS_ABS_MIN
        f_clip[mask] = np.where(f_clip[mask] >= 0, F_CORIOLIS_ABS_MIN, -F_CORIOLIS_ABS_MIN)
        self.inv_f = 1.0 / f_clip

    def ddx(self, field):
        left = np.concatenate([field[:, :, :1], field[:, :, :-1]], axis=2)
        right = np.concatenate([field[:, :, 1:], field[:, :, -1:]], axis=2)
        return 0.5 * (right - left) * self.inv_dx[None, :, None]

    def ddy(self, field):
        top = np.concatenate([field[:, :1, :], field[:, :-1, :]], axis=1)
        bottom = np.concatenate([field[:, 1:, :], field[:, -1:, :]], axis=1)
        if self.lat_decreasing:
            return 0.5 * (top - bottom) * self.inv_dy
        return 0.5 * (bottom - top) * self.inv_dy

    def divergence(self, u, v):
        return self.ddx(u) + self.ddy(v) - v * self.tan_over_r[None, :, None]

    def vorticity(self, u, v):
        return self.ddx(v) - self.ddy(u) + u * self.tan_over_r[None, :, None]

    def geostrophic_wind(self, z):
        dz_dy = self.ddy(z)
        dz_dx = self.ddx(z)
        return -self.inv_f[None, :, None] * dz_dy, self.inv_f[None, :, None] * dz_dx

    def moist_static_energy(self, t, z, q):
        return CP_AIR * t + z + LV_WATER * q

    def moist_static_energy_gradients(self, t, z, q):
        mse = self.moist_static_energy(t, z, q)
        return self.ddx(mse), self.ddy(mse)


def compute_physics_derived(batch_5ch, ops: SphericalPhysicsOps):
    z, q, t, u, v = (batch_5ch[:, i] for i in range(5))
    div = ops.divergence(u, v)
    vor = ops.vorticity(u, v)
    ug, vg = ops.geostrophic_wind(z)
    ageo_mag = np.sqrt((u - ug) ** 2 + (v - vg) ** 2)
    mse_gx, mse_gy = ops.moist_static_energy_gradients(t, z, q)
    return np.stack([div, vor, ageo_mag, mse_gx, mse_gy], axis=1).astype(np.float32)


# =============================================================================
# 4) Metric Core: Closed-form Computation from Sufficient Statistics
# =============================================================================
BASE_STAT_KEYS = ["count", "sum_abs", "sum_sq", "sum_rel",
                  "sum_pred", "sum_true", "sum_pred2", "sum_true2", "sum_pred_true"]
IMG_STAT_KEYS = ["ssim_sum", "ssim_count"]


def metrics_from_sums(sums, data_range=None, enable_image_metrics=False):
    """
    Compute metrics analytically from sufficient statistics dictionary.
    sums: dict[str] -> ndarray, shape (..., C)
    Returns dict[metric] -> ndarray, shape (..., C)
    """
    cnt = np.maximum(sums["count"], 1.0)

    mse = sums["sum_sq"] / cnt
    rmse = np.sqrt(mse)
    mae = sums["sum_abs"] / cnt
    mre = sums["sum_rel"] / cnt

    n = cnt
    num = n * sums["sum_pred_true"] - sums["sum_pred"] * sums["sum_true"]
    den = np.sqrt(
        np.maximum(n * sums["sum_pred2"] - sums["sum_pred"] ** 2, 0.0) *
        np.maximum(n * sums["sum_true2"] - sums["sum_true"] ** 2, 0.0)
    )
    corr = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    out = {"RMSE": rmse, "MSE": mse, "MAE": mae, "MRE": mre, "Correlation": corr}

    if enable_image_metrics:
        dr = np.maximum(np.asarray(data_range, dtype=np.float64), 1e-12)   # (C,)
        out["PSNR"] = 20.0 * np.log10(dr) - 10.0 * np.log10(np.maximum(mse, 1e-12))
        out["SSIM"] = np.divide(sums["ssim_sum"], sums["ssim_count"],
                                out=np.full_like(sums["ssim_sum"], np.nan),
                                where=sums["ssim_count"] > 0)
    return out


class PerSampleMetricCollector:
    """
    Maintains both:
      (a) Global accumulator (exact point estimates)
      (b) Per-sample statistics matrix (N, C) for bootstrap resampling
    """
    def __init__(self, var_names, mre_eps, n_samples,
                 data_range=None, enable_image_metrics=False):
        self.var_names = list(var_names)
        self.nv = len(self.var_names)
        self.n_samples = int(n_samples)

        self.mre_eps = np.asarray(mre_eps, dtype=np.float64).reshape(-1)
        assert self.mre_eps.shape[0] == self.nv

        self.data_range = None
        if data_range is not None:
            self.data_range = np.asarray(data_range, dtype=np.float64).reshape(-1)
            assert self.data_range.shape[0] == self.nv

        self.enable_image_metrics = bool(enable_image_metrics)
        self.stat_keys = list(BASE_STAT_KEYS) + (IMG_STAT_KEYS if self.enable_image_metrics else [])

        # (a) Global accumulators
        self.g = {k: np.zeros(self.nv, dtype=np.float64) for k in self.stat_keys}
        # (b) Per-sample accumulators
        self.ps = {k: np.zeros((self.n_samples, self.nv), dtype=np.float64) for k in self.stat_keys}

    def update(self, pred, true, start_index):
        pred = pred.astype(np.float64, copy=False)
        true = true.astype(np.float64, copy=False)
        b, c, h, w = pred.shape
        sl = slice(start_index, start_index + b)

        err = pred - true
        abs_err = np.abs(err)
        sq_err = err ** 2
        denom = np.maximum(np.abs(true), self.mre_eps.reshape(1, -1, 1, 1))
        rel_err = abs_err / denom

        # ---- (a) Global Accumulation ----
        self.g["sum_abs"] += np.sum(abs_err, axis=(0, 2, 3))
        self.g["sum_sq"]  += np.sum(sq_err,  axis=(0, 2, 3))
        self.g["sum_rel"] += np.sum(rel_err, axis=(0, 2, 3))
        self.g["count"]   += b * h * w

        pf = pred.reshape(b, c, -1)
        tf = true.reshape(b, c, -1)
        self.g["sum_pred"]      += pf.sum(axis=(0, 2))
        self.g["sum_true"]      += tf.sum(axis=(0, 2))
        self.g["sum_pred2"]     += (pf ** 2).sum(axis=(0, 2))
        self.g["sum_true2"]     += (tf ** 2).sum(axis=(0, 2))
        self.g["sum_pred_true"] += (pf * tf).sum(axis=(0, 2))

        # ---- (b) Per-Sample Accumulation ----
        self.ps["sum_abs"][sl] = abs_err.sum(axis=(2, 3))
        self.ps["sum_sq"][sl]  = sq_err.sum(axis=(2, 3))
        self.ps["sum_rel"][sl] = rel_err.sum(axis=(2, 3))
        self.ps["count"][sl]   = float(h * w)
        self.ps["sum_pred"][sl]      = pf.sum(axis=2)
        self.ps["sum_true"][sl]      = tf.sum(axis=2)
        self.ps["sum_pred2"][sl]     = (pf ** 2).sum(axis=2)
        self.ps["sum_true2"][sl]     = (tf ** 2).sum(axis=2)
        self.ps["sum_pred_true"][sl] = (pf * tf).sum(axis=2)

        if self.enable_image_metrics:
            self._update_ssim(pred, true, start_index)

    def _update_ssim(self, pred, true, start_index):
        bsz, ch, _, _ = pred.shape
        for c in range(ch):
            dr = float(self.data_range[c]) if self.data_range is not None else None
            if dr is None or (not np.isfinite(dr)) or dr <= 0:
                continue
            for b in range(bsz):
                val = ssim_2d(true[b, c], pred[b, c], data_range=dr, win_size=SSIM_WIN_SIZE)
                if np.isfinite(val):
                    self.g["ssim_sum"][c] += float(val)
                    self.g["ssim_count"][c] += 1.0
                    self.ps["ssim_sum"][start_index + b, c] = float(val)
                    self.ps["ssim_count"][start_index + b, c] = 1.0

    def finalize(self):
        """Finalize global point estimates."""
        return metrics_from_sums(self.g, self.data_range, self.enable_image_metrics)

    def finalize_from_per_sample(self):
        """Self-check consistency by summing per-sample statistics."""
        tot = {k: self.ps[k].sum(axis=0) for k in self.stat_keys}
        return metrics_from_sums(tot, self.data_range, self.enable_image_metrics)

    def stacked_matrix(self):
        """Returns shape (N, K*C); K = len(stat_keys)"""
        return np.concatenate([self.ps[k] for k in self.stat_keys], axis=1)

    def unstack(self, mat):
        """Unstacks matrix: (..., K*C) -> dict[key] -> (..., C)"""
        out = {}
        for i, k in enumerate(self.stat_keys):
            out[k] = mat[..., i * self.nv:(i + 1) * self.nv]
        return out


def build_metrics_dataframe(result_dict, var_names, metric_order):
    rows = []
    for model_name, metric_map in result_dict.items():
        row = {"Model": model_name}
        for m in metric_order:
            values = metric_map[m]
            for i, vname in enumerate(var_names):
                row[f"{m}_{vname}"] = float(values[i])
        rows.append(row)
    columns = ["Model"] + [f"{m}_{v}" for m in metric_order for v in var_names]
    return pd.DataFrame(rows, columns=columns)


# =============================================================================
# 5) Bootstrap Engine
# =============================================================================
def resolve_block_length(n, spec):
    if isinstance(spec, str):
        s = spec.lower()
        if s == "auto":
            L = int(max(1, round(n ** (1.0 / 3.0))))
        elif s == "sqrt":
            L = int(max(1, round(math.sqrt(n))))
        else:
            raise ValueError(f"Unknown BOOTSTRAP_BLOCK_LENGTH option: {spec}")
    else:
        L = int(spec)
    return int(np.clip(L, 1, max(1, n)))


def gen_bootstrap_indices(n, n_rep, scheme, block_length, rng):
    """Generates an (n_rep, n) int64 matrix of sampled indices."""
    if scheme == "iid":
        return rng.integers(0, n, size=(n_rep, n), dtype=np.int64)

    L = int(block_length)
    if scheme in ("moving_block", "circular_block"):
        nb = int(math.ceil(n / L))
        offs = np.arange(L, dtype=np.int64)[None, None, :]
        if scheme == "moving_block":
            hi = max(1, n - L + 1)
            starts = rng.integers(0, hi, size=(n_rep, nb)).astype(np.int64)
            idx = (starts[:, :, None] + offs).reshape(n_rep, -1)[:, :n]
            idx = np.minimum(idx, n - 1)
        else:
            starts = rng.integers(0, n, size=(n_rep, nb)).astype(np.int64)
            idx = ((starts[:, :, None] + offs) % n).reshape(n_rep, -1)[:, :n]
        return idx

    if scheme == "stationary":
        p = 1.0 / float(L)
        t = np.arange(n, dtype=np.int64)[None, :]
        newblock = rng.random((n_rep, n)) < p
        newblock[:, 0] = True
        starts = rng.integers(0, n, size=(n_rep, n)).astype(np.int64)
        last = np.maximum.accumulate(np.where(newblock, np.broadcast_to(t, newblock.shape), -1), axis=1)
        offset = np.broadcast_to(t, newblock.shape) - last
        start_vals = np.take_along_axis(starts, last, axis=1)
        return (start_vals + offset) % n

    raise ValueError(f"Unknown BOOTSTRAP_SCHEME: {scheme}")


def indices_to_counts(idx, n):
    """Converts index matrix (R, n) to frequency counts matrix (R, n)."""
    r = idx.shape[0]
    counts = np.empty((r, n), dtype=np.float64)
    for i in range(r):
        counts[i] = np.bincount(idx[i], minlength=n).astype(np.float64)
    return counts


def make_jackknife_groups(n, mode, block_length):
    if mode == "delete1":
        return None
    L = int(block_length)
    return np.arange(0, n, L, dtype=np.int64)


class BootstrapEngine:
    def __init__(self, n_samples, n_rep, scheme, block_length, jackknife_mode,
                 seed, chunk, confidence, ci_method):
        self.n = int(n_samples)
        self.n_rep = int(n_rep)
        self.scheme = scheme
        self.L = resolve_block_length(self.n, block_length)
        if jackknife_mode == "auto":
            jackknife_mode = "delete1" if scheme == "iid" else "block"
        self.jk_mode = jackknife_mode
        self.jk_starts = make_jackknife_groups(self.n, self.jk_mode, self.L)
        self.rng = np.random.default_rng(seed)
        self.chunk = int(chunk)
        self.alpha = 1.0 - float(confidence)
        self.ci_method = ci_method
        self.confidence = float(confidence)

    def replicate_metrics(self, collector: PerSampleMetricCollector, desc=""):
        M = collector.stacked_matrix()                     # (N, K*C)
        total = M.sum(axis=0)                              # (K*C,)

        # ---- Bootstrap resampling ----
        boot_mat = np.empty((self.n_rep, M.shape[1]), dtype=np.float64)
        pos = 0
        pbar = tqdm(total=self.n_rep, desc=f"Bootstrap [{desc}]", leave=False)
        while pos < self.n_rep:
            r = min(self.chunk, self.n_rep - pos)
            idx = gen_bootstrap_indices(self.n, r, self.scheme, self.L, self.rng)
            counts = indices_to_counts(idx, self.n)        # (r, N)
            boot_mat[pos:pos + r] = counts @ M
            pos += r
            pbar.update(r)
        pbar.close()

        # ---- Jackknife estimation (for BCa acceleration) ----
        if self.jk_mode == "delete1":
            jack_mat = total[None, :] - M                  # (N, K*C)
        else:
            grp = np.add.reduceat(M, self.jk_starts, axis=0)   # (J, K*C)
            jack_mat = total[None, :] - grp

        boot_sums = collector.unstack(boot_mat)
        jack_sums = collector.unstack(jack_mat)

        boot_reps = metrics_from_sums(boot_sums, collector.data_range, collector.enable_image_metrics)
        jack_reps = metrics_from_sums(jack_sums, collector.data_range, collector.enable_image_metrics)
        return boot_reps, jack_reps

    def reset_rng(self, seed):
        """Reset RNG state to ensure identical resample sequences across models."""
        self.rng = np.random.default_rng(seed)

    def ci(self, theta_hat, boot, jack=None, method=None):
        method = method or self.ci_method
        a = self.alpha
        boot = np.asarray(boot, dtype=np.float64)
        good = boot[np.isfinite(boot)]
        res = {"CI_Low": np.nan, "CI_High": np.nan, "CI_Method": method,
               "Boot_Mean": np.nan, "Boot_SE": np.nan, "Boot_Bias": np.nan,
               "N_Valid_Resamples": int(good.size), "Note": ""}

        if good.size < 20 or not np.isfinite(theta_hat):
            res["Note"] = "insufficient_valid_resamples"
            return res

        res["Boot_Mean"] = float(good.mean())
        res["Boot_SE"] = float(good.std(ddof=1))
        res["Boot_Bias"] = float(good.mean() - theta_hat)

        lo_p = float(np.percentile(good, 100.0 * a / 2.0))
        hi_p = float(np.percentile(good, 100.0 * (1.0 - a / 2.0)))

        if method == "percentile":
            res["CI_Low"], res["CI_High"] = lo_p, hi_p
            return res

        if method == "basic":
            res["CI_Low"], res["CI_High"] = 2.0 * theta_hat - hi_p, 2.0 * theta_hat - lo_p
            return res

        # ---- BCa Confidence Interval ----
        prop = (np.sum(good < theta_hat) + 0.5 * np.sum(good == theta_hat)) / good.size
        prop = float(np.clip(prop, 1.0 / (2 * good.size), 1.0 - 1.0 / (2 * good.size)))
        z0 = _norm.ppf(prop)

        acc = 0.0
        if jack is not None:
            jk = np.asarray(jack, dtype=np.float64)
            jk = jk[np.isfinite(jk)]
            if jk.size >= 3:
                d = jk.mean() - jk
                s2 = np.sum(d ** 2)
                if s2 > 0:
                    acc = float(np.sum(d ** 3) / (6.0 * (s2 ** 1.5)))
        if not np.isfinite(acc):
            acc = 0.0

        z_lo, z_hi = _norm.ppf(a / 2.0), _norm.ppf(1.0 - a / 2.0)

        def _adj(z):
            dd = 1.0 - acc * (z0 + z)
            if abs(dd) < 1e-12:
                return np.nan
            return _norm.cdf(z0 + (z0 + z) / dd)

        a1, a2 = _adj(z_lo), _adj(z_hi)
        if (not np.isfinite(a1)) or (not np.isfinite(a2)) or a1 >= a2:
            res["CI_Low"], res["CI_High"] = lo_p, hi_p
            res["CI_Method"] = "percentile(BCa_fallback)"
            res["Note"] = "BCa_degenerate"
            return res

        a1 = float(np.clip(a1, 1e-6, 1 - 1e-6))
        a2 = float(np.clip(a2, 1e-6, 1 - 1e-6))
        lo = float(np.percentile(good, 100.0 * a1))
        hi = float(np.percentile(good, 100.0 * a2))
        if lo > hi:
            lo, hi = hi, lo
        res["CI_Low"], res["CI_High"] = lo, hi
        res["BCa_z0"] = float(z0)
        res["BCa_a"] = float(acc)
        return res


def bootstrap_pvalue(diff_boot):
    """Two-tailed achieved significance level (H0: Delta = 0)."""
    d = np.asarray(diff_boot, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 20:
        return np.nan
    b = d.size
    p_le = (1.0 + np.sum(d <= 0.0)) / (b + 1.0)
    p_ge = (1.0 + np.sum(d >= 0.0)) / (b + 1.0)
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def bh_fdr(pvals):
    """Benjamini-Hochberg False Discovery Rate adjustment."""
    p = np.asarray(pvals, dtype=np.float64)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = pv.size
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * m / (np.arange(m) + 1.0)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qq = np.empty(m, dtype=np.float64)
    qq[order] = np.clip(ranked, 0.0, 1.0)
    q[ok] = qq
    return q


# =============================================================================
# 6) Target Pre-scan (Single pass to compute ranges & normalizations)
# =============================================================================
def precompute_target_stats(ds_y, variables, batch_size, max_samples, physics_ops):
    n_total = int(ds_y.sizes["time"])
    if max_samples is not None:
        n_total = min(n_total, int(max_samples))

    nv = len(variables)
    npv = len(PHYS_DERIVED_NAMES)

    vmin = np.full(nv, np.inf, dtype=np.float64)
    vmax = np.full(nv, -np.inf, dtype=np.float64)
    sum_abs_raw = np.zeros(nv, dtype=np.float64)
    sum_abs_phy = np.zeros(npv, dtype=np.float64)
    cnt_raw = 0.0
    cnt_phy = 0.0

    for st in tqdm(range(0, n_total, batch_size), desc="Pre-scanning Y_test (raw + physics)"):
        ed = min(st + batch_size, n_total)
        y = read_batch_from_ds(ds_y, variables, st, ed)

        vmin = np.minimum(vmin, y.min(axis=(0, 2, 3)))
        vmax = np.maximum(vmax, y.max(axis=(0, 2, 3)))
        sum_abs_raw += np.sum(np.abs(y), axis=(0, 2, 3))
        cnt_raw += y.shape[0] * y.shape[2] * y.shape[3]

        phy = compute_physics_derived(y, physics_ops)
        sum_abs_phy += np.sum(np.abs(phy), axis=(0, 2, 3))
        cnt_phy += phy.shape[0] * phy.shape[2] * phy.shape[3]

    return (vmax - vmin,
            sum_abs_raw / max(cnt_raw, 1.0),
            sum_abs_phy / max(cnt_phy, 1.0))


# =============================================================================
# 7) Main Evaluation Loop
# =============================================================================
def evaluate_all_methods(ds_x, ds_y, variables, models, cubic_interpolator, physics_ops,
                         scaler_x_stats, scaler_y_stats, raw_data_range, raw_mre_eps,
                         phys_mre_eps, batch_size, max_samples, device):
    method_names = ["cubic spline"] + list(models.keys())

    n_total = int(ds_x.sizes["time"])
    if max_samples is not None:
        n_total = min(n_total, int(max_samples))

    raw_acc, phys_acc = OrderedDict(), OrderedDict()
    for m in method_names:
        raw_acc[m] = PerSampleMetricCollector(VAR_DISPLAY, raw_mre_eps, n_total,
                                              raw_data_range, enable_image_metrics=True)
        phys_acc[m] = PerSampleMetricCollector(PHYS_DERIVED_NAMES, phys_mre_eps, n_total,
                                               None, enable_image_metrics=False)

    x_mean, x_std = scaler_x_stats["mean"], scaler_x_stats["std"]
    y_mean, y_std = scaler_y_stats["mean"], scaler_y_stats["std"]

    for st in tqdm(range(0, n_total, batch_size), desc="Evaluating methods"):
        ed = min(st + batch_size, n_total)

        x_raw = read_batch_from_ds(ds_x, variables, st, ed)
        y_raw = read_batch_from_ds(ds_y, variables, st, ed)
        true_phys = compute_physics_derived(y_raw, physics_ops)

        # -------- Cubic Spline Baseline --------
        pred_cubic = cubic_interpolator(x_raw)
        raw_acc["cubic spline"].update(pred_cubic, y_raw, st)
        phys_acc["cubic spline"].update(compute_physics_derived(pred_cubic, physics_ops),
                                        true_phys, st)

        # -------- AI Models --------
        if len(models) > 0:
            x_tensor = torch.from_numpy(standardize_batch(x_raw, x_mean, x_std)).to(device)
            with torch.no_grad():
                for name, model in models.items():
                    pred_std_np = model(x_tensor).detach().cpu().numpy().astype(np.float32)
                    pred_raw = inverse_standardize_batch(pred_std_np, y_mean, y_std)
                    raw_acc[name].update(pred_raw, y_raw, st)
                    phys_acc[name].update(compute_physics_derived(pred_raw, physics_ops),
                                          true_phys, st)

    raw_result = OrderedDict((k, v.finalize()) for k, v in raw_acc.items())
    phys_result = OrderedDict((k, v.finalize()) for k, v in phys_acc.items())
    return raw_result, phys_result, raw_acc, phys_acc, n_total


# =============================================================================
# 8) Bootstrap Post-Processing & Output Generation
# =============================================================================
def fmt_ci(est, lo, hi, sig=4):
    if not np.isfinite(est):
        return "nan"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{est:.{sig}g}"
    return f"{est:.{sig}g} [{lo:.{sig}g}, {hi:.{sig}g}]"


def run_bootstrap_block(engine, collectors, point_results, var_names, metric_order,
                        group_tag, base_seed):
    ci_rows = []
    boot_store, jack_store = {}, {}

    for model_name, coll in collectors.items():
        engine.reset_rng(base_seed)                  # Shared resampling sequence across methods
        boot_reps, jack_reps = engine.replicate_metrics(coll, desc=f"{group_tag}/{model_name}")

        for m in metric_order:
            boot_store[(model_name, m)] = boot_reps[m]
            jack_store[(model_name, m)] = jack_reps[m]
            for ci_v, vname in enumerate(var_names):
                theta = float(point_results[model_name][m][ci_v])
                r = engine.ci(theta, boot_reps[m][:, ci_v], jack_reps[m][:, ci_v])
                ci_rows.append({
                    "Group": group_tag, "Model": model_name, "Variable": vname, "Metric": m,
                    "Estimate": theta,
                    "CI_Low": r["CI_Low"], "CI_High": r["CI_High"],
                    "CI_HalfWidth": (r["CI_High"] - r["CI_Low"]) / 2.0
                                    if np.isfinite(r["CI_High"]) and np.isfinite(r["CI_Low"]) else np.nan,
                    "Boot_Mean": r["Boot_Mean"], "Boot_Bias": r["Boot_Bias"], "Boot_SE": r["Boot_SE"],
                    "CI_Method": r["CI_Method"],
                    "Confidence": engine.confidence,
                    "Scheme": engine.scheme, "Block_Length": engine.L,
                    "Jackknife": engine.jk_mode,
                    "N_Resamples": engine.n_rep, "N_Samples": engine.n,
                    "Note": r["Note"],
                    "Pretty": fmt_ci(theta, r["CI_Low"], r["CI_High"]),
                })
    return ci_rows, boot_store, jack_store


def run_paired_comparison(engine, point_results, boot_store, jack_store,
                           var_names, metric_order, group_tag, baselines):
    rows = []
    models = list(point_results.keys())
    for base in baselines:
        if base not in models:
            continue
        for mdl in models:
            if mdl == base:
                continue
            for m in metric_order:
                bm = boot_store[(mdl, m)] - boot_store[(base, m)]     # Paired difference
                jm = jack_store[(mdl, m)] - jack_store[(base, m)]
                for ci_v, vname in enumerate(var_names):
                    e_m = float(point_results[mdl][m][ci_v])
                    e_b = float(point_results[base][m][ci_v])
                    d_hat = e_m - e_b
                    r = engine.ci(d_hat, bm[:, ci_v], jm[:, ci_v])
                    pval = bootstrap_pvalue(bm[:, ci_v])

                    # Relative improvement % (positive = model is better)
                    sign = 1.0 if m in HIGHER_IS_BETTER else -1.0
                    denom = abs(e_b) if abs(e_b) > 1e-30 else np.nan
                    rel_hat = sign * d_hat / denom * 100.0
                    rel_boot = sign * bm[:, ci_v] / denom * 100.0
                    rel_jack = sign * jm[:, ci_v] / denom * 100.0
                    rr = engine.ci(rel_hat, rel_boot, rel_jack)

                    sig = bool(np.isfinite(r["CI_Low"]) and np.isfinite(r["CI_High"])
                               and (r["CI_Low"] > 0 or r["CI_High"] < 0))
                    rows.append({
                        "Group": group_tag, "Baseline": base, "Model": mdl,
                        "Variable": vname, "Metric": m,
                        "Est_Model": e_m, "Est_Baseline": e_b,
                        "Diff": d_hat, "Diff_CI_Low": r["CI_Low"], "Diff_CI_High": r["CI_High"],
                        "Diff_Boot_SE": r["Boot_SE"],
                        "RelImprove_%": rel_hat,
                        "RelImprove_CI_Low": rr["CI_Low"], "RelImprove_CI_High": rr["CI_High"],
                        "p_value_boot": pval,
                        "Better": ("model" if (sign * d_hat) > 0 else "baseline"),
                        "Significant_at_95": sig,
                        "CI_Method": r["CI_Method"],
                        "Pretty_Diff": fmt_ci(d_hat, r["CI_Low"], r["CI_High"]),
                        "Pretty_RelImprove_%": fmt_ci(rel_hat, rr["CI_Low"], rr["CI_High"], sig=3),
                    })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        # Perform BH-FDR within each (Group, Baseline, Metric, Variable) family across models
        df["q_value_BH"] = np.nan
        for key, sub in df.groupby(["Group", "Baseline", "Metric", "Variable"], sort=False):
            df.loc[sub.index, "q_value_BH"] = bh_fdr(sub["p_value_boot"].values)
    return df


def ci_long_to_wide(df_long, var_names, metric_order, group_tag):
    sub = df_long[df_long["Group"] == group_tag]
    models = list(OrderedDict.fromkeys(sub["Model"].tolist()))
    rows = []
    for mdl in models:
        row = {"Model": mdl}
        s_m = sub[sub["Model"] == mdl]
        for m in metric_order:
            for v in var_names:
                r = s_m[(s_m["Metric"] == m) & (s_m["Variable"] == v)]
                if len(r) == 0:
                    row[f"{m}_{v}"] = ""
                    continue
                r = r.iloc[0]
                row[f"{m}_{v}"] = fmt_ci(r["Estimate"], r["CI_Low"], r["CI_High"])
        rows.append(row)
    cols = ["Model"] + [f"{m}_{v}" for m in metric_order for v in var_names]
    return pd.DataFrame(rows, columns=cols)


def make_error_bar_plots(df_long, group_tag, metrics, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Warning] Matplotlib plot creation failed: {e}")
        return
    os.makedirs(out_dir, exist_ok=True)

    sub_all = df_long[df_long["Group"] == group_tag]
    variables = list(OrderedDict.fromkeys(sub_all["Variable"].tolist()))
    models = list(OrderedDict.fromkeys(sub_all["Model"].tolist()))

    for m in metrics:
        s = sub_all[sub_all["Metric"] == m]
        if len(s) == 0:
            continue
        ncol = len(variables)
        fig, axes = plt.subplots(1, ncol, figsize=(3.2 * ncol, 0.55 * len(models) + 2.2))
        axes = np.atleast_1d(axes)
        for ax, v in zip(axes, variables):
            sv = s[s["Variable"] == v].set_index("Model").reindex(models)
            y = np.arange(len(models))
            est = sv["Estimate"].values.astype(float)
            lo = sv["CI_Low"].values.astype(float)
            hi = sv["CI_High"].values.astype(float)
            err = np.vstack([np.clip(est - lo, 0, None), np.clip(hi - est, 0, None)])
            ax.errorbar(est, y, xerr=err, fmt="o", capsize=3, ms=4, lw=1.2)
            ax.set_yticks(y)
            ax.set_yticklabels(models, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(f"{m} - {v}", fontsize=10)
            ax.grid(alpha=0.3, axis="x")
        fig.suptitle(f"{group_tag} - {m} with {int(BOOTSTRAP_CONFIDENCE*100)}% Bootstrap CI", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        path = os.path.join(out_dir, f"{group_tag}_{m}_CI.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"  Figure saved: {path}")


# =============================================================================
# 9) Main Execution Routine
# =============================================================================
def main():
    set_seed(RANDOM_SEED)
    device = torch.device(DEVICE)

    print("=" * 96)
    print("ERA5 Super-Resolution Model Evaluation v2.0")
    print(f"Device: {device}")
    print("=" * 96)

    must_exist = [X_TEST_PATH, Y_TEST_PATH, SCALER_PATH] + list(MODEL_PATHS.values())
    for p in must_exist:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    ds_x = xr.open_dataset(X_TEST_PATH, decode_times=False)
    ds_y = xr.open_dataset(Y_TEST_PATH, decode_times=False)

    for v in VARIABLES:
        if v not in ds_x.variables:
            raise KeyError(f"Variable missing in {X_TEST_PATH}: {v}")
        if v not in ds_y.variables:
            raise KeyError(f"Variable missing in {Y_TEST_PATH}: {v}")

    n_x, n_y = int(ds_x.sizes["time"]), int(ds_y.sizes["time"])
    if n_x != n_y:
        raise ValueError(f"Time dimension mismatch: X={n_x}, Y={n_y}")

    lat_lr = ds_x["latitude"].values.astype(np.float64)
    lon_lr = ds_x["longitude"].values.astype(np.float64)
    lat_hr = ds_y["latitude"].values.astype(np.float64)
    lon_hr = ds_y["longitude"].values.astype(np.float64)

    print(f"X_test shape: time={n_x}, lat={len(lat_lr)}, lon={len(lon_lr)}")
    print(f"Y_test shape: time={n_y}, lat={len(lat_hr)}, lon={len(lon_hr)}")

    scaler_x_stats, scaler_y_stats = load_scaler_stats(SCALER_PATH)
    print("Successfully loaded scalers.joblib")

    models = OrderedDict()
    print("\nLoading model checkpoints...")
    for name, path in MODEL_PATHS.items():
        model = load_one_model(path, scaler_x_stats, scaler_y_stats, device)
        models[name] = model
        print(f"  - {name:6s} | {path} | params={sum(p.numel() for p in model.parameters()):,}")

    cubic_interpolator = CubicSplineInterpolator2D(lat_lr, lon_lr, lat_hr, lon_hr,
                                                   kx=CUBIC_SPLINE_KX, ky=CUBIC_SPLINE_KY)
    print("\nInitialized cubic spline interpolator.")

    physics_ops = SphericalPhysicsOps(latitudes=lat_hr, longitudes=lon_hr)
    print("Initialized spherical physics operators.")

    # ---------------- Target Dataset Pre-scan ----------------
    raw_data_range, raw_mean_abs, phys_mean_abs = precompute_target_stats(
        ds_y=ds_y, variables=VARIABLES, batch_size=BATCH_SIZE,
        max_samples=EVAL_MAX_SAMPLES, physics_ops=physics_ops)

    raw_mre_eps = np.maximum(MRE_EPS_ABS, MRE_EPS_REL * raw_mean_abs)
    phys_mre_eps = np.maximum(MRE_EPS_ABS, MRE_EPS_REL * phys_mean_abs)

    print("\nRaw Variables data_range and MRE epsilon:")
    for i, name in enumerate(VAR_DISPLAY):
        print(f"  {name}: data_range={raw_data_range[i]:.6e}, mre_eps={raw_mre_eps[i]:.6e}")
    print("\nPhysics Variables MRE epsilon:")
    for i, name in enumerate(PHYS_DERIVED_NAMES):
        print(f"  {name}: mre_eps={phys_mre_eps[i]:.6e}")

    # ---------------- Evaluation Execution ----------------
    raw_result, phys_result, raw_acc, phys_acc, n_used = evaluate_all_methods(
        ds_x=ds_x, ds_y=ds_y, variables=VARIABLES, models=models,
        cubic_interpolator=cubic_interpolator, physics_ops=physics_ops,
        scaler_x_stats=scaler_x_stats, scaler_y_stats=scaler_y_stats,
        raw_data_range=raw_data_range, raw_mre_eps=raw_mre_eps, phys_mre_eps=phys_mre_eps,
        batch_size=BATCH_SIZE, max_samples=EVAL_MAX_SAMPLES, device=device)

    # ---------------- Consistency Self-Check ----------------
    max_rel = 0.0
    for acc_map, order in ((raw_acc, RAW_METRIC_ORDER), (phys_acc, PHYS_METRIC_ORDER)):
        for name, coll in acc_map.items():
            a, b = coll.finalize(), coll.finalize_from_per_sample()
            for m in order:
                x, y = np.asarray(a[m], float), np.asarray(b[m], float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.any():
                    rel = np.abs(x[ok] - y[ok]) / np.maximum(np.abs(x[ok]), 1e-30)
                    max_rel = max(max_rel, float(rel.max()))
    print(f"\n[Self-Check] Global vs. per-sample sum max relative difference = {max_rel:.3e} (Expected < 1e-10)")

    # ---------------- Primary CSV Outputs ----------------
    df_raw = build_metrics_dataframe(raw_result, VAR_DISPLAY, RAW_METRIC_ORDER)
    df_phys = build_metrics_dataframe(phys_result, PHYS_DERIVED_NAMES, PHYS_METRIC_ORDER)
    df_raw.to_csv(CSV_RAW_METRICS_PATH, index=False, float_format="%.6e")
    df_phys.to_csv(CSV_PHYS_METRICS_PATH, index=False, float_format="%.6e")
    df_all = pd.concat([df_raw.set_index("Model"), df_phys.set_index("Model")], axis=1).reset_index()
    df_all.to_csv(CSV_ALL_IN_ONE_PATH, index=False, float_format="%.6e")

    print("\n" + "=" * 96)
    print("Point estimate evaluation finished. CSV files exported:")
    print(f"  1) {CSV_RAW_METRICS_PATH}")
    print(f"  2) {CSV_PHYS_METRICS_PATH}")
    print(f"  3) {CSV_ALL_IN_ONE_PATH}")
    print("=" * 96)

    # =========================================================================
    # Bootstrap Confidence Interval Calculation
    # =========================================================================
    if not ENABLE_BOOTSTRAP:
        ds_x.close()
        ds_y.close()
        return

    engine = BootstrapEngine(
        n_samples=n_used, n_rep=BOOTSTRAP_N_RESAMPLES, scheme=BOOTSTRAP_SCHEME,
        block_length=BOOTSTRAP_BLOCK_LENGTH, jackknife_mode=BOOTSTRAP_JACKKNIFE,
        seed=BOOTSTRAP_SEED, chunk=BOOTSTRAP_CHUNK,
        confidence=BOOTSTRAP_CONFIDENCE, ci_method=BOOTSTRAP_CI_METHOD)

    print("\n" + "=" * 96)
    print("Bootstrap Configuration")
    print(f"  Sampling unit      : Time samples (N = {engine.n})")
    print(f"  Resampling scheme  : {engine.scheme} (Block length L = {engine.L})")
    print(f"  Resample iterations: {engine.n_rep}")
    print(f"  CI Method          : {engine.ci_method} ({int(engine.confidence*100)}%)")
    print(f"  Jackknife scheme   : {engine.jk_mode} (for BCa acceleration)")
    print("=" * 96)

    ci_rows_raw, boot_raw, jack_raw = run_bootstrap_block(
        engine, raw_acc, raw_result, VAR_DISPLAY, RAW_METRIC_ORDER, "raw", BOOTSTRAP_SEED)
    ci_rows_phys, boot_phys, jack_phys = run_bootstrap_block(
        engine, phys_acc, phys_result, PHYS_DERIVED_NAMES, PHYS_METRIC_ORDER, "physics",
        BOOTSTRAP_SEED)

    df_ci = pd.DataFrame(ci_rows_raw + ci_rows_phys)
    df_ci.to_csv(CSV_CI_LONG_PATH, index=False, float_format="%.6e")

    df_ci_raw_wide = ci_long_to_wide(df_ci, VAR_DISPLAY, RAW_METRIC_ORDER, "raw")
    df_ci_phys_wide = ci_long_to_wide(df_ci, PHYS_DERIVED_NAMES, PHYS_METRIC_ORDER, "physics")
    df_ci_raw_wide.to_csv(CSV_CI_RAW_WIDE_PATH, index=False)
    df_ci_phys_wide.to_csv(CSV_CI_PHYS_WIDE_PATH, index=False)

    # ---------------- Paired Differences Analysis ----------------
    df_diff_raw = run_paired_comparison(engine, raw_result, boot_raw, jack_raw,
                                        VAR_DISPLAY, RAW_METRIC_ORDER, "raw",
                                        BASELINES_FOR_DIFF)
    df_diff_phys = run_paired_comparison(engine, phys_result, boot_phys, jack_phys,
                                         PHYS_DERIVED_NAMES, PHYS_METRIC_ORDER, "physics",
                                         BASELINES_FOR_DIFF)
    df_diff = pd.concat([df_diff_raw, df_diff_phys], ignore_index=True)
    df_diff.to_csv(CSV_PAIRED_DIFF_PATH, index=False, float_format="%.6e")

    # ---------------- Save Bootstrap Distributions ----------------
    if SAVE_BOOT_DISTRIBUTIONS:
        payload = {}
        for (mdl, met), arr in boot_raw.items():
            payload[f"raw|{mdl}|{met}"] = arr.astype(np.float32)
        for (mdl, met), arr in boot_phys.items():
            payload[f"physics|{mdl}|{met}"] = arr.astype(np.float32)
        payload["_meta_var_raw"] = np.array(VAR_DISPLAY)
        payload["_meta_var_phys"] = np.array(PHYS_DERIVED_NAMES)
        payload["_meta_config"] = np.array([engine.scheme, str(engine.L), str(engine.n_rep),
                                            engine.ci_method, str(engine.n)])
        np.savez_compressed(NPZ_BOOT_DIST_PATH, **payload)

    # ---------------- Plotting ----------------
    if MAKE_PLOTS:
        print("\nGenerating error bar figures...")
        make_error_bar_plots(df_ci, "raw", PLOT_RAW_METRICS, FIG_DIR)
        make_error_bar_plots(df_ci, "physics", PLOT_PHYS_METRICS, FIG_DIR)

    # ---------------- Console Summaries ----------------
    print("\n" + "=" * 96)
    print(f"Summary: RAW / RMSE (Point Estimate [{int(engine.confidence*100)}% CI])")
    print("=" * 96)
    s = df_ci[(df_ci["Group"] == "raw") & (df_ci["Metric"] == "RMSE")]
    print(s.pivot(index="Model", columns="Variable", values="Pretty")
           .reindex(list(raw_result.keys()))[VAR_DISPLAY].to_string())

    print("\n" + "=" * 96)
    print("Summary: Paired Difference vs. noPhy Baseline (RAW / RMSE, Delta < 0 denotes improvement)")
    print("=" * 96)
    d = df_diff[(df_diff["Group"] == "raw") & (df_diff["Metric"] == "RMSE") &
                (df_diff["Baseline"] == "noPhy")]
    if len(d) > 0:
        show = d[["Model", "Variable", "Pretty_Diff", "Pretty_RelImprove_%",
                  "p_value_boot", "q_value_BH", "Significant_at_95"]]
        print(show.to_string(index=False))
    else:
        print("(No 'noPhy' baseline found in results)")

    print("\n" + "=" * 96)
    print("Bootstrap pipeline complete. Generated output files:")
    print(f"  4) {CSV_CI_LONG_PATH}       (Long format CI table)")
    print(f"  5) {CSV_CI_RAW_WIDE_PATH}    (Publication wide format: Est [Lo, Hi])")
    print(f"  6) {CSV_CI_PHYS_WIDE_PATH}   (Publication wide format: Est [Lo, Hi])")
    print(f"  7) {CSV_PAIRED_DIFF_PATH}    (Paired diffs + p-values + BH-FDR q-values)")
    if SAVE_BOOT_DISTRIBUTIONS:
        print(f"  8) {NPZ_BOOT_DIST_PATH}     (Bootstrap distributions for custom analysis)")
    if MAKE_PLOTS:
        print(f"  9) {FIG_DIR}/               (Error bar figures)")
    print("=" * 96)

    ds_x.close()
    ds_y.close()


if __name__ == "__main__":
    main()