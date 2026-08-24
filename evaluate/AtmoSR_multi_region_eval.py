#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA5 Super-Resolution Model Multi-Region Evaluation Script with Bootstrap 95% CI
---------------------------------------------------------------------------------
Evaluation Targets:
  1) Cubic spline baseline
  2) Two optimal models: noPhy / AllPhy

Data:
  - Multi-region test datasets (identical grid dimensions, different geographical extents)
  - Options: Global scalers.joblib OR region-specific scalers computed from test sets
  - Checkpoints: best_model_noPhy.pth / best_model_AllPhy.pth

Metrics:
  A. Raw 5 Variables (Z, Q, T, U, V):
     RMSE, MSE, MAE, MRE, Correlation, PSNR, SSIM
  B. Physically Derived 5 Variables:
     DIV, VOR, AGEO_MAG, MSE_GRAD_X, MSE_GRAD_Y
     RMSE, MSE, MAE, MRE, Correlation

Bootstrap:
  - CI Methods: percentile / basic / BCa
  - Resampling Schemes: iid / moving_block / circular_block / stationary
  - Default Configuration: circular_block + BCa
  - Acceleration parameter (a) in BCa calculated via jackknife 
    (auto: iid -> delete1, others -> block jackknife)

References:
  - Moving block bootstrap: Künsch (1989)
  - Circular block bootstrap: Politis & Romano (1992)
  - Stationary bootstrap: Politis & Romano (1994)
  - BCa interval: Efron (1987)
"""

import os
import math
import random
import hashlib
from collections import OrderedDict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm
from skimage.metrics import structural_similarity as ssim_2d
from tqdm import tqdm


# =============================================================================
# 0) User Hyperparameters & Global Configuration
# =============================================================================
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Batch size: Reduce if GPU memory is constrained
BATCH_SIZE = 32

# Max sample limit for quick debugging (e.g., 512); set to None for full evaluation
EVAL_MAX_SAMPLES = None

# Cubic spline interpolation degrees
CUBIC_SPLINE_KX = 3
CUBIC_SPLINE_KY = 3

# MRE stabilization parameters: eps_var = max(MRE_EPS_ABS, MRE_EPS_REL * mean_abs_truth_var)
MRE_EPS_REL = 1e-3
MRE_EPS_ABS = 1e-8

# SSIM window size
SSIM_WIN_SIZE = 7

# Physical constants
CP_AIR = 1004.0
LV_WATER = 2.501e6
EARTH_RADIUS = 6371000.0
OMEGA = 7.2921e-5
F_CORIOLIS_ABS_MIN = 2.5e-5
TAN_CLIP = 50.0

# -------------------------------------------------------------------------
# Scaler Mode:
#   1) "global_joblib"    : Use global scalers.joblib (shared across regions)
#   2) "per_region_test"  : Compute mean/std for each region individually (default)
# -------------------------------------------------------------------------
SCALER_MODE = "per_region_test"  # "global_joblib" or "per_region_test"
VALID_SCALER_MODES = {"global_joblib", "per_region_test"}

# Used when SCALER_MODE = "global_joblib"
SCALER_PATH = "scalers.joblib"

# Max samples to calculate scaler statistics when SCALER_MODE = "per_region_test"
SCALER_STATS_MAX_SAMPLES = None

# -------------------- Bootstrap Configuration --------------------
ENABLE_BOOTSTRAP = True
BOOTSTRAP_N_RESAMPLES = 2000          # BCa recommended >= 2000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_CI_METHOD = "BCa"           # {"percentile", "basic", "BCa"}
BOOTSTRAP_SCHEME = "circular_block"
#   "iid"            : Independent and identically distributed
#   "moving_block"   : Moving block bootstrap (Künsch 1989)
#   "circular_block" : Circular block bootstrap (Politis & Romano 1992) <- Recommended
#   "stationary"     : Stationary bootstrap with geometric block lengths (Politis & Romano 1994)
BOOTSTRAP_BLOCK_LENGTH = "auto"       # int / "auto" (N^(1/3)) / "sqrt" (N^(1/2))
BOOTSTRAP_JACKKNIFE = "auto"          # "auto" / "delete1" / "block"
BOOTSTRAP_SEED = 42
BOOTSTRAP_CHUNK = 256                 # Number of resamples processed per chunk (memory control)

VALID_BOOTSTRAP_METHODS = {"percentile", "basic", "bca"}
VALID_BOOTSTRAP_SCHEMES = {"iid", "moving_block", "circular_block", "stationary"}
VALID_BOOTSTRAP_JACKKNIFE = {"auto", "delete1", "block"}

# Multi-region test datasets
REGION_DATASETS = OrderedDict([
    ("US_East", (
        "X2_test_US_East_500hpa_2025.nc",
        "Y_test_US_East_500hpa_2025.nc",
    )),
    ("US_West", (
        "X2_test_US_West_500hpa_2025.nc",
        "Y_test_US_West_500hpa_2025.nc",
    )),
    ("Alaska", (
        "X2_test_Alaska_500hpa_2025.nc",
        "Y_test_Alaska_500hpa_2025.nc",
    )),
    ("Europe", (
        "X2_test_Europe_500hpa_2025.nc",
        "Y_test_Europe_500hpa_2025.nc",
    )),
    ("Australia", (
        "X2_test_Australia_500hpa_2025.nc",
        "Y_test_Australia_500hpa_2025.nc",
    )),
    ("Africa", (
        "X2_test_Africa_500hpa_2025.nc",
        "Y_test_Africa_500hpa_2025.nc",
    )),
])

# Evaluation model checkpoints
MODEL_PATHS = OrderedDict([
    ("noPhy", "best_model_noPhy.pth"),
    ("AllPhy", "best_model_AllPhy.pth"),
])

# Output CSV prefix definitions
CSV_RAW_METRICS_PREFIX = "test_metrics_raw_by_region"
CSV_PHYS_METRICS_PREFIX = "test_metrics_physics_by_region"
CSV_ALL_IN_ONE_PREFIX = "test_metrics_all_in_one_by_region"
CSV_BOOTSTRAP_CI_LONG_PREFIX = "test_metrics_bootstrap_ci_long_by_region"

# Variable definitions
VARIABLES = ["z", "q", "t", "u", "v"]
VAR_DISPLAY = ["Z", "Q", "T", "U", "V"]
PHYS_DERIVED_NAMES = ["DIV", "VOR", "AGEO_MAG", "MSE_GRAD_X", "MSE_GRAD_Y"]

RAW_METRIC_ORDER = ["RMSE", "MSE", "MAE", "MRE", "Correlation", "PSNR", "SSIM"]
PHYS_METRIC_ORDER = ["RMSE", "MSE", "MAE", "MRE", "Correlation"]

BASE_COMPONENT_KEYS = [
    "sum_abs", "sum_sq", "sum_rel",
    "sum_pred", "sum_true", "sum_pred2", "sum_true2", "sum_pred_true"
]


# =============================================================================
# 1) Core Utilities & Helpers
# =============================================================================
def set_seed(seed: int):
    """Set random seed for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ChannelScaler:
    """Class placeholder required for deserialization if joblib objects rely on it."""
    def __init__(self):
        self.mean = None
        self.std = None


def stable_int_hash(text: str) -> int:
    """Generate a deterministic 32-bit integer hash from a string."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def make_deterministic_seed(base_seed: int, *tokens) -> int:
    """Create a deterministic seed by mixing base seed with string tokens."""
    mix = "|".join(str(t) for t in tokens)
    h = stable_int_hash(mix)
    return int((int(base_seed) + h) % (2**32 - 1))


def normalize_ci_method_name(method: str):
    """Normalize and validate confidence interval method names."""
    m = str(method).strip().lower()
    if m not in VALID_BOOTSTRAP_METHODS:
        raise ValueError(f"Unsupported BOOTSTRAP_CI_METHOD: {method}. Available: {sorted(VALID_BOOTSTRAP_METHODS)}")
    if m == "bca":
        return m, "BCa"
    return m, m


def load_scaler_stats(scaler_path: str):
    """Extract mean and std statistics for X and Y from scalers.joblib."""
    obj = joblib.load(scaler_path)

    if isinstance(obj, dict) and ("scaler_X" in obj) and ("scaler_Y" in obj):
        sx = obj["scaler_X"]
        sy = obj["scaler_Y"]
        x_mean = np.asarray(sx.mean, dtype=np.float32)
        x_std = np.asarray(sx.std, dtype=np.float32)
        y_mean = np.asarray(sy.mean, dtype=np.float32)
        y_std = np.asarray(sy.std, dtype=np.float32)
    else:
        raise ValueError("Invalid scalers.joblib format: expected scaler_X and scaler_Y keys.")

    x_std = np.where(np.abs(x_std) < 1e-12, 1.0, x_std)
    y_std = np.where(np.abs(y_std) < 1e-12, 1.0, y_std)

    scaler_x_stats = {"mean": x_mean, "std": x_std}
    scaler_y_stats = {"mean": y_mean, "std": y_std}
    return scaler_x_stats, scaler_y_stats


def read_batch_from_ds(ds, variables, start, end):
    """Slice a batch of data array from xarray Dataset."""
    arr = np.stack(
        [ds[v].isel(time=slice(start, end)).values.astype(np.float32) for v in variables],
        axis=1
    )  # Shape: (B, C, H, W)
    return arr


def compute_channel_mean_std_from_ds(ds, variables, batch_size, max_samples=None, desc="dataset"):
    """Compute per-channel mean and standard deviation directly from dataset."""
    n_total = int(ds.sizes["time"])
    if max_samples is not None:
        n_total = min(n_total, int(max_samples))

    c = len(variables)
    sum_val = np.zeros(c, dtype=np.float64)
    sum_sq = np.zeros(c, dtype=np.float64)
    count = 0.0

    for st in tqdm(range(0, n_total, batch_size), desc=f"Prescanning {desc} (mean/std)"):
        ed = min(st + batch_size, n_total)
        arr = read_batch_from_ds(ds, variables, st, ed).astype(np.float64, copy=False)
        sum_val += arr.sum(axis=(0, 2, 3))
        sum_sq += (arr ** 2).sum(axis=(0, 2, 3))
        count += arr.shape[0] * arr.shape[2] * arr.shape[3]

    count = max(count, 1.0)
    mean = sum_val / count
    var = np.maximum(sum_sq / count - mean ** 2, 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-12, 1.0, std)

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def standardize_batch(x_raw, mean, std):
    """Standardize batch using channel mean and standard deviation."""
    return ((x_raw - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)).astype(np.float32)


def inverse_standardize_batch(y_std, mean, std):
    """Revert standardized batch to physical units."""
    return (y_std * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)).astype(np.float32)


def resolve_scaler_stats_for_region(
    scaler_mode,
    ds_x,
    ds_y,
    variables,
    batch_size,
    scaler_stats_max_samples,
    global_scaler_stats=None,
    region_name=""
):
    """Resolve scaler statistics according to SCALER_MODE."""
    if scaler_mode == "global_joblib":
        if global_scaler_stats is None:
            raise ValueError("global_joblib mode requires loaded global_scaler_stats")
        return global_scaler_stats

    if scaler_mode == "per_region_test":
        sx = compute_channel_mean_std_from_ds(
            ds=ds_x,
            variables=variables,
            batch_size=batch_size,
            max_samples=scaler_stats_max_samples,
            desc=f"{region_name} X_test"
        )
        sy = compute_channel_mean_std_from_ds(
            ds=ds_y,
            variables=variables,
            batch_size=batch_size,
            max_samples=scaler_stats_max_samples,
            desc=f"{region_name} Y_test"
        )
        return sx, sy

    raise ValueError(f"Unsupported SCALER_MODE: {scaler_mode}")


# =============================================================================
# 2) Model Architecture Definitions
# =============================================================================
class ResidualBlock(nn.Module):
    """Standard 2D Residual Block with optional Dropout."""
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
    """Deep residual network architecture for super-resolution residual estimation."""
    def __init__(
        self,
        in_channels=5,
        out_channels=5,
        num_features=64,
        num_res_blocks=8,
        upscale_factor=2,
        dropout_rate=0.0,
        residual_scale=1.0,
        zero_init_last=False
    ):
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
    """End-to-End Super-Resolution model combining bicubic baseline and deep residual network."""
    def __init__(
        self,
        in_channels,
        out_channels,
        num_features,
        num_res_blocks,
        upscale_factor,
        dropout_rate,
        scaler_X_mean,
        scaler_X_std,
        scaler_Y_mean,
        scaler_Y_std,
        align_corners=False,
        residual_scale=1.0,
        zero_init_last=False
    ):
        super().__init__()
        self.upscale_factor = int(upscale_factor)
        self.align_corners = bool(align_corners)

        self.backbone = ResidualCorrectionBackbone(
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=num_features,
            num_res_blocks=num_res_blocks,
            upscale_factor=upscale_factor,
            dropout_rate=dropout_rate,
            residual_scale=residual_scale,
            zero_init_last=zero_init_last
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
        base_x_std = F.interpolate(
            x_std,
            scale_factor=self.upscale_factor,
            mode="bicubic",
            align_corners=self.align_corners
        )
        return base_x_std * self.x2y_alpha + self.x2y_beta

    def forward(self, x_std):
        base_y_std = self.bicubic_baseline_ystd(x_std)
        residual_y_std = self.backbone(x_std)
        return base_y_std + residual_y_std


def infer_model_hparams_from_state_dict(state_dict):
    """Infer model structural hyperparameters directly from state_dict."""
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

    return {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "num_features": num_features,
        "num_res_blocks": num_res_blocks,
        "upscale_factor": upscale_factor,
    }


def _strip_module_prefix_if_needed(state_dict):
    """Remove 'module.' prefix from state_dict keys if present (DataParallel)."""
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_one_model(ckpt_path, scaler_x_stats, scaler_y_stats, device):
    """Load model checkpoint and apply target regional scaler statistics."""
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = _strip_module_prefix_if_needed(state_dict)

    hp = infer_model_hparams_from_state_dict(state_dict)

    align_corners = bool(ckpt.get("bicubic_align_corners", False))
    residual_scale = float(ckpt.get("residual_scale", 1.0))
    zero_init_last = bool(ckpt.get("zero_init_last", False))

    model = BicubicResidualSRModel(
        in_channels=hp["in_channels"],
        out_channels=hp["out_channels"],
        num_features=hp["num_features"],
        num_res_blocks=hp["num_res_blocks"],
        upscale_factor=hp["upscale_factor"],
        dropout_rate=0.0,  # Dropout disabled during evaluation
        scaler_X_mean=scaler_x_stats["mean"],
        scaler_X_std=scaler_x_stats["std"],
        scaler_Y_mean=scaler_y_stats["mean"],
        scaler_Y_std=scaler_y_stats["std"],
        align_corners=align_corners,
        residual_scale=residual_scale,
        zero_init_last=zero_init_last
    )

    # Re-evaluate x2y_alpha/x2y_beta from updated regional scalers
    state_dict_no_x2y = dict(state_dict)
    state_dict_no_x2y.pop("x2y_alpha", None)
    state_dict_no_x2y.pop("x2y_beta", None)

    missing, unexpected = model.load_state_dict(state_dict_no_x2y, strict=False)

    unexpected = set(unexpected)
    missing = set(missing)
    expected_missing = {"x2y_alpha", "x2y_beta"}

    if unexpected:
        raise RuntimeError(f"{ckpt_path} contains unexpected keys: {sorted(unexpected)}")
    if missing != expected_missing:
        raise RuntimeError(
            f"{ckpt_path} contains unexpected missing keys: {sorted(missing)}, expected only {sorted(expected_missing)}"
        )

    model.to(device)
    model.eval()
    return model


# =============================================================================
# 3) Cubic Spline & Spherical Physics Operators
# =============================================================================
def to_increasing_axis(axis_1d):
    """Ensure 1D coordinate array is monotonically increasing."""
    axis_1d = np.asarray(axis_1d, dtype=np.float64)
    if axis_1d[1] > axis_1d[0]:
        return axis_1d, False
    return axis_1d[::-1], True


class CubicSplineInterpolator2D:
    """2D Cubic Spline interpolator from LR to HR spatial grids."""
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
        # x_batch shape: (B, C, H_lr, W_lr)
        bsz, ch, _, _ = x_batch.shape
        out = np.empty((bsz, ch, self.h_hr, self.w_hr), dtype=np.float32)

        for b in range(bsz):
            for c in range(ch):
                field = x_batch[b, c]

                if self.flip_lat_lr:
                    field = field[::-1, :]
                if self.flip_lon_lr:
                    field = field[:, ::-1]

                spline = RectBivariateSpline(
                    self.lat_lr_inc, self.lon_lr_inc, field,
                    kx=self.kx, ky=self.ky, s=0.0
                )
                pred = spline(self.lat_hr_inc, self.lon_hr_inc)

                if self.flip_lat_hr:
                    pred = pred[::-1, :]
                if self.flip_lon_hr:
                    pred = pred[:, ::-1]

                out[b, c] = pred.astype(np.float32)

        return out


class SphericalPhysicsOps:
    """Spherical differential operators and atmospheric physical diagnostics."""
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
        self.inv_dx = 1.0 / dx  # Shape: (H,)

        tan_lat = np.clip(np.tan(lat_rad), -TAN_CLIP, TAN_CLIP)
        self.tan_over_r = tan_lat / EARTH_RADIUS  # Shape: (H,)

        f_raw = 2.0 * OMEGA * np.sin(lat_rad)
        f_clip = f_raw.copy()
        mask = np.abs(f_clip) < F_CORIOLIS_ABS_MIN
        f_clip[mask] = np.where(f_clip[mask] >= 0, F_CORIOLIS_ABS_MIN, -F_CORIOLIS_ABS_MIN)
        self.inv_f = 1.0 / f_clip  # Shape: (H,)

    def ddx(self, field):
        """Zonal central difference derivative d/dx."""
        left = np.concatenate([field[:, :, :1], field[:, :, :-1]], axis=2)
        right = np.concatenate([field[:, :, 1:], field[:, :, -1:]], axis=2)
        return 0.5 * (right - left) * self.inv_dx[None, :, None]

    def ddy(self, field):
        """Meridional central difference derivative d/dy."""
        top = np.concatenate([field[:, :1, :], field[:, :-1, :]], axis=1)
        bottom = np.concatenate([field[:, 1:, :], field[:, -1:, :]], axis=1)

        if self.lat_decreasing:
            return 0.5 * (top - bottom) * self.inv_dy
        return 0.5 * (bottom - top) * self.inv_dy

    def divergence(self, u, v):
        """Horizontal wind divergence."""
        du_dx = self.ddx(u)
        dv_dy = self.ddy(v)
        metric = v * self.tan_over_r[None, :, None]
        return du_dx + dv_dy - metric

    def vorticity(self, u, v):
        """Relative vorticity."""
        dv_dx = self.ddx(v)
        du_dy = self.ddy(u)
        metric = u * self.tan_over_r[None, :, None]
        return dv_dx - du_dy + metric

    def geostrophic_wind(self, z):
        """Geostrophic wind components (ug, vg)."""
        dz_dy = self.ddy(z)
        dz_dx = self.ddx(z)
        ug = -self.inv_f[None, :, None] * dz_dy
        vg = self.inv_f[None, :, None] * dz_dx
        return ug, vg

    def moist_static_energy(self, t, z, q):
        """Moist static energy (MSE)."""
        return CP_AIR * t + z + LV_WATER * q

    def moist_static_energy_gradients(self, t, z, q):
        """Horizontal gradients of moist static energy."""
        mse = self.moist_static_energy(t, z, q)
        return self.ddx(mse), self.ddy(mse)


def compute_physics_derived(batch_5ch, ops: SphericalPhysicsOps):
    """
    Compute physically derived variables from raw 5 variables.
    Inputs:
      batch_5ch: (B, 5, H, W) -> [Z, Q, T, U, V]
    Outputs:
      (B, 5, H, W) -> [DIV, VOR, AGEO_MAG, MSE_GRAD_X, MSE_GRAD_Y]
    """
    z = batch_5ch[:, 0]
    q = batch_5ch[:, 1]
    t = batch_5ch[:, 2]
    u = batch_5ch[:, 3]
    v = batch_5ch[:, 4]

    div = ops.divergence(u, v)
    vor = ops.vorticity(u, v)

    ug, vg = ops.geostrophic_wind(z)
    ageo_mag = np.sqrt((u - ug) ** 2 + (v - vg) ** 2)

    mse_grad_x, mse_grad_y = ops.moist_static_energy_gradients(t, z, q)

    return np.stack([div, vor, ageo_mag, mse_grad_x, mse_grad_y], axis=1).astype(np.float32)


# =============================================================================
# 4) Per-Sample Metric Accumulation & Bootstrap Inference
# =============================================================================
class PerSampleContributionCollector:
    """
    Collect per-sample metric components to enable point estimation and 
    fast bootstrap resampling calculations.
    """
    def __init__(self, var_names, mre_eps, data_range=None, enable_image_metrics=False):
        self.var_names = list(var_names)
        self.nv = len(self.var_names)
        self.mre_eps = np.asarray(mre_eps, dtype=np.float64).reshape(-1)
        assert self.mre_eps.shape[0] == self.nv

        self.data_range = None
        if data_range is not None:
            self.data_range = np.asarray(data_range, dtype=np.float64).reshape(-1)
            assert self.data_range.shape[0] == self.nv

        self.enable_image_metrics = bool(enable_image_metrics)

        self.buf = {
            "sum_abs": [],
            "sum_sq": [],
            "sum_rel": [],
            "sum_pred": [],
            "sum_true": [],
            "sum_pred2": [],
            "sum_true2": [],
            "sum_pred_true": [],
            "count_pix": [],
        }
        if self.enable_image_metrics:
            self.buf["ssim_sum"] = []
            self.buf["ssim_count"] = []

    def update(self, pred, true):
        """
        Accumulate batch predictions and ground truth metrics.
        pred/true shape: (B, C, H, W)
        """
        pred64 = pred.astype(np.float64, copy=False)
        true64 = true.astype(np.float64, copy=False)

        err = pred64 - true64
        bsz, ch, hh, ww = pred64.shape

        sum_abs = np.sum(np.abs(err), axis=(2, 3))
        sum_sq = np.sum(err ** 2, axis=(2, 3))

        denom = np.maximum(np.abs(true64), self.mre_eps.reshape(1, -1, 1, 1))
        sum_rel = np.sum(np.abs(err) / denom, axis=(2, 3))

        sum_pred = np.sum(pred64, axis=(2, 3))
        sum_true = np.sum(true64, axis=(2, 3))
        sum_pred2 = np.sum(pred64 ** 2, axis=(2, 3))
        sum_true2 = np.sum(true64 ** 2, axis=(2, 3))
        sum_pred_true = np.sum(pred64 * true64, axis=(2, 3))
        count_pix = np.full((bsz,), hh * ww, dtype=np.float64)

        self.buf["sum_abs"].append(sum_abs)
        self.buf["sum_sq"].append(sum_sq)
        self.buf["sum_rel"].append(sum_rel)
        self.buf["sum_pred"].append(sum_pred)
        self.buf["sum_true"].append(sum_true)
        self.buf["sum_pred2"].append(sum_pred2)
        self.buf["sum_true2"].append(sum_true2)
        self.buf["sum_pred_true"].append(sum_pred_true)
        self.buf["count_pix"].append(count_pix)

        if self.enable_image_metrics:
            ssim_sum = np.zeros((bsz, ch), dtype=np.float64)
            ssim_count = np.zeros((bsz, ch), dtype=np.float64)

            for c in range(ch):
                dr = float(self.data_range[c]) if self.data_range is not None else None
                if dr is None or (not np.isfinite(dr)) or dr <= 0:
                    continue
                for b in range(bsz):
                    val = ssim_2d(
                        true64[b, c], pred64[b, c],
                        data_range=dr,
                        win_size=SSIM_WIN_SIZE
                    )
                    if np.isfinite(val):
                        ssim_sum[b, c] = float(val)
                        ssim_count[b, c] = 1.0

            self.buf["ssim_sum"].append(ssim_sum)
            self.buf["ssim_count"].append(ssim_count)

    def to_payload(self):
        """Export accumulated buffers into a persistent payload dictionary."""
        if len(self.buf["count_pix"]) == 0:
            raise RuntimeError("Collector buffer is empty. Cannot generate payload.")

        payload = {}
        for k, vv in self.buf.items():
            payload[k] = np.concatenate(vv, axis=0)

        payload["mre_eps"] = self.mre_eps.copy()
        payload["data_range"] = None if self.data_range is None else self.data_range.copy()
        payload["enable_image_metrics"] = self.enable_image_metrics
        payload["var_names"] = list(self.var_names)
        payload["n_time"] = int(payload["count_pix"].shape[0])
        payload["totals"] = compute_payload_totals(payload)
        return payload


def compute_payload_totals(payload):
    """Compute overall scalar sums from payload array buffers."""
    totals = {}
    for k in BASE_COMPONENT_KEYS:
        totals[k] = np.sum(payload[k], axis=0, dtype=np.float64)
    totals["count_pix"] = float(np.sum(payload["count_pix"], dtype=np.float64))

    if payload["enable_image_metrics"]:
        totals["ssim_sum"] = np.sum(payload["ssim_sum"], axis=0, dtype=np.float64)
        totals["ssim_count"] = np.sum(payload["ssim_count"], axis=0, dtype=np.float64)

    return totals


def compute_metric_map_from_aggregates(agg, data_range=None, enable_image_metrics=False):
    """Derive evaluation metrics from aggregated scalar components."""
    cnt = max(float(agg["count_pix"]), 1.0)

    mse = agg["sum_sq"] / cnt
    rmse = np.sqrt(mse)
    mae = agg["sum_abs"] / cnt
    mre = agg["sum_rel"] / cnt

    n = cnt
    num = n * agg["sum_pred_true"] - agg["sum_pred"] * agg["sum_true"]
    den = np.sqrt(
        np.maximum(n * agg["sum_pred2"] - agg["sum_pred"] ** 2, 0.0) *
        np.maximum(n * agg["sum_true2"] - agg["sum_true"] ** 2, 0.0)
    )
    corr = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    out = OrderedDict([
        ("RMSE", rmse),
        ("MSE", mse),
        ("MAE", mae),
        ("MRE", mre),
        ("Correlation", corr),
    ])

    if enable_image_metrics:
        dr = np.asarray(data_range, dtype=np.float64)
        dr = np.maximum(dr, 1e-12)
        psnr = 20.0 * np.log10(dr) - 10.0 * np.log10(np.maximum(mse, 1e-12))
        ssim = np.divide(
            agg["ssim_sum"],
            agg["ssim_count"],
            out=np.full_like(agg["ssim_sum"], np.nan),
            where=agg["ssim_count"] > 0
        )
        out["PSNR"] = psnr
        out["SSIM"] = ssim

    return out


def compute_metric_map_from_payload(payload):
    """Compute full dataset evaluation metrics directly from payload."""
    return compute_metric_map_from_aggregates(
        payload["totals"],
        data_range=payload["data_range"],
        enable_image_metrics=payload["enable_image_metrics"]
    )


def compute_metric_map_from_indices(payload, idx_1d):
    """Compute evaluation metrics from a subset of sample indices."""
    idx = np.asarray(idx_1d, dtype=np.int64)

    agg = {}
    for k in BASE_COMPONENT_KEYS:
        agg[k] = np.sum(payload[k][idx], axis=0, dtype=np.float64)
    agg["count_pix"] = float(np.sum(payload["count_pix"][idx], dtype=np.float64))

    if payload["enable_image_metrics"]:
        agg["ssim_sum"] = np.sum(payload["ssim_sum"][idx], axis=0, dtype=np.float64)
        agg["ssim_count"] = np.sum(payload["ssim_count"][idx], axis=0, dtype=np.float64)

    return compute_metric_map_from_aggregates(
        agg,
        data_range=payload["data_range"],
        enable_image_metrics=payload["enable_image_metrics"]
    )


def flatten_metric_map(metric_map, metric_order):
    """Flatten structured metric map into a 1D vector."""
    vals = []
    for m in metric_order:
        vals.append(np.asarray(metric_map[m], dtype=np.float64).reshape(-1))
    return np.concatenate(vals, axis=0)


def build_stat_specs(metric_order, var_names):
    """Generate (Metric, Variable) specification tuples."""
    specs = []
    for m in metric_order:
        for v in var_names:
            specs.append((m, v))
    return specs


def build_metrics_dataframe(result_dict, var_names, metric_order):
    """Build pandas DataFrame from metric maps."""
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


def resolve_block_length(n, block_length_rule):
    """Determine block length L based on sequence length and block rule."""
    if n <= 0:
        raise ValueError("Sequence length n must be positive.")

    if isinstance(block_length_rule, int):
        L = int(block_length_rule)
    elif isinstance(block_length_rule, str):
        rr = block_length_rule.strip().lower()
        if rr == "auto":
            L = int(round(n ** (1.0 / 3.0)))
        elif rr == "sqrt":
            L = int(round(n ** 0.5))
        else:
            raise ValueError(f"Unsupported BOOTSTRAP_BLOCK_LENGTH rule: {block_length_rule}")
    else:
        raise ValueError(f"Unsupported type for BOOTSTRAP_BLOCK_LENGTH: {type(block_length_rule)}")

    L = max(1, min(int(n), int(L)))
    return L


def resolve_jackknife_mode(scheme, jk_rule):
    """Determine jackknife mode ('delete1' or 'block')."""
    jk = str(jk_rule).strip().lower()
    if jk not in VALID_BOOTSTRAP_JACKKNIFE:
        raise ValueError(f"Unsupported BOOTSTRAP_JACKKNIFE: {jk_rule}. Available: {sorted(VALID_BOOTSTRAP_JACKKNIFE)}")

    if jk == "auto":
        if scheme == "iid":
            return "delete1"
        return "block"
    return jk


def iter_bootstrap_index_chunks(n, n_resamples, scheme, block_length, chunk_size, rng):
    """Yield resample index chunks of shape (m, n) according to bootstrap scheme."""
    produced = 0
    L = max(1, min(n, int(block_length)))

    while produced < n_resamples:
        m = min(chunk_size, n_resamples - produced)

        if scheme == "iid":
            idx = rng.integers(0, n, size=(m, n), dtype=np.int32)

        elif scheme == "moving_block":
            n_blocks = int(math.ceil(n / L))
            max_start = max(0, n - L)
            starts = rng.integers(0, max_start + 1, size=(m, n_blocks), dtype=np.int32)
            offs = np.arange(L, dtype=np.int32)[None, None, :]
            idx = (starts[:, :, None] + offs).reshape(m, -1)[:, :n]

        elif scheme == "circular_block":
            n_blocks = int(math.ceil(n / L))
            starts = rng.integers(0, n, size=(m, n_blocks), dtype=np.int32)
            offs = np.arange(L, dtype=np.int32)[None, None, :]
            idx = ((starts[:, :, None] + offs) % n).reshape(m, -1)[:, :n]

        elif scheme == "stationary":
            p = min(max(1.0 / L, 1e-12), 1.0)
            idx = np.empty((m, n), dtype=np.int32)
            idx[:, 0] = rng.integers(0, n, size=(m,), dtype=np.int32)

            if n > 1:
                starts = rng.integers(0, n, size=(m, n - 1), dtype=np.int32)
                new_block = rng.random((m, n - 1)) < p
                for t in range(1, n):
                    cont = (idx[:, t - 1] + 1) % n
                    idx[:, t] = np.where(new_block[:, t - 1], starts[:, t - 1], cont)
        else:
            raise ValueError(f"Unsupported BOOTSTRAP_SCHEME: {scheme}")

        yield idx
        produced += m


def bootstrap_stat_matrix(
    payload,
    metric_order,
    var_names,
    n_resamples,
    scheme,
    block_length,
    chunk_size,
    seed,
    desc=""
):
    """Compute bootstrap metric matrix of shape (n_resamples, n_metrics)."""
    p = len(metric_order) * len(var_names)
    out = np.empty((n_resamples, p), dtype=np.float64)

    n = int(payload["n_time"])
    rng = np.random.default_rng(int(seed))

    n_chunks = int(math.ceil(n_resamples / float(chunk_size)))
    ptr = 0

    for idx_chunk in tqdm(
        iter_bootstrap_index_chunks(n, n_resamples, scheme, block_length, chunk_size, rng),
        total=n_chunks,
        desc=desc,
        leave=False
    ):
        m = idx_chunk.shape[0]
        chunk_vals = np.empty((m, p), dtype=np.float64)

        for i in range(m):
            metric_map = compute_metric_map_from_indices(payload, idx_chunk[i])
            chunk_vals[i] = flatten_metric_map(metric_map, metric_order)

        out[ptr:ptr + m] = chunk_vals
        ptr += m

    return out


def jackknife_stat_matrix(payload, metric_order, var_names, jk_mode, block_length, desc=""):
    """
    Compute Jackknife estimates for BCa acceleration parameter.
    Returns:
      jk_mat: (N_jk, P)
      jk_n: N_jk
      jk_block_used: L if block mode, else 1
    """
    n = int(payload["n_time"])
    p = len(metric_order) * len(var_names)

    if n <= 1:
        return np.full((0, p), np.nan, dtype=np.float64), 0, np.nan

    totals = payload["totals"]

    if jk_mode == "delete1":
        out = np.empty((n, p), dtype=np.float64)
        for i in tqdm(range(n), desc=desc, leave=False):
            agg = {}
            for k in BASE_COMPONENT_KEYS:
                agg[k] = totals[k] - payload[k][i]
            agg["count_pix"] = float(totals["count_pix"] - payload["count_pix"][i])

            if payload["enable_image_metrics"]:
                agg["ssim_sum"] = totals["ssim_sum"] - payload["ssim_sum"][i]
                agg["ssim_count"] = totals["ssim_count"] - payload["ssim_count"][i]

            mm = compute_metric_map_from_aggregates(
                agg,
                data_range=payload["data_range"],
                enable_image_metrics=payload["enable_image_metrics"]
            )
            out[i] = flatten_metric_map(mm, metric_order)

        return out, n, 1

    if jk_mode == "block":
        L = int(block_length)
        L = max(1, min(L, n - 1))
        out = np.empty((n, p), dtype=np.float64)
        offs = np.arange(L, dtype=np.int64)

        for s in tqdm(range(n), desc=f"{desc}(L={L})", leave=False):
            idx_del = (s + offs) % n
            agg = {}
            for k in BASE_COMPONENT_KEYS:
                agg[k] = totals[k] - np.sum(payload[k][idx_del], axis=0, dtype=np.float64)
            agg["count_pix"] = float(totals["count_pix"] - np.sum(payload["count_pix"][idx_del], dtype=np.float64))

            if payload["enable_image_metrics"]:
                agg["ssim_sum"] = totals["ssim_sum"] - np.sum(payload["ssim_sum"][idx_del], axis=0, dtype=np.float64)
                agg["ssim_count"] = totals["ssim_count"] - np.sum(payload["ssim_count"][idx_del], axis=0, dtype=np.float64)

            mm = compute_metric_map_from_aggregates(
                agg,
                data_range=payload["data_range"],
                enable_image_metrics=payload["enable_image_metrics"]
            )
            out[s] = flatten_metric_map(mm, metric_order)

        return out, n, L

    raise ValueError(f"Unsupported jackknife mode: {jk_mode}")


def ci_from_bootstrap_scalar(theta_hat, boot_vals, confidence, method_key, method_disp, jk_vals=None):
    """Compute confidence interval bounds for a scalar point estimate."""
    alpha = 1.0 - float(confidence)

    res = {
        "lower": np.nan,
        "upper": np.nan,
        "method_used": method_disp,
        "status": "ok",
        "z0": np.nan,
        "acceleration": np.nan,
        "adj_alpha_low": np.nan,
        "adj_alpha_high": np.nan,
    }

    boot = np.asarray(boot_vals, dtype=np.float64)
    boot = boot[np.isfinite(boot)]

    if (boot.size < 2) or (not np.isfinite(theta_hat)):
        res["status"] = "insufficient_bootstrap_or_nan_point"
        return res

    # --- Percentile ---
    if method_key == "percentile":
        ql = alpha / 2.0
        qh = 1.0 - alpha / 2.0
        res["lower"] = float(np.quantile(boot, ql))
        res["upper"] = float(np.quantile(boot, qh))
        return res

    # --- Basic ---
    if method_key == "basic":
        ql = alpha / 2.0
        qh = 1.0 - alpha / 2.0
        boot_lo = float(np.quantile(boot, ql))
        boot_hi = float(np.quantile(boot, qh))
        res["lower"] = float(2.0 * theta_hat - boot_hi)
        res["upper"] = float(2.0 * theta_hat - boot_lo)
        if res["lower"] > res["upper"]:
            res["lower"], res["upper"] = res["upper"], res["lower"]
        return res

    # --- BCa ---
    if method_key == "bca":
        if jk_vals is None:
            ql = alpha / 2.0
            qh = 1.0 - alpha / 2.0
            res["lower"] = float(np.quantile(boot, ql))
            res["upper"] = float(np.quantile(boot, qh))
            res["method_used"] = "percentile(fallback_no_jackknife)"
            res["status"] = "fallback"
            return res

        jk = np.asarray(jk_vals, dtype=np.float64)
        jk = jk[np.isfinite(jk)]
        if jk.size < 3:
            ql = alpha / 2.0
            qh = 1.0 - alpha / 2.0
            res["lower"] = float(np.quantile(boot, ql))
            res["upper"] = float(np.quantile(boot, qh))
            res["method_used"] = "percentile(fallback_insufficient_jackknife)"
            res["status"] = "fallback"
            return res

        # Bias correction parameter z0
        less = np.sum(boot < theta_hat)
        equal = np.sum(boot == theta_hat)
        prop = (less + 0.5 * equal) / float(boot.size)
        eps = 1.0 / (2.0 * boot.size)
        prop = float(np.clip(prop, eps, 1.0 - eps))
        z0 = float(norm.ppf(prop))

        # Acceleration parameter a
        jk_mean = float(np.mean(jk))
        diff = jk_mean - jk
        num = float(np.sum(diff ** 3))
        den = float(6.0 * (np.sum(diff ** 2) ** 1.5))
        if (den == 0.0) or (not np.isfinite(den)):
            a = 0.0
        else:
            a = num / den

        z_alpha_low = float(norm.ppf(alpha / 2.0))
        z_alpha_high = float(norm.ppf(1.0 - alpha / 2.0))

        def _adj_alpha(z):
            denom = 1.0 - a * (z0 + z)
            if (not np.isfinite(denom)) or abs(denom) < 1e-14:
                return np.nan
            return float(norm.cdf(z0 + (z0 + z) / denom))

        adj_low = _adj_alpha(z_alpha_low)
        adj_high = _adj_alpha(z_alpha_high)

        if (not np.isfinite(adj_low)) or (not np.isfinite(adj_high)):
            ql = alpha / 2.0
            qh = 1.0 - alpha / 2.0
            res["lower"] = float(np.quantile(boot, ql))
            res["upper"] = float(np.quantile(boot, qh))
            res["method_used"] = "percentile(fallback_bad_bca_adjust)"
            res["status"] = "fallback"
            return res

        adj_low = float(np.clip(adj_low, 0.0, 1.0))
        adj_high = float(np.clip(adj_high, 0.0, 1.0))

        ql = min(adj_low, adj_high)
        qh = max(adj_low, adj_high)
        lo = float(np.quantile(boot, ql))
        hi = float(np.quantile(boot, qh))

        res["lower"] = lo
        res["upper"] = hi
        res["z0"] = z0
        res["acceleration"] = a
        res["adj_alpha_low"] = adj_low
        res["adj_alpha_high"] = adj_high
        return res

    raise ValueError(f"Unsupported CI method: {method_key}")


def bootstrap_ci_dataframe_for_method_domain(
    region_name,
    model_name,
    domain_name,            # "raw" or "physics"
    payload,
    point_metric_map,
    metric_order,
    var_names,
    n_resamples,
    confidence,
    ci_method,
    scheme,
    block_length_rule,
    jackknife_rule,
    chunk_size,
    seed
):
    """Generate pandas DataFrame containing confidence intervals for specified domain."""
    method_key, method_disp = normalize_ci_method_name(ci_method)

    n = int(payload["n_time"])
    block_len = resolve_block_length(n, block_length_rule)

    boot_desc = f"[{region_name}][{model_name}][{domain_name}] bootstrap"
    boot_mat = bootstrap_stat_matrix(
        payload=payload,
        metric_order=metric_order,
        var_names=var_names,
        n_resamples=n_resamples,
        scheme=scheme,
        block_length=block_len,
        chunk_size=chunk_size,
        seed=seed,
        desc=boot_desc
    )

    point_vec = flatten_metric_map(point_metric_map, metric_order)

    jk_mode = "not_used"
    jk_block_used = np.nan
    jk_mat = None
    jk_n = 0

    if method_key == "bca":
        jk_mode = resolve_jackknife_mode(scheme, jackknife_rule)
        jk_desc = f"[{region_name}][{model_name}][{domain_name}] jackknife-{jk_mode}"
        jk_mat, jk_n, jk_block_used = jackknife_stat_matrix(
            payload=payload,
            metric_order=metric_order,
            var_names=var_names,
            jk_mode=jk_mode,
            block_length=block_len,
            desc=jk_desc
        )

    specs = build_stat_specs(metric_order, var_names)
    rows = []

    for j, (metric_name, var_name) in enumerate(specs):
        theta_hat = float(point_vec[j])
        boot_vals = boot_mat[:, j]
        jk_vals = jk_mat[:, j] if (jk_mat is not None and jk_mat.shape[0] > 0) else None

        ci = ci_from_bootstrap_scalar(
            theta_hat=theta_hat,
            boot_vals=boot_vals,
            confidence=confidence,
            method_key=method_key,
            method_disp=method_disp,
            jk_vals=jk_vals
        )

        row = {
            "Region": region_name,
            "Model": model_name,
            "Domain": domain_name,
            "Metric": metric_name,
            "Variable": var_name,
            "PointEstimate": theta_hat,
            "CI_Lower": ci["lower"],
            "CI_Upper": ci["upper"],
            "Confidence": float(confidence),
            "CI_Method_Requested": method_disp,
            "CI_Method_Used": ci["method_used"],
            "CI_Status": ci["status"],
            "Bootstrap_Scheme": scheme,
            "Bootstrap_Block_Length": int(block_len),
            "Bootstrap_Resamples": int(n_resamples),
            "Bootstrap_Seed": int(seed),
            "Bootstrap_Effective_N": int(boot_vals[np.isfinite(boot_vals)].size),
            "Jackknife_Mode": jk_mode,
            "Jackknife_Block_Length": jk_block_used,
            "Jackknife_Effective_N": int(jk_n),
            "BCa_z0": ci["z0"],
            "BCa_acceleration": ci["acceleration"],
            "BCa_adj_alpha_low": ci["adj_alpha_low"],
            "BCa_adj_alpha_high": ci["adj_alpha_high"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# 5) Prescanning Statistics & Evaluation Loops
# =============================================================================
def precompute_raw_target_stats(ds_y, variables, batch_size, max_samples=None):
    """
    Prescan Y_test for:
      - Variable data ranges (max - min) for PSNR/SSIM.
      - Variable mean absolute values for MRE stabilization epsilon.
    """
    n_total = int(ds_y.sizes["time"])
    if max_samples is not None:
        n_total = min(n_total, int(max_samples))

    vmin = np.full(len(variables), np.inf, dtype=np.float64)
    vmax = np.full(len(variables), -np.inf, dtype=np.float64)
    sum_abs = np.zeros(len(variables), dtype=np.float64)
    count = 0.0

    for st in tqdm(range(0, n_total, batch_size), desc="Prescanning Y_test (raw)"):
        ed = min(st + batch_size, n_total)
        y = read_batch_from_ds(ds_y, variables, st, ed)

        vmin = np.minimum(vmin, y.min(axis=(0, 2, 3)))
        vmax = np.maximum(vmax, y.max(axis=(0, 2, 3)))

        sum_abs += np.sum(np.abs(y), axis=(0, 2, 3))
        count += y.shape[0] * y.shape[2] * y.shape[3]

    data_range = vmax - vmin
    mean_abs = sum_abs / max(count, 1.0)
    return data_range, mean_abs


def precompute_physics_target_mean_abs(ds_y, variables, batch_size, max_samples, physics_ops):
    """Prescan Y_test physically derived variables to obtain mean absolute values."""
    n_total = int(ds_y.sizes["time"])
    if max_samples is not None:
        n_total = min(n_total, int(max_samples))

    sum_abs = np.zeros(len(PHYS_DERIVED_NAMES), dtype=np.float64)
    count = 0.0

    for st in tqdm(range(0, n_total, batch_size), desc="Prescanning Y_test (physics)"):
        ed = min(st + batch_size, n_total)
        y = read_batch_from_ds(ds_y, variables, st, ed)
        phy = compute_physics_derived(y, physics_ops)

        sum_abs += np.sum(np.abs(phy), axis=(0, 2, 3))
        count += phy.shape[0] * phy.shape[2] * phy.shape[3]

    mean_abs = sum_abs / max(count, 1.0)
    return mean_abs


def evaluate_all_methods(
    ds_x, ds_y, variables,
    models, cubic_interpolator, physics_ops,
    scaler_x_stats, scaler_y_stats,
    raw_data_range, raw_mre_eps, phys_mre_eps,
    batch_size, max_samples, device
):
    """Run batch-wise inference across cubic spline baseline and AI SR models."""
    method_names = ["cubic spline"] + list(models.keys())

    raw_collectors = OrderedDict()
    phys_collectors = OrderedDict()

    for m in method_names:
        raw_collectors[m] = PerSampleContributionCollector(
            var_names=VAR_DISPLAY,
            mre_eps=raw_mre_eps,
            data_range=raw_data_range,
            enable_image_metrics=True
        )
        phys_collectors[m] = PerSampleContributionCollector(
            var_names=PHYS_DERIVED_NAMES,
            mre_eps=phys_mre_eps,
            data_range=None,
            enable_image_metrics=False
        )

    x_mean = scaler_x_stats["mean"]
    x_std = scaler_x_stats["std"]
    y_mean = scaler_y_stats["mean"]
    y_std = scaler_y_stats["std"]

    n_total = int(ds_x.sizes["time"])
    if max_samples is not None:
        n_total = min(n_total, int(max_samples))

    for st in tqdm(range(0, n_total, batch_size), desc="Main evaluation loop"):
        ed = min(st + batch_size, n_total)

        x_raw = read_batch_from_ds(ds_x, variables, st, ed)  # (B, 5, 50, 50)
        y_raw = read_batch_from_ds(ds_y, variables, st, ed)  # (B, 5, 100, 100)

        # Ground truth physical diagnostics (computed once per batch)
        true_phys = compute_physics_derived(y_raw, physics_ops)

        # -------------------- Cubic Spline --------------------
        pred_cubic = cubic_interpolator(x_raw)
        raw_collectors["cubic spline"].update(pred_cubic, y_raw)

        pred_cubic_phys = compute_physics_derived(pred_cubic, physics_ops)
        phys_collectors["cubic spline"].update(pred_cubic_phys, true_phys)

        # -------------------- AI Models --------------------
        if len(models) > 0:
            x_std_batch = standardize_batch(x_raw, x_mean, x_std)
            x_tensor = torch.from_numpy(x_std_batch).to(device, non_blocking=True)

            with torch.no_grad():
                for model_name, model in models.items():
                    pred_std = model(x_tensor)
                    pred_std_np = pred_std.detach().cpu().numpy().astype(np.float32)
                    pred_raw = inverse_standardize_batch(pred_std_np, y_mean, y_std)

                    raw_collectors[model_name].update(pred_raw, y_raw)

                    pred_phys = compute_physics_derived(pred_raw, physics_ops)
                    phys_collectors[model_name].update(pred_phys, true_phys)

    raw_result = OrderedDict()
    phys_result = OrderedDict()
    raw_payloads = OrderedDict()
    phys_payloads = OrderedDict()

    for m in method_names:
        rp = raw_collectors[m].to_payload()
        pp = phys_collectors[m].to_payload()
        raw_payloads[m] = rp
        phys_payloads[m] = pp
        raw_result[m] = compute_metric_map_from_payload(rp)
        phys_result[m] = compute_metric_map_from_payload(pp)

    return raw_result, phys_result, raw_payloads, phys_payloads


def evaluate_one_region(
    region_name,
    x_path,
    y_path,
    model_paths,
    scaler_mode,
    global_scaler_stats,
    batch_size,
    eval_max_samples,
    scaler_stats_max_samples,
    device
):
    """Perform evaluation and Bootstrap CI estimation for a single geographic region."""
    print("\n" + "=" * 90)
    print(f"Evaluating region: {region_name}")
    print(f"X: {x_path}")
    print(f"Y: {y_path}")
    print("=" * 90)

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"[{region_name}] File not found: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"[{region_name}] File not found: {y_path}")

    ds_x = xr.open_dataset(x_path, decode_times=False)
    ds_y = xr.open_dataset(y_path, decode_times=False)

    try:
        for v in VARIABLES:
            if v not in ds_x.variables:
                raise KeyError(f"[{region_name}] Variable {v} missing in {x_path}")
            if v not in ds_y.variables:
                raise KeyError(f"[{region_name}] Variable {v} missing in {y_path}")

        n_x = int(ds_x.sizes["time"])
        n_y = int(ds_y.sizes["time"])
        if n_x != n_y:
            raise ValueError(f"[{region_name}] Mismatched time steps: X={n_x}, Y={n_y}")

        lat_lr = ds_x["latitude"].values.astype(np.float64)
        lon_lr = ds_x["longitude"].values.astype(np.float64)
        lat_hr = ds_y["latitude"].values.astype(np.float64)
        lon_hr = ds_y["longitude"].values.astype(np.float64)

        print(f"[{region_name}] X_test shape: time={n_x}, lat={len(lat_lr)}, lon={len(lon_lr)}")
        print(f"[{region_name}] Y_test shape: time={n_y}, lat={len(lat_hr)}, lon={len(lon_hr)}")
        print(f"[{region_name}] Scaler mode: {scaler_mode}")

        # Resolve regional scalers
        scaler_x_stats, scaler_y_stats = resolve_scaler_stats_for_region(
            scaler_mode=scaler_mode,
            ds_x=ds_x,
            ds_y=ds_y,
            variables=VARIABLES,
            batch_size=batch_size,
            scaler_stats_max_samples=scaler_stats_max_samples,
            global_scaler_stats=global_scaler_stats,
            region_name=region_name
        )

        # Load models
        models = OrderedDict()
        print(f"[{region_name}] Loading models...")
        for name, path in model_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"[{region_name}] Model checkpoint not found: {path}")
            model = load_one_model(path, scaler_x_stats, scaler_y_stats, device)
            models[name] = model
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  - {name:6s} | {path} | params={n_params:,}")

        # Initialize cubic spline interpolator
        cubic_interpolator = CubicSplineInterpolator2D(
            lat_lr=lat_lr, lon_lr=lon_lr,
            lat_hr=lat_hr, lon_hr=lon_hr,
            kx=CUBIC_SPLINE_KX, ky=CUBIC_SPLINE_KY
        )

        # Initialize spherical physics operator
        physics_ops = SphericalPhysicsOps(latitudes=lat_hr, longitudes=lon_hr)

        # Prescan raw dataset parameters
        raw_data_range, raw_mean_abs = precompute_raw_target_stats(
            ds_y=ds_y,
            variables=VARIABLES,
            batch_size=batch_size,
            max_samples=eval_max_samples
        )
        raw_mre_eps = np.maximum(MRE_EPS_ABS, MRE_EPS_REL * raw_mean_abs)

        print(f"[{region_name}] Raw data_range / mre_eps:")
        for i, name in enumerate(VAR_DISPLAY):
            print(f"  {name}: data_range={raw_data_range[i]:.6e}, mre_eps={raw_mre_eps[i]:.6e}")

        # Prescan physics parameters
        phys_mean_abs = precompute_physics_target_mean_abs(
            ds_y=ds_y,
            variables=VARIABLES,
            batch_size=batch_size,
            max_samples=eval_max_samples,
            physics_ops=physics_ops
        )
        phys_mre_eps = np.maximum(MRE_EPS_ABS, MRE_EPS_REL * phys_mean_abs)

        print(f"[{region_name}] Physics mre_eps:")
        for i, name in enumerate(PHYS_DERIVED_NAMES):
            print(f"  {name}: mre_eps={phys_mre_eps[i]:.6e}")

        # Execute primary model inference
        raw_result, phys_result, raw_payloads, phys_payloads = evaluate_all_methods(
            ds_x=ds_x,
            ds_y=ds_y,
            variables=VARIABLES,
            models=models,
            cubic_interpolator=cubic_interpolator,
            physics_ops=physics_ops,
            scaler_x_stats=scaler_x_stats,
            scaler_y_stats=scaler_y_stats,
            raw_data_range=raw_data_range,
            raw_mre_eps=raw_mre_eps,
            phys_mre_eps=phys_mre_eps,
            batch_size=batch_size,
            max_samples=eval_max_samples,
            device=device
        )

        # Construct point estimate DataFrames
        df_raw = build_metrics_dataframe(
            result_dict=raw_result,
            var_names=VAR_DISPLAY,
            metric_order=RAW_METRIC_ORDER
        )
        df_phys = build_metrics_dataframe(
            result_dict=phys_result,
            var_names=PHYS_DERIVED_NAMES,
            metric_order=PHYS_METRIC_ORDER
        )

        df_raw.insert(0, "Region", region_name)
        df_phys.insert(0, "Region", region_name)

        # Compute Bootstrap Confidence Intervals
        df_ci_long = None
        if ENABLE_BOOTSTRAP:
            scheme = BOOTSTRAP_SCHEME.strip().lower()
            ci_frames = []

            for method_name in raw_result.keys():
                # RAW variables CI
                seed_raw = make_deterministic_seed(
                    BOOTSTRAP_SEED, region_name, method_name, "raw"
                )
                df_ci_raw = bootstrap_ci_dataframe_for_method_domain(
                    region_name=region_name,
                    model_name=method_name,
                    domain_name="raw",
                    payload=raw_payloads[method_name],
                    point_metric_map=raw_result[method_name],
                    metric_order=RAW_METRIC_ORDER,
                    var_names=VAR_DISPLAY,
                    n_resamples=BOOTSTRAP_N_RESAMPLES,
                    confidence=BOOTSTRAP_CONFIDENCE,
                    ci_method=BOOTSTRAP_CI_METHOD,
                    scheme=scheme,
                    block_length_rule=BOOTSTRAP_BLOCK_LENGTH,
                    jackknife_rule=BOOTSTRAP_JACKKNIFE,
                    chunk_size=BOOTSTRAP_CHUNK,
                    seed=seed_raw
                )
                ci_frames.append(df_ci_raw)

                # PHYSICS derived variables CI
                seed_phys = make_deterministic_seed(
                    BOOTSTRAP_SEED, region_name, method_name, "physics"
                )
                df_ci_phys = bootstrap_ci_dataframe_for_method_domain(
                    region_name=region_name,
                    model_name=method_name,
                    domain_name="physics",
                    payload=phys_payloads[method_name],
                    point_metric_map=phys_result[method_name],
                    metric_order=PHYS_METRIC_ORDER,
                    var_names=PHYS_DERIVED_NAMES,
                    n_resamples=BOOTSTRAP_N_RESAMPLES,
                    confidence=BOOTSTRAP_CONFIDENCE,
                    ci_method=BOOTSTRAP_CI_METHOD,
                    scheme=scheme,
                    block_length_rule=BOOTSTRAP_BLOCK_LENGTH,
                    jackknife_rule=BOOTSTRAP_JACKKNIFE,
                    chunk_size=BOOTSTRAP_CHUNK,
                    seed=seed_phys
                )
                ci_frames.append(df_ci_phys)

            df_ci_long = pd.concat(ci_frames, ignore_index=True)

        return df_raw, df_phys, df_ci_long

    finally:
        ds_x.close()
        ds_y.close()


# =============================================================================
# 6) Entry Point
# =============================================================================
def main():
    if SCALER_MODE not in VALID_SCALER_MODES:
        raise ValueError(f"SCALER_MODE must be one of {sorted(VALID_SCALER_MODES)}, got {SCALER_MODE}")

    if ENABLE_BOOTSTRAP:
        ci_method_key, ci_method_disp = normalize_ci_method_name(BOOTSTRAP_CI_METHOD)
        scheme = BOOTSTRAP_SCHEME.strip().lower()
        jk_rule = BOOTSTRAP_JACKKNIFE.strip().lower()

        if scheme not in VALID_BOOTSTRAP_SCHEMES:
            raise ValueError(f"BOOTSTRAP_SCHEME must be one of {sorted(VALID_BOOTSTRAP_SCHEMES)}, got {BOOTSTRAP_SCHEME}")
        if jk_rule not in VALID_BOOTSTRAP_JACKKNIFE:
            raise ValueError(f"BOOTSTRAP_JACKKNIFE must be one of {sorted(VALID_BOOTSTRAP_JACKKNIFE)}, got {BOOTSTRAP_JACKKNIFE}")
        if BOOTSTRAP_N_RESAMPLES <= 0:
            raise ValueError("BOOTSTRAP_N_RESAMPLES must be positive.")
        if BOOTSTRAP_CHUNK <= 0:
            raise ValueError("BOOTSTRAP_CHUNK must be positive.")

    set_seed(RANDOM_SEED)
    device = torch.device(DEVICE)

    print("=" * 90)
    print("ERA5 Multi-Region Super-Resolution Model Evaluation Started")
    print(f"Device: {device}")
    print(f"SCALER_MODE: {SCALER_MODE}")
    if ENABLE_BOOTSTRAP:
        print(
            f"Bootstrap Configuration: ENABLE={ENABLE_BOOTSTRAP}, "
            f"METHOD={BOOTSTRAP_CI_METHOD}, CONF={BOOTSTRAP_CONFIDENCE}, "
            f"N={BOOTSTRAP_N_RESAMPLES}, SCHEME={BOOTSTRAP_SCHEME}, "
            f"BLOCK={BOOTSTRAP_BLOCK_LENGTH}, JK={BOOTSTRAP_JACKKNIFE}, CHUNK={BOOTSTRAP_CHUNK}"
        )
    print("=" * 90)

    # Check model checkpoints
    for p in MODEL_PATHS.values():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Model checkpoint missing: {p}")

    # Load global scalers if configured
    global_scaler_stats = None
    if SCALER_MODE == "global_joblib":
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Global scaler file missing: {SCALER_PATH}")
        global_scaler_stats = load_scaler_stats(SCALER_PATH)
        print(f"Loaded global scaler file: {SCALER_PATH}")

    all_raw_df = []
    all_phys_df = []
    all_ci_df = []

    for region_name, (x_path, y_path) in REGION_DATASETS.items():
        df_raw_region, df_phys_region, df_ci_region = evaluate_one_region(
            region_name=region_name,
            x_path=x_path,
            y_path=y_path,
            model_paths=MODEL_PATHS,
            scaler_mode=SCALER_MODE,
            global_scaler_stats=global_scaler_stats,
            batch_size=BATCH_SIZE,
            eval_max_samples=EVAL_MAX_SAMPLES,
            scaler_stats_max_samples=SCALER_STATS_MAX_SAMPLES,
            device=device
        )
        all_raw_df.append(df_raw_region)
        all_phys_df.append(df_phys_region)

        if ENABLE_BOOTSTRAP and (df_ci_region is not None):
            all_ci_df.append(df_ci_region)

    df_raw_all = pd.concat(all_raw_df, ignore_index=True)
    df_phys_all = pd.concat(all_phys_df, ignore_index=True)

    # Merge raw and physical metrics by Region + Model
    df_all = pd.concat(
        [
            df_raw_all.set_index(["Region", "Model"]),
            df_phys_all.set_index(["Region", "Model"]),
        ],
        axis=1
    ).reset_index()

    csv_raw_path = f"{CSV_RAW_METRICS_PREFIX}_{SCALER_MODE}.csv"
    csv_phys_path = f"{CSV_PHYS_METRICS_PREFIX}_{SCALER_MODE}.csv"
    csv_all_path = f"{CSV_ALL_IN_ONE_PREFIX}_{SCALER_MODE}.csv"

    df_raw_all.to_csv(csv_raw_path, index=False, float_format="%.6e")
    df_phys_all.to_csv(csv_phys_path, index=False, float_format="%.6e")
    df_all.to_csv(csv_all_path, index=False, float_format="%.6e")

    csv_ci_path = None
    if ENABLE_BOOTSTRAP and len(all_ci_df) > 0:
        ci_method_key, ci_method_disp = normalize_ci_method_name(BOOTSTRAP_CI_METHOD)
        scheme = BOOTSTRAP_SCHEME.strip().lower()

        df_ci_all = pd.concat(all_ci_df, ignore_index=True)
        csv_ci_path = (
            f"{CSV_BOOTSTRAP_CI_LONG_PREFIX}_"
            f"{SCALER_MODE}_"
            f"{ci_method_disp}_"
            f"{scheme}_"
            f"B{BOOTSTRAP_N_RESAMPLES}.csv"
        )
        df_ci_all.to_csv(csv_ci_path, index=False, float_format="%.6e")

    print("\n" + "=" * 90)
    print("Evaluation completed across all regions. Output CSV files saved:")
    print(f"  1) {csv_raw_path}")
    print(f"  2) {csv_phys_path}")
    print(f"  3) {csv_all_path}")
    if csv_ci_path is not None:
        print(f"  4) {csv_ci_path}   (Bootstrap CI long format)")
    print("=" * 90)


if __name__ == "__main__":
    main()