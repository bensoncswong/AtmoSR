#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA5 VDSR Super-Resolution Model Evaluation Script with Bootstrap 95% CIs
------------------------------------------------------------------------
Features:
  1) Cubic spline baseline evaluation
  2) (Optional) Bicubic baseline evaluation
  3) Multiple VDSR model evaluation & ablation studies
  4) Bootstrap 95% Confidence Interval (CI) estimation:
       - CI Methods: percentile / basic / BCa
       - Resampling Schemes: iid / moving_block / circular_block / stationary
       - BCa Jackknife Modes: delete1 / block / auto
  5) Resampling performed along the time dimension (spatial structure preserved)
  6) Fast bootstrap re-computation via 'Sufficient Statistics', avoiding repeated model inference.
"""

import os
import math
import random
import warnings
from collections import OrderedDict
from typing import Dict, List, Union, Optional

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
# 0) User-Defined Hyperparameters & Settings
# =============================================================================
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EVAL_MAX_SAMPLES = None  # Set to an integer (e.g., 512) for quick tests; None for full run

# Baseline options
ENABLE_BICUBIC_BASELINE = True
BICUBIC_BASELINE_ALIGN_CORNERS = False

# Cubic spline interpolation degrees
CUBIC_SPLINE_KX = 3
CUBIC_SPLINE_KY = 3

# MRE Stabilization parameters: eps_var = max(MRE_EPS_ABS, MRE_EPS_REL * mean_abs_truth_var)
MRE_EPS_REL = 1e-3
MRE_EPS_ABS = 1e-8

# SSIM Window Size
SSIM_WIN_SIZE = 7

# Physical Constants
CP_AIR = 1004.0
LV_WATER = 2.501e6
EARTH_RADIUS = 6371000.0
OMEGA = 7.2921e-5
F_CORIOLIS_ABS_MIN = 2.5e-5
TAN_CLIP = 50.0

# File Paths
X_TEST_PATH = "X2_test.nc"
Y_TEST_PATH = "Y_test.nc"
SCALER_PATH = "runs_vdsr_ablation/scalers.joblib"

MODEL_PATHS = OrderedDict([
    ("vdsr_l1_only",    "runs_vdsr_ablation/vdsr_l1_only/best_model.pth"),
    ("vdsr_l1_physics", "runs_vdsr_ablation/vdsr_l1_physics/best_model.pth"),
])

# Strict Check: If True, raises FileNotFoundError if any model file is missing
STRICT_MODEL_CHECK = False

# Output CSV Paths (Point Estimates)
CSV_RAW_METRICS_PATH = "vdsr_test_metrics_raw.csv"
CSV_PHYS_METRICS_PATH = "vdsr_test_metrics_physics.csv"
CSV_ALL_IN_ONE_PATH = "vdsr_test_metrics_all_in_one.csv"

# Output CSV Paths (Bootstrap CIs)
CSV_RAW_BOOTSTRAP_CI_PATH = "vdsr_test_metrics_raw_bootstrap_ci.csv"
CSV_PHYS_BOOTSTRAP_CI_PATH = "vdsr_test_metrics_physics_bootstrap_ci.csv"
CSV_BOOTSTRAP_LONG_PATH = "vdsr_test_metrics_bootstrap_ci_long.csv"
CSV_ALL_IN_ONE_BOOTSTRAP_CI_PATH = "vdsr_test_metrics_all_in_one_bootstrap_ci.csv"

# Bootstrap Configurations
ENABLE_BOOTSTRAP = True
BOOTSTRAP_N_RESAMPLES = 2000          # Recommended >= 2000 for BCa
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_CI_METHOD = "BCa"           # Options: {"percentile", "basic", "BCa"}
BOOTSTRAP_SCHEME = "circular_block"   # Options: {"iid", "moving_block", "circular_block", "stationary"}
BOOTSTRAP_BLOCK_LENGTH = "auto"       # Options: int / "auto" (N^(1/3)) / "sqrt" (N^(1/2))
BOOTSTRAP_JACKKNIFE = "auto"          # Options: "auto" / "delete1" / "block"
BOOTSTRAP_SEED = 42
BOOTSTRAP_CHUNK = 256                 # Number of resamples processed per batch (memory control)

# Variable Configurations
VARIABLES = ["z", "q", "t", "u", "v"]
VAR_DISPLAY = ["Z", "Q", "T", "U", "V"]
PHYS_DERIVED_NAMES = ["DIV", "VOR", "AGEO_MAG", "MSE_GRAD_X", "MSE_GRAD_Y"]

# Internal Constants
SUM_KEYS_2D = [
    "sum_abs", "sum_sq", "sum_rel",
    "sum_pred", "sum_true", "sum_pred2", "sum_true2", "sum_pred_true"
]


# =============================================================================
# 1) Core Utilities & Data Processing
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_compat(path: str, map_location):
    """
    Load PyTorch state dictionary with compatibility across PyTorch versions.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class ChannelScaler:
    """
    Dummy class for joblib deserialization compatibility with scaler objects.
    """
    def __init__(self):
        self.mean = None
        self.std = None
        self.n_channels = None


def _extract_scaler_stats(obj, obj_name="scaler"):
    if hasattr(obj, "mean") and hasattr(obj, "std"):
        mean = np.asarray(obj.mean, dtype=np.float32)
        std = np.asarray(obj.std, dtype=np.float32)
    elif isinstance(obj, dict) and ("mean" in obj) and ("std" in obj):
        mean = np.asarray(obj["mean"], dtype=np.float32)
        std = np.asarray(obj["std"], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported scaler format for {obj_name}, missing mean/std attributes.")
    std = np.where(np.abs(std) < 1e-12, 1.0, std)
    return mean, std


def load_scaler_stats(scaler_path: str):
    """
    Load channel-wise mean and std for input/target datasets from scalers.joblib.
    """
    obj = joblib.load(scaler_path)

    if isinstance(obj, dict) and ("scaler_X" in obj) and ("scaler_Y" in obj):
        x_mean, x_std = _extract_scaler_stats(obj["scaler_X"], "scaler_X")
        y_mean, y_std = _extract_scaler_stats(obj["scaler_Y"], "scaler_Y")
    elif isinstance(obj, dict) and all(k in obj for k in ["x_mean", "x_std", "y_mean", "y_std"]):
        x_mean = np.asarray(obj["x_mean"], dtype=np.float32)
        x_std = np.asarray(obj["x_std"], dtype=np.float32)
        y_mean = np.asarray(obj["y_mean"], dtype=np.float32)
        y_std = np.asarray(obj["y_std"], dtype=np.float32)
        x_std = np.where(np.abs(x_std) < 1e-12, 1.0, x_std)
        y_std = np.where(np.abs(y_std) < 1e-12, 1.0, y_std)
    else:
        raise ValueError("Invalid format in scalers.joblib: expected scaler_X and scaler_Y.")
    return {"mean": x_mean, "std": x_std}, {"mean": y_mean, "std": y_std}


def standardize_batch(x_raw, mean, std):
    return ((x_raw - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)).astype(np.float32)


def inverse_standardize_batch(y_std, mean, std):
    return (y_std * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)).astype(np.float32)


def read_batch_from_ds(ds, variables, start, end):
    """
    Read batch data from xarray dataset into array of shape: (B, C, H, W).
    """
    arr = np.stack(
        [ds[v].isel(time=slice(start, end)).values.astype(np.float32) for v in variables],
        axis=1
    )
    return arr


def collect_available_models(model_paths: OrderedDict, strict: bool = False):
    available = OrderedDict()
    missing = []
    for name, path in model_paths.items():
        if os.path.exists(path):
            available[name] = path
        else:
            missing.append((name, path))

    if missing:
        msg = "The following model paths do not exist:\n" + "\n".join([f"  - {n}: {p}" for n, p in missing])
        if strict:
            raise FileNotFoundError(msg)
        warnings.warn(msg + "\nSkipping missing models automatically.")
    return available


# =============================================================================
# 2) Neural Network Architecture (VDSR)
# =============================================================================
class VDSRResidualNet(nn.Module):
    """
    Deep CNN architecture without Batch Normalization.
    Cascaded Conv3x3 + ReLU layers predicting residual features.
    """
    def __init__(self, in_channels=5, out_channels=5, num_features=64, num_layers=18, zero_init_last=True):
        super().__init__()
        if num_layers < 3:
            raise ValueError("VDSR num_layers must be >= 3.")

        layers = [
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ]

        self.features = nn.Sequential(*layers)
        self.output_conv = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
        self.zero_init_last = bool(zero_init_last)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.zero_init_last:
            nn.init.zeros_(self.output_conv.weight)
            if self.output_conv.bias is not None:
                nn.init.zeros_(self.output_conv.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.output_conv(x)
        return x


class VDSRSRModel(nn.Module):
    """
    Complete VDSR Model:
      baseline = bicubic(x_std) -> affine transformed to y_std
      residual = residual_net(baseline)
      output   = baseline + residual
    """
    def __init__(
        self,
        in_channels, out_channels,
        num_features, num_layers, upscale_factor,
        scaler_X_mean, scaler_X_std,
        scaler_Y_mean, scaler_Y_std,
        align_corners=False,
        zero_init_last=True
    ):
        super().__init__()
        self.upscale_factor = int(upscale_factor)
        self.align_corners = bool(align_corners)

        self.residual_net = VDSRResidualNet(
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=num_features,
            num_layers=num_layers,
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
        base_y_std = base_x_std * self.x2y_alpha + self.x2y_beta
        return base_y_std

    def forward(self, x_std):
        base = self.bicubic_baseline_ystd(x_std)
        residual = self.residual_net(base)
        out = base + residual
        return out


def infer_vdsr_hparams_from_state_dict(state_dict):
    """
    Infer VDSR hyper-parameters directly from state_dict.
    """
    key_in = "residual_net.features.0.weight"
    key_out = "residual_net.output_conv.weight"

    if key_in not in state_dict or key_out not in state_dict:
        raise KeyError("State dict missing required keys for VDSR initialization.")

    in_channels = state_dict[key_in].shape[1]
    num_features = state_dict[key_in].shape[0]
    out_channels = state_dict[key_out].shape[0]

    n_feature_convs = 0
    for k, v in state_dict.items():
        if k.startswith("residual_net.features.") and k.endswith(".weight"):
            if hasattr(v, "ndim") and int(v.ndim) == 4:
                n_feature_convs += 1

    num_layers = n_feature_convs + 1

    return {
        "in_channels": int(in_channels),
        "out_channels": int(out_channels),
        "num_features": int(num_features),
        "num_layers": int(num_layers),
    }


def load_one_model(ckpt_path, scaler_x_fallback, scaler_y_fallback, device, default_upscale_factor):
    ckpt = torch_load_compat(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    if not isinstance(state_dict, dict):
        raise TypeError(f"Invalid checkpoint format in {ckpt_path}")

    # Remove DataParallel prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    hp = infer_vdsr_hparams_from_state_dict(state_dict)

    if isinstance(ckpt, dict):
        x_mean = np.asarray(ckpt.get("scaler_X_mean", scaler_x_fallback["mean"]), dtype=np.float32)
        x_std = np.asarray(ckpt.get("scaler_X_std", scaler_x_fallback["std"]), dtype=np.float32)
        y_mean = np.asarray(ckpt.get("scaler_Y_mean", scaler_y_fallback["mean"]), dtype=np.float32)
        y_std = np.asarray(ckpt.get("scaler_Y_std", scaler_y_fallback["std"]), dtype=np.float32)

        upscale_factor = int(ckpt.get("upscale_factor", default_upscale_factor))
        align_corners = bool(ckpt.get("bicubic_align_corners", False))
        zero_init_last = bool(ckpt.get("zero_init_last", False))
    else:
        x_mean = scaler_x_fallback["mean"]
        x_std = scaler_x_fallback["std"]
        y_mean = scaler_y_fallback["mean"]
        y_std = scaler_y_fallback["std"]
        upscale_factor = int(default_upscale_factor)
        align_corners = False
        zero_init_last = False

    x_std = np.where(np.abs(x_std) < 1e-12, 1.0, x_std)
    y_std = np.where(np.abs(y_std) < 1e-12, 1.0, y_std)

    model = VDSRSRModel(
        in_channels=hp["in_channels"],
        out_channels=hp["out_channels"],
        num_features=hp["num_features"],
        num_layers=hp["num_layers"],
        upscale_factor=upscale_factor,
        scaler_X_mean=x_mean,
        scaler_X_std=x_std,
        scaler_Y_mean=y_mean,
        scaler_Y_std=y_std,
        align_corners=align_corners,
        zero_init_last=zero_init_last
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return {
        "model": model,
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
        "upscale_factor": int(upscale_factor),
        "align_corners": bool(align_corners),
        "num_layers": int(hp["num_layers"]),
        "num_features": int(hp["num_features"]),
    }


# =============================================================================
# 3) Interpolators & Spherical Physical Operations
# =============================================================================
def to_increasing_axis(axis_1d):
    axis_1d = np.asarray(axis_1d, dtype=np.float64)
    if axis_1d[1] > axis_1d[0]:
        return axis_1d, False
    return axis_1d[::-1], True


class CubicSplineInterpolator2D:
    """
    2D Cubic Spline Interpolator mapping LR grids to HR grids.
    """
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


class BicubicTorchInterpolator:
    """
    PyTorch Bicubic Interpolator for baseline comparison.
    """
    def __init__(self, out_h, out_w, align_corners=False, device="cpu"):
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        self.align_corners = bool(align_corners)
        self.device = torch.device(device)

    def __call__(self, x_batch):
        x_t = torch.from_numpy(x_batch).to(self.device, non_blocking=True)
        with torch.no_grad():
            y_t = F.interpolate(
                x_t,
                size=(self.out_h, self.out_w),
                mode="bicubic",
                align_corners=self.align_corners
            )
        return y_t.detach().cpu().numpy().astype(np.float32)


class SphericalPhysicsOps:
    """
    Spherical coordinate differential operators and physical diagnostic calculators (NumPy).
    """
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
        self.inv_dx = 1.0 / dx  # (H,)

        tan_lat = np.clip(np.tan(lat_rad), -TAN_CLIP, TAN_CLIP)
        self.tan_over_r = tan_lat / EARTH_RADIUS  # (H,)

        f_raw = 2.0 * OMEGA * np.sin(lat_rad)
        f_clip = f_raw.copy()
        mask = np.abs(f_clip) < F_CORIOLIS_ABS_MIN
        f_clip[mask] = np.where(f_clip[mask] >= 0, F_CORIOLIS_ABS_MIN, -F_CORIOLIS_ABS_MIN)
        self.inv_f = 1.0 / f_clip  # (H,)

    def ddx(self, field):
        # field shape: (B, H, W)
        left = np.concatenate([field[:, :, :1], field[:, :, :-1]], axis=2)
        right = np.concatenate([field[:, :, 1:], field[:, :, -1:]], axis=2)
        return 0.5 * (right - left) * self.inv_dx[None, :, None]

    def ddy(self, field):
        # field shape: (B, H, W)
        top = np.concatenate([field[:, :1, :], field[:, :-1, :]], axis=1)
        bottom = np.concatenate([field[:, 1:, :], field[:, -1:, :]], axis=1)

        if self.lat_decreasing:
            return 0.5 * (top - bottom) * self.inv_dy
        return 0.5 * (bottom - top) * self.inv_dy

    def divergence(self, u, v):
        du_dx = self.ddx(u)
        dv_dy = self.ddy(v)
        metric = v * self.tan_over_r[None, :, None]
        return du_dx + dv_dy - metric

    def vorticity(self, u, v):
        dv_dx = self.ddx(v)
        du_dy = self.ddy(u)
        metric = u * self.tan_over_r[None, :, None]
        return dv_dx - du_dy + metric

    def geostrophic_wind(self, z):
        dz_dy = self.ddy(z)
        dz_dx = self.ddx(z)
        ug = -self.inv_f[None, :, None] * dz_dy
        vg = self.inv_f[None, :, None] * dz_dx
        return ug, vg

    def moist_static_energy(self, t, z, q):
        return CP_AIR * t + z + LV_WATER * q

    def moist_static_energy_gradients(self, t, z, q):
        mse = self.moist_static_energy(t, z, q)
        return self.ddx(mse), self.ddy(mse)


def compute_physics_derived(batch_5ch, ops: SphericalPhysicsOps):
    """
    Input:
      batch_5ch: (B, 5, H, W) ordered as [z, q, t, u, v]
    Output:
      (B, 5, H, W) ordered as [DIV, VOR, AGEO_MAG, MSE_GRAD_X, MSE_GRAD_Y]
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
# 4) Sufficient Statistics Collector
# =============================================================================
class SufficientStatsCollector:
    """
    Accumulates per-time-sample sufficient statistics required for evaluation metrics.
    Enables instant point estimation and rapid Bootstrap re-weighting without re-running model inference.
    """
    def __init__(self, var_names, mre_eps, data_range=None, enable_image_metrics=False, ssim_win_size=7):
        self.var_names = list(var_names)
        self.nv = len(self.var_names)

        self.mre_eps = np.asarray(mre_eps, dtype=np.float64).reshape(-1)
        if self.mre_eps.shape[0] != self.nv:
            raise ValueError("mre_eps length must equal the number of variables.")

        self.data_range = None
        if data_range is not None:
            self.data_range = np.asarray(data_range, dtype=np.float64).reshape(-1)
            if self.data_range.shape[0] != self.nv:
                raise ValueError("data_range length must equal the number of variables.")

        self.enable_image_metrics = bool(enable_image_metrics)
        self.ssim_win_size = int(ssim_win_size)

        self._lists = {
            "sum_abs": [],
            "sum_sq": [],
            "sum_rel": [],
            "count": [],
            "sum_pred": [],
            "sum_true": [],
            "sum_pred2": [],
            "sum_true2": [],
            "sum_pred_true": [],
        }
        if self.enable_image_metrics:
            self._lists["ssim_sum"] = []
            self._lists["ssim_count"] = []

        self._arrays = None

    def update(self, pred, true):
        """
        pred/true shape: (B, C, H, W)
        """
        pred = pred.astype(np.float64, copy=False)
        true = true.astype(np.float64, copy=False)

        bsz, ch, h, w = pred.shape
        if ch != self.nv:
            raise ValueError(f"Channel dimension mismatch: expected {self.nv}, got {ch}")

        err = pred - true

        sum_abs = np.sum(np.abs(err), axis=(2, 3))
        sum_sq = np.sum(err ** 2, axis=(2, 3))

        denom = np.maximum(np.abs(true), self.mre_eps.reshape(1, -1, 1, 1))
        sum_rel = np.sum(np.abs(err) / denom, axis=(2, 3))

        pred_flat = pred.reshape(bsz, ch, -1)
        true_flat = true.reshape(bsz, ch, -1)

        sum_pred = pred_flat.sum(axis=2)
        sum_true = true_flat.sum(axis=2)
        sum_pred2 = (pred_flat ** 2).sum(axis=2)
        sum_true2 = (true_flat ** 2).sum(axis=2)
        sum_pred_true = (pred_flat * true_flat).sum(axis=2)

        n_pix = float(h * w)
        count = np.full((bsz,), n_pix, dtype=np.float64)

        self._lists["sum_abs"].append(sum_abs)
        self._lists["sum_sq"].append(sum_sq)
        self._lists["sum_rel"].append(sum_rel)
        self._lists["count"].append(count)
        self._lists["sum_pred"].append(sum_pred)
        self._lists["sum_true"].append(sum_true)
        self._lists["sum_pred2"].append(sum_pred2)
        self._lists["sum_true2"].append(sum_true2)
        self._lists["sum_pred_true"].append(sum_pred_true)

        if self.enable_image_metrics:
            ssim_sum, ssim_count = self._compute_ssim_batch(pred, true)
            self._lists["ssim_sum"].append(ssim_sum)
            self._lists["ssim_count"].append(ssim_count)

        self._arrays = None  # Clear cached arrays

    def _compute_ssim_batch(self, pred, true):
        bsz, ch, h, w = pred.shape
        ssim_sum = np.zeros((bsz, ch), dtype=np.float64)
        ssim_count = np.zeros((bsz, ch), dtype=np.float64)

        win = min(self.ssim_win_size, h, w)
        if win % 2 == 0:
            win -= 1
        if win < 3:
            return ssim_sum, ssim_count

        for c in range(ch):
            dr = float(self.data_range[c]) if self.data_range is not None else None
            if dr is None or (not np.isfinite(dr)) or dr <= 0:
                continue
            for b in range(bsz):
                val = ssim_2d(
                    true[b, c], pred[b, c],
                    data_range=dr,
                    win_size=win
                )
                if np.isfinite(val):
                    ssim_sum[b, c] = float(val)
                    ssim_count[b, c] = 1.0
        return ssim_sum, ssim_count

    def get_arrays(self):
        if self._arrays is not None:
            return self._arrays

        if len(self._lists["count"]) == 0:
            raise RuntimeError("No accumulated sample stats available.")

        arrays = {}
        for k in SUM_KEYS_2D:
            arrays[k] = np.concatenate(self._lists[k], axis=0).astype(np.float64, copy=False)
        arrays["count"] = np.concatenate(self._lists["count"], axis=0).astype(np.float64, copy=False)

        if self.enable_image_metrics:
            arrays["ssim_sum"] = np.concatenate(self._lists["ssim_sum"], axis=0).astype(np.float64, copy=False)
            arrays["ssim_count"] = np.concatenate(self._lists["ssim_count"], axis=0).astype(np.float64, copy=False)

        self._arrays = arrays
        return self._arrays

    def total_over_all(self):
        arr = self.get_arrays()
        totals = {}
        for k in SUM_KEYS_2D:
            totals[k] = arr[k].sum(axis=0)
        totals["count"] = float(arr["count"].sum())

        if self.enable_image_metrics:
            totals["ssim_sum"] = arr["ssim_sum"].sum(axis=0)
            totals["ssim_count"] = arr["ssim_count"].sum(axis=0)
        return totals

    def totals_from_counts(self, counts):
        """
        counts shape: (R, N_samples), containing frequency of each sample per bootstrap drawing.
        Returns: totals dictionary (batch format, shapes (R, C) or (R,))
        """
        arr = self.get_arrays()
        counts = np.asarray(counts, dtype=np.float64)

        totals = {}
        for k in SUM_KEYS_2D:
            totals[k] = counts @ arr[k]  # (R, N) @ (N, C) => (R, C)
        totals["count"] = counts @ arr["count"]  # (R, N) @ (N,) => (R,)

        if self.enable_image_metrics:
            totals["ssim_sum"] = counts @ arr["ssim_sum"]
            totals["ssim_count"] = counts @ arr["ssim_count"]
        return totals

    def metrics_from_totals(self, totals):
        cnt = max(float(totals["count"]), 1.0)

        mse = totals["sum_sq"] / cnt
        rmse = np.sqrt(mse)
        mae = totals["sum_abs"] / cnt
        mre = totals["sum_rel"] / cnt

        n = cnt
        num = n * totals["sum_pred_true"] - totals["sum_pred"] * totals["sum_true"]
        den = np.sqrt(
            np.maximum(n * totals["sum_pred2"] - totals["sum_pred"] ** 2, 0.0) *
            np.maximum(n * totals["sum_true2"] - totals["sum_true"] ** 2, 0.0)
        )
        corr = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

        out = {
            "RMSE": rmse,
            "MSE": mse,
            "MAE": mae,
            "MRE": mre,
            "Correlation": corr
        }

        if self.enable_image_metrics:
            dr = np.maximum(self.data_range, 1e-12)
            psnr = 20.0 * np.log10(dr) - 10.0 * np.log10(np.maximum(mse, 1e-12))
            ssim = np.divide(
                totals["ssim_sum"], totals["ssim_count"],
                out=np.full_like(totals["ssim_sum"], np.nan),
                where=totals["ssim_count"] > 0
            )
            out["PSNR"] = psnr
            out["SSIM"] = ssim

        return out

    def metrics_from_totals_batch(self, totals):
        cnt = np.maximum(np.asarray(totals["count"], dtype=np.float64), 1.0).reshape(-1, 1)

        mse = totals["sum_sq"] / cnt
        rmse = np.sqrt(mse)
        mae = totals["sum_abs"] / cnt
        mre = totals["sum_rel"] / cnt

        n = cnt
        num = n * totals["sum_pred_true"] - totals["sum_pred"] * totals["sum_true"]
        den = np.sqrt(
            np.maximum(n * totals["sum_pred2"] - totals["sum_pred"] ** 2, 0.0) *
            np.maximum(n * totals["sum_true2"] - totals["sum_true"] ** 2, 0.0)
        )
        corr = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

        out = {
            "RMSE": rmse,
            "MSE": mse,
            "MAE": mae,
            "MRE": mre,
            "Correlation": corr
        }

        if self.enable_image_metrics:
            dr = np.maximum(self.data_range.reshape(1, -1), 1e-12)
            psnr = 20.0 * np.log10(dr) - 10.0 * np.log10(np.maximum(mse, 1e-12))
            ssim = np.divide(
                totals["ssim_sum"], totals["ssim_count"],
                out=np.full_like(totals["ssim_sum"], np.nan),
                where=totals["ssim_count"] > 0
            )
            out["PSNR"] = psnr
            out["SSIM"] = ssim

        return out

    def point_estimate(self):
        totals = self.total_over_all()
        return self.metrics_from_totals(totals)


def flatten_metric_map(metric_map, metric_order):
    return np.concatenate([np.asarray(metric_map[m], dtype=np.float64).reshape(-1) for m in metric_order], axis=0)


def flatten_metric_batch(metric_batch_map, metric_order):
    return np.concatenate([np.asarray(metric_batch_map[m], dtype=np.float64) for m in metric_order], axis=1)


def unflatten_metric_vector(flat_vec, metric_order, n_vars):
    out = {}
    p = 0
    for m in metric_order:
        out[m] = np.asarray(flat_vec[p:p + n_vars], dtype=np.float64)
        p += n_vars
    return out


def build_metrics_dataframe(result_dict, var_names, metric_order):
    rows = []
    for model_name, metric_map in result_dict.items():
        row = {"Model": model_name}
        for m in metric_order:
            vals = metric_map[m]
            for i, vname in enumerate(var_names):
                row[f"{m}_{vname}"] = float(vals[i])
        rows.append(row)

    columns = ["Model"] + [f"{m}_{v}" for m in metric_order for v in var_names]
    return pd.DataFrame(rows, columns=columns)


# =============================================================================
# 5) Bootstrap Algorithms & Utilities
# =============================================================================
def normalize_ci_method(method: str) -> str:
    m = str(method).strip().lower()
    if m == "bca":
        return "bca"
    if m in ("percentile", "basic"):
        return m
    raise ValueError(f"Unknown BOOTSTRAP_CI_METHOD: {method}")


def normalize_bootstrap_scheme(scheme: str) -> str:
    s = str(scheme).strip().lower()
    valid = {"iid", "moving_block", "circular_block", "stationary"}
    if s not in valid:
        raise ValueError(f"Unknown BOOTSTRAP_SCHEME: {scheme}. Expected one of {sorted(valid)}")
    return s


def resolve_block_length(cfg: Union[int, str], n_samples: int) -> int:
    if isinstance(cfg, str):
        key = cfg.strip().lower()
        if key == "auto":
            L = int(round(n_samples ** (1.0 / 3.0)))
        elif key == "sqrt":
            L = int(round(n_samples ** 0.5))
        else:
            raise ValueError(f"Unknown BOOTSTRAP_BLOCK_LENGTH: {cfg}")
    else:
        L = int(cfg)

    L = max(1, min(L, n_samples))
    return int(L)


def resolve_jackknife_mode(cfg: str, scheme: str, n_samples: int) -> str:
    key = str(cfg).strip().lower()
    if key in ("delete1", "block"):
        return key
    if key == "auto":
        if scheme == "iid" and n_samples <= 4000:
            return "delete1"
        return "block"
    raise ValueError(f"Unknown BOOTSTRAP_JACKKNIFE: {cfg}")


def generate_bootstrap_indices_chunk(
    n_samples: int,
    n_resamples_chunk: int,
    scheme: str,
    block_length: int,
    rng: np.random.Generator
):
    scheme = normalize_bootstrap_scheme(scheme)
    n = int(n_samples)
    r = int(n_resamples_chunk)

    if scheme == "iid":
        return rng.integers(0, n, size=(r, n), dtype=np.int64)

    L = max(1, min(int(block_length), n))

    if scheme in ("moving_block", "circular_block"):
        n_blocks = int(math.ceil(n / L))
        offsets = np.arange(L, dtype=np.int64)

        if scheme == "moving_block":
            start_high = max(1, n - L + 1)
            starts = rng.integers(0, start_high, size=(r, n_blocks), dtype=np.int64)
            idx = starts[:, :, None] + offsets[None, None, :]
        else:  # circular_block
            starts = rng.integers(0, n, size=(r, n_blocks), dtype=np.int64)
            idx = (starts[:, :, None] + offsets[None, None, :]) % n

        idx = idx.reshape(r, -1)[:, :n]
        return idx

    if scheme == "stationary":
        p = 1.0 / float(max(L, 1))
        idx = np.empty((r, n), dtype=np.int64)
        idx[:, 0] = rng.integers(0, n, size=r, dtype=np.int64)

        if n == 1:
            return idx

        restart = rng.random((r, n - 1)) < p
        new_start = rng.integers(0, n, size=(r, n - 1), dtype=np.int64)

        for t in range(1, n):
            cont = (idx[:, t - 1] + 1) % n
            idx[:, t] = np.where(restart[:, t - 1], new_start[:, t - 1], cont)
        return idx

    raise RuntimeError("Unreachable bootstrap scheme state.")


def bootstrap_index_generator(
    n_samples: int,
    n_resamples: int,
    chunk: int,
    scheme: str,
    block_length: int,
    rng: np.random.Generator
):
    done = 0
    while done < n_resamples:
        r = min(chunk, n_resamples - done)
        idx = generate_bootstrap_indices_chunk(
            n_samples=n_samples,
            n_resamples_chunk=r,
            scheme=scheme,
            block_length=block_length,
            rng=rng
        )
        yield idx
        done += r


def indices_to_counts(indices: np.ndarray, n_samples: int):
    r = indices.shape[0]
    counts = np.zeros((r, n_samples), dtype=np.float64)
    for i in range(r):
        counts[i] = np.bincount(indices[i], minlength=n_samples).astype(np.float64)
    return counts


def colwise_quantile(data: np.ndarray, q):
    data = np.asarray(data, dtype=np.float64)
    d = data.shape[1]

    if np.isscalar(q):
        q_arr = np.full(d, float(q), dtype=np.float64)
    else:
        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if q_arr.shape[0] != d:
            raise ValueError("Length of q must equal the number of columns in data.")

    out = np.full(d, np.nan, dtype=np.float64)
    for j in range(d):
        col = data[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        qq = q_arr[j]
        if not np.isfinite(qq):
            continue
        qq = float(np.clip(qq, 0.0, 1.0))
        try:
            out[j] = np.quantile(col, qq, method="linear")
        except TypeError:
            out[j] = np.quantile(col, qq, interpolation="linear")
    return out


def _circular_block_sum_2d(arr: np.ndarray, L: int):
    n = arr.shape[0]
    if L <= 0:
        raise ValueError("L must be > 0")
    ext = np.concatenate([arr, arr[:max(L - 1, 0)]], axis=0)
    csum = np.vstack([np.zeros((1, arr.shape[1]), dtype=np.float64), np.cumsum(ext, axis=0)])
    starts = np.arange(n)
    return csum[starts + L] - csum[starts]


def _circular_block_sum_1d(arr: np.ndarray, L: int):
    n = arr.shape[0]
    if L <= 0:
        raise ValueError("L must be > 0")
    ext = np.concatenate([arr, arr[:max(L - 1, 0)]], axis=0)
    csum = np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(ext, axis=0)])
    starts = np.arange(n)
    return csum[starts + L] - csum[starts]


def build_jackknife_distribution(
    collector: SufficientStatsCollector,
    metric_order: List[str],
    mode: str,
    block_length: int
):
    arr = collector.get_arrays()
    n = arr["sum_abs"].shape[0]
    if n <= 2:
        return None, {"jackknife_mode": "none", "jackknife_block_length": np.nan, "n_jackknife": 0}

    totals_all = collector.total_over_all()
    mode = str(mode).strip().lower()

    if mode == "delete1":
        totals_batch = {}
        for k in SUM_KEYS_2D:
            totals_batch[k] = totals_all[k][None, :] - arr[k]
        totals_batch["count"] = totals_all["count"] - arr["count"]

        if collector.enable_image_metrics:
            totals_batch["ssim_sum"] = totals_all["ssim_sum"][None, :] - arr["ssim_sum"]
            totals_batch["ssim_count"] = totals_all["ssim_count"][None, :] - arr["ssim_count"]

        metric_batch = collector.metrics_from_totals_batch(totals_batch)
        jk_theta = flatten_metric_batch(metric_batch, metric_order)
        return jk_theta, {
            "jackknife_mode": "delete1",
            "jackknife_block_length": 1,
            "n_jackknife": int(jk_theta.shape[0])
        }

    if mode == "block":
        L = max(1, min(int(block_length), n - 1))
        totals_batch = {}

        for k in SUM_KEYS_2D:
            blk = _circular_block_sum_2d(arr[k], L)
            totals_batch[k] = totals_all[k][None, :] - blk

        blk_count = _circular_block_sum_1d(arr["count"], L)
        totals_batch["count"] = totals_all["count"] - blk_count

        if collector.enable_image_metrics:
            blk_ssim_sum = _circular_block_sum_2d(arr["ssim_sum"], L)
            blk_ssim_cnt = _circular_block_sum_2d(arr["ssim_count"], L)
            totals_batch["ssim_sum"] = totals_all["ssim_sum"][None, :] - blk_ssim_sum
            totals_batch["ssim_count"] = totals_all["ssim_count"][None, :] - blk_ssim_cnt

        metric_batch = collector.metrics_from_totals_batch(totals_batch)
        jk_theta = flatten_metric_batch(metric_batch, metric_order)
        return jk_theta, {
            "jackknife_mode": "block",
            "jackknife_block_length": int(L),
            "n_jackknife": int(jk_theta.shape[0])
        }

    raise ValueError(f"Unknown jackknife mode: {mode}")


def compute_bca_acceleration(jk_theta: np.ndarray):
    d = jk_theta.shape[1]
    acc = np.full(d, np.nan, dtype=np.float64)

    for j in range(d):
        col = jk_theta[:, j]
        col = col[np.isfinite(col)]
        if col.size < 3:
            continue
        tdot = np.mean(col)
        diff = tdot - col
        num = np.sum(diff ** 3)
        den = 6.0 * (np.sum(diff ** 2) ** 1.5)
        if den > 0:
            acc[j] = num / den
        else:
            acc[j] = 0.0
    return acc


def compute_bca_z0(theta_hat: np.ndarray, boot_theta: np.ndarray):
    d = theta_hat.shape[0]
    z0 = np.full(d, np.nan, dtype=np.float64)

    for j in range(d):
        th = theta_hat[j]
        if not np.isfinite(th):
            continue
        col = boot_theta[:, j]
        col = col[np.isfinite(col)]
        m = col.size
        if m < 10:
            continue
        p = (np.sum(col < th) + 0.5 * np.sum(col == th)) / float(m)
        p = np.clip(p, 1.0 / (2.0 * m), 1.0 - 1.0 / (2.0 * m))
        z0[j] = norm.ppf(p)
    return z0


def bca_interval(theta_hat: np.ndarray, boot_theta: np.ndarray, jk_theta: np.ndarray, confidence: float):
    alpha = (1.0 - confidence) / 2.0
    z_alpha_low = norm.ppf(alpha)
    z_alpha_high = norm.ppf(1.0 - alpha)

    z0 = compute_bca_z0(theta_hat, boot_theta)
    acc = compute_bca_acceleration(jk_theta)

    d = theta_hat.shape[0]
    p_low = np.full(d, np.nan, dtype=np.float64)
    p_high = np.full(d, np.nan, dtype=np.float64)

    for j in range(d):
        if not np.isfinite(theta_hat[j]) or not np.isfinite(z0[j]):
            continue

        a = acc[j] if np.isfinite(acc[j]) else 0.0
        z = z0[j]

        den_l = 1.0 - a * (z + z_alpha_low)
        den_h = 1.0 - a * (z + z_alpha_high)

        if abs(den_l) < 1e-12 or abs(den_h) < 1e-12:
            continue

        pl = norm.cdf(z + (z + z_alpha_low) / den_l)
        ph = norm.cdf(z + (z + z_alpha_high) / den_h)

        pl = float(np.clip(pl, 0.0, 1.0))
        ph = float(np.clip(ph, 0.0, 1.0))

        if pl <= ph:
            p_low[j], p_high[j] = pl, ph
        else:
            p_low[j], p_high[j] = ph, pl

    lower = colwise_quantile(boot_theta, p_low)
    upper = colwise_quantile(boot_theta, p_high)

    extra = {
        "z0": z0,
        "acceleration": acc,
        "adj_alpha_low": p_low,
        "adj_alpha_high": p_high
    }
    return lower, upper, extra


def bootstrap_distribution(
    collector: SufficientStatsCollector,
    metric_order: List[str],
    n_resamples: int,
    scheme: str,
    block_length: int,
    chunk: int,
    seed: int,
    desc: str
):
    arr = collector.get_arrays()
    n_samples = arr["sum_abs"].shape[0]
    d = len(metric_order) * collector.nv

    boot_theta = np.full((n_resamples, d), np.nan, dtype=np.float64)

    rng = np.random.default_rng(seed)
    offset = 0

    pbar = tqdm(total=n_resamples, desc=desc)
    for idx_chunk in bootstrap_index_generator(
        n_samples=n_samples,
        n_resamples=n_resamples,
        chunk=chunk,
        scheme=scheme,
        block_length=block_length,
        rng=rng
    ):
        counts = indices_to_counts(idx_chunk, n_samples)
        totals_batch = collector.totals_from_counts(counts)
        metric_batch = collector.metrics_from_totals_batch(totals_batch)
        flat_chunk = flatten_metric_batch(metric_batch, metric_order)

        r = flat_chunk.shape[0]
        boot_theta[offset:offset + r] = flat_chunk
        offset += r
        pbar.update(r)
    pbar.close()

    return boot_theta


def run_bootstrap_ci_for_collector(
    collector: SufficientStatsCollector,
    metric_order: List[str],
    ci_method: str = "BCa",
    confidence: float = 0.95,
    n_resamples: int = 2000,
    scheme: str = "circular_block",
    block_length: Union[int, str] = "auto",
    jackknife: str = "auto",
    chunk: int = 256,
    seed: int = 42,
    desc: str = "Bootstrap"
):
    if n_resamples <= 0:
        raise ValueError("n_resamples must be > 0.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between (0, 1).")

    ci_method_norm = normalize_ci_method(ci_method)
    scheme_norm = normalize_bootstrap_scheme(scheme)

    n_samples = collector.get_arrays()["sum_abs"].shape[0]
    if n_samples < 2:
        raise ValueError("Insufficient sample count for bootstrap estimation.")

    block_len = resolve_block_length(block_length, n_samples)
    if scheme_norm == "iid":
        block_len = 1

    theta_hat_map = collector.point_estimate()
    theta_hat = flatten_metric_map(theta_hat_map, metric_order)

    boot_theta = bootstrap_distribution(
        collector=collector,
        metric_order=metric_order,
        n_resamples=n_resamples,
        scheme=scheme_norm,
        block_length=block_len,
        chunk=int(chunk),
        seed=int(seed),
        desc=desc
    )

    alpha = (1.0 - confidence) / 2.0
    lower = np.full_like(theta_hat, np.nan, dtype=np.float64)
    upper = np.full_like(theta_hat, np.nan, dtype=np.float64)

    ci_method_used = ci_method_norm
    jk_mode_used = "none"
    jk_block_len_used = np.nan

    if ci_method_norm == "percentile":
        lower = colwise_quantile(boot_theta, alpha)
        upper = colwise_quantile(boot_theta, 1.0 - alpha)

    elif ci_method_norm == "basic":
        ql = colwise_quantile(boot_theta, alpha)
        qh = colwise_quantile(boot_theta, 1.0 - alpha)
        lower = 2.0 * theta_hat - qh
        upper = 2.0 * theta_hat - ql

    elif ci_method_norm == "bca":
        jk_mode = resolve_jackknife_mode(jackknife, scheme_norm, n_samples)
        jk_block_len = 1 if jk_mode == "delete1" else max(1, min(block_len, n_samples - 1))

        jk_theta, jk_meta = build_jackknife_distribution(
            collector=collector,
            metric_order=metric_order,
            mode=jk_mode,
            block_length=jk_block_len
        )

        if (jk_theta is None) or (jk_theta.shape[0] < 3):
            warnings.warn("Insufficient jackknife samples for BCa; falling back to Percentile CI.")
            ci_method_used = "percentile(fallback)"
            lower = colwise_quantile(boot_theta, alpha)
            upper = colwise_quantile(boot_theta, 1.0 - alpha)
            jk_mode_used = "none"
            jk_block_len_used = np.nan
        else:
            lower, upper, _ = bca_interval(
                theta_hat=theta_hat,
                boot_theta=boot_theta,
                jk_theta=jk_theta,
                confidence=confidence
            )
            jk_mode_used = jk_meta.get("jackknife_mode", jk_mode)
            jk_block_len_used = jk_meta.get("jackknife_block_length", jk_block_len)

    # Ensure lower <= upper
    swap = np.isfinite(lower) & np.isfinite(upper) & (lower > upper)
    if np.any(swap):
        tmp = lower.copy()
        lower[swap] = upper[swap]
        upper[swap] = tmp[swap]

    lower_map = unflatten_metric_vector(lower, metric_order, collector.nv)
    upper_map = unflatten_metric_vector(upper, metric_order, collector.nv)

    valid_counts = np.sum(np.isfinite(boot_theta), axis=0)
    n_valid_min = int(valid_counts.min()) if valid_counts.size > 0 else 0

    return {
        "estimate": theta_hat_map,
        "ci_lower": lower_map,
        "ci_upper": upper_map,
        "meta": {
            "ci_method_requested": str(ci_method),
            "ci_method_used": str(ci_method_used),
            "bootstrap_scheme": str(scheme_norm),
            "block_length": int(block_len),
            "jackknife_mode": str(jk_mode_used),
            "jackknife_block_length": jk_block_len_used,
            "n_resamples": int(n_resamples),
            "confidence": float(confidence),
            "effective_n_samples": int(n_samples),
            "n_valid_bootstrap_min": int(n_valid_min),
            "seed": int(seed)
        }
    }


def run_bootstrap_for_methods(
    collectors: OrderedDict,
    metric_order: List[str],
    group_name: str
):
    out = OrderedDict()

    for idx, (method_name, collector) in enumerate(collectors.items()):
        seed_i = int(BOOTSTRAP_SEED + idx * 10007 + (0 if group_name == "raw" else 500000))
        desc = f"Bootstrap-{group_name}:{method_name}"

        print(f"\n[Bootstrap-{group_name}] {method_name}")
        out[method_name] = run_bootstrap_ci_for_collector(
            collector=collector,
            metric_order=metric_order,
            ci_method=BOOTSTRAP_CI_METHOD,
            confidence=BOOTSTRAP_CONFIDENCE,
            n_resamples=BOOTSTRAP_N_RESAMPLES,
            scheme=BOOTSTRAP_SCHEME,
            block_length=BOOTSTRAP_BLOCK_LENGTH,
            jackknife=BOOTSTRAP_JACKKNIFE,
            chunk=BOOTSTRAP_CHUNK,
            seed=seed_i,
            desc=desc
        )

    return out


def build_bootstrap_wide_dataframe(bootstrap_result_dict, var_names, metric_order, confidence):
    ci_tag = f"CI{int(round(confidence * 100))}"
    rows = []

    for model_name, res in bootstrap_result_dict.items():
        est = res["estimate"]
        low = res["ci_lower"]
        up = res["ci_upper"]
        meta = res["meta"]

        row = {
            "Model": model_name,
            "CI_Method_Requested": meta.get("ci_method_requested"),
            "CI_Method_Used": meta.get("ci_method_used"),
            "Bootstrap_Scheme": meta.get("bootstrap_scheme"),
            "Block_Length": meta.get("block_length"),
            "Jackknife_Used": meta.get("jackknife_mode"),
            "Jackknife_Block_Length": meta.get("jackknife_block_length"),
            "N_Resamples": meta.get("n_resamples"),
            "Confidence": meta.get("confidence"),
            "Effective_N_Samples": meta.get("effective_n_samples"),
            "N_Valid_Bootstrap_Min": meta.get("n_valid_bootstrap_min"),
            "Seed": meta.get("seed"),
        }

        for m in metric_order:
            for i, v in enumerate(var_names):
                base = f"{m}_{v}"
                ev = est[m][i]
                lv = low[m][i]
                uv = up[m][i]

                row[base] = float(ev) if np.isfinite(ev) else np.nan
                row[f"{base}_{ci_tag}_LOW"] = float(lv) if np.isfinite(lv) else np.nan
                row[f"{base}_{ci_tag}_HIGH"] = float(uv) if np.isfinite(uv) else np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    base_cols = [
        "Model",
        "CI_Method_Requested", "CI_Method_Used",
        "Bootstrap_Scheme", "Block_Length",
        "Jackknife_Used", "Jackknife_Block_Length",
        "N_Resamples", "Confidence", "Effective_N_Samples",
        "N_Valid_Bootstrap_Min", "Seed"
    ]
    metric_cols = []
    for m in metric_order:
        for v in var_names:
            base = f"{m}_{v}"
            metric_cols += [base, f"{base}_{ci_tag}_LOW", f"{base}_{ci_tag}_HIGH"]

    for c in base_cols + metric_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[base_cols + metric_cols]


def build_bootstrap_long_dataframe(bootstrap_result_dict, var_names, metric_order, group_name):
    rows = []
    for model_name, res in bootstrap_result_dict.items():
        est = res["estimate"]
        low = res["ci_lower"]
        up = res["ci_upper"]
        meta = res["meta"]

        for m in metric_order:
            for i, v in enumerate(var_names):
                rows.append({
                    "Group": group_name,
                    "Model": model_name,
                    "Metric": m,
                    "Variable": v,
                    "Estimate": float(est[m][i]) if np.isfinite(est[m][i]) else np.nan,
                    "CI_Lower": float(low[m][i]) if np.isfinite(low[m][i]) else np.nan,
                    "CI_Upper": float(up[m][i]) if np.isfinite(up[m][i]) else np.nan,
                    "Confidence": meta.get("confidence"),
                    "CI_Method_Requested": meta.get("ci_method_requested"),
                    "CI_Method_Used": meta.get("ci_method_used"),
                    "Bootstrap_Scheme": meta.get("bootstrap_scheme"),
                    "Block_Length": meta.get("block_length"),
                    "Jackknife_Used": meta.get("jackknife_mode"),
                    "Jackknife_Block_Length": meta.get("jackknife_block_length"),
                    "N_Resamples": meta.get("n_resamples"),
                    "Effective_N_Samples": meta.get("effective_n_samples"),
                    "N_Valid_Bootstrap_Min": meta.get("n_valid_bootstrap_min"),
                    "Seed": meta.get("seed"),
                })

    cols = [
        "Group", "Model", "Metric", "Variable",
        "Estimate", "CI_Lower", "CI_Upper", "Confidence",
        "CI_Method_Requested", "CI_Method_Used",
        "Bootstrap_Scheme", "Block_Length",
        "Jackknife_Used", "Jackknife_Block_Length",
        "N_Resamples", "Effective_N_Samples",
        "N_Valid_Bootstrap_Min", "Seed"
    ]
    return pd.DataFrame(rows, columns=cols)


# =============================================================================
# 6) Pre-scanning & Evaluation Pipeline
# =============================================================================
def precompute_raw_target_stats(ds_y, variables, batch_size, max_samples=None):
    """
    Pre-scans Y_test to compute data_range (PSNR/SSIM) and mean_abs (MRE eps).
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
    """
    Pre-scans Y_test physics derived variables for mean_abs (MRE eps).
    """
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
    model_bundles,
    cubic_interpolator,
    physics_ops,
    raw_data_range, raw_mre_eps, phys_mre_eps,
    batch_size, max_samples,
    device,
    enable_bicubic_baseline=False,
    bicubic_interpolator=None
):
    method_names = ["cubic spline"]
    if enable_bicubic_baseline:
        method_names.append("bicubic")
    method_names += list(model_bundles.keys())

    raw_collectors = OrderedDict()
    phys_collectors = OrderedDict()

    for m in method_names:
        raw_collectors[m] = SufficientStatsCollector(
            var_names=VAR_DISPLAY,
            mre_eps=raw_mre_eps,
            data_range=raw_data_range,
            enable_image_metrics=True,
            ssim_win_size=SSIM_WIN_SIZE
        )
        phys_collectors[m] = SufficientStatsCollector(
            var_names=PHYS_DERIVED_NAMES,
            mre_eps=phys_mre_eps,
            data_range=None,
            enable_image_metrics=False
        )

    n_total = int(ds_x.sizes["time"])
    if max_samples is not None:
        n_total = min(n_total, int(max_samples))

    for st in tqdm(range(0, n_total, batch_size), desc="Main Evaluation Loop"):
        ed = min(st + batch_size, n_total)

        x_raw = read_batch_from_ds(ds_x, variables, st, ed)
        y_raw = read_batch_from_ds(ds_y, variables, st, ed)

        true_phys = compute_physics_derived(y_raw, physics_ops)

        # ---- Cubic spline baseline ----
        pred_cubic = cubic_interpolator(x_raw)
        raw_collectors["cubic spline"].update(pred_cubic, y_raw)

        pred_cubic_phys = compute_physics_derived(pred_cubic, physics_ops)
        phys_collectors["cubic spline"].update(pred_cubic_phys, true_phys)

        # ---- Bicubic baseline (optional) ----
        if enable_bicubic_baseline:
            if bicubic_interpolator is None:
                raise ValueError("enable_bicubic_baseline=True but bicubic_interpolator is None.")

            pred_bicubic = bicubic_interpolator(x_raw)
            raw_collectors["bicubic"].update(pred_bicubic, y_raw)

            pred_bicubic_phys = compute_physics_derived(pred_bicubic, physics_ops)
            phys_collectors["bicubic"].update(pred_bicubic_phys, true_phys)

        # ---- Neural Network Models (VDSR) ----
        if len(model_bundles) > 0:
            with torch.no_grad():
                for model_name, bundle in model_bundles.items():
                    x_std_batch = standardize_batch(x_raw, bundle["x_mean"], bundle["x_std"])
                    x_tensor = torch.from_numpy(x_std_batch).to(device, non_blocking=True)

                    pred_std = bundle["model"](x_tensor)
                    pred_std_np = pred_std.detach().cpu().numpy().astype(np.float32)
                    pred_raw = inverse_standardize_batch(pred_std_np, bundle["y_mean"], bundle["y_std"])

                    raw_collectors[model_name].update(pred_raw, y_raw)

                    pred_phys = compute_physics_derived(pred_raw, physics_ops)
                    phys_collectors[model_name].update(pred_phys, true_phys)

    raw_result = OrderedDict((k, v.point_estimate()) for k, v in raw_collectors.items())
    phys_result = OrderedDict((k, v.point_estimate()) for k, v in phys_collectors.items())

    return raw_result, phys_result, raw_collectors, phys_collectors


# =============================================================================
# 7) Main Routine
# =============================================================================
def main():
    set_seed(RANDOM_SEED)
    device = torch.device(DEVICE)

    print("=" * 100)
    print("ERA5 VDSR Super-Resolution Evaluation Started")
    print(f"Device: {device}")
    print("=" * 100)

    # File existence check
    must_exist = [X_TEST_PATH, Y_TEST_PATH, SCALER_PATH]
    for p in must_exist:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    # Check available model checkpoints
    available_model_paths = collect_available_models(MODEL_PATHS, strict=STRICT_MODEL_CHECK)

    ds_x = xr.open_dataset(X_TEST_PATH, decode_times=False)
    ds_y = xr.open_dataset(Y_TEST_PATH, decode_times=False)

    try:
        # Variable checks
        for v in VARIABLES:
            if v not in ds_x.variables:
                raise KeyError(f"{X_TEST_PATH} missing variable: {v}")
            if v not in ds_y.variables:
                raise KeyError(f"{Y_TEST_PATH} missing variable: {v}")

        n_x = int(ds_x.sizes["time"])
        n_y = int(ds_y.sizes["time"])
        if n_x != n_y:
            raise ValueError(f"Time dimension mismatch: X={n_x}, Y={n_y}")

        lat_lr = ds_x["latitude"].values.astype(np.float64)
        lon_lr = ds_x["longitude"].values.astype(np.float64)
        lat_hr = ds_y["latitude"].values.astype(np.float64)
        lon_hr = ds_y["longitude"].values.astype(np.float64)

        h_lr, w_lr = len(lat_lr), len(lon_lr)
        h_hr, w_hr = len(lat_hr), len(lon_hr)

        if (h_hr % h_lr != 0) or (w_hr % w_lr != 0):
            raise ValueError(f"HR/LR resolution factor non-integer: LR={h_lr}x{w_lr}, HR={h_hr}x{w_hr}")
        scale_h = h_hr // h_lr
        scale_w = w_hr // w_lr
        if scale_h != scale_w:
            raise ValueError(f"H/W scaling factor mismatch: scale_h={scale_h}, scale_w={scale_w}")
        actual_scale = int(scale_h)

        print(f"X_test shape: time={n_x}, lat={h_lr}, lon={w_lr}")
        print(f"Y_test shape: time={n_y}, lat={h_hr}, lon={w_hr}")
        print(f"Dataset scale factor: {actual_scale}x")

        # Load Scalers
        scaler_x_stats, scaler_y_stats = load_scaler_stats(SCALER_PATH)
        print("Loaded scalers.joblib successfully.")

        # Load Models
        model_bundles = OrderedDict()
        print("\nLoading VDSR models...")
        if len(available_model_paths) == 0:
            print("  (No model checkpoints found, evaluating baselines only)")

        for name, path in available_model_paths.items():
            bundle = load_one_model(
                ckpt_path=path,
                scaler_x_fallback=scaler_x_stats,
                scaler_y_fallback=scaler_y_stats,
                device=device,
                default_upscale_factor=actual_scale
            )
            if bundle["upscale_factor"] != actual_scale:
                warnings.warn(
                    f"[{name}] Checkpoint upscale_factor={bundle['upscale_factor']} "
                    f"differs from data scale {actual_scale}x."
                )

            model_bundles[name] = bundle
            n_params = sum(p.numel() for p in bundle["model"].parameters())
            print(
                f"  - {name:16s} | {path} | params={n_params:,} | "
                f"layers={bundle['num_layers']} | feat={bundle['num_features']} | "
                f"align_corners={bundle['align_corners']}"
            )

        # Baseline Interpolators
        cubic_interpolator = CubicSplineInterpolator2D(
            lat_lr=lat_lr, lon_lr=lon_lr,
            lat_hr=lat_hr, lon_hr=lon_hr,
            kx=CUBIC_SPLINE_KX, ky=CUBIC_SPLINE_KY
        )
        print("\nCreated cubic spline interpolator.")

        bicubic_interpolator = None
        if ENABLE_BICUBIC_BASELINE:
            bicubic_interpolator = BicubicTorchInterpolator(
                out_h=h_hr, out_w=w_hr,
                align_corners=BICUBIC_BASELINE_ALIGN_CORNERS,
                device=device
            )
            print(f"Created bicubic interpolator (align_corners={BICUBIC_BASELINE_ALIGN_CORNERS}).")

        # Physics Operator
        physics_ops = SphericalPhysicsOps(latitudes=lat_hr, longitudes=lon_hr)
        print("Created spherical physics operations handler.")

        # Precompute Stats for Raw Variables
        raw_data_range, raw_mean_abs = precompute_raw_target_stats(
            ds_y=ds_y,
            variables=VARIABLES,
            batch_size=BATCH_SIZE,
            max_samples=EVAL_MAX_SAMPLES
        )
        raw_mre_eps = np.maximum(MRE_EPS_ABS, MRE_EPS_REL * raw_mean_abs)

        print("\nRaw variables data_range and MRE epsilon:")
        for i, name in enumerate(VAR_DISPLAY):
            print(f"  {name}: data_range={raw_data_range[i]:.6e}, mre_eps={raw_mre_eps[i]:.6e}")

        # Precompute Stats for Physics Variables
        phys_mean_abs = precompute_physics_target_mean_abs(
            ds_y=ds_y,
            variables=VARIABLES,
            batch_size=BATCH_SIZE,
            max_samples=EVAL_MAX_SAMPLES,
            physics_ops=physics_ops
        )
        phys_mre_eps = np.maximum(MRE_EPS_ABS, MRE_EPS_REL * phys_mean_abs)

        print("\nPhysics variables MRE epsilon:")
        for i, name in enumerate(PHYS_DERIVED_NAMES):
            print(f"  {name}: mre_eps={phys_mre_eps[i]:.6e}")

        # Main Evaluation Execution
        raw_result, phys_result, raw_collectors, phys_collectors = evaluate_all_methods(
            ds_x=ds_x,
            ds_y=ds_y,
            variables=VARIABLES,
            model_bundles=model_bundles,
            cubic_interpolator=cubic_interpolator,
            physics_ops=physics_ops,
            raw_data_range=raw_data_range,
            raw_mre_eps=raw_mre_eps,
            phys_mre_eps=phys_mre_eps,
            batch_size=BATCH_SIZE,
            max_samples=EVAL_MAX_SAMPLES,
            device=device,
            enable_bicubic_baseline=ENABLE_BICUBIC_BASELINE,
            bicubic_interpolator=bicubic_interpolator
        )

        # Output Point Estimates to CSV
        raw_metric_order = ["RMSE", "MSE", "MAE", "MRE", "Correlation", "PSNR", "SSIM"]
        phys_metric_order = ["RMSE", "MSE", "MAE", "MRE", "Correlation"]

        df_raw = build_metrics_dataframe(
            result_dict=raw_result,
            var_names=VAR_DISPLAY,
            metric_order=raw_metric_order
        )
        df_phys = build_metrics_dataframe(
            result_dict=phys_result,
            var_names=PHYS_DERIVED_NAMES,
            metric_order=phys_metric_order
        )

        df_raw.to_csv(CSV_RAW_METRICS_PATH, index=False, float_format="%.6e")
        df_phys.to_csv(CSV_PHYS_METRICS_PATH, index=False, float_format="%.6e")

        df_all = pd.concat([df_raw.set_index("Model"), df_phys.set_index("Model")], axis=1).reset_index()
        df_all.to_csv(CSV_ALL_IN_ONE_PATH, index=False, float_format="%.6e")

        print("\n" + "=" * 100)
        print("Point estimate evaluation completed. CSV files saved:")
        print(f"  1) {CSV_RAW_METRICS_PATH}")
        print(f"  2) {CSV_PHYS_METRICS_PATH}")
        print(f"  3) {CSV_ALL_IN_ONE_PATH}")
        print("=" * 100)

        print("\nResults Preview (All-in-One):")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(df_all)

        # ---------------------------------------------------------------------
        # Bootstrap Confidence Intervals Computation
        # ---------------------------------------------------------------------
        if ENABLE_BOOTSTRAP:
            scheme_norm = normalize_bootstrap_scheme(BOOTSTRAP_SCHEME)
            if scheme_norm == "iid":
                warnings.warn("BOOTSTRAP_SCHEME='iid' is intended for datasets where the time dimension is shuffled or i.i.d.")

            print("\n" + "=" * 100)
            print("Starting Bootstrap CI computation")
            print(f"  Method         : {BOOTSTRAP_CI_METHOD}")
            print(f"  Confidence     : {BOOTSTRAP_CONFIDENCE}")
            print(f"  Resamples      : {BOOTSTRAP_N_RESAMPLES}")
            print(f"  Scheme         : {BOOTSTRAP_SCHEME}")
            print(f"  Block Length   : {BOOTSTRAP_BLOCK_LENGTH}")
            print(f"  Jackknife      : {BOOTSTRAP_JACKKNIFE}")
            print(f"  Chunk Size     : {BOOTSTRAP_CHUNK}")
            print("=" * 100)

            raw_boot = run_bootstrap_for_methods(
                collectors=raw_collectors,
                metric_order=raw_metric_order,
                group_name="raw"
            )
            phys_boot = run_bootstrap_for_methods(
                collectors=phys_collectors,
                metric_order=phys_metric_order,
                group_name="physics"
            )

            df_raw_boot = build_bootstrap_wide_dataframe(
                bootstrap_result_dict=raw_boot,
                var_names=VAR_DISPLAY,
                metric_order=raw_metric_order,
                confidence=BOOTSTRAP_CONFIDENCE
            )
            df_phys_boot = build_bootstrap_wide_dataframe(
                bootstrap_result_dict=phys_boot,
                var_names=PHYS_DERIVED_NAMES,
                metric_order=phys_metric_order,
                confidence=BOOTSTRAP_CONFIDENCE
            )
            df_boot_long = pd.concat([
                build_bootstrap_long_dataframe(raw_boot, VAR_DISPLAY, raw_metric_order, group_name="raw"),
                build_bootstrap_long_dataframe(phys_boot, PHYS_DERIVED_NAMES, phys_metric_order, group_name="physics")
            ], axis=0, ignore_index=True)

            df_raw_boot.to_csv(CSV_RAW_BOOTSTRAP_CI_PATH, index=False, float_format="%.6e")
            df_phys_boot.to_csv(CSV_PHYS_BOOTSTRAP_CI_PATH, index=False, float_format="%.6e")
            df_boot_long.to_csv(CSV_BOOTSTRAP_LONG_PATH, index=False, float_format="%.6e")

            df_boot_all = pd.merge(
                df_raw_boot, df_phys_boot,
                on="Model", how="outer", suffixes=("_RAW", "_PHYS")
            )
            df_boot_all.to_csv(CSV_ALL_IN_ONE_BOOTSTRAP_CI_PATH, index=False, float_format="%.6e")

            print("\n" + "=" * 100)
            print("Bootstrap CI completed. CSV files saved:")
            print(f"  1) {CSV_RAW_BOOTSTRAP_CI_PATH}")
            print(f"  2) {CSV_PHYS_BOOTSTRAP_CI_PATH}")
            print(f"  3) {CSV_BOOTSTRAP_LONG_PATH}")
            print(f"  4) {CSV_ALL_IN_ONE_BOOTSTRAP_CI_PATH}")
            print("=" * 100)

            print("\nBootstrap Results Preview (Long format):")
            with pd.option_context("display.max_columns", None, "display.width", 200):
                print(df_boot_long.head(30))

    finally:
        ds_x.close()
        ds_y.close()


if __name__ == "__main__":
    main()