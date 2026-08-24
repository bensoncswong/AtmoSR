#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA5 Multi-Level Pressure Super-Resolution Model Evaluation Script
(Includes Standard Evaluation + Bootstrap 95% Confidence Intervals)
--------------------------------------------------------------------
Evaluation Targets:
  1) Cubic spline baseline
  2) Two best models: noPhy / AllPhy

Multi-Level Data:
  - 925hPa: X2_test_925hpa_2025.nc, Y_test_925hpa_2025.nc
  - 850hPa: X2_test_850hpa_2025.nc, Y_test_850hpa_2025.nc
  - 700hPa: X2_test_700hpa_2025.nc, Y_test_700hpa_2025.nc
  - 500hPa: X2_test_500hpa_2025.nc, Y_test_500hpa_2025.nc
  - 300hPa: X2_test_300hpa_2025.nc, Y_test_300hpa_2025.nc
  - 100hPa: X2_test_100hpa_2025.nc, Y_test_100hpa_2025.nc

Scaler Mode (Flag):
  - SCALER_MODE = "joblib"
      Use scalers.joblib as standard scaling info across all pressure levels.
  - SCALER_MODE = "layer_test"
      Estimate channel mean/std directly from each layer's test set.

Metrics:
  A. Raw 5 Variables (Z, Q, T, U, V):
      RMSE, MSE, MAE, MRE, Correlation, PSNR, SSIM
  B. Physically Derived 5 Variables:
      DIV, VOR, AGEO_MAG, MSE_GRAD_X, MSE_GRAD_Y
      RMSE, MSE, MAE, MRE, Correlation

Outputs:
  - Standard Point Estimates:
    1) test_metrics_raw_multilevel.csv
    2) test_metrics_physics_multilevel.csv
    3) test_metrics_all_in_one_multilevel.csv
  - Extended Estimates (with Bootstrap CIs):
    4) test_metrics_raw_multilevel_bootstrap_CI95.csv
    5) test_metrics_physics_multilevel_bootstrap_CI95.csv
    6) test_metrics_all_in_one_multilevel_bootstrap_CI95.csv
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
from skimage.metrics import structural_similarity as ssim_2d
from tqdm import tqdm


# =============================================================================
# 0) Configurable Parameters
# =============================================================================
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EVAL_MAX_SAMPLES = None  # Set to an integer (e.g., 256/512) for quick debugging; None for full evaluation

# Scaler Mode Configuration:
#   "joblib"     -> Load global scaler from SCALER_PATH
#   "layer_test" -> Estimate mean and std from each layer's test set
SCALER_MODE = "layer_test"
SCALER_PATH = "scalers.joblib"

# If SCALER_MODE="layer_test", whether to use only EVAL_MAX_SAMPLES to compute statistics
SCALER_FOLLOW_EVAL_MAX_SAMPLES = False

# Cubic spline interpolation degree
CUBIC_SPLINE_KX = 3
CUBIC_SPLINE_KY = 3

# MRE Stabilization Parameters: eps_var = max(MRE_EPS_ABS, MRE_EPS_REL * mean_abs_truth_var)
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

# Bootstrap Configurations
BOOTSTRAP_ENABLE = True
BOOTSTRAP_N_RESAMPLES = 2000
BOOTSTRAP_CI_ALPHA = 0.05          # 95% CI -> alpha=0.05
BOOTSTRAP_RANDOM_SEED = 2025
BOOTSTRAP_CHUNK_SIZE = 16          # Controls memory allocation per chunk
BOOTSTRAP_SHOW_PROGRESS = True

CI_TAG = f"CI{int(round((1.0 - BOOTSTRAP_CI_ALPHA) * 100))}"

# Multi-level test dataset paths
LAYER_DATASETS = OrderedDict([
    ("925hpa", ("X2_test_925hpa_2025.nc", "Y_test_925hpa_2025.nc")),
    ("850hpa", ("X2_test_850hpa_2025.nc", "Y_test_850hpa_2025.nc")),
    ("700hpa", ("X2_test_700hpa_2025.nc", "Y_test_700hpa_2025.nc")),
    ("500hpa", ("X2_test_500hpa_2025.nc", "Y_test_500hpa_2025.nc")),
    ("300hpa", ("X2_test_300hpa_2025.nc", "Y_test_300hpa_2025.nc")),
    ("100hpa", ("X2_test_100hpa_2025.nc", "Y_test_100hpa_2025.nc")),
])

# Evaluation models
MODEL_PATHS = OrderedDict([
    ("noPhy",  "best_model_noPhy.pth"),
    ("AllPhy", "best_model_AllPhy.pth"),
])

# CSV Output Paths (Standard Point Estimates)
CSV_RAW_METRICS_PATH = "test_metrics_raw_multilevel.csv"
CSV_PHYS_METRICS_PATH = "test_metrics_physics_multilevel.csv"
CSV_ALL_IN_ONE_PATH = "test_metrics_all_in_one_multilevel.csv"

# CSV Output Paths (Bootstrap Confidence Intervals)
CSV_RAW_METRICS_CI_PATH = f"test_metrics_raw_multilevel_bootstrap_{CI_TAG}.csv"
CSV_PHYS_METRICS_CI_PATH = f"test_metrics_physics_multilevel_bootstrap_{CI_TAG}.csv"
CSV_ALL_IN_ONE_CI_PATH = f"test_metrics_all_in_one_multilevel_bootstrap_{CI_TAG}.csv"

# Whether to append a MEAN_OVER_LEVELS summary row to all tables
APPEND_MEAN_OVER_LEVELS = True

VARIABLES = ["z", "q", "t", "u", "v"]
VAR_DISPLAY = ["Z", "Q", "T", "U", "V"]
PHYS_DERIVED_NAMES = ["DIV", "VOR", "AGEO_MAG", "MSE_GRAD_X", "MSE_GRAD_Y"]

RAW_METRIC_ORDER = ["RMSE", "MSE", "MAE", "MRE", "Correlation", "PSNR", "SSIM"]
PHYS_METRIC_ORDER = ["RMSE", "MSE", "MAE", "MRE", "Correlation"]


# =============================================================================
# 1) Utility Functions
# =============================================================================
def set_seed(seed: int):
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ChannelScaler:
    """Dummy class structure to maintain joblib deserialization compatibility."""
    def __init__(self):
        self.mean = None
        self.std = None


def sanitize_std(std_arr):
    """Replace standard deviations near zero with 1.0 to prevent division by zero."""
    std_arr = np.asarray(std_arr, dtype=np.float32)
    return np.where(np.abs(std_arr) < 1e-12, 1.0, std_arr).astype(np.float32)


def load_scaler_stats_from_joblib(scaler_path: str):
    """Load pre-computed mean and standard deviation statistics from a joblib file."""
    obj = joblib.load(scaler_path)

    if not (isinstance(obj, dict) and "scaler_X" in obj and "scaler_Y" in obj):
        raise ValueError("Invalid scalers.joblib format: expected keys 'scaler_X' and 'scaler_Y'")

    sx = obj["scaler_X"]
    sy = obj["scaler_Y"]

    x_mean = np.asarray(sx.mean, dtype=np.float32)
    x_std = sanitize_std(sx.std)
    y_mean = np.asarray(sy.mean, dtype=np.float32)
    y_std = sanitize_std(sy.std)

    return {"mean": x_mean, "std": x_std}, {"mean": y_mean, "std": y_std}


def read_batch_from_ds(ds, variables, start, end):
    """Extract a batch of data from an xarray Dataset and format as (B, C, H, W)."""
    arr = np.stack(
        [ds[v].isel(time=slice(start, end)).values.astype(np.float32) for v in variables],
        axis=1
    )  # Shape: (B, C, H, W)
    return arr


def compute_channel_stats_from_dataset(ds, variables, batch_size, max_samples=None, desc="Estimating scaler"):
    """Compute per-channel mean and standard deviation incrementally across a dataset."""
    n_total = int(ds.sizes["time"])
    if max_samples is not None:
        n_total = min(n_total, int(max_samples))

    sum_ = np.zeros(len(variables), dtype=np.float64)
    sum2 = np.zeros(len(variables), dtype=np.float64)
    count = 0.0

    for st in tqdm(range(0, n_total, batch_size), desc=desc):
        ed = min(st + batch_size, n_total)
        x = read_batch_from_ds(ds, variables, st, ed)  # (B, C, H, W)
        sum_ += np.sum(x, axis=(0, 2, 3), dtype=np.float64)
        sum2 += np.sum(x * x, axis=(0, 2, 3), dtype=np.float64)
        count += x.shape[0] * x.shape[2] * x.shape[3]

    mean = sum_ / max(count, 1.0)
    var = np.maximum(sum2 / max(count, 1.0) - mean ** 2, 0.0)
    std = np.sqrt(var)

    return {
        "mean": mean.astype(np.float32),
        "std": sanitize_std(std),
    }


def build_scalers_from_layer_test_data(ds_x, ds_y, variables, batch_size, max_samples=None):
    """Estimate input (X) and target (Y) scaler statistics directly from the current layer test data."""
    scaler_x = compute_channel_stats_from_dataset(
        ds_x, variables, batch_size, max_samples=max_samples, desc="Estimating scaler_X (layer_test)"
    )
    scaler_y = compute_channel_stats_from_dataset(
        ds_y, variables, batch_size, max_samples=max_samples, desc="Estimating scaler_Y (layer_test)"
    )
    return scaler_x, scaler_y


def standardize_batch(x_raw, mean, std):
    """Apply z-score standardization to a raw input batch."""
    return ((x_raw - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)).astype(np.float32)


def inverse_standardize_batch(y_std, mean, std):
    """Revert standardized predictions back to original physical scales."""
    return (y_std * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)).astype(np.float32)


def get_lat_lon(ds):
    """Extract latitude and longitude arrays from an xarray Dataset."""
    lat_name = "latitude" if "latitude" in ds.variables else ("lat" if "lat" in ds.variables else None)
    lon_name = "longitude" if "longitude" in ds.variables else ("lon" if "lon" in ds.variables else None)
    if lat_name is None or lon_name is None:
        raise KeyError("Dataset is missing latitude/longitude (or lat/lon) coordinate variables.")
    lat = ds[lat_name].values.astype(np.float64)
    lon = ds[lon_name].values.astype(np.float64)
    return lat, lon


# =============================================================================
# 2) Model Architectures (Aligned with Training Pipeline)
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
        x_std = sanitize_std(scaler_X_std)
        y_mean = np.asarray(scaler_Y_mean, dtype=np.float32)
        y_std = sanitize_std(scaler_Y_std)

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
    """Infer hyper-parameter configurations automatically from a loaded state dict."""
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


def load_model_payloads(model_paths):
    """Load checkpoint states and extract model metadata into memory."""
    payloads = OrderedDict()
    for name, path in model_paths.items():
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)

        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        hp = infer_model_hparams_from_state_dict(state_dict)
        payloads[name] = {
            "path": path,
            "state_dict": state_dict,
            "hp": hp,
            "align_corners": bool(ckpt.get("bicubic_align_corners", False)),
            "residual_scale": float(ckpt.get("residual_scale", 1.0)),
            "zero_init_last": bool(ckpt.get("zero_init_last", False)),
        }
    return payloads


def force_apply_scaler_buffers(model, scaler_x_stats, scaler_y_stats):
    """Force update x2y_alpha and x2y_beta buffers to match specified target scalers."""
    x_mean = np.asarray(scaler_x_stats["mean"], dtype=np.float32)
    x_std = sanitize_std(scaler_x_stats["std"])
    y_mean = np.asarray(scaler_y_stats["mean"], dtype=np.float32)
    y_std = sanitize_std(scaler_y_stats["std"])

    alpha = x_std / y_std
    beta = (x_mean - y_mean) / y_std

    alpha_t = torch.from_numpy(alpha).view(1, -1, 1, 1).to(model.x2y_alpha.device)
    beta_t = torch.from_numpy(beta).view(1, -1, 1, 1).to(model.x2y_beta.device)

    with torch.no_grad():
        model.x2y_alpha.copy_(alpha_t)
        model.x2y_beta.copy_(beta_t)


def build_models_from_payloads(payloads, scaler_x_stats, scaler_y_stats, device):
    """Instantiate and initialize neural network models using given checkpoint payloads."""
    models = OrderedDict()
    for name, p in payloads.items():
        hp = p["hp"]
        model = BicubicResidualSRModel(
            in_channels=hp["in_channels"],
            out_channels=hp["out_channels"],
            num_features=hp["num_features"],
            num_res_blocks=hp["num_res_blocks"],
            upscale_factor=hp["upscale_factor"],
            dropout_rate=0.0,
            scaler_X_mean=scaler_x_stats["mean"],
            scaler_X_std=scaler_x_stats["std"],
            scaler_Y_mean=scaler_y_stats["mean"],
            scaler_Y_std=scaler_y_stats["std"],
            align_corners=p["align_corners"],
            residual_scale=p["residual_scale"],
            zero_init_last=p["zero_init_last"],
        )
        model.load_state_dict(p["state_dict"], strict=True)
        force_apply_scaler_buffers(model, scaler_x_stats, scaler_y_stats)
        model.to(device)
        model.eval()
        models[name] = model
    return models


# =============================================================================
# 3) Interpolation & Physical Computations
# =============================================================================
def to_increasing_axis(axis_1d):
    """Ensure coordinate axis is strictly monotonically increasing."""
    axis_1d = np.asarray(axis_1d, dtype=np.float64)
    if axis_1d[1] > axis_1d[0]:
        return axis_1d, False
    return axis_1d[::-1], True


class CubicSplineInterpolator2D:
    """Perform 2D cubic spline spatial interpolation from low to high resolution grid."""
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
        # x_batch: (B, C, H_lr, W_lr)
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
    """Spherical spatial derivatives and physical quantity diagnostic operators."""
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
    Compute derived physical quantities from state variables.

    Input:
      batch_5ch: (B, 5, H, W), ordered as [Z, Q, T, U, V]
    Output:
      (B, 5, H, W), ordered as [DIV, VOR, AGEO_MAG, MSE_GRAD_X, MSE_GRAD_Y]
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
# 4) Metric Accumulator & Bootstrap
# =============================================================================
class MetricAccumulator:
    """
    Accumulates statistical metrics across batches:
      - Core: RMSE, MSE, MAE, MRE, Correlation
      - Optional: PSNR, SSIM
      - Optional: Per-sample sufficient statistics for bootstrap sampling
    """
    def __init__(self, var_names, mre_eps, data_range=None, enable_image_metrics=False, store_per_sample=False):
        self.var_names = list(var_names)
        self.nv = len(self.var_names)

        self.mre_eps = np.asarray(mre_eps, dtype=np.float64).reshape(-1)
        assert self.mre_eps.shape[0] == self.nv

        self.data_range = None
        if data_range is not None:
            self.data_range = np.asarray(data_range, dtype=np.float64).reshape(-1)
            assert self.data_range.shape[0] == self.nv

        self.enable_image_metrics = bool(enable_image_metrics)
        self.store_per_sample = bool(store_per_sample)

        self.sum_abs = np.zeros(self.nv, dtype=np.float64)
        self.sum_sq = np.zeros(self.nv, dtype=np.float64)
        self.sum_rel = np.zeros(self.nv, dtype=np.float64)
        self.count = np.zeros(self.nv, dtype=np.float64)

        # Correlation accumulators
        self.sum_pred = np.zeros(self.nv, dtype=np.float64)
        self.sum_true = np.zeros(self.nv, dtype=np.float64)
        self.sum_pred2 = np.zeros(self.nv, dtype=np.float64)
        self.sum_true2 = np.zeros(self.nv, dtype=np.float64)
        self.sum_pred_true = np.zeros(self.nv, dtype=np.float64)

        # SSIM accumulators
        self.ssim_sum = np.zeros(self.nv, dtype=np.float64)
        self.ssim_count = np.zeros(self.nv, dtype=np.float64)

        # Per-sample statistics for bootstrap estimation
        if self.store_per_sample:
            self.ps_sum_abs = []
            self.ps_sum_sq = []
            self.ps_sum_rel = []
            self.ps_count = []
            self.ps_sum_pred = []
            self.ps_sum_true = []
            self.ps_sum_pred2 = []
            self.ps_sum_true2 = []
            self.ps_sum_pred_true = []
            if self.enable_image_metrics:
                self.ps_ssim = []
                self.ps_ssim_valid = []

    def _compute_ssim_per_sample(self, pred, true):
        bsz, ch, h, w = pred.shape
        ssim_vals = np.full((bsz, ch), np.nan, dtype=np.float64)
        ssim_valid = np.zeros((bsz, ch), dtype=bool)

        win = min(SSIM_WIN_SIZE, h, w)
        if win % 2 == 0:
            win -= 1
        if win < 3:
            return ssim_vals, ssim_valid

        for c in range(ch):
            dr = float(self.data_range[c]) if self.data_range is not None else None
            if dr is None or (not np.isfinite(dr)) or dr <= 0:
                continue

            for b in range(bsz):
                try:
                    ssim_val = ssim_2d(
                        true[b, c], pred[b, c],
                        data_range=dr,
                        win_size=win
                    )
                except Exception:
                    ssim_val = np.nan

                if np.isfinite(ssim_val):
                    ssim_vals[b, c] = float(ssim_val)
                    ssim_valid[b, c] = True

        return ssim_vals, ssim_valid

    def update(self, pred, true):
        pred = pred.astype(np.float64, copy=False)
        true = true.astype(np.float64, copy=False)

        err = pred - true

        # Per-sample statistics
        ps_abs = np.sum(np.abs(err), axis=(2, 3), dtype=np.float64)      # (B, C)
        ps_sq = np.sum(err ** 2, axis=(2, 3), dtype=np.float64)          # (B, C)

        denom = np.maximum(np.abs(true), self.mre_eps.reshape(1, -1, 1, 1))
        ps_rel = np.sum(np.abs(err) / denom, axis=(2, 3), dtype=np.float64)  # (B, C)

        n_pix = pred.shape[2] * pred.shape[3]
        ps_count = np.full((pred.shape[0], self.nv), n_pix, dtype=np.float64)

        ps_pred = np.sum(pred, axis=(2, 3), dtype=np.float64)
        ps_true = np.sum(true, axis=(2, 3), dtype=np.float64)
        ps_pred2 = np.sum(pred ** 2, axis=(2, 3), dtype=np.float64)
        ps_true2 = np.sum(true ** 2, axis=(2, 3), dtype=np.float64)
        ps_pred_true = np.sum(pred * true, axis=(2, 3), dtype=np.float64)

        # Global aggregations
        self.sum_abs += ps_abs.sum(axis=0)
        self.sum_sq += ps_sq.sum(axis=0)
        self.sum_rel += ps_rel.sum(axis=0)
        self.count += ps_count.sum(axis=0)

        self.sum_pred += ps_pred.sum(axis=0)
        self.sum_true += ps_true.sum(axis=0)
        self.sum_pred2 += ps_pred2.sum(axis=0)
        self.sum_true2 += ps_true2.sum(axis=0)
        self.sum_pred_true += ps_pred_true.sum(axis=0)

        # SSIM calculation
        ssim_vals = None
        ssim_valid = None
        if self.enable_image_metrics:
            ssim_vals, ssim_valid = self._compute_ssim_per_sample(pred, true)
            self.ssim_sum += np.where(ssim_valid, ssim_vals, 0.0).sum(axis=0)
            self.ssim_count += ssim_valid.astype(np.float64).sum(axis=0)

        # Store per-sample metrics if enabled
        if self.store_per_sample:
            self.ps_sum_abs.append(ps_abs)
            self.ps_sum_sq.append(ps_sq)
            self.ps_sum_rel.append(ps_rel)
            self.ps_count.append(ps_count)
            self.ps_sum_pred.append(ps_pred)
            self.ps_sum_true.append(ps_true)
            self.ps_sum_pred2.append(ps_pred2)
            self.ps_sum_true2.append(ps_true2)
            self.ps_sum_pred_true.append(ps_pred_true)
            if self.enable_image_metrics:
                self.ps_ssim.append(ssim_vals)
                self.ps_ssim_valid.append(ssim_valid)

    def finalize(self):
        cnt = np.maximum(self.count, 1.0)

        mse = self.sum_sq / cnt
        rmse = np.sqrt(mse)
        mae = self.sum_abs / cnt
        mre = self.sum_rel / cnt

        n = cnt
        num = n * self.sum_pred_true - self.sum_pred * self.sum_true
        den = np.sqrt(
            np.maximum(n * self.sum_pred2 - self.sum_pred ** 2, 0.0) *
            np.maximum(n * self.sum_true2 - self.sum_true ** 2, 0.0)
        )
        corr = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

        out = OrderedDict()
        out["RMSE"] = rmse
        out["MSE"] = mse
        out["MAE"] = mae
        out["MRE"] = mre
        out["Correlation"] = corr

        if self.enable_image_metrics:
            if self.data_range is None:
                psnr = np.full(self.nv, np.nan, dtype=np.float64)
            else:
                dr = np.maximum(self.data_range, 1e-12)
                psnr = 20.0 * np.log10(dr) - 10.0 * np.log10(np.maximum(mse, 1e-12))

            ssim = np.divide(
                self.ssim_sum, self.ssim_count,
                out=np.full_like(self.ssim_sum, np.nan),
                where=self.ssim_count > 0
            )
            out["PSNR"] = psnr
            out["SSIM"] = ssim

        return out

    def get_sample_stats(self):
        if not self.store_per_sample:
            return None

        def _cat(lst, dtype=np.float64):
            if len(lst) == 0:
                return np.zeros((0, self.nv), dtype=dtype)
            return np.concatenate(lst, axis=0).astype(dtype, copy=False)

        out = {
            "sum_abs": _cat(self.ps_sum_abs, np.float64),
            "sum_sq": _cat(self.ps_sum_sq, np.float64),
            "sum_rel": _cat(self.ps_sum_rel, np.float64),
            "count": _cat(self.ps_count, np.float64),
            "sum_pred": _cat(self.ps_sum_pred, np.float64),
            "sum_true": _cat(self.ps_sum_true, np.float64),
            "sum_pred2": _cat(self.ps_sum_pred2, np.float64),
            "sum_true2": _cat(self.ps_sum_true2, np.float64),
            "sum_pred_true": _cat(self.ps_sum_pred_true, np.float64),
        }

        if self.enable_image_metrics:
            out["ssim"] = _cat(self.ps_ssim, np.float64)
            out["ssim_valid"] = _cat(self.ps_ssim_valid, bool)

        return out


def _as_2d(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def compute_metrics_from_aggregates(
    sum_abs, sum_sq, sum_rel, count,
    sum_pred, sum_true, sum_pred2, sum_true2, sum_pred_true,
    data_range=None,
    ssim_sum=None, ssim_count=None
):
    sum_abs = _as_2d(sum_abs)
    sum_sq = _as_2d(sum_sq)
    sum_rel = _as_2d(sum_rel)
    count = _as_2d(count)

    sum_pred = _as_2d(sum_pred)
    sum_true = _as_2d(sum_true)
    sum_pred2 = _as_2d(sum_pred2)
    sum_true2 = _as_2d(sum_true2)
    sum_pred_true = _as_2d(sum_pred_true)

    cnt = np.maximum(count, 1.0)

    mse = sum_sq / cnt
    rmse = np.sqrt(mse)
    mae = sum_abs / cnt
    mre = sum_rel / cnt

    num = cnt * sum_pred_true - sum_pred * sum_true
    den = np.sqrt(
        np.maximum(cnt * sum_pred2 - sum_pred ** 2, 0.0) *
        np.maximum(cnt * sum_true2 - sum_true ** 2, 0.0)
    )
    corr = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    out = OrderedDict()
    out["RMSE"] = rmse
    out["MSE"] = mse
    out["MAE"] = mae
    out["MRE"] = mre
    out["Correlation"] = corr

    if data_range is not None:
        dr = np.maximum(np.asarray(data_range, dtype=np.float64).reshape(1, -1), 1e-12)
        psnr = 20.0 * np.log10(dr) - 10.0 * np.log10(np.maximum(mse, 1e-12))
        out["PSNR"] = psnr

    if (ssim_sum is not None) and (ssim_count is not None):
        ssim_sum = _as_2d(ssim_sum)
        ssim_count = _as_2d(ssim_count)
        ssim = np.divide(
            ssim_sum, ssim_count,
            out=np.full_like(ssim_sum, np.nan),
            where=ssim_count > 0
        )
        out["SSIM"] = ssim

    return out


def nan_percentile(arr, q, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanpercentile(arr, q, axis=axis)


def bootstrap_ci_from_sample_stats(
    sample_stats,
    metric_order,
    n_resamples,
    ci_alpha,
    seed,
    chunk_size,
    data_range=None,
    include_psnr_ssim=False,
    show_progress=True,
    desc="Bootstrap"
):
    """Compute empirical bootstrap distributions and percentile confidence intervals."""
    if sample_stats is None:
        return None, None

    n_samples = int(sample_stats["sum_sq"].shape[0])
    n_var = int(sample_stats["sum_sq"].shape[1])

    n_resamples = int(n_resamples)
    chunk_size = max(int(chunk_size), 1)

    dist = OrderedDict()
    for m in metric_order:
        dist[m] = np.full((max(n_resamples, 1), n_var), np.nan, dtype=np.float64)

    ci = OrderedDict()
    q_low = 100.0 * (ci_alpha / 2.0)
    q_high = 100.0 * (1.0 - ci_alpha / 2.0)

    if n_samples <= 0 or n_resamples <= 0:
        for m in metric_order:
            ci[m] = {
                "low": np.full(n_var, np.nan, dtype=np.float64),
                "high": np.full(n_var, np.nan, dtype=np.float64),
            }
        for m in metric_order:
            dist[m] = np.zeros((0, n_var), dtype=np.float64)
        return ci, dist

    rng = np.random.default_rng(seed)
    base_keys = [
        "sum_abs", "sum_sq", "sum_rel", "count",
        "sum_pred", "sum_true", "sum_pred2", "sum_true2", "sum_pred_true"
    ]

    iterator = range(0, n_resamples, chunk_size)
    if show_progress:
        iterator = tqdm(iterator, desc=desc, leave=False)

    for st in iterator:
        ed = min(st + chunk_size, n_resamples)
        r = ed - st

        idx = rng.integers(0, n_samples, size=(r, n_samples), endpoint=False)

        agg = {}
        for k in base_keys:
            agg[k] = np.take(sample_stats[k], idx, axis=0).sum(axis=1)  # (r, C)

        ssim_sum = None
        ssim_count = None
        if include_psnr_ssim and ("ssim" in sample_stats) and ("ssim_valid" in sample_stats):
            ssim_take = np.take(sample_stats["ssim"], idx, axis=0)               # (r, n_samples, C)
            valid_take = np.take(sample_stats["ssim_valid"], idx, axis=0)        # bool
            ssim_sum = np.where(valid_take, ssim_take, 0.0).sum(axis=1)         # (r, C)
            ssim_count = valid_take.astype(np.float64).sum(axis=1)              # (r, C)

        m_batch = compute_metrics_from_aggregates(
            sum_abs=agg["sum_abs"],
            sum_sq=agg["sum_sq"],
            sum_rel=agg["sum_rel"],
            count=agg["count"],
            sum_pred=agg["sum_pred"],
            sum_true=agg["sum_true"],
            sum_pred2=agg["sum_pred2"],
            sum_true2=agg["sum_true2"],
            sum_pred_true=agg["sum_pred_true"],
            data_range=(data_range if include_psnr_ssim else None),
            ssim_sum=ssim_sum,
            ssim_count=ssim_count,
        )

        for m in metric_order:
            if m in m_batch:
                dist[m][st:ed] = m_batch[m]

    for m in metric_order:
        ci[m] = {
            "low": nan_percentile(dist[m], q_low, axis=0),
            "high": nan_percentile(dist[m], q_high, axis=0),
        }

    return ci, dist


def build_metrics_dataframe_multilevel(result_by_level, var_names, metric_order):
    """Format point estimation metric dictionary into a pandas DataFrame."""
    rows = []
    for level_name, result_dict in result_by_level.items():
        for model_name, metric_map in result_dict.items():
            row = {"Level": level_name, "Model": model_name}
            for m in metric_order:
                values = metric_map[m]
                for i, vname in enumerate(var_names):
                    row[f"{m}_{vname}"] = float(values[i])
            rows.append(row)

    columns = ["Level", "Model"] + [f"{m}_{v}" for m in metric_order for v in var_names]
    return pd.DataFrame(rows, columns=columns)


def append_mean_over_levels(df):
    """Append a summary row containing metric averages across all pressure levels."""
    metric_cols = [c for c in df.columns if c not in ("Level", "Model")]
    df_mean = df.groupby("Model", as_index=False)[metric_cols].mean()
    df_mean.insert(0, "Level", "MEAN_OVER_LEVELS")
    out = pd.concat([df, df_mean], axis=0, ignore_index=True)
    return out


def build_metrics_dataframe_with_ci_multilevel(
    result_by_level,
    bootstrap_by_level,
    var_names,
    metric_order,
    ci_alpha=0.05,
    append_mean=True
):
    """Format metrics and their corresponding lower/upper Bootstrap CIs into a pandas DataFrame."""
    ci_tag = f"CI{int(round((1.0 - ci_alpha) * 100))}"
    q_low = 100.0 * (ci_alpha / 2.0)
    q_high = 100.0 * (1.0 - ci_alpha / 2.0)

    level_order = list(result_by_level.keys())

    model_order = []
    for lv in level_order:
        for m in result_by_level[lv].keys():
            if m not in model_order:
                model_order.append(m)

    rows = []

    # Process each pressure level
    for lv in level_order:
        for model_name in model_order:
            if model_name not in result_by_level[lv]:
                continue

            row = {"Level": lv, "Model": model_name}
            point_map = result_by_level[lv][model_name]

            boot_info = None
            if bootstrap_by_level is not None:
                boot_info = bootstrap_by_level.get(lv, {}).get(model_name, None)

            for metric_name in metric_order:
                point_vals = np.asarray(point_map[metric_name], dtype=np.float64)

                if (boot_info is not None) and ("ci" in boot_info) and (metric_name in boot_info["ci"]):
                    low_vals = np.asarray(boot_info["ci"][metric_name]["low"], dtype=np.float64)
                    high_vals = np.asarray(boot_info["ci"][metric_name]["high"], dtype=np.float64)
                else:
                    low_vals = np.full_like(point_vals, np.nan)
                    high_vals = np.full_like(point_vals, np.nan)

                for i, vname in enumerate(var_names):
                    row[f"{metric_name}_{vname}"] = float(point_vals[i])
                    row[f"{metric_name}_{vname}_{ci_tag}_LOW"] = float(low_vals[i])
                    row[f"{metric_name}_{vname}_{ci_tag}_HIGH"] = float(high_vals[i])

            rows.append(row)

    # Aggregate across pressure levels
    if append_mean:
        for model_name in model_order:
            row = {"Level": "MEAN_OVER_LEVELS", "Model": model_name}

            for metric_name in metric_order:
                for i, vname in enumerate(var_names):
                    point_vals = []
                    for lv in level_order:
                        if model_name in result_by_level[lv]:
                            point_vals.append(result_by_level[lv][model_name][metric_name][i])
                    point_mean = float(np.nanmean(point_vals)) if len(point_vals) > 0 else np.nan

                    ci_low = np.nan
                    ci_high = np.nan
                    if bootstrap_by_level is not None:
                        dist_list = []
                        for lv in level_order:
                            bi = bootstrap_by_level.get(lv, {}).get(model_name, None)
                            if bi is None:
                                continue
                            dist_map = bi.get("dist", {})
                            if metric_name not in dist_map:
                                continue
                            dist_list.append(np.asarray(dist_map[metric_name][:, i], dtype=np.float64))

                        if len(dist_list) > 0:
                            min_len = min(d.shape[0] for d in dist_list)
                            if min_len > 0:
                                stack = np.stack([d[:min_len] for d in dist_list], axis=0)  # (L, B)
                                mean_dist = np.nanmean(stack, axis=0)                        # (B,)
                                ci_low = float(nan_percentile(mean_dist, q_low, axis=0))
                                ci_high = float(nan_percentile(mean_dist, q_high, axis=0))

                    row[f"{metric_name}_{vname}"] = point_mean
                    row[f"{metric_name}_{vname}_{ci_tag}_LOW"] = ci_low
                    row[f"{metric_name}_{vname}_{ci_tag}_HIGH"] = ci_high

            rows.append(row)

    columns = ["Level", "Model"]
    for metric_name in metric_order:
        for vname in var_names:
            columns.extend([
                f"{metric_name}_{vname}",
                f"{metric_name}_{vname}_{ci_tag}_LOW",
                f"{metric_name}_{vname}_{ci_tag}_HIGH",
            ])

    return pd.DataFrame(rows, columns=columns)


# =============================================================================
# 5) Pre-scanning & Main Evaluation Logic
# =============================================================================
def precompute_raw_target_stats(ds_y, variables, batch_size, max_samples=None):
    """Pre-scan Y_test dataset to obtain max-min range (PSNR/SSIM) and mean_abs (MRE epsilon)."""
    n_total = int(ds_y.sizes["time"])
    if max_samples is not None:
        n_total = min(n_total, int(max_samples))

    vmin = np.full(len(variables), np.inf, dtype=np.float64)
    vmax = np.full(len(variables), -np.inf, dtype=np.float64)
    sum_abs = np.zeros(len(variables), dtype=np.float64)
    count = 0.0

    for st in tqdm(range(0, n_total, batch_size), desc="Pre-scanning Y_test (raw)"):
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
    """Pre-scan Y_test physical derived variables to obtain mean_abs for MRE stabilization."""
    n_total = int(ds_y.sizes["time"])
    if max_samples is not None:
        n_total = min(n_total, int(max_samples))

    sum_abs = np.zeros(len(PHYS_DERIVED_NAMES), dtype=np.float64)
    count = 0.0

    for st in tqdm(range(0, n_total, batch_size), desc="Pre-scanning Y_test (physics)"):
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
    batch_size, max_samples, device,
    bootstrap_enable=False,
    bootstrap_n_resamples=1000,
    bootstrap_ci_alpha=0.05,
    bootstrap_seed=2025,
    bootstrap_chunk_size=16,
    bootstrap_show_progress=True
):
    """Execute evaluation loop across baselines and AI models for a single pressure level."""
    method_names = ["cubic spline"] + list(models.keys())

    raw_acc = OrderedDict()
    phys_acc = OrderedDict()

    for m in method_names:
        raw_acc[m] = MetricAccumulator(
            var_names=VAR_DISPLAY,
            mre_eps=raw_mre_eps,
            data_range=raw_data_range,
            enable_image_metrics=True,
            store_per_sample=bootstrap_enable
        )
        phys_acc[m] = MetricAccumulator(
            var_names=PHYS_DERIVED_NAMES,
            mre_eps=phys_mre_eps,
            data_range=None,
            enable_image_metrics=False,
            store_per_sample=bootstrap_enable
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

        x_raw = read_batch_from_ds(ds_x, variables, st, ed)
        y_raw = read_batch_from_ds(ds_y, variables, st, ed)

        # Compute ground truth physical derived variables once
        true_phys = compute_physics_derived(y_raw, physics_ops)

        # ----- Cubic Spline Baseline -----
        pred_cubic = cubic_interpolator(x_raw)
        raw_acc["cubic spline"].update(pred_cubic, y_raw)

        pred_cubic_phys = compute_physics_derived(pred_cubic, physics_ops)
        phys_acc["cubic spline"].update(pred_cubic_phys, true_phys)

        # ----- AI Models -----
        if len(models) > 0:
            x_std_batch = standardize_batch(x_raw, x_mean, x_std)
            x_tensor = torch.from_numpy(x_std_batch).to(device, non_blocking=True)

            with torch.no_grad():
                for model_name, model in models.items():
                    pred_std = model(x_tensor)
                    pred_std_np = pred_std.detach().cpu().numpy().astype(np.float32)
                    pred_raw = inverse_standardize_batch(pred_std_np, y_mean, y_std)

                    raw_acc[model_name].update(pred_raw, y_raw)

                    pred_phys = compute_physics_derived(pred_raw, physics_ops)
                    phys_acc[model_name].update(pred_phys, true_phys)

    raw_result = OrderedDict((k, v.finalize()) for k, v in raw_acc.items())
    phys_result = OrderedDict((k, v.finalize()) for k, v in phys_acc.items())

    raw_bootstrap = None
    phys_bootstrap = None

    if bootstrap_enable:
        raw_bootstrap = OrderedDict()
        phys_bootstrap = OrderedDict()

        for method_name in method_names:
            # Raw variable bootstrap estimation
            raw_ci, raw_dist = bootstrap_ci_from_sample_stats(
                sample_stats=raw_acc[method_name].get_sample_stats(),
                metric_order=RAW_METRIC_ORDER,
                n_resamples=bootstrap_n_resamples,
                ci_alpha=bootstrap_ci_alpha,
                seed=bootstrap_seed,
                chunk_size=bootstrap_chunk_size,
                data_range=raw_data_range,
                include_psnr_ssim=True,
                show_progress=bootstrap_show_progress,
                desc=f"Bootstrap raw [{method_name}]"
            )
            raw_bootstrap[method_name] = {"ci": raw_ci, "dist": raw_dist}

            # Physical variable bootstrap estimation
            phys_ci, phys_dist = bootstrap_ci_from_sample_stats(
                sample_stats=phys_acc[method_name].get_sample_stats(),
                metric_order=PHYS_METRIC_ORDER,
                n_resamples=bootstrap_n_resamples,
                ci_alpha=bootstrap_ci_alpha,
                seed=bootstrap_seed,
                chunk_size=bootstrap_chunk_size,
                data_range=None,
                include_psnr_ssim=False,
                show_progress=bootstrap_show_progress,
                desc=f"Bootstrap physics [{method_name}]"
            )
            phys_bootstrap[method_name] = {"ci": phys_ci, "dist": phys_dist}

    return raw_result, phys_result, raw_bootstrap, phys_bootstrap


# =============================================================================
# 6) Main Execution Routine
# =============================================================================
def main():
    set_seed(RANDOM_SEED)
    device = torch.device(DEVICE)

    print("=" * 110)
    print("ERA5 Multi-Level Super-Resolution Evaluation Started")
    print(f"Device: {device}")
    print(f"SCALER_MODE: {SCALER_MODE}")
    print(f"Bootstrap Enabled: {BOOTSTRAP_ENABLE}")
    if BOOTSTRAP_ENABLE:
        print(f"  - Resamples  : {BOOTSTRAP_N_RESAMPLES}")
        print(f"  - CI Alpha   : {BOOTSTRAP_CI_ALPHA} ({CI_TAG})")
        print(f"  - Chunk Size : {BOOTSTRAP_CHUNK_SIZE}")
    print("=" * 110)

    if SCALER_MODE not in ("joblib", "layer_test"):
        raise ValueError("SCALER_MODE must be either 'joblib' or 'layer_test'")

    if BOOTSTRAP_ENABLE and BOOTSTRAP_N_RESAMPLES <= 0:
        raise ValueError("BOOTSTRAP_N_RESAMPLES must be > 0 when Bootstrap is enabled")

    # File availability checks
    for _, (x_path, y_path) in LAYER_DATASETS.items():
        if not os.path.exists(x_path):
            raise FileNotFoundError(f"Dataset file not found: {x_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"Dataset file not found: {y_path}")

    for _, mpath in MODEL_PATHS.items():
        if not os.path.exists(mpath):
            raise FileNotFoundError(f"Model file not found: {mpath}")

    if SCALER_MODE == "joblib" and (not os.path.exists(SCALER_PATH)):
        raise FileNotFoundError(f"SCALER_MODE='joblib' but file not found: {SCALER_PATH}")

    # Load model checkpoint payloads once
    print("\nLoading model checkpoint payloads...")
    model_payloads = load_model_payloads(MODEL_PATHS)
    for name, p in model_payloads.items():
        hp = p["hp"]
        print(f"  - {name:6s} | {p['path']} | hparams={hp}")

    # Global scaler initialization for 'joblib' mode
    global_scaler_x_stats, global_scaler_y_stats = None, None
    global_models = None
    if SCALER_MODE == "joblib":
        global_scaler_x_stats, global_scaler_y_stats = load_scaler_stats_from_joblib(SCALER_PATH)
        global_models = build_models_from_payloads(
            model_payloads, global_scaler_x_stats, global_scaler_y_stats, device
        )
        print("Applied scalers.joblib to initialize models for all levels.")

    # Storage containers for results across pressure levels
    raw_results_by_level = OrderedDict()
    phys_results_by_level = OrderedDict()
    raw_bootstrap_by_level = OrderedDict() if BOOTSTRAP_ENABLE else None
    phys_bootstrap_by_level = OrderedDict() if BOOTSTRAP_ENABLE else None

    # Evaluate pressure levels sequentially
    for li, (level_name, (x_path, y_path)) in enumerate(LAYER_DATASETS.items()):
        print("\n" + "-" * 110)
        print(f"Evaluating Pressure Level: {level_name}")
        print(f"  X: {x_path}")
        print(f"  Y: {y_path}")

        ds_x = xr.open_dataset(x_path, decode_times=False)
        ds_y = xr.open_dataset(y_path, decode_times=False)

        # Check required variables
        for v in VARIABLES:
            if v not in ds_x.variables:
                raise KeyError(f"{x_path} is missing variable: {v}")
            if v not in ds_y.variables:
                raise KeyError(f"{y_path} is missing variable: {v}")

        # Check dimension consistency
        n_x = int(ds_x.sizes["time"])
        n_y = int(ds_y.sizes["time"])
        if n_x != n_y:
            raise ValueError(f"[{level_name}] Time step mismatch: X={n_x}, Y={n_y}")

        lat_lr, lon_lr = get_lat_lon(ds_x)
        lat_hr, lon_hr = get_lat_lon(ds_y)

        print(f"  X Dimension: time={n_x}, lat={len(lat_lr)}, lon={len(lon_lr)}")
        print(f"  Y Dimension: time={n_y}, lat={len(lat_hr)}, lon={len(lon_hr)}")

        # Configure scalers and models for current layer
        if SCALER_MODE == "joblib":
            scaler_x_stats = global_scaler_x_stats
            scaler_y_stats = global_scaler_y_stats
            models = global_models
        else:
            scaler_max_samples = EVAL_MAX_SAMPLES if SCALER_FOLLOW_EVAL_MAX_SAMPLES else None
            scaler_x_stats, scaler_y_stats = build_scalers_from_layer_test_data(
                ds_x=ds_x,
                ds_y=ds_y,
                variables=VARIABLES,
                batch_size=BATCH_SIZE,
                max_samples=scaler_max_samples,
            )
            models = build_models_from_payloads(
                model_payloads, scaler_x_stats, scaler_y_stats, device
            )
            print("  Estimated scaler statistics using current layer test data and updated models.")

        # Instantiate spatial interpolator and spherical physics operator
        cubic_interpolator = CubicSplineInterpolator2D(
            lat_lr=lat_lr, lon_lr=lon_lr,
            lat_hr=lat_hr, lon_hr=lon_hr,
            kx=CUBIC_SPLINE_KX, ky=CUBIC_SPLINE_KY
        )
        physics_ops = SphericalPhysicsOps(latitudes=lat_hr, longitudes=lon_hr)

        # Pre-scan target stats
        raw_data_range, raw_mean_abs = precompute_raw_target_stats(
            ds_y=ds_y,
            variables=VARIABLES,
            batch_size=BATCH_SIZE,
            max_samples=EVAL_MAX_SAMPLES
        )
        raw_mre_eps = np.maximum(MRE_EPS_ABS, MRE_EPS_REL * raw_mean_abs)

        phys_mean_abs = precompute_physics_target_mean_abs(
            ds_y=ds_y,
            variables=VARIABLES,
            batch_size=BATCH_SIZE,
            max_samples=EVAL_MAX_SAMPLES,
            physics_ops=physics_ops
        )
        phys_mre_eps = np.maximum(MRE_EPS_ABS, MRE_EPS_REL * phys_mean_abs)

        # Execute evaluation
        layer_boot_seed = BOOTSTRAP_RANDOM_SEED + li * 100003
        raw_result, phys_result, raw_boot, phys_boot = evaluate_all_methods(
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
            batch_size=BATCH_SIZE,
            max_samples=EVAL_MAX_SAMPLES,
            device=device,
            bootstrap_enable=BOOTSTRAP_ENABLE,
            bootstrap_n_resamples=BOOTSTRAP_N_RESAMPLES,
            bootstrap_ci_alpha=BOOTSTRAP_CI_ALPHA,
            bootstrap_seed=layer_boot_seed,
            bootstrap_chunk_size=BOOTSTRAP_CHUNK_SIZE,
            bootstrap_show_progress=BOOTSTRAP_SHOW_PROGRESS
        )

        raw_results_by_level[level_name] = raw_result
        phys_results_by_level[level_name] = phys_result

        if BOOTSTRAP_ENABLE:
            raw_bootstrap_by_level[level_name] = raw_boot
            phys_bootstrap_by_level[level_name] = phys_boot

        ds_x.close()
        ds_y.close()

        if SCALER_MODE == "layer_test":
            del models
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Save point estimates to CSV
    df_raw = build_metrics_dataframe_multilevel(
        result_by_level=raw_results_by_level,
        var_names=VAR_DISPLAY,
        metric_order=RAW_METRIC_ORDER
    )
    df_phys = build_metrics_dataframe_multilevel(
        result_by_level=phys_results_by_level,
        var_names=PHYS_DERIVED_NAMES,
        metric_order=PHYS_METRIC_ORDER
    )

    if APPEND_MEAN_OVER_LEVELS:
        df_raw = append_mean_over_levels(df_raw)
        df_phys = append_mean_over_levels(df_phys)

    df_raw.to_csv(CSV_RAW_METRICS_PATH, index=False, float_format="%.6e")
    df_phys.to_csv(CSV_PHYS_METRICS_PATH, index=False, float_format="%.6e")

    df_all = pd.merge(df_raw, df_phys, on=["Level", "Model"], how="inner")
    df_all.to_csv(CSV_ALL_IN_ONE_PATH, index=False, float_format="%.6e")

    # Save Bootstrap Confidence Interval tables to CSV
    if BOOTSTRAP_ENABLE:
        df_raw_ci = build_metrics_dataframe_with_ci_multilevel(
            result_by_level=raw_results_by_level,
            bootstrap_by_level=raw_bootstrap_by_level,
            var_names=VAR_DISPLAY,
            metric_order=RAW_METRIC_ORDER,
            ci_alpha=BOOTSTRAP_CI_ALPHA,
            append_mean=APPEND_MEAN_OVER_LEVELS
        )

        df_phys_ci = build_metrics_dataframe_with_ci_multilevel(
            result_by_level=phys_results_by_level,
            bootstrap_by_level=phys_bootstrap_by_level,
            var_names=PHYS_DERIVED_NAMES,
            metric_order=PHYS_METRIC_ORDER,
            ci_alpha=BOOTSTRAP_CI_ALPHA,
            append_mean=APPEND_MEAN_OVER_LEVELS
        )

        df_raw_ci.to_csv(CSV_RAW_METRICS_CI_PATH, index=False, float_format="%.6e")
        df_phys_ci.to_csv(CSV_PHYS_METRICS_CI_PATH, index=False, float_format="%.6e")

        df_all_ci = pd.merge(df_raw_ci, df_phys_ci, on=["Level", "Model"], how="inner")
        df_all_ci.to_csv(CSV_ALL_IN_ONE_CI_PATH, index=False, float_format="%.6e")

    print("\n" + "=" * 110)
    print("Evaluation completed successfully. CSV files generated:")
    print(f"  1) {CSV_RAW_METRICS_PATH}")
    print(f"  2) {CSV_PHYS_METRICS_PATH}")
    print(f"  3) {CSV_ALL_IN_ONE_PATH}")
    if BOOTSTRAP_ENABLE:
        print(f"  4) {CSV_RAW_METRICS_CI_PATH}")
        print(f"  5) {CSV_PHYS_METRICS_CI_PATH}")
        print(f"  6) {CSV_ALL_IN_ONE_CI_PATH}")
    print("=" * 110)


if __name__ == "__main__":
    main()