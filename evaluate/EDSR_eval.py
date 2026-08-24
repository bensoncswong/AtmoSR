#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA5 ClassicEDSR Super-Resolution Evaluation Suite (Bootstrap CI Enhanced)
--------------------------------------------------------------------------
Features:
A) Standard Variable Evaluation:
   - RMSE, MSE, MAE, MRE, Correlation, PSNR, SSIM
B) Physics-Derived Pointwise Quantities:
   - Divergence (DIV), Vorticity (VOR), Ageostrophic Wind Components (AGEO_U, AGEO_V, AGEO_MAG),
     and Moist Static Energy Gradients (MSE_GRAD_X, MSE_GRAD_Y)
C) Global Relative Error (GRE) Physical Terms:
   - DIV_GRE, VOR_GRE, GEO_GRE, MSE_GRE
D) Statistical Uncertainty Estimation:
   - Nonparametric Bootstrap 95% Confidence Intervals (Percentile / Basic / BCa)
   - Block Bootstrap Schemes for Spatial-Temporal Dependence:
     (iid / moving_block / circular_block / stationary)
   - Jackknife Replicates for BCa: (auto / delete1 / block)
"""

import os
import re
import math
import random
import warnings
from collections import OrderedDict
from typing import Dict, Any, Tuple, List, Optional

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
# 0) Configuration and Hyperparameters
# =============================================================================
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Evaluation batch size (adjust according to GPU memory)
BATCH_SIZE = 32

# Sample limit for debugging/testing (set to None for full evaluation)
EVAL_MAX_SAMPLES = None

# Baseline switches
ENABLE_BICUBIC_BASELINE = True
ENABLE_CUBIC_SPLINE_BASELINE = True

# 2D Cubic Spline degrees
CUBIC_SPLINE_KX = 3
CUBIC_SPLINE_KY = 3

# MRE Stabilization Epsilon: eps_var = max(MRE_EPS_ABS, MRE_EPS_REL * mean_abs_truth_var)
MRE_EPS_REL = 1e-3
MRE_EPS_ABS = 1e-8

# SSIM Window Size (automatically capped to <= min(H, W) and forced to be odd)
SSIM_WIN_SIZE = 7

# Physical Constants (Aligned with Training Protocol)
CP_AIR = 1004.0
LV_WATER = 2.501e6
EARTH_RADIUS = 6371000.0
OMEGA = 7.2921e-5
F_CORIOLIS_ABS_MIN = 2.5e-5
TAN_CLIP = 50.0

# Geostrophic Loss Weighting Configuration
GEO_WEIGHT_MODE = "sin_global"   # Options: ["f_ratio", "sin_global", "cos", "one_minus_cos"]
GEO_WEIGHT_ALPHA = 1.0
GEO_WEIGHT_MIN = 0.10

# Global Relative Error (GRE) Epsilon Constants
EPS_DIV = 1e-8
EPS_VOR = 1e-8
EPS_GEO = 1e-4
EPS_MSE = 1e-4

# Dataset and Checkpoint Paths
X_TEST_PATH = "X2_test.nc"  # Change to X4_test.nc for 4x downsampling
Y_TEST_PATH = "Y_test.nc"
SCALER_PATH = "runs_classic_edsr/scalers.joblib"

# Model Checkpoints to Evaluate
MODEL_PATHS = OrderedDict([
    ("ClassicEDSR_L1_only", "runs_classic_edsr/ClassicEDSR_L1_only/best_model.pth"),
    ("ClassicEDSR_L1_Physics", "runs_classic_edsr/ClassicEDSR_L1_Physics_GRE/best_model.pth"),
])

# Output CSV Files (Point Estimates)
CSV_RAW_METRICS_PATH = "test_metrics_raw_classic_edsr.csv"
CSV_PHYS_POINTWISE_METRICS_PATH = "test_metrics_physics_pointwise_classic_edsr.csv"
CSV_PHYS_GRE_METRICS_PATH = "test_metrics_physics_gre_classic_edsr.csv"
CSV_ALL_IN_ONE_PATH = "test_metrics_all_in_one_classic_edsr.csv"

# Output CSV Files (Bootstrap Confidence Intervals)
CSV_RAW_BOOTSTRAP_CI_PATH = "test_metrics_raw_bootstrap_ci_classic_edsr.csv"
CSV_PHYS_POINTWISE_BOOTSTRAP_CI_PATH = "test_metrics_physics_pointwise_bootstrap_ci_classic_edsr.csv"
CSV_PHYS_GRE_BOOTSTRAP_CI_PATH = "test_metrics_physics_gre_bootstrap_ci_classic_edsr.csv"
CSV_BOOTSTRAP_ALL_IN_ONE_PATH = "test_metrics_bootstrap_all_in_one_classic_edsr.csv"

# Variable Target Configurations
VARIABLES = ["z", "q", "t", "u", "v"]
VAR_DISPLAY = ["Z", "Q", "T", "U", "V"]

PHYS_POINTWISE_NAMES = [
    "DIV", "VOR", "AGEO_U", "AGEO_V", "AGEO_MAG", "MSE_GRAD_X", "MSE_GRAD_Y"
]
GRE_TERM_NAMES = ["DIV_GRE", "VOR_GRE", "GEO_GRE", "MSE_GRE"]

# ---------------- Bootstrap Settings ----------------
ENABLE_BOOTSTRAP = True
BOOTSTRAP_N_RESAMPLES = 2000          # Recommended >= 2000 for BCa
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_CI_METHOD = "BCa"           # Options: {"percentile", "basic", "BCa"}
BOOTSTRAP_SCHEME = "circular_block"   # Options: {"iid", "moving_block", "circular_block", "stationary"}
BOOTSTRAP_BLOCK_LENGTH = "auto"       # int / "auto"(N^(1/3)) / "sqrt"(N^(1/2))
BOOTSTRAP_JACKKNIFE = "auto"          # Options: "auto" / "delete1" / "block"
BOOTSTRAP_SEED = 42
BOOTSTRAP_CHUNK = 256                 # Resample batch size for memory management


# =============================================================================
# 1) Core Utilities
# =============================================================================
def set_seed(seed: int):
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ChannelScaler:
    """Dummy class to enable joblib unpickling of custom scaler objects."""
    def __init__(self):
        self.mean = None
        self.std = None


def get_coord_1d(ds: xr.Dataset, candidates: List[str]) -> Tuple[np.ndarray, str]:
    """Retrieve 1D coordinate array from Dataset searching candidate names."""
    for name in candidates:
        if name in ds.coords:
            return ds[name].values.astype(np.float64), name
        if name in ds.variables:
            return ds[name].values.astype(np.float64), name
    raise KeyError(f"Coordinate candidates not found in dataset: {candidates}")


def load_scaler_stats(scaler_path: str):
    """Load mean and standard deviation statistics from joblib file."""
    obj = joblib.load(scaler_path)

    if isinstance(obj, dict) and ("scaler_X" in obj) and ("scaler_Y" in obj):
        sx = obj["scaler_X"]
        sy = obj["scaler_Y"]
        x_mean = np.asarray(sx.mean, dtype=np.float32).reshape(-1)
        x_std = np.asarray(sx.std, dtype=np.float32).reshape(-1)
        y_mean = np.asarray(sy.mean, dtype=np.float32).reshape(-1)
        y_std = np.asarray(sy.std, dtype=np.float32).reshape(-1)
    else:
        raise ValueError("Invalid scalers.joblib format. Expected 'scaler_X' and 'scaler_Y' keys.")

    x_std = np.where(np.abs(x_std) < 1e-12, 1.0, x_std)
    y_std = np.where(np.abs(y_std) < 1e-12, 1.0, y_std)

    return {"mean": x_mean, "std": x_std}, {"mean": y_mean, "std": y_std}


def standardize_batch(x_raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize input array using feature statistics."""
    return ((x_raw - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)).astype(np.float32)


def inverse_standardize_batch(y_std: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Invert standardization to restore original physical scale."""
    return (y_std * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)).astype(np.float32)


def read_batch_from_ds(ds: xr.Dataset, variables: List[str], start: int, end: int) -> np.ndarray:
    """Read a batch slice across specified variables from xarray dataset."""
    arr = np.stack(
        [ds[v].isel(time=slice(start, end)).values.astype(np.float32) for v in variables],
        axis=1
    )  # Shape: (B, C, H, W)
    return arr


def normalize_ci_method(ci_method: str) -> str:
    """Normalize Confidence Interval method string."""
    s = str(ci_method).strip().lower()
    if s == "bca":
        return "BCa"
    if s in {"percentile", "basic"}:
        return s
    raise ValueError(f"Unknown BOOTSTRAP_CI_METHOD: {ci_method}")


def normalize_bootstrap_scheme(scheme: str) -> str:
    """Normalize Bootstrap resampling scheme identifier."""
    s = str(scheme).strip().lower().replace("-", "_").replace(" ", "_")
    allowed = {"iid", "moving_block", "circular_block", "stationary"}
    if s not in allowed:
        raise ValueError(f"Unknown BOOTSTRAP_SCHEME: {scheme}. Allowed: {allowed}")
    return s


def resolve_block_length(n_samples: int, block_length_cfg: Any) -> int:
    """Determine block length integer based on specification or dynamic rule."""
    if n_samples <= 1:
        return 1

    if isinstance(block_length_cfg, str):
        s = block_length_cfg.strip().lower()
        if s == "auto":
            l = int(round(n_samples ** (1.0 / 3.0)))
        elif s == "sqrt":
            l = int(round(math.sqrt(n_samples)))
        else:
            raise ValueError(f"Unknown BOOTSTRAP_BLOCK_LENGTH string: {block_length_cfg}")
    else:
        l = int(block_length_cfg)

    l = max(1, min(l, n_samples))
    return l


def resolve_jackknife_mode(jk_mode_cfg: str, scheme: str) -> str:
    """Determine jackknife mode ('delete1' vs 'block') based on configuration and bootstrap scheme."""
    s = str(jk_mode_cfg).strip().lower()
    if s == "auto":
        return "delete1" if scheme == "iid" else "block"
    if s not in {"delete1", "block"}:
        raise ValueError("BOOTSTRAP_JACKKNIFE must be one of {'auto', 'delete1', 'block'}")
    return s


# =============================================================================
# 2) Classic EDSR Model Architecture
# =============================================================================
def default_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    """Default 2D convolution with standard padding."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=kernel_size // 2, bias=bias
    )


class ResBlock(nn.Module):
    """Classic EDSR Residual Block with optional Batch Normalization and scaling."""
    def __init__(self, conv, n_feats, kernel_size=3, bias=True, bn=False, res_scale=1.0):
        super().__init__()
        modules = [conv(n_feats, n_feats, kernel_size, bias=bias)]
        if bn:
            modules.append(nn.BatchNorm2d(n_feats))
        modules.append(nn.ReLU(True))
        modules.append(conv(n_feats, n_feats, kernel_size, bias=bias))
        if bn:
            modules.append(nn.BatchNorm2d(n_feats))
        self.body = nn.Sequential(*modules)
        self.res_scale = float(res_scale)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        return x + res


class Upsampler(nn.Sequential):
    """PixelShuffle Upsampling module supporting scale factors 2, 3, 4, 8."""
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Power of 2 (2, 4, 8)
            for _ in range(int(math.log2(scale))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise ValueError(f"Unsupported upscaling factor: {scale}")
        super().__init__(*m)


class ClassicEDSR(nn.Module):
    """
    Classic EDSR Super-Resolution Architecture:
    Head -> Residual Body -> Global Residual Addition -> Upsampler -> Tail
    """
    def __init__(
        self,
        in_channels=5,
        out_channels=5,
        n_feats=64,
        n_resblocks=16,
        scale=2,
        res_scale=1.0,
        use_bn=False
    ):
        super().__init__()
        if scale not in [2, 3, 4, 8]:
            raise ValueError(f"Unsupported upscaling factor: {scale}")

        self.use_bn = bool(use_bn)
        self.head = default_conv(in_channels, n_feats, 3)

        body = []
        for _ in range(n_resblocks):
            body.append(ResBlock(default_conv, n_feats, kernel_size=3, bn=self.use_bn, res_scale=res_scale))
        body.append(default_conv(n_feats, n_feats, 3))
        self.body = nn.Sequential(*body)

        self.tail = nn.Sequential(
            Upsampler(default_conv, scale, n_feats, bn=False, act=False),
            default_conv(n_feats, out_channels, 3)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = res + x
        out = self.tail(res)
        return out


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove DataParallel 'module.' prefix from model state dictionary."""
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def infer_num_resblocks_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int:
    """Infer the number of residual blocks from checkpoint weight keys."""
    pat = re.compile(r"^body\.(\d+)\.body\.0\.weight$")
    rb_idx = []
    for k in state_dict.keys():
        m = pat.match(k)
        if m:
            rb_idx.append(int(m.group(1)))
    if rb_idx:
        return max(rb_idx) + 1
    raise ValueError("Failed to infer num_res_blocks from state_dict.")


def infer_scale_from_state_dict(state_dict: Dict[str, torch.Tensor], n_feats: int) -> int:
    """Infer the upscale factor from tail layer weight dimensions."""
    pat = re.compile(r"^tail\.0\.(\d+)\.weight$")
    conv_entries = []
    for k, v in state_dict.items():
        m = pat.match(k)
        if m and v.ndim == 4:
            conv_entries.append((int(m.group(1)), int(v.shape[0])))

    if len(conv_entries) == 0:
        raise ValueError("Cannot find tail.0.*.weight keys to infer scale factor.")

    conv_entries.sort(key=lambda x: x[0])
    out_channels = [oc for _, oc in conv_entries]

    if len(out_channels) == 1 and out_channels[0] == 9 * n_feats:
        return 3

    if all(oc == 4 * n_feats for oc in out_channels):
        return int(2 ** len(out_channels))

    raise ValueError(f"Failed to infer scale factor from tail weights: channels={out_channels}, n_feats={n_feats}")


def infer_use_bn_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Detect presence of Batch Normalization layers in state dictionary."""
    return any(k.startswith("body.") and k.endswith("running_mean") for k in state_dict.keys())


def infer_classic_edsr_hparams(ckpt: Dict[str, Any], state_dict: Dict[str, torch.Tensor]):
    """Infer model hyperparameters directly from checkpoint object or state dict."""
    if "head.weight" not in state_dict:
        raise KeyError("State dict missing 'head.weight'. Is this a valid ClassicEDSR checkpoint?")

    tail_key = "tail.1.weight" if "tail.1.weight" in state_dict else None
    if tail_key is None:
        raise KeyError("State dict missing 'tail.1.weight'. Is this a valid ClassicEDSR checkpoint?")

    in_channels = int(ckpt.get("in_channels", state_dict["head.weight"].shape[1]))
    out_channels = int(ckpt.get("out_channels", state_dict[tail_key].shape[0]))
    n_feats = int(ckpt.get("num_features", state_dict["head.weight"].shape[0]))

    if "num_res_blocks" in ckpt:
        n_resblocks = int(ckpt["num_res_blocks"])
    else:
        n_resblocks = infer_num_resblocks_from_state_dict(state_dict)

    if "upscale_factor" in ckpt:
        scale = int(ckpt["upscale_factor"])
    else:
        scale = infer_scale_from_state_dict(state_dict, n_feats)

    res_scale = float(ckpt.get("res_block_scale", ckpt.get("res_scale", 1.0)))

    if "use_bn" in ckpt:
        use_bn = bool(ckpt["use_bn"])
    elif "use_bn_resblock" in ckpt:
        use_bn = bool(ckpt["use_bn_resblock"])
    else:
        use_bn = infer_use_bn_from_state_dict(state_dict)

    return {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "n_feats": n_feats,
        "n_resblocks": n_resblocks,
        "scale": scale,
        "res_scale": res_scale,
        "use_bn": use_bn
    }


def load_one_classic_edsr_model(ckpt_path: str, default_scaler_x: dict, default_scaler_y: dict, device: torch.device):
    """Load a single ClassicEDSR model checkpoint and configure standardization statistics."""
    ckpt_obj = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt_obj, dict) and ("model_state_dict" in ckpt_obj):
        ckpt = ckpt_obj
        state_dict = ckpt_obj["model_state_dict"]
    elif isinstance(ckpt_obj, dict):
        ckpt = {}
        state_dict = ckpt_obj
    else:
        raise TypeError(f"Unsupported checkpoint format in {ckpt_path}")

    state_dict = strip_module_prefix(state_dict)
    hp = infer_classic_edsr_hparams(ckpt, state_dict)

    model = ClassicEDSR(
        in_channels=hp["in_channels"],
        out_channels=hp["out_channels"],
        n_feats=hp["n_feats"],
        n_resblocks=hp["n_resblocks"],
        scale=hp["scale"],
        res_scale=hp["res_scale"],
        use_bn=hp["use_bn"]
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    x_mean = np.asarray(ckpt.get("scaler_x_mean", ckpt.get("scaler_X_mean", default_scaler_x["mean"])), dtype=np.float32).reshape(-1)
    x_std = np.asarray(ckpt.get("scaler_x_std", ckpt.get("scaler_X_std", default_scaler_x["std"])), dtype=np.float32).reshape(-1)
    y_mean = np.asarray(ckpt.get("scaler_y_mean", ckpt.get("scaler_Y_mean", default_scaler_y["mean"])), dtype=np.float32).reshape(-1)
    y_std = np.asarray(ckpt.get("scaler_y_std", ckpt.get("scaler_Y_std", default_scaler_y["std"])), dtype=np.float32).reshape(-1)

    x_std = np.where(np.abs(x_std) < 1e-12, 1.0, x_std)
    y_std = np.where(np.abs(y_std) < 1e-12, 1.0, y_std)

    if x_mean.size != len(VARIABLES) or y_mean.size != len(VARIABLES):
        raise ValueError(f"Scaler channel count mismatch in {ckpt_path}. Expected {len(VARIABLES)}.")

    info = {
        "architecture": ckpt.get("architecture", "ClassicEDSR"),
        "epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
        "hparams": hp,
        "experiment_name": ckpt.get("experiment_name", os.path.basename(os.path.dirname(ckpt_path))),
    }

    return {
        "model": model,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "info": info
    }


# =============================================================================
# 3) Baselines and Spherical Physics Calculations
# =============================================================================
def to_increasing_axis(axis_1d: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Ensure coordinate axis is strictly increasing for RectBivariateSpline."""
    axis_1d = np.asarray(axis_1d, dtype=np.float64)
    if axis_1d[1] > axis_1d[0]:
        return axis_1d, False
    return axis_1d[::-1], True


class BicubicInterpolatorTorch:
    """Bicubic interpolation baseline wrapper via PyTorch grid interpolation."""
    def __init__(self, out_h: int, out_w: int, device: torch.device):
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        self.device = torch.device(device)

    def __call__(self, x_batch: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x_batch).to(self.device, non_blocking=True)
        with torch.no_grad():
            out = F.interpolate(x, size=(self.out_h, self.out_w), mode="bicubic", align_corners=False)
        return out.detach().cpu().numpy().astype(np.float32)


class CubicSplineInterpolator2D:
    """2D Cubic Spline interpolation baseline using RectBivariateSpline."""
    def __init__(self, lat_lr, lon_lr, lat_hr, lon_hr, kx=3, ky=3):
        self.lat_lr_inc, self.flip_lat_lr = to_increasing_axis(lat_lr)
        self.lon_lr_inc, self.flip_lon_lr = to_increasing_axis(lon_lr)
        self.lat_hr_inc, self.flip_lat_hr = to_increasing_axis(lat_hr)
        self.lon_hr_inc, self.flip_lon_hr = to_increasing_axis(lon_hr)

        self.kx = int(kx)
        self.ky = int(ky)
        self.h_hr = len(lat_hr)
        self.w_hr = len(lon_hr)

    def __call__(self, x_batch: np.ndarray) -> np.ndarray:
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


def build_geo_weight(lat_rad: np.ndarray, mode: str, alpha: float, min_val: float) -> np.ndarray:
    """Build latitude-dependent weighting for geostrophic wind components."""
    alpha = float(alpha)
    min_val = float(min_val)

    if mode == "f_ratio":
        f_raw = 2.0 * OMEGA * np.sin(lat_rad)
        f_abs = np.abs(f_raw)
        geo_w = (f_abs / (f_abs.max() + 1e-12)) ** alpha
    elif mode == "sin_global":
        geo_w = (np.abs(np.sin(lat_rad))) ** alpha
    elif mode == "cos":
        geo_w = (np.abs(np.cos(lat_rad))) ** alpha
    elif mode == "one_minus_cos":
        geo_w = (1.0 - np.abs(np.cos(lat_rad))) ** alpha
        m = geo_w.max()
        if m > 0:
            geo_w = geo_w / m
    else:
        raise ValueError(f"Unknown GEO_WEIGHT_MODE: {mode}")

    geo_w = np.clip(geo_w, min_val, 1.0)
    return geo_w.astype(np.float64)


class SphericalPhysicsOps:
    """
    Spherical Coordinate Physics Operators (NumPy Implementation).
    Calculates differential operators and atmospheric physical quantities.
    """
    def __init__(self, latitudes, longitudes, geo_weight_mode, geo_weight_alpha, geo_weight_min):
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

        self.geo_weight = build_geo_weight(
            lat_rad,
            mode=geo_weight_mode,
            alpha=geo_weight_alpha,
            min_val=geo_weight_min
        ).reshape(-1, 1)  # (H, 1)

    def ddx(self, field: np.ndarray) -> np.ndarray:
        """Zonal derivative with central difference (periodic boundaries)."""
        left = np.concatenate([field[:, :, :1], field[:, :, :-1]], axis=2)
        right = np.concatenate([field[:, :, 1:], field[:, :, -1:]], axis=2)
        return 0.5 * (right - left) * self.inv_dx[None, :, None]

    def ddy(self, field: np.ndarray) -> np.ndarray:
        """Meridional derivative with central difference."""
        top = np.concatenate([field[:, :1, :], field[:, :-1, :]], axis=1)
        bottom = np.concatenate([field[:, 1:, :], field[:, -1:, :]], axis=1)

        if self.lat_decreasing:
            return 0.5 * (top - bottom) * self.inv_dy
        return 0.5 * (bottom - top) * self.inv_dy

    def divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Spherical divergence of horizontal wind field."""
        du_dx = self.ddx(u)
        dv_dy = self.ddy(v)
        metric = v * self.tan_over_r[None, :, None]
        return du_dx + dv_dy - metric

    def vorticity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Spherical relative vorticity of horizontal wind field."""
        dv_dx = self.ddx(v)
        du_dy = self.ddy(u)
        metric = u * self.tan_over_r[None, :, None]
        return dv_dx - du_dy + metric

    def geostrophic_wind(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Geostrophic wind components derived from geopotential z."""
        dz_dy = self.ddy(z)
        dz_dx = self.ddx(z)
        ug = -self.inv_f[None, :, None] * dz_dy
        vg = self.inv_f[None, :, None] * dz_dx
        return ug, vg

    def moist_static_energy(self, t: np.ndarray, z: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Moist Static Energy: MSE = Cp*T + z + Lv*q."""
        return CP_AIR * t + z + LV_WATER * q

    def moist_static_energy_gradients(self, t: np.ndarray, z: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Horizontal spatial gradients of Moist Static Energy."""
        mse = self.moist_static_energy(t, z, q)
        return self.ddx(mse), self.ddy(mse)

    def ageostrophic_components(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ageostrophic wind components: (u - ug, v - vg)."""
        ug, vg = self.geostrophic_wind(z)
        return u - ug, v - vg


def compute_pointwise_physics_derived(batch_5ch: np.ndarray, ops: SphericalPhysicsOps) -> np.ndarray:
    """
    Compute point-by-point physical derived features.
    Input: batch_5ch (B, 5, H, W) -> [z, q, t, u, v]
    Output: (B, 7, H, W) -> [DIV, VOR, AGEO_U, AGEO_V, AGEO_MAG, MSE_GRAD_X, MSE_GRAD_Y]
    """
    z = batch_5ch[:, 0]
    q = batch_5ch[:, 1]
    t = batch_5ch[:, 2]
    u = batch_5ch[:, 3]
    v = batch_5ch[:, 4]

    div = ops.divergence(u, v)
    vor = ops.vorticity(u, v)

    ageo_u, ageo_v = ops.ageostrophic_components(u, v, z)
    ageo_mag = np.sqrt(ageo_u ** 2 + ageo_v ** 2)

    mse_grad_x, mse_grad_y = ops.moist_static_energy_gradients(t, z, q)

    return np.stack(
        [div, vor, ageo_u, ageo_v, ageo_mag, mse_grad_x, mse_grad_y],
        axis=1
    ).astype(np.float32)


def global_relative_error(pred_qty: np.ndarray, target_qty: np.ndarray, eps: float, weight: Optional[np.ndarray] = None) -> float:
    """Compute Global Relative Error (GRE) over all spatial points and samples."""
    err = np.abs(pred_qty - target_qty)
    ref = np.abs(target_qty)
    if weight is not None:
        w = weight[None, :, :]  # Shape: (1, H, 1)
        err = err * w
        ref = ref * w
    return float(err.sum() / (ref.sum() + float(eps)))


def compute_physics_gre_terms(pred_5ch: np.ndarray, true_5ch: np.ndarray, ops: SphericalPhysicsOps) -> OrderedDict:
    """Compute the 4 Global Relative Error physical terms matching training loss."""
    z_p, q_p, t_p = pred_5ch[:, 0], pred_5ch[:, 1], pred_5ch[:, 2]
    u_p, v_p = pred_5ch[:, 3], pred_5ch[:, 4]

    z_t, q_t, t_t = true_5ch[:, 0], true_5ch[:, 1], true_5ch[:, 2]
    u_t, v_t = true_5ch[:, 3], true_5ch[:, 4]

    div_p = ops.divergence(u_p, v_p)
    div_t = ops.divergence(u_t, v_t)
    div_gre = global_relative_error(div_p, div_t, EPS_DIV)

    vor_p = ops.vorticity(u_p, v_p)
    vor_t = ops.vorticity(u_t, v_t)
    vor_gre = global_relative_error(vor_p, vor_t, EPS_VOR)

    ageo_u_p, ageo_v_p = ops.ageostrophic_components(u_p, v_p, z_p)
    ageo_u_t, ageo_v_t = ops.ageostrophic_components(u_t, v_t, z_t)

    w_geo = ops.geo_weight  # (H, 1)
    geo_u = global_relative_error(ageo_u_p, ageo_u_t, EPS_GEO, weight=w_geo)
    geo_v = global_relative_error(ageo_v_p, ageo_v_t, EPS_GEO, weight=w_geo)
    geo_gre = 0.5 * (geo_u + geo_v)

    mse_x_p, mse_y_p = ops.moist_static_energy_gradients(t_p, z_p, q_p)
    mse_x_t, mse_y_t = ops.moist_static_energy_gradients(t_t, z_t, q_t)

    mse_x = global_relative_error(mse_x_p, mse_x_t, EPS_MSE)
    mse_y = global_relative_error(mse_y_p, mse_y_t, EPS_MSE)
    mse_gre = 0.5 * (mse_x + mse_y)

    return OrderedDict([
        ("DIV_GRE", div_gre),
        ("VOR_GRE", vor_gre),
        ("GEO_GRE", geo_gre),
        ("MSE_GRE", mse_gre),
    ])


def compute_physics_gre_sample_components(pred_5ch: np.ndarray, true_5ch: np.ndarray, ops: SphericalPhysicsOps) -> Dict[str, np.ndarray]:
    """Compute GRE numerator and denominator terms per sample to facilitate vectorised Bootstrap."""
    z_p, q_p, t_p = pred_5ch[:, 0], pred_5ch[:, 1], pred_5ch[:, 2]
    u_p, v_p = pred_5ch[:, 3], pred_5ch[:, 4]

    z_t, q_t, t_t = true_5ch[:, 0], true_5ch[:, 1], true_5ch[:, 2]
    u_t, v_t = true_5ch[:, 3], true_5ch[:, 4]

    div_p = ops.divergence(u_p, v_p)
    div_t = ops.divergence(u_t, v_t)

    vor_p = ops.vorticity(u_p, v_p)
    vor_t = ops.vorticity(u_t, v_t)

    ageo_u_p, ageo_v_p = ops.ageostrophic_components(u_p, v_p, z_p)
    ageo_u_t, ageo_v_t = ops.ageostrophic_components(u_t, v_t, z_t)

    mse_x_p, mse_y_p = ops.moist_static_energy_gradients(t_p, z_p, q_p)
    mse_x_t, mse_y_t = ops.moist_static_energy_gradients(t_t, z_t, q_t)

    w_geo = ops.geo_weight[None, :, :]  # Shape: (1, H, 1)

    out = {
        "div_num": np.sum(np.abs(div_p - div_t), axis=(1, 2)).astype(np.float64),
        "div_den": np.sum(np.abs(div_t), axis=(1, 2)).astype(np.float64),

        "vor_num": np.sum(np.abs(vor_p - vor_t), axis=(1, 2)).astype(np.float64),
        "vor_den": np.sum(np.abs(vor_t), axis=(1, 2)).astype(np.float64),

        "geo_u_num": np.sum(np.abs(ageo_u_p - ageo_u_t) * w_geo, axis=(1, 2)).astype(np.float64),
        "geo_u_den": np.sum(np.abs(ageo_u_t) * w_geo, axis=(1, 2)).astype(np.float64),

        "geo_v_num": np.sum(np.abs(ageo_v_p - ageo_v_t) * w_geo, axis=(1, 2)).astype(np.float64),
        "geo_v_den": np.sum(np.abs(ageo_v_t) * w_geo, axis=(1, 2)).astype(np.float64),

        "mse_x_num": np.sum(np.abs(mse_x_p - mse_x_t), axis=(1, 2)).astype(np.float64),
        "mse_x_den": np.sum(np.abs(mse_x_t), axis=(1, 2)).astype(np.float64),

        "mse_y_num": np.sum(np.abs(mse_y_p - mse_y_t), axis=(1, 2)).astype(np.float64),
        "mse_y_den": np.sum(np.abs(mse_y_t), axis=(1, 2)).astype(np.float64),
    }
    return out


# =============================================================================
# 4) Metric Contribution and Accumulation Engine
# =============================================================================
def compute_ssim_per_sample(pred: np.ndarray, true: np.ndarray, data_range: np.ndarray) -> np.ndarray:
    """Compute SSIM per sample and channel using skimage metrics."""
    bsz, ch, h, w = pred.shape
    dr = np.asarray(data_range, dtype=np.float64).reshape(-1)

    out = np.full((bsz, ch), np.nan, dtype=np.float64)
    win = min(int(SSIM_WIN_SIZE), int(h), int(w))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return out

    for c in range(ch):
        if (not np.isfinite(dr[c])) or dr[c] <= 0:
            continue
        for b in range(bsz):
            s = ssim_2d(true[b, c], pred[b, c], data_range=float(dr[c]), win_size=win)
            if np.isfinite(s):
                out[b, c] = float(s)

    return out


def compute_metric_contrib(
    pred: np.ndarray,
    true: np.ndarray,
    mre_eps: np.ndarray,
    data_range: Optional[np.ndarray] = None,
    enable_image_metrics: bool = False
) -> Dict[str, np.ndarray]:
    """Compute sufficient metric components per sample for incremental updates and bootstrapping."""
    pred = pred.astype(np.float64, copy=False)
    true = true.astype(np.float64, copy=False)
    mre_eps = np.asarray(mre_eps, dtype=np.float64).reshape(1, -1, 1, 1)

    err = pred - true

    out = {
        "sum_abs": np.sum(np.abs(err), axis=(2, 3)).astype(np.float64),
        "sum_sq": np.sum(err ** 2, axis=(2, 3)).astype(np.float64),
        "sum_rel": np.sum(np.abs(err) / np.maximum(np.abs(true), mre_eps), axis=(2, 3)).astype(np.float64),
        "count": np.full((pred.shape[0],), pred.shape[2] * pred.shape[3], dtype=np.float64),

        "sum_pred": np.sum(pred, axis=(2, 3)).astype(np.float64),
        "sum_true": np.sum(true, axis=(2, 3)).astype(np.float64),
        "sum_pred2": np.sum(pred ** 2, axis=(2, 3)).astype(np.float64),
        "sum_true2": np.sum(true ** 2, axis=(2, 3)).astype(np.float64),
        "sum_pred_true": np.sum(pred * true, axis=(2, 3)).astype(np.float64),
    }

    if enable_image_metrics:
        if data_range is None:
            raise ValueError("Data range must be provided when enable_image_metrics=True")
        out["ssim"] = compute_ssim_per_sample(pred, true, data_range)

    return out


def finalize_metric_arrays(
    sum_abs, sum_sq, sum_rel, count,
    sum_pred, sum_true, sum_pred2, sum_true2, sum_pred_true,
    data_range=None,
    ssim_sum=None, ssim_count=None,
    enable_image_metrics=False
) -> OrderedDict:
    """Finalize metrics from aggregated sufficient statistics (supports multi-replicate matrices)."""
    cnt = np.maximum(np.asarray(count, dtype=np.float64).reshape(-1, 1), 1.0)

    mse = sum_sq / cnt
    rmse = np.sqrt(mse)
    mae = sum_abs / cnt
    mre = sum_rel / cnt

    n = cnt
    num = n * sum_pred_true - sum_pred * sum_true
    den = np.sqrt(
        np.maximum(n * sum_pred2 - sum_pred ** 2, 0.0) *
        np.maximum(n * sum_true2 - sum_true ** 2, 0.0)
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
        if data_range is None:
            raise ValueError("Data range must be provided when enable_image_metrics=True")
        dr = np.maximum(np.asarray(data_range, dtype=np.float64).reshape(1, -1), 1e-12)
        psnr = 20.0 * np.log10(dr) - 10.0 * np.log10(np.maximum(mse, 1e-12))
        out["PSNR"] = psnr

        if (ssim_sum is None) or (ssim_count is None):
            raise ValueError("SSIM sums and counts required when enable_image_metrics=True")
        ssim = np.divide(
            ssim_sum, ssim_count,
            out=np.full_like(ssim_sum, np.nan),
            where=ssim_count > 0
        )
        out["SSIM"] = ssim

    return out


class MetricAccumulator:
    """Accumulates batch metric statistics across evaluation dataset."""
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

        self.sum_abs = np.zeros(self.nv, dtype=np.float64)
        self.sum_sq = np.zeros(self.nv, dtype=np.float64)
        self.sum_rel = np.zeros(self.nv, dtype=np.float64)
        self.count = np.zeros(self.nv, dtype=np.float64)

        self.sum_pred = np.zeros(self.nv, dtype=np.float64)
        self.sum_true = np.zeros(self.nv, dtype=np.float64)
        self.sum_pred2 = np.zeros(self.nv, dtype=np.float64)
        self.sum_true2 = np.zeros(self.nv, dtype=np.float64)
        self.sum_pred_true = np.zeros(self.nv, dtype=np.float64)

        self.ssim_sum = np.zeros(self.nv, dtype=np.float64)
        self.ssim_count = np.zeros(self.nv, dtype=np.float64)

    def update_from_contrib(self, contrib: Dict[str, np.ndarray]):
        self.sum_abs += contrib["sum_abs"].sum(axis=0)
        self.sum_sq += contrib["sum_sq"].sum(axis=0)
        self.sum_rel += contrib["sum_rel"].sum(axis=0)

        total_count = float(np.sum(contrib["count"]))
        self.count += total_count

        self.sum_pred += contrib["sum_pred"].sum(axis=0)
        self.sum_true += contrib["sum_true"].sum(axis=0)
        self.sum_pred2 += contrib["sum_pred2"].sum(axis=0)
        self.sum_true2 += contrib["sum_true2"].sum(axis=0)
        self.sum_pred_true += contrib["sum_pred_true"].sum(axis=0)

        if self.enable_image_metrics and ("ssim" in contrib):
            ssim_vals = contrib["ssim"]
            valid = np.isfinite(ssim_vals)
            self.ssim_sum += np.where(valid, ssim_vals, 0.0).sum(axis=0)
            self.ssim_count += valid.sum(axis=0)

    def finalize(self) -> OrderedDict:
        out = finalize_metric_arrays(
            sum_abs=self.sum_abs.reshape(1, -1),
            sum_sq=self.sum_sq.reshape(1, -1),
            sum_rel=self.sum_rel.reshape(1, -1),
            count=self.count.reshape(1, -1)[:, :1],
            sum_pred=self.sum_pred.reshape(1, -1),
            sum_true=self.sum_true.reshape(1, -1),
            sum_pred2=self.sum_pred2.reshape(1, -1),
            sum_true2=self.sum_true2.reshape(1, -1),
            sum_pred_true=self.sum_pred_true.reshape(1, -1),
            data_range=self.data_range,
            ssim_sum=self.ssim_sum.reshape(1, -1) if self.enable_image_metrics else None,
            ssim_count=self.ssim_count.reshape(1, -1) if self.enable_image_metrics else None,
            enable_image_metrics=self.enable_image_metrics
        )
        return OrderedDict((k, v.reshape(-1)) for k, v in out.items())


class PhysicsGREAccumulator:
    """Accumulates GRE physics loss terms across batch predictions."""
    def __init__(self, physics_ops: SphericalPhysicsOps):
        self.physics_ops = physics_ops
        self.sum_terms = OrderedDict((k, 0.0) for k in GRE_TERM_NAMES)
        self.n_batches = 0

    def update(self, pred_5ch: np.ndarray, true_5ch: np.ndarray):
        terms = compute_physics_gre_terms(pred_5ch, true_5ch, self.physics_ops)
        for k in GRE_TERM_NAMES:
            self.sum_terms[k] += float(terms[k])
        self.n_batches += 1

    def finalize( -> OrderedDict:
        nb = max(self.n_batches, 1)
        return OrderedDict((k, self.sum_terms[k] / nb) for k in GRE_TERM_NAMES)


class BootstrapStore:
    """Buffer to retain per-sample statistics across evaluation methods for fast bootstrap resampling."""
    RAW_KEYS = [
        "sum_abs", "sum_sq", "sum_rel", "count",
        "sum_pred", "sum_true", "sum_pred2", "sum_true2", "sum_pred_true",
        "ssim"
    ]
    PHYS_KEYS = [
        "sum_abs", "sum_sq", "sum_rel", "count",
        "sum_pred", "sum_true", "sum_pred2", "sum_true2", "sum_pred_true"
    ]
    GRE_KEYS = [
        "div_num", "div_den", "vor_num", "vor_den",
        "geo_u_num", "geo_u_den", "geo_v_num", "geo_v_den",
        "mse_x_num", "mse_x_den", "mse_y_num", "mse_y_den"
    ]

    def __init__(self, method_names: List[str]):
        self.buffers = OrderedDict()
        for m in method_names:
            self.buffers[m] = {
                "raw": {k: [] for k in self.RAW_KEYS},
                "phys": {k: [] for k in self.PHYS_KEYS},
                "gre": {k: [] for k in self.GRE_KEYS},
            }

    def append_raw(self, method_name: str, contrib: dict):
        for k in self.RAW_KEYS:
            self.buffers[method_name]["raw"][k].append(np.asarray(contrib[k], dtype=np.float64))

    def append_phys(self, method_name: str, contrib: dict):
        for k in self.PHYS_KEYS:
            self.buffers[method_name]["phys"][k].append(np.asarray(contrib[k], dtype=np.float64))

    def append_gre(self, method_name: str, gre_comp: dict):
        for k in self.GRE_KEYS:
            self.buffers[method_name]["gre"][k].append(np.asarray(gre_comp[k], dtype=np.float64))

    def finalize( -> OrderedDict:
        out = OrderedDict()
        for m, parts in self.buffers.items():
            out[m] = {"raw": {}, "phys": {}, "gre": {}}

            for k in self.RAW_KEYS:
                out[m]["raw"][k] = np.concatenate(parts["raw"][k], axis=0) if len(parts["raw"][k]) else np.empty((0,), dtype=np.float64)

            for k in self.PHYS_KEYS:
                out[m]["phys"][k] = np.concatenate(parts["phys"][k], axis=0) if len(parts["phys"][k]) else np.empty((0,), dtype=np.float64)

            for k in self.GRE_KEYS:
                out[m]["gre"][k] = np.concatenate(parts["gre"][k], axis=0) if len(parts["gre"][k]) else np.empty((0,), dtype=np.float64)

        return out


def build_metrics_dataframe(result_dict: dict, var_names: list, metric_order: list) -> pd.DataFrame:
    """Format pointwise metric results dictionary into Pandas DataFrame."""
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


def build_gre_dataframe(gre_result_dict: dict) -> pd.DataFrame:
    """Format GRE physics metric results into Pandas DataFrame."""
    rows = []
    for model_name, term_map in gre_result_dict.items():
        row = {"Model": model_name}
        for k in GRE_TERM_NAMES:
            row[k] = float(term_map[k])
        rows.append(row)
    return pd.DataFrame(rows, columns=["Model"] + GRE_TERM_NAMES)


# =============================================================================
# 5) Bootstrap Engine & Resampling Algorithms
# =============================================================================
def sample_bootstrap_indices_one(n: int, scheme: str, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a single bootstrap index array according to the configured scheme."""
    if scheme == "iid":
        return rng.integers(0, n, size=n, dtype=np.int64)

    if scheme == "moving_block":
        l = int(block_length)
        n_blocks = int(math.ceil(n / l))
        max_start = max(0, n - l)
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        base = np.arange(l, dtype=np.int64)
        idx = (starts[:, None] + base[None, :]).reshape(-1)[:n]
        return idx

    if scheme == "circular_block":
        l = int(block_length)
        n_blocks = int(math.ceil(n / l))
        starts = rng.integers(0, n, size=n_blocks)
        base = np.arange(l, dtype=np.int64)
        idx = ((starts[:, None] + base[None, :]) % n).reshape(-1)[:n]
        return idx

    if scheme == "stationary":
        l = max(1, int(block_length))
        p = 1.0 / float(l)
        idx = np.empty(n, dtype=np.int64)
        idx[0] = rng.integers(0, n)
        new_flags = rng.random(n - 1) < p
        new_starts = rng.integers(0, n, size=n - 1)

        for t in range(1, n):
            if new_flags[t - 1]:
                idx[t] = new_starts[t - 1]
            else:
                idx[t] = (idx[t - 1] + 1) % n
        return idx

    raise ValueError(f"Unknown bootstrap scheme: {scheme}")


def bootstrap_indices_generator(
    n_samples: int,
    n_resamples: int,
    scheme: str,
    block_length: int,
    chunk_size: int,
    rng: np.random.Generator
):
    """Chunked generator yielding batches of bootstrap index matrices."""
    done = 0
    while done < n_resamples:
        m = min(int(chunk_size), n_resamples - done)
        idx_chunk = np.empty((m, n_samples), dtype=np.int64)
        for i in range(m):
            idx_chunk[i] = sample_bootstrap_indices_one(n_samples, scheme, block_length, rng)
        yield idx_chunk
        done += m


def compute_metrics_from_counts_matrix(store: dict, counts_matrix: np.ndarray, data_range=None, enable_image_metrics=False):
    """Vectorised metric recalculation across multiple bootstrap replicates using multiplicity matrix."""
    sum_abs = counts_matrix @ store["sum_abs"]
    sum_sq = counts_matrix @ store["sum_sq"]
    sum_rel = counts_matrix @ store["sum_rel"]

    count = (counts_matrix @ store["count"]).reshape(-1, 1)

    sum_pred = counts_matrix @ store["sum_pred"]
    sum_true = counts_matrix @ store["sum_true"]
    sum_pred2 = counts_matrix @ store["sum_pred2"]
    sum_true2 = counts_matrix @ store["sum_true2"]
    sum_pred_true = counts_matrix @ store["sum_pred_true"]

    if enable_image_metrics:
        ssim_vals = store["ssim"]
        ssim_safe = np.where(np.isfinite(ssim_vals), ssim_vals, 0.0)
        ssim_valid = np.isfinite(ssim_vals).astype(np.float64)

        ssim_sum = counts_matrix @ ssim_safe
        ssim_count = counts_matrix @ ssim_valid
    else:
        ssim_sum = None
        ssim_count = None

    return finalize_metric_arrays(
        sum_abs=sum_abs,
        sum_sq=sum_sq,
        sum_rel=sum_rel,
        count=count,
        sum_pred=sum_pred,
        sum_true=sum_true,
        sum_pred2=sum_pred2,
        sum_true2=sum_true2,
        sum_pred_true=sum_pred_true,
        data_range=data_range,
        ssim_sum=ssim_sum,
        ssim_count=ssim_count,
        enable_image_metrics=enable_image_metrics
    )


def compute_gre_batchmean_from_indices(gre_store: dict, indices: np.ndarray, batch_size: int) -> OrderedDict:
    """Compute average batch GRE loss over a specified set of sampled indices."""
    n = int(indices.size)
    if n <= 0:
        return OrderedDict((k, np.nan) for k in GRE_TERM_NAMES)

    starts = np.arange(0, n, batch_size, dtype=np.int64)

    def blk_mean(num_arr, den_arr, eps):
        v_num = num_arr[indices]
        v_den = den_arr[indices]
        num_blk = np.add.reduceat(v_num, starts)
        den_blk = np.add.reduceat(v_den, starts)
        return float(np.mean(num_blk / (den_blk + eps)))

    div = blk_mean(gre_store["div_num"], gre_store["div_den"], EPS_DIV)
    vor = blk_mean(gre_store["vor_num"], gre_store["vor_den"], EPS_VOR)

    geo_u = blk_mean(gre_store["geo_u_num"], gre_store["geo_u_den"], EPS_GEO)
    geo_v = blk_mean(gre_store["geo_v_num"], gre_store["geo_v_den"], EPS_GEO)
    geo = 0.5 * (geo_u + geo_v)

    mse_x = blk_mean(gre_store["mse_x_num"], gre_store["mse_x_den"], EPS_MSE)
    mse_y = blk_mean(gre_store["mse_y_num"], gre_store["mse_y_den"], EPS_MSE)
    mse = 0.5 * (mse_x + mse_y)

    return OrderedDict([
        ("DIV_GRE", div),
        ("VOR_GRE", vor),
        ("GEO_GRE", geo),
        ("MSE_GRE", mse),
    ])


def compute_gre_batchmean_chunk(gre_store: dict, idx_chunk: np.ndarray, batch_size: int) -> OrderedDict:
    """Compute batch GRE metrics for a chunk of resampled indices."""
    m = idx_chunk.shape[0]
    out = OrderedDict((k, np.empty((m,), dtype=np.float64)) for k in GRE_TERM_NAMES)
    for i in range(m):
        terms = compute_gre_batchmean_from_indices(gre_store, idx_chunk[i], batch_size)
        for k in GRE_TERM_NAMES:
            out[k][i] = float(terms[k])
    return out


def build_jackknife_blocks(n_samples: int, block_length: int, circular: bool) -> List[np.ndarray]:
    """Generate non-overlapping contiguous index blocks to drop for block-jackknife estimation."""
    if n_samples <= 1:
        return [np.array([0], dtype=np.int64)]

    l = max(1, min(int(block_length), n_samples - 1))
    starts = np.arange(0, n_samples, l, dtype=np.int64)

    blocks = []
    for s in starts:
        if circular:
            blk = (s + np.arange(l, dtype=np.int64)) % n_samples
            blk = np.unique(blk)
        else:
            blk = np.arange(s, min(s + l, n_samples), dtype=np.int64)
        if blk.size > 0:
            blocks.append(blk)
    return blocks


def compute_jackknife_metrics_from_store(
    store: dict,
    data_range,
    enable_image_metrics: bool,
    mode: str,
    block_length: int,
    scheme: str,
    desc: str
) -> OrderedDict:
    """Compute leave-one-out or leave-block-out Jackknife replicates for BCa acceleration computation."""
    n = int(store["count"].shape[0])
    c = int(store["sum_abs"].shape[1])

    total_abs = store["sum_abs"].sum(axis=0)
    total_sq = store["sum_sq"].sum(axis=0)
    total_rel = store["sum_rel"].sum(axis=0)
    total_count = float(store["count"].sum())

    total_pred = store["sum_pred"].sum(axis=0)
    total_true = store["sum_true"].sum(axis=0)
    total_pred2 = store["sum_pred2"].sum(axis=0)
    total_true2 = store["sum_true2"].sum(axis=0)
    total_pred_true = store["sum_pred_true"].sum(axis=0)

    if enable_image_metrics:
        ssim_vals = store["ssim"]
        ssim_safe = np.where(np.isfinite(ssim_vals), ssim_vals, 0.0)
        ssim_valid = np.isfinite(ssim_vals).astype(np.float64)
        total_ssim_sum = ssim_safe.sum(axis=0)
        total_ssim_count = ssim_valid.sum(axis=0)
    else:
        ssim_safe = ssim_valid = total_ssim_sum = total_ssim_count = None

    if mode == "delete1":
        sum_abs = total_abs.reshape(1, c) - store["sum_abs"]
        sum_sq = total_sq.reshape(1, c) - store["sum_sq"]
        sum_rel = total_rel.reshape(1, c) - store["sum_rel"]
        count = (total_count - store["count"]).reshape(-1, 1)

        sum_pred = total_pred.reshape(1, c) - store["sum_pred"]
        sum_true = total_true.reshape(1, c) - store["sum_true"]
        sum_pred2 = total_pred2.reshape(1, c) - store["sum_pred2"]
        sum_true2 = total_true2.reshape(1, c) - store["sum_true2"]
        sum_pred_true = total_pred_true.reshape(1, c) - store["sum_pred_true"]

        if enable_image_metrics:
            ssum = total_ssim_sum.reshape(1, c) - ssim_safe
            scnt = total_ssim_count.reshape(1, c) - ssim_valid
        else:
            ssum = scnt = None

        return finalize_metric_arrays(
            sum_abs=sum_abs,
            sum_sq=sum_sq,
            sum_rel=sum_rel,
            count=count,
            sum_pred=sum_pred,
            sum_true=sum_true,
            sum_pred2=sum_pred2,
            sum_true2=sum_true2,
            sum_pred_true=sum_pred_true,
            data_range=data_range,
            ssim_sum=ssum,
            ssim_count=scnt,
            enable_image_metrics=enable_image_metrics
        )

    if mode == "block":
        circular = scheme in {"circular_block", "stationary"}
        blocks = build_jackknife_blocks(n_samples=n, block_length=block_length, circular=circular)
        k = len(blocks)

        sum_abs = np.empty((k, c), dtype=np.float64)
        sum_sq = np.empty((k, c), dtype=np.float64)
        sum_rel = np.empty((k, c), dtype=np.float64)
        count = np.empty((k, 1), dtype=np.float64)

        sum_pred = np.empty((k, c), dtype=np.float64)
        sum_true = np.empty((k, c), dtype=np.float64)
        sum_pred2 = np.empty((k, c), dtype=np.float64)
        sum_true2 = np.empty((k, c), dtype=np.float64)
        sum_pred_true = np.empty((k, c), dtype=np.float64)

        if enable_image_metrics:
            ssum = np.empty((k, c), dtype=np.float64)
            scnt = np.empty((k, c), dtype=np.float64)
        else:
            ssum = scnt = None

        for i, rem in enumerate(tqdm(blocks, desc=desc, leave=False)):
            sum_abs[i] = total_abs - store["sum_abs"][rem].sum(axis=0)
            sum_sq[i] = total_sq - store["sum_sq"][rem].sum(axis=0)
            sum_rel[i] = total_rel - store["sum_rel"][rem].sum(axis=0)
            count[i, 0] = total_count - store["count"][rem].sum()

            sum_pred[i] = total_pred - store["sum_pred"][rem].sum(axis=0)
            sum_true[i] = total_true - store["sum_true"][rem].sum(axis=0)
            sum_pred2[i] = total_pred2 - store["sum_pred2"][rem].sum(axis=0)
            sum_true2[i] = total_true2 - store["sum_true2"][rem].sum(axis=0)
            sum_pred_true[i] = total_pred_true - store["sum_pred_true"][rem].sum(axis=0)

            if enable_image_metrics:
                ssum[i] = total_ssim_sum - ssim_safe[rem].sum(axis=0)
                scnt[i] = total_ssim_count - ssim_valid[rem].sum(axis=0)

        return finalize_metric_arrays(
            sum_abs=sum_abs,
            sum_sq=sum_sq,
            sum_rel=sum_rel,
            count=count,
            sum_pred=sum_pred,
            sum_true=sum_true,
            sum_pred2=sum_pred2,
            sum_true2=sum_true2,
            sum_pred_true=sum_pred_true,
            data_range=data_range,
            ssim_sum=ssum,
            ssim_count=scnt,
            enable_image_metrics=enable_image_metrics
        )

    raise ValueError(f"Unknown jackknife mode: {mode}")


def compute_gre_jackknife_replicates(
    gre_store: dict,
    n_samples: int,
    mode: str,
    block_length: int,
    scheme: str,
    batch_size: int,
    desc: str
) -> OrderedDict:
    """Compute jackknife replicates specifically for GRE physical metrics."""
    out = OrderedDict((k, []) for k in GRE_TERM_NAMES)
    if n_samples <= 1:
        return OrderedDict((k, np.array([np.nan], dtype=np.float64)) for k in GRE_TERM_NAMES)

    base = np.arange(n_samples, dtype=np.int64)

    if mode == "delete1":
        iterator = tqdm(range(n_samples), desc=desc, leave=False)
        for i in iterator:
            idx = np.empty((n_samples - 1,), dtype=np.int64)
            idx[:i] = base[:i]
            idx[i:] = base[i + 1:]
            terms = compute_gre_batchmean_from_indices(gre_store, idx, batch_size)
            for k in GRE_TERM_NAMES:
                out[k].append(float(terms[k]))

    elif mode == "block":
        circular = scheme in {"circular_block", "stationary"}
        blocks = build_jackknife_blocks(n_samples, block_length, circular)

        iterator = tqdm(blocks, desc=desc, leave=False)
        for rem in iterator:
            mask = np.ones(n_samples, dtype=bool)
            mask[rem] = False
            idx = base[mask]
            terms = compute_gre_batchmean_from_indices(gre_store, idx, batch_size)
            for k in GRE_TERM_NAMES:
                out[k].append(float(terms[k]))
    else:
        raise ValueError(f"Unknown jackknife mode: {mode}")

    return OrderedDict((k, np.asarray(v, dtype=np.float64)) for k, v in out.items())


def bootstrap_ci_1d(
    theta_hat: float,
    boot_vec: np.ndarray,
    confidence: float,
    method: str,
    jack_vec: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Calculate Confidence Interval bounds for a scalar metric estimate.
    Supported methods: 'percentile', 'basic', 'BCa' (Bias-Corrected and Accelerated).
    """
    if not np.isfinite(theta_hat):
        return np.nan, np.nan

    alpha = (1.0 - float(confidence)) / 2.0

    b = np.asarray(boot_vec, dtype=np.float64)
    b = b[np.isfinite(b)]
    B = b.size
    if B < 20:
        return np.nan, np.nan

    m = method.lower()
    if m == "percentile":
        low = np.quantile(b, alpha)
        high = np.quantile(b, 1.0 - alpha)

    elif m == "basic":
        q_hi = np.quantile(b, alpha)
        q_lo = np.quantile(b, 1.0 - alpha)
        low = 2.0 * theta_hat - q_lo
        high = 2.0 * theta_hat - q_hi

    elif m == "bca":
        if jack_vec is None:
            low = np.quantile(b, alpha)
            high = np.quantile(b, 1.0 - alpha)
        else:
            j = np.asarray(jack_vec, dtype=np.float64)
            j = j[np.isfinite(j)]
            if j.size < 8:
                low = np.quantile(b, alpha)
                high = np.quantile(b, 1.0 - alpha)
            else:
                prop = np.mean(b < theta_hat)
                prop = np.clip(prop, 1.0 / (2.0 * B), 1.0 - 1.0 / (2.0 * B))
                z0 = norm.ppf(prop)

                j_bar = np.mean(j)
                dif = j_bar - j
                num = np.sum(dif ** 3)
                den = 6.0 * (np.sum(dif ** 2) ** 1.5)
                a = num / den if den > 1e-30 else 0.0

                z_low = norm.ppf(alpha)
                z_high = norm.ppf(1.0 - alpha)

                adj_low = norm.cdf(z0 + (z0 + z_low) / (1.0 - a * (z0 + z_low)))
                adj_high = norm.cdf(z0 + (z0 + z_high) / (1.0 - a * (z0 + z_high)))

                adj_low = float(np.clip(adj_low, 1.0 / (B + 1.0), B / (B + 1.0)))
                adj_high = float(np.clip(adj_high, 1.0 / (B + 1.0), B / (B + 1.0)))

                low = np.quantile(b, adj_low)
                high = np.quantile(b, adj_high)
    else:
        raise ValueError(f"Unknown CI method: {method}")

    if low > high:
        low, high = high, low
    return float(low), float(high)


def build_bootstrap_ci_dataframe(
    point_result: Dict[str, Dict[str, np.ndarray]],
    boot_result: Dict[str, Dict[str, np.ndarray]],
    jack_result: Optional[Dict[str, Dict[str, np.ndarray]]],
    metric_order: List[str],
    var_names: List[str],
    group_name: str,
    ci_method: str,
    confidence: float,
    scheme: str,
    block_length_used,
    jackknife_mode_used: str,
    n_resamples: int,
    n_eval_samples: int
) -> pd.DataFrame:
    """Format Pointwise Bootstrap Confidence Intervals into long-format Pandas DataFrame."""
    rows = []
    for model_name, pmap in point_result.items():
        for metric in metric_order:
            theta = np.asarray(pmap[metric], dtype=np.float64).reshape(-1)
            boot_mat = np.asarray(boot_result[model_name][metric], dtype=np.float64)

            if jack_result is not None:
                jack_mat = np.asarray(jack_result[model_name][metric], dtype=np.float64)
            else:
                jack_mat = None

            for i, vname in enumerate(var_names):
                jack_vec = jack_mat[:, i] if jack_mat is not None else None
                low, high = bootstrap_ci_1d(
                    theta_hat=float(theta[i]),
                    boot_vec=boot_mat[:, i],
                    confidence=confidence,
                    method=ci_method,
                    jack_vec=jack_vec
                )
                rows.append({
                    "Model": model_name,
                    "Group": group_name,
                    "Metric": metric,
                    "Variable": vname,
                    "PointEstimate": float(theta[i]),
                    "CI_Low": low,
                    "CI_High": high,
                    "CI_Method": ci_method,
                    "Confidence": float(confidence),
                    "Bootstrap_Scheme": scheme,
                    "Block_Length_Used": block_length_used,
                    "Jackknife_Mode_Used": jackknife_mode_used,
                    "N_Resamples": int(n_resamples),
                    "N_EvalSamples": int(n_eval_samples),
                })

    return pd.DataFrame(rows)


def build_bootstrap_ci_dataframe_gre(
    point_result: Dict[str, Dict[str, float]],
    boot_result: Dict[str, Dict[str, np.ndarray]],
    jack_result: Optional[Dict[str, Dict[str, np.ndarray]]],
    ci_method: str,
    confidence: float,
    scheme: str,
    block_length_used,
    jackknife_mode_used: str,
    n_resamples: int,
    n_eval_samples: int
) -> pd.DataFrame:
    """Format GRE Bootstrap Confidence Intervals into Pandas DataFrame."""
    rows = []
    for model_name, pmap in point_result.items():
        for term in GRE_TERM_NAMES:
            theta = float(pmap[term])
            boot_vec = np.asarray(boot_result[model_name][term], dtype=np.float64)
            jack_vec = None
            if jack_result is not None:
                jack_vec = np.asarray(jack_result[model_name][term], dtype=np.float64)

            low, high = bootstrap_ci_1d(
                theta_hat=theta,
                boot_vec=boot_vec,
                confidence=confidence,
                method=ci_method,
                jack_vec=jack_vec
            )
            rows.append({
                "Model": model_name,
                "Group": "PHYS_GRE",
                "Metric": term,
                "Variable": "-",
                "PointEstimate": theta,
                "CI_Low": low,
                "CI_High": high,
                "CI_Method": ci_method,
                "Confidence": float(confidence),
                "Bootstrap_Scheme": scheme,
                "Block_Length_Used": block_length_used,
                "Jackknife_Mode_Used": jackknife_mode_used,
                "N_Resamples": int(n_resamples),
                "N_EvalSamples": int(n_eval_samples),
            })
    return pd.DataFrame(rows)


def run_bootstrap_ci(
    bootstrap_data,
    raw_result,
    phys_point_result,
    phys_gre_result,
    raw_data_range,
    batch_size,
    n_resamples,
    confidence,
    ci_method,
    scheme,
    block_length_cfg,
    jackknife_cfg,
    seed,
    chunk_size
):
    """Execute the master Bootstrap resampling pipeline across all models and metrics."""
    ci_method = normalize_ci_method(ci_method)
    scheme = normalize_bootstrap_scheme(scheme)

    method_names = list(raw_result.keys())
    if len(method_names) == 0:
        raise ValueError("No evaluation methods available for bootstrapping.")

    n_eval = int(bootstrap_data[method_names[0]]["raw"]["count"].shape[0])
    if n_eval <= 1:
        raise ValueError("Evaluation sample size is too small for Bootstrap CI calculation.")

    if scheme == "iid":
        block_len_used = 1
    else:
        block_len_used = resolve_block_length(n_eval, block_length_cfg)

    print("\n[Bootstrap Configuration]")
    print(f"  Method            : {ci_method}")
    print(f"  Confidence Level  : {confidence}")
    print(f"  Resampling Scheme : {scheme}")
    print(f"  Num Resamples (B) : {n_resamples}")
    print(f"  Eval Samples (N)  : {n_eval}")
    print(f"  Block Length      : {block_len_used if scheme != 'iid' else 'N/A'}")
    print(f"  Chunk Size        : {chunk_size}")

    raw_metric_order = ["RMSE", "MSE", "MAE", "MRE", "Correlation", "PSNR", "SSIM"]
    phys_metric_order = ["RMSE", "MSE", "MAE", "MRE", "Correlation"]

    # Pre-allocate bootstrap result buffers
    boot_raw = OrderedDict()
    boot_phys = OrderedDict()
    boot_gre = OrderedDict()

    for m in method_names:
        c_raw = len(VAR_DISPLAY)
        c_phys = len(PHYS_POINTWISE_NAMES)

        boot_raw[m] = OrderedDict((k, np.empty((n_resamples, c_raw), dtype=np.float64)) for k in raw_metric_order)
        boot_phys[m] = OrderedDict((k, np.empty((n_resamples, c_phys), dtype=np.float64)) for k in phys_metric_order)
        boot_gre[m] = OrderedDict((k, np.empty((n_resamples,), dtype=np.float64)) for k in GRE_TERM_NAMES)

    rng = np.random.default_rng(seed)
    offset = 0

    pbar = tqdm(total=n_resamples, desc=f"Bootstrap Resampling [{scheme}]")
    for idx_chunk in bootstrap_indices_generator(
        n_samples=n_eval,
        n_resamples=n_resamples,
        scheme=scheme,
        block_length=block_len_used,
        chunk_size=chunk_size,
        rng=rng
    ):
        m_chunk = idx_chunk.shape[0]

        # Convert indices to multiplicity matrix via bincount for vectorized metric calculation
        counts_chunk = np.stack(
            [np.bincount(idx_chunk[i], minlength=n_eval) for i in range(m_chunk)],
            axis=0
        ).astype(np.float64)

        for method_name in method_names:
            raw_chunk_map = compute_metrics_from_counts_matrix(
                store=bootstrap_data[method_name]["raw"],
                counts_matrix=counts_chunk,
                data_range=raw_data_range,
                enable_image_metrics=True
            )
            for met in raw_metric_order:
                boot_raw[method_name][met][offset:offset + m_chunk] = raw_chunk_map[met]

            phys_chunk_map = compute_metrics_from_counts_matrix(
                store=bootstrap_data[method_name]["phys"],
                counts_matrix=counts_chunk,
                data_range=None,
                enable_image_metrics=False
            )
            for met in phys_metric_order:
                boot_phys[method_name][met][offset:offset + m_chunk] = phys_chunk_map[met]

            gre_chunk_map = compute_gre_batchmean_chunk(
                gre_store=bootstrap_data[method_name]["gre"],
                idx_chunk=idx_chunk,
                batch_size=batch_size
            )
            for k in GRE_TERM_NAMES:
                boot_gre[method_name][k][offset:offset + m_chunk] = gre_chunk_map[k]

        offset += m_chunk
        pbar.update(m_chunk)

    pbar.close()

    # BCa: Compute Jackknife replicates for acceleration parameter estimation
    if ci_method == "BCa":
        jk_mode = resolve_jackknife_mode(jackknife_cfg, scheme)

        if jk_mode == "delete1" and n_eval > 5000:
            warnings.warn("delete1 Jackknife may be slow for N > 5000. Consider setting BOOTSTRAP_JACKKNIFE='block'.")

        if jk_mode == "block":
            jk_block_len = resolve_block_length(n_eval, block_length_cfg)
            jk_block_len = max(1, min(jk_block_len, n_eval - 1))
        else:
            jk_block_len = 1

        print(f"\n[BCa Jackknife Configuration] Mode={jk_mode}, Block Length={jk_block_len if jk_mode=='block' else 'N/A'}")

        jack_raw = OrderedDict()
        jack_phys = OrderedDict()
        jack_gre = OrderedDict()

        for method_name in method_names:
            jack_raw[method_name] = compute_jackknife_metrics_from_store(
                store=bootstrap_data[method_name]["raw"],
                data_range=raw_data_range,
                enable_image_metrics=True,
                mode=jk_mode,
                block_length=jk_block_len,
                scheme=scheme,
                desc=f"Jackknife RAW [{method_name}]"
            )

            jack_phys[method_name] = compute_jackknife_metrics_from_store(
                store=bootstrap_data[method_name]["phys"],
                data_range=None,
                enable_image_metrics=False,
                mode=jk_mode,
                block_length=jk_block_len,
                scheme=scheme,
                desc=f"Jackknife PHYS [{method_name}]"
            )

            jack_gre[method_name] = compute_gre_jackknife_replicates(
                gre_store=bootstrap_data[method_name]["gre"],
                n_samples=n_eval,
                mode=jk_mode,
                block_length=jk_block_len,
                scheme=scheme,
                batch_size=batch_size,
                desc=f"Jackknife GRE [{method_name}]"
            )
        jackknife_mode_used = jk_mode
    else:
        jack_raw = None
        jack_phys = None
        jack_gre = None
        jackknife_mode_used = "N/A"

    # Assemble long-format Confidence Interval DataFrames
    df_raw_ci = build_bootstrap_ci_dataframe(
        point_result=raw_result,
        boot_result=boot_raw,
        jack_result=jack_raw,
        metric_order=raw_metric_order,
        var_names=VAR_DISPLAY,
        group_name="RAW",
        ci_method=ci_method,
        confidence=confidence,
        scheme=scheme,
        block_length_used=(block_len_used if scheme != "iid" else "N/A"),
        jackknife_mode_used=jackknife_mode_used,
        n_resamples=n_resamples,
        n_eval_samples=n_eval
    )

    df_phys_ci = build_bootstrap_ci_dataframe(
        point_result=phys_point_result,
        boot_result=boot_phys,
        jack_result=jack_phys,
        metric_order=phys_metric_order,
        var_names=PHYS_POINTWISE_NAMES,
        group_name="PHYS_POINTWISE",
        ci_method=ci_method,
        confidence=confidence,
        scheme=scheme,
        block_length_used=(block_len_used if scheme != "iid" else "N/A"),
        jackknife_mode_used=jackknife_mode_used,
        n_resamples=n_resamples,
        n_eval_samples=n_eval
    )

    df_gre_ci = build_bootstrap_ci_dataframe_gre(
        point_result=phys_gre_result,
        boot_result=boot_gre,
        jack_result=jack_gre,
        ci_method=ci_method,
        confidence=confidence,
        scheme=scheme,
        block_length_used=(block_len_used if scheme != "iid" else "N/A"),
        jackknife_mode_used=jackknife_mode_used,
        n_resamples=n_resamples,
        n_eval_samples=n_eval
    )

    df_boot_all = pd.concat([df_raw_ci, df_phys_ci, df_gre_ci], axis=0, ignore_index=True)

    meta = {
        "ci_method": ci_method,
        "scheme": scheme,
        "block_len_used": (block_len_used if scheme != "iid" else "N/A"),
        "jackknife_mode_used": jackknife_mode_used,
        "n_resamples": n_resamples,
        "n_eval_samples": n_eval
    }

    return df_raw_ci, df_phys_ci, df_gre_ci, df_boot_all, meta


# =============================================================================
# 6) Target Pre-scanning and Evaluation Workflow
# =============================================================================
def precompute_raw_target_stats(ds_y: xr.Dataset, variables: List[str], batch_size: int, max_samples: Optional[int]):
    """
    Pre-scan Y_test target dataset:
    1. Channel-wise dynamic range (max - min) for PSNR/SSIM calculation.
    2. Channel-wise mean absolute values for adaptive MRE epsilon stabilization.
    """
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


def precompute_phys_pointwise_target_mean_abs(
    ds_y: xr.Dataset,
    variables: List[str],
    batch_size: int,
    max_samples: Optional[int],
    physics_ops: SphericalPhysicsOps
) -> np.ndarray:
    """Pre-scan Y_test physical derived variables to determine mean absolute values for MRE epsilon."""
    n_total = int(ds_y.sizes["time"])
    if max_samples is not None:
        n_total = min(n_total, int(max_samples))

    sum_abs = np.zeros(len(PHYS_POINTWISE_NAMES), dtype=np.float64)
    count = 0.0

    for st in tqdm(range(0, n_total, batch_size), desc="Pre-scanning Y_test (physics pointwise)"):
        ed = min(st + batch_size, n_total)
        y = read_batch_from_ds(ds_y, variables, st, ed)
        phy = compute_pointwise_physics_derived(y, physics_ops)

        sum_abs += np.sum(np.abs(phy), axis=(0, 2, 3))
        count += phy.shape[0] * phy.shape[2] * phy.shape[3]

    mean_abs = sum_abs / max(count, 1.0)
    return mean_abs


def evaluate_all_methods(
    ds_x, ds_y, variables,
    model_specs,
    baseline_predictors,
    physics_ops,
    raw_data_range, raw_mre_eps, phys_point_mre_eps,
    batch_size, max_samples, device,
    enable_bootstrap=False
):
    """Execute main evaluation loop across baselines and EDSR models."""
    method_names = list(baseline_predictors.keys()) + list(model_specs.keys())
    if len(method_names) == 0:
        raise ValueError("No methods registered for evaluation.")

    raw_acc = OrderedDict()
    phys_point_acc = OrderedDict()
    phys_gre_acc = OrderedDict()

    for m in method_names:
        raw_acc[m] = MetricAccumulator(
            var_names=VAR_DISPLAY,
            mre_eps=raw_mre_eps,
            data_range=raw_data_range,
            enable_image_metrics=True
        )
        phys_point_acc[m] = MetricAccumulator(
            var_names=PHYS_POINTWISE_NAMES,
            mre_eps=phys_point_mre_eps,
            data_range=None,
            enable_image_metrics=False
        )
        phys_gre_acc[m] = PhysicsGREAccumulator(physics_ops=physics_ops)

    boot_collector = BootstrapStore(method_names) if enable_bootstrap else None

    n_total = int(ds_x.sizes["time"])
    if max_samples is not None:
        n_total = min(n_total, int(max_samples))

    for st in tqdm(range(0, n_total, batch_size), desc="Main Evaluation Pipeline"):
        ed = min(st + batch_size, n_total)

        x_raw = read_batch_from_ds(ds_x, variables, st, ed)  # (B, 5, H_lr, W_lr)
        y_raw = read_batch_from_ds(ds_y, variables, st, ed)  # (B, 5, H_hr, W_hr)

        true_phys_point = compute_pointwise_physics_derived(y_raw, physics_ops)

        # -------------------- Baseline Evaluations --------------------
        for base_name, predictor in baseline_predictors.items():
            pred_raw = predictor(x_raw)
            if pred_raw.shape != y_raw.shape:
                raise ValueError(f"{base_name} output shape {pred_raw.shape} mismatches ground truth {y_raw.shape}")

            raw_contrib = compute_metric_contrib(
                pred=pred_raw, true=y_raw, mre_eps=raw_mre_eps,
                data_range=raw_data_range, enable_image_metrics=True
            )
            raw_acc[base_name].update_from_contrib(raw_contrib)
            if enable_bootstrap:
                boot_collector.append_raw(base_name, raw_contrib)

            pred_phys_point = compute_pointwise_physics_derived(pred_raw, physics_ops)
            phys_contrib = compute_metric_contrib(
                pred=pred_phys_point, true=true_phys_point, mre_eps=phys_point_mre_eps,
                data_range=None, enable_image_metrics=False
            )
            phys_point_acc[base_name].update_from_contrib(phys_contrib)
            if enable_bootstrap:
                boot_collector.append_phys(base_name, phys_contrib)

            phys_gre_acc[base_name].update(pred_raw, y_raw)
            if enable_bootstrap:
                gre_comp = compute_physics_gre_sample_components(pred_raw, y_raw, physics_ops)
                boot_collector.append_gre(base_name, gre_comp)

        # -------------------- ClassicEDSR Model Evaluations --------------------
        if len(model_specs) > 0:
            with torch.no_grad():
                for model_name, spec in model_specs.items():
                    x_std_batch = standardize_batch(x_raw, spec["x_mean"], spec["x_std"])
                    x_tensor = torch.from_numpy(x_std_batch).to(device, non_blocking=True)

                    pred_std = spec["model"](x_tensor)
                    pred_std_np = pred_std.detach().cpu().numpy().astype(np.float32)
                    pred_raw = inverse_standardize_batch(pred_std_np, spec["y_mean"], spec["y_std"])

                    if pred_raw.shape != y_raw.shape:
                        raise ValueError(
                            f"{model_name} output shape {pred_raw.shape} mismatches ground truth {y_raw.shape}. "
                            "Please verify model upscale factor matches the test dataset magnification."
                        )

                    raw_contrib = compute_metric_contrib(
                        pred=pred_raw, true=y_raw, mre_eps=raw_mre_eps,
                        data_range=raw_data_range, enable_image_metrics=True
                    )
                    raw_acc[model_name].update_from_contrib(raw_contrib)
                    if enable_bootstrap:
                        boot_collector.append_raw(model_name, raw_contrib)

                    pred_phys_point = compute_pointwise_physics_derived(pred_raw, physics_ops)
                    phys_contrib = compute_metric_contrib(
                        pred=pred_phys_point, true=true_phys_point, mre_eps=phys_point_mre_eps,
                        data_range=None, enable_image_metrics=False
                    )
                    phys_point_acc[model_name].update_from_contrib(phys_contrib)
                    if enable_bootstrap:
                        boot_collector.append_phys(model_name, phys_contrib)

                    phys_gre_acc[model_name].update(pred_raw, y_raw)
                    if enable_bootstrap:
                        gre_comp = compute_physics_gre_sample_components(pred_raw, y_raw, physics_ops)
                        boot_collector.append_gre(model_name, gre_comp)

    raw_result = OrderedDict((k, v.finalize()) for k, v in raw_acc.items())
    phys_point_result = OrderedDict((k, v.finalize()) for k, v in phys_point_acc.items())
    phys_gre_result = OrderedDict((k, v.finalize()) for k, v in phys_gre_acc.items())

    bootstrap_data = boot_collector.finalize() if enable_bootstrap else None
    return raw_result, phys_point_result, phys_gre_result, bootstrap_data


def print_quick_summary(raw_result: dict, phys_point_result: dict, phys_gre_result: dict):
    """Print high-level mean performance metrics across evaluation methods to terminal."""
    print("\n" + "=" * 120)
    print("Summary Performance Overview (Mean across variables):")
    print("-" * 120)
    for model_name in raw_result.keys():
        raw_rmse_mean = float(np.nanmean(raw_result[model_name]["RMSE"]))
        raw_psnr_mean = float(np.nanmean(raw_result[model_name]["PSNR"]))
        raw_ssim_mean = float(np.nanmean(raw_result[model_name]["SSIM"]))

        phy_rmse_mean = float(np.nanmean(phys_point_result[model_name]["RMSE"]))
        phy_corr_mean = float(np.nanmean(phys_point_result[model_name]["Correlation"]))

        gre = phys_gre_result[model_name]
        print(
            f"{model_name:22s} | "
            f"Raw RMSE={raw_rmse_mean:.6e}, PSNR={raw_psnr_mean:.4f}, SSIM={raw_ssim_mean:.4f} | "
            f"PhysPoint RMSE={phy_rmse_mean:.6e}, Corr={phy_corr_mean:.4f} | "
            f"GRE: DIV={gre['DIV_GRE']:.6e}, VOR={gre['VOR_GRE']:.6e}, GEO={gre['GEO_GRE']:.6e}, MSE={gre['MSE_GRE']:.6e}"
        )
    print("-" * 120)


# =============================================================================
# 7) Main Program Execution
# =============================================================================
def main():
    set_seed(RANDOM_SEED)
    device = torch.device(DEVICE)

    print("=" * 120)
    print("ERA5 ClassicEDSR Super-Resolution Evaluation (with Bootstrap CI)")
    print(f"Device: {device}")
    print("=" * 120)

    # File Existence Checks
    must_exist = [X_TEST_PATH, Y_TEST_PATH, SCALER_PATH] + list(MODEL_PATHS.values())
    for p in must_exist:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    ds_x = xr.open_dataset(X_TEST_PATH, decode_times=False)
    ds_y = xr.open_dataset(Y_TEST_PATH, decode_times=False)

    try:
        # Check NetCDF variables
        for v in VARIABLES:
            if v not in ds_x.variables:
                raise KeyError(f"{X_TEST_PATH} missing required variable: {v}")
            if v not in ds_y.variables:
                raise KeyError(f"{Y_TEST_PATH} missing required variable: {v}")

        n_x = int(ds_x.sizes["time"])
        n_y = int(ds_y.sizes["time"])
        if n_x != n_y:
            raise ValueError(f"Time dimension mismatch: X={n_x}, Y={n_y}")

        lat_lr, lat_lr_name = get_coord_1d(ds_x, ["latitude", "lat"])
        lon_lr, lon_lr_name = get_coord_1d(ds_x, ["longitude", "lon"])
        lat_hr, lat_hr_name = get_coord_1d(ds_y, ["latitude", "lat"])
        lon_hr, lon_hr_name = get_coord_1d(ds_y, ["longitude", "lon"])

        print(f"X_test shape: time={n_x}, lat={len(lat_lr)}, lon={len(lon_lr)}")
        print(f"Y_test shape: time={n_y}, lat={len(lat_hr)}, lon={len(lon_hr)}")
        print(f"X Coordinate names: lat='{lat_lr_name}', lon='{lon_lr_name}'")
        print(f"Y Coordinate names: lat='{lat_hr_name}', lon='{lon_hr_name}'")

        if (len(lat_hr) % len(lat_lr) == 0) and (len(lon_hr) % len(lon_lr) == 0):
            scale_lat = len(lat_hr) // len(lat_lr)
            scale_lon = len(lon_hr) // len(lon_lr)
            print(f"Dataset magnification factor: lat x{scale_lat}, lon x{scale_lon}")
        else:
            print("[Warning] HR/LR dimensions are not integer multiples. Please verify alignment.")

        # Load Scalers
        scaler_x_stats, scaler_y_stats = load_scaler_stats(SCALER_PATH)
        print("Loaded scalers.joblib successfully.")

        # Load ClassicEDSR Models
        model_specs = OrderedDict()
        print("\nLoading ClassicEDSR models...")
        for name, path in MODEL_PATHS.items():
            spec = load_one_classic_edsr_model(path, scaler_x_stats, scaler_y_stats, device)
            model_specs[name] = spec

            hp = spec["info"]["hparams"]
            n_params = sum(p.numel() for p in spec["model"].parameters())
            print(
                f"  - {name:22s} | {path}\n"
                f"      arch={spec['info']['architecture']}, epoch={spec['info']['epoch']}, "
                f"scale={hp['scale']}, feats={hp['n_feats']}, blocks={hp['n_resblocks']}, "
                f"use_bn={hp['use_bn']}, res_scale={hp['res_scale']}, params={n_params:,}"
            )

            dx = float(np.max(np.abs(spec["x_mean"] - scaler_x_stats["mean"])))
            dy = float(np.max(np.abs(spec["y_mean"] - scaler_y_stats["mean"])))
            if dx > 1e-6 or dy > 1e-6:
                print(f"      [Note] Scaler statistics in {name} checkpoint slightly differ from joblib file. Using checkpoint scalers.")

        # Instantiate Baselines
        baseline_predictors = OrderedDict()

        if ENABLE_BICUBIC_BASELINE:
            baseline_predictors["bicubic"] = BicubicInterpolatorTorch(
                out_h=len(lat_hr),
                out_w=len(lon_hr),
                device=device
            )

        if ENABLE_CUBIC_SPLINE_BASELINE:
            baseline_predictors["cubic spline"] = CubicSplineInterpolator2D(
                lat_lr=lat_lr, lon_lr=lon_lr,
                lat_hr=lat_hr, lon_hr=lon_hr,
                kx=CUBIC_SPLINE_KX, ky=CUBIC_SPLINE_KY
            )

        print(f"\nRegistered baselines: {list(baseline_predictors.keys())}")

        # Instantiate Spherical Physics Operator
        physics_ops = SphericalPhysicsOps(
            latitudes=lat_hr,
            longitudes=lon_hr,
            geo_weight_mode=GEO_WEIGHT_MODE,
            geo_weight_alpha=GEO_WEIGHT_ALPHA,
            geo_weight_min=GEO_WEIGHT_MIN
        )
        print("Built spherical physics operator with latitudinal weighting.")

        # Precompute Target Data Range and Epsilon Parameters
        raw_data_range, raw_mean_abs = precompute_raw_target_stats(
            ds_y=ds_y,
            variables=VARIABLES,
            batch_size=BATCH_SIZE,
            max_samples=EVAL_MAX_SAMPLES
        )
        raw_mre_eps = np.maximum(MRE_EPS_ABS, MRE_EPS_REL * raw_mean_abs)

        print("\nRaw variable data_range and MRE epsilon:")
        for i, name in enumerate(VAR_DISPLAY):
            print(f"  {name}: data_range={raw_data_range[i]:.6e}, mre_eps={raw_mre_eps[i]:.6e}")

        # Precompute Pointwise Physics Epsilon
        phys_point_mean_abs = precompute_phys_pointwise_target_mean_abs(
            ds_y=ds_y,
            variables=VARIABLES,
            batch_size=BATCH_SIZE,
            max_samples=EVAL_MAX_SAMPLES,
            physics_ops=physics_ops
        )
        phys_point_mre_eps = np.maximum(MRE_EPS_ABS, MRE_EPS_REL * phys_point_mean_abs)

        print("\nPhysics (pointwise) variable MRE epsilon:")
        for i, name in enumerate(PHYS_POINTWISE_NAMES):
            print(f"  {name}: mre_eps={phys_point_mre_eps[i]:.6e}")

        # Run Primary Evaluation Loop
        raw_result, phys_point_result, phys_gre_result, bootstrap_data = evaluate_all_methods(
            ds_x=ds_x,
            ds_y=ds_y,
            variables=VARIABLES,
            model_specs=model_specs,
            baseline_predictors=baseline_predictors,
            physics_ops=physics_ops,
            raw_data_range=raw_data_range,
            raw_mre_eps=raw_mre_eps,
            phys_point_mre_eps=phys_point_mre_eps,
            batch_size=BATCH_SIZE,
            max_samples=EVAL_MAX_SAMPLES,
            device=device,
            enable_bootstrap=ENABLE_BOOTSTRAP
        )

        # Output Standard CSV Metrics
        raw_metric_order = ["RMSE", "MSE", "MAE", "MRE", "Correlation", "PSNR", "SSIM"]
        phys_metric_order = ["RMSE", "MSE", "MAE", "MRE", "Correlation"]

        df_raw = build_metrics_dataframe(
            result_dict=raw_result,
            var_names=VAR_DISPLAY,
            metric_order=raw_metric_order
        )
        df_phys_point = build_metrics_dataframe(
            result_dict=phys_point_result,
            var_names=PHYS_POINTWISE_NAMES,
            metric_order=phys_metric_order
        )
        df_phys_gre = build_gre_dataframe(phys_gre_result)

        df_raw.to_csv(CSV_RAW_METRICS_PATH, index=False, float_format="%.6e")
        df_phys_point.to_csv(CSV_PHYS_POINTWISE_METRICS_PATH, index=False, float_format="%.6e")
        df_phys_gre.to_csv(CSV_PHYS_GRE_METRICS_PATH, index=False, float_format="%.6e")

        df_all = (
            df_raw.set_index("Model")
            .join(df_phys_point.set_index("Model"), how="outer")
            .join(df_phys_gre.set_index("Model"), how="outer")
            .reset_index()
        )
        df_all.to_csv(CSV_ALL_IN_ONE_PATH, index=False, float_format="%.6e")

        print_quick_summary(raw_result, phys_point_result, phys_gre_result)

        # Output Bootstrap Confidence Interval CSVs
        if ENABLE_BOOTSTRAP:
            df_raw_ci, df_phys_ci, df_gre_ci, df_boot_all, boot_meta = run_bootstrap_ci(
                bootstrap_data=bootstrap_data,
                raw_result=raw_result,
                phys_point_result=phys_point_result,
                phys_gre_result=phys_gre_result,
                raw_data_range=raw_data_range,
                batch_size=BATCH_SIZE,
                n_resamples=int(BOOTSTRAP_N_RESAMPLES),
                confidence=float(BOOTSTRAP_CONFIDENCE),
                ci_method=BOOTSTRAP_CI_METHOD,
                scheme=BOOTSTRAP_SCHEME,
                block_length_cfg=BOOTSTRAP_BLOCK_LENGTH,
                jackknife_cfg=BOOTSTRAP_JACKKNIFE,
                seed=int(BOOTSTRAP_SEED),
                chunk_size=int(BOOTSTRAP_CHUNK)
            )

            df_raw_ci.to_csv(CSV_RAW_BOOTSTRAP_CI_PATH, index=False, float_format="%.6e")
            df_phys_ci.to_csv(CSV_PHYS_POINTWISE_BOOTSTRAP_CI_PATH, index=False, float_format="%.6e")
            df_gre_ci.to_csv(CSV_PHYS_GRE_BOOTSTRAP_CI_PATH, index=False, float_format="%.6e")
            df_boot_all.to_csv(CSV_BOOTSTRAP_ALL_IN_ONE_PATH, index=False, float_format="%.6e")

            print("\nBootstrap Metadata:")
            for k, v in boot_meta.items():
                print(f"  {k}: {v}")

        print("\n" + "=" * 120)
        print("Evaluation complete. Exported CSV files:")
        print(f"  1) {CSV_RAW_METRICS_PATH}")
        print(f"  2) {CSV_PHYS_POINTWISE_METRICS_PATH}")
        print(f"  3) {CSV_PHYS_GRE_METRICS_PATH}")
        print(f"  4) {CSV_ALL_IN_ONE_PATH}")

        if ENABLE_BOOTSTRAP:
            print(f"  5) {CSV_RAW_BOOTSTRAP_CI_PATH}")
            print(f"  6) {CSV_PHYS_POINTWISE_BOOTSTRAP_CI_PATH}")
            print(f"  7) {CSV_PHYS_GRE_BOOTSTRAP_CI_PATH}")
            print(f"  8) {CSV_BOOTSTRAP_ALL_IN_ONE_PATH}")

        print("=" * 120)

    finally:
        ds_x.close()
        ds_y.close()


if __name__ == "__main__":
    main()