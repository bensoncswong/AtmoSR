#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA5 Super-Resolution Model Training (VDSR with Physical Constraint Ablation)

VDSR Architecture:
  1) Low-resolution input X_std (B, C, h, w)
  2) Bicubic upsampling to high-resolution resolution (B, C, H, W)
  3) Map to Y_std space as baseline
  4) Deep CNN (15-20 conv layers of 3x3, without BN) to predict residual
  5) Output = baseline + residual (Global Residual)

Physical Constraints:
  - Divergence, vorticity, geostrophic (ageostrophic), moist static energy gradient
  - Comparative formulation: Predicted physical quantity vs. Target physical quantity
  - Calculated in physical space after inverse normalization
  - Global Relative Error (GRE): sum|pred - target| / (sum|target| + eps) per batch

Ablation Experiments:
  - vdsr_l1_only     (L1 loss only, no physics constraints)
  - vdsr_l1_physics  (L1 loss + physics constraints)
"""

import os
import gc
import json
import time
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from datetime import datetime


# =============================================================================
# 0) Global Configurations
# =============================================================================
NUM_EPOCHS         = 200
LEARNING_RATE      = 1e-4
BATCH_SIZE         = 32
RANDOM_SEED        = 42
WEIGHT_DECAY       = 1e-4
SCHEDULER_FACTOR   = 0.1
SCHEDULER_PATIENCE = 3
NUM_WORKERS        = 2
CHECK_MISSING      = False

# ---------- VDSR Configuration ----------
NUM_FEATURES           = 64
NUM_VDSR_LAYERS        = 18   # Recommended range: 15-20
UPSCALE_FACTOR         = 2    # Supported: 2 or 4 (verified against data resolution)
BICUBIC_ALIGN_CORNERS  = False
ZERO_INIT_LAST         = True  # Ensures initial output closely follows bicubic baseline

# ---------- Thermodynamic Constants ----------
CP_AIR   = 1004.0     # Specific heat capacity of air [J/(kg K)]
LV_WATER = 2.501e6    # Latent heat of vaporization [J/kg]

# ---------- GRE Epsilon Constants ----------
EPS_DIV = 1e-8
EPS_VOR = 1e-8
EPS_GEO = 1e-4
EPS_MSE = 1e-4

# ---------- Geostrophic Weighting Settings ----------
GEO_WEIGHT_MODE  = "sin_global"  # Options: ['f_ratio', 'sin_global', 'cos', 'one_minus_cos']
GEO_WEIGHT_ALPHA = 1.0
GEO_WEIGHT_MIN   = 0.10

# ---------- Lambda Geo Auto-tuning (Optional) ----------
AUTO_TUNE_LAMBDA_GEO = False
GEO_TARGET_REL_TO_L1 = 0.10
GEO_LAMBDA_MIN       = 1e-3
GEO_LAMBDA_MAX       = 2.0

# ---------- Data Paths ----------
X_TRAIN_PATH = "X2_train.nc"
Y_TRAIN_PATH = "Y_train.nc"
X_VALID_PATH = "X2_valid.nc"
Y_VALID_PATH = "Y_valid.nc"
X_TEST_PATH  = "X2_test.nc"
Y_TEST_PATH  = "Y_test.nc"

VARIABLES    = ["z", "q", "t", "u", "v"]
VAR_NAMES    = ["Z", "Q", "T", "U", "V"]
VAR_UNITS    = ["m^2/s^2", "kg/kg", "K", "m/s", "m/s"]
NUM_CHANNELS = len(VARIABLES)

PHYSICS_KEYS = ["div", "vor", "geo", "mse"]

OUTPUT_ROOT      = "runs_vdsr_ablation"
SCALER_SAVE_PATH = os.path.join(OUTPUT_ROOT, "scalers.joblib")
SUMMARY_CSV_PATH = os.path.join(OUTPUT_ROOT, "ablation_summary.csv")

# ---------- Ablation Experiment Setup ----------
EXPERIMENTS = [
    {
        "name": "vdsr_l1_only",
        "use_physics": False,
        "lam_l1": 1.0, "lam_div": 0.0, "lam_vor": 0.0, "lam_geo": 0.0, "lam_mse": 0.0
    },
    {
        "name": "vdsr_l1_physics",
        "use_physics": True,
        "lam_l1": 1.0, "lam_div": 0.1, "lam_vor": 0.1, "lam_geo": 0.1, "lam_mse": 0.1
    },
]

# Set to specific experiment names to run subset (e.g., ["vdsr_l1_only"]), or None for all
RUN_ONLY_EXPERIMENTS = ["vdsr_l1_only"]


# =============================================================================
# 1) Utility Functions
# =============================================================================
def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_missing_values(file_paths, variables, missing_value=-999):
    """Check NetCDF files for NaN or placeholder missing values."""
    print("Checking NetCDF files for missing values (-999/NaN)...")
    print("-" * 90)
    found_missing = False
    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        ds = xr.open_dataset(path)
        file_has_issue = False
        for var in variables:
            data = ds[var].values
            miss_count = int(np.sum(np.isclose(data, missing_value, atol=0.1)))
            nan_count  = int(np.sum(np.isnan(data)))
            if miss_count > 0:
                print(f"[WARNING] {path} | {var}: Found {miss_count:,} missing values (-999)")
                found_missing = True
                file_has_issue = True
            if nan_count > 0:
                print(f"[WARNING] {path} | {var}: Found {nan_count:,} NaNs")
                found_missing = True
                file_has_issue = True
        if not file_has_issue:
            print(f"[PASSED] {path} has no missing values.")
        ds.close()

    print("-" * 90)
    if found_missing:
        raise ValueError("Missing or invalid values found in dataset. Clean data before proceeding.")
    print("[RESULT] All missing value checks passed successfully.")
    print("=" * 90)


def format_value(val):
    """Format numerical values dynamically for clear logging."""
    abs_val = abs(val)
    if abs_val >= 100:
        return f"{val:.2f}"
    elif abs_val >= 0.01:
        return f"{val:.4f}"
    else:
        return f"{val:.4e}"


def format_rmse_str(rmse_array, var_names, overall_rmse):
    """Format variable-wise and overall RMSE values into a single string."""
    parts = [f"{name}: {format_value(rmse_array[i])}" for i, name in enumerate(var_names)]
    return ", ".join(parts) + f", Overall: {format_value(overall_rmse)}"


def format_physics_str(phys_dict):
    """Format physical loss components into a single string."""
    items = []
    for k in PHYSICS_KEYS:
        v = phys_dict[k]
        if v >= 0.01:
            items.append(f"{k}={v:.4f}")
        else:
            items.append(f"{k}={v:.2e}")
    return ", ".join(items)


def suggest_lambda_geo(diag_loss_dict, lam_l1, target_rel_to_l1, lam_min, lam_max):
    """Suggest balanced lambda_geo dynamic scaling factor based on initial loss ratio."""
    l1  = max(float(diag_loss_dict["l1"]), 1e-12)
    geo = max(float(diag_loss_dict["geo"]), 1e-12)
    raw = (target_rel_to_l1 * float(lam_l1) * l1) / geo
    clipped = float(np.clip(raw, lam_min, lam_max))
    return raw, clipped


# =============================================================================
# 2) Channel Scaler
# =============================================================================
class ChannelScaler:
    """Channel-wise Z-score Normalization."""
    def __init__(self):
        self.mean = None
        self.std = None
        self.n_channels = None

    def fit(self, data):
        # data shape: (N, C, H, W)
        self.n_channels = data.shape[1]
        self.mean = data.mean(axis=(0, 2, 3)).astype(np.float64)
        self.std = data.std(axis=(0, 2, 3)).astype(np.float64)
        self.std[self.std < 1e-10] = 1.0
        return self

    def transform(self, data):
        mean = self.mean.reshape(1, -1, 1, 1).astype(np.float32)
        std  = self.std.reshape(1, -1, 1, 1).astype(np.float32)
        data = data.astype(np.float32, copy=False)
        return ((data - mean) / std).astype(np.float32)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform_numpy(self, data):
        mean = self.mean.reshape(1, -1, 1, 1).astype(np.float32)
        std  = self.std.reshape(1, -1, 1, 1).astype(np.float32)
        return data * std + mean

    def __repr__(self):
        if self.mean is None:
            return "ChannelScaler(Unfitted)"
        mean_str = ", ".join([f"{m:.4f}" for m in self.mean])
        std_str  = ", ".join([f"{s:.4f}" for s in self.std])
        return f"ChannelScaler(mean=[{mean_str}], std=[{std_str}])"


# =============================================================================
# 3) PyTorch Dataset
# =============================================================================
class ERA5Dataset(Dataset):
    """PyTorch Dataset wrapper for ERA5 multidimensional tensors."""
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# =============================================================================
# 4) VDSR Architecture
# =============================================================================
class VDSRResidualNet(nn.Module):
    """
    Deep Convolutional Network for VDSR (without Batch Normalization):
      Cascaded Conv3x3 + ReLU blocks (15-20 layers), ending with a single Conv3x3 layer.
    """
    def __init__(self, in_channels=5, out_channels=5, num_features=64, num_layers=18, zero_init_last=True):
        super().__init__()
        if num_layers < 3:
            raise ValueError("VDSR num_layers must be >= 3.")

        layers = [
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        # Intermediate layers
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
    VDSR Super-Resolution Pipeline:
      baseline = bicubic(x_std) -> Affine transform to y_std
      residual = residual_net(baseline)
      output   = baseline + residual
    """
    def __init__(self,
                 in_channels, out_channels,
                 num_features, num_layers, upscale_factor,
                 scaler_X_mean, scaler_X_std,
                 scaler_Y_mean, scaler_Y_std,
                 align_corners=False,
                 zero_init_last=True):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.align_corners = align_corners

        self.residual_net = VDSRResidualNet(
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=num_features,
            num_layers=num_layers,
            zero_init_last=zero_init_last
        )

        # Affine coefficients mapping X_std space to Y_std baseline space
        x_mean = np.asarray(scaler_X_mean, dtype=np.float32)
        x_std  = np.asarray(scaler_X_std, dtype=np.float32)
        y_mean = np.asarray(scaler_Y_mean, dtype=np.float32)
        y_std  = np.asarray(scaler_Y_std, dtype=np.float32)

        alpha = x_std / y_std
        beta  = (x_mean - y_mean) / y_std

        self.register_buffer("x2y_alpha", torch.tensor(alpha, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("x2y_beta",  torch.tensor(beta,  dtype=torch.float32).view(1, -1, 1, 1))

    def bicubic_baseline_ystd(self, x_std):
        # Scale (B, C, h, w) -> (B, C, H, W)
        base_x_std = F.interpolate(
            x_std,
            scale_factor=self.upscale_factor,
            mode="bicubic",
            align_corners=self.align_corners
        )
        # Linear transform into Y_std domain
        base_y_std = base_x_std * self.x2y_alpha + self.x2y_beta
        return base_y_std

    def forward(self, x_std):
        base = self.bicubic_baseline_ystd(x_std)
        residual = self.residual_net(base)
        out = base + residual
        return out

    @torch.no_grad()
    def forward_with_components(self, x_std):
        self.eval()
        base = self.bicubic_baseline_ystd(x_std)
        res  = self.residual_net(base)
        out  = base + res
        return out, base, res


# =============================================================================
# 5) Physical Constraint Loss
# =============================================================================
class PhysicsConstraintLoss(nn.Module):
    """
    Computes physical constraints after un-normalizing back into physical units:
      L = sum(w * |A_pred - A_target|) / (sum(w * |A_target|) + eps)

    Physical Quantities Evaluated:
      1) Divergence
      2) Vorticity
      3) Geostrophic balance (Ageostrophic residual, latitude-weighted)
      4) Moist static energy (MSE) horizontal gradient
    """
    def __init__(self,
                 scaler_Y_mean, scaler_Y_std,
                 latitudes, longitudes,
                 cp_air=1004.0, lv=2.501e6,
                 eps_div=1e-8, eps_vor=1e-8, eps_geo=1e-4, eps_mse=1e-4,
                 geo_weight_mode="sin_global",
                 geo_weight_alpha=1.0,
                 geo_weight_min=0.10):
        super().__init__()

        self.idx_z, self.idx_q, self.idx_t = 0, 1, 2
        self.idx_u, self.idx_v = 3, 4

        self.eps_div = eps_div
        self.eps_vor = eps_vor
        self.eps_geo = eps_geo
        self.eps_mse = eps_mse

        self.register_buffer("cp_air", torch.tensor(cp_air, dtype=torch.float32))
        self.register_buffer("lv", torch.tensor(lv, dtype=torch.float32))

        self.register_buffer("y_mean", torch.tensor(scaler_Y_mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("y_std",  torch.tensor(scaler_Y_std,  dtype=torch.float32).view(1, -1, 1, 1))

        R = 6371000.0
        omega = 7.2921e-5
        self.R_earth = R

        latitudes = latitudes.astype(np.float64)
        longitudes = longitudes.astype(np.float64)
        lat_rad = np.deg2rad(latitudes)
        lon_rad = np.deg2rad(longitudes)

        dlat = np.abs(np.mean(np.diff(lat_rad)))
        dlon = np.abs(np.mean(np.diff(lon_rad)))

        lat_decreasing = (latitudes[1] < latitudes[0])  # True if ERA5 latitude is North to South

        dy = R * dlat
        dx = R * np.cos(lat_rad) * dlon
        dx = np.maximum(dx, 100.0)

        inv_dx_arr = (1.0 / dx).reshape(1, 1, -1, 1)  # (1, 1, H, 1)
        self.register_buffer("inv_dx", torch.tensor(inv_dx_arr, dtype=torch.float32))
        self.register_buffer("inv_dy", torch.tensor(1.0 / dy, dtype=torch.float32))

        # Coriolis parameter f (clipped at low latitudes to prevent extreme values)
        f_raw = 2.0 * omega * np.sin(lat_rad)
        f_cor = f_raw.copy()
        f_min = 2.5e-5
        f_cor = np.where(np.abs(f_cor) < f_min, np.where(f_cor >= 0, f_min, -f_min), f_cor)
        inv_f_arr = (1.0 / f_cor).reshape(1, 1, -1, 1)
        self.register_buffer("inv_f", torch.tensor(inv_f_arr, dtype=torch.float32))

        # Latitude weighting for geostrophic loss
        if geo_weight_mode == "f_ratio":
            f_abs = np.abs(f_raw)
            denom = np.max(f_abs) + 1e-12
            geo_w = (f_abs / denom) ** float(geo_weight_alpha)
        elif geo_weight_mode == "sin_global":
            geo_w = (np.abs(np.sin(lat_rad))) ** float(geo_weight_alpha)
        elif geo_weight_mode == "cos":
            geo_w = (np.abs(np.cos(lat_rad))) ** float(geo_weight_alpha)
        elif geo_weight_mode == "one_minus_cos":
            geo_w = (1.0 - np.abs(np.cos(lat_rad))) ** float(geo_weight_alpha)
            mx = np.max(geo_w)
            if mx > 0:
                geo_w = geo_w / mx
        else:
            raise ValueError(f"Unknown geo_weight_mode: {geo_weight_mode}")

        geo_w = np.clip(geo_w, float(geo_weight_min), 1.0)
        self.register_buffer("geo_weight", torch.tensor(geo_w.reshape(1, -1, 1), dtype=torch.float32))

        # Spherical metric correction term: tan(phi) / R
        tan_lat = np.tan(lat_rad)
        tan_lat = np.clip(tan_lat, -50.0, 50.0)
        tan_over_R = (tan_lat / R).reshape(1, 1, -1, 1)
        self.register_buffer("tan_over_R", torch.tensor(tan_over_R, dtype=torch.float32))

        # Central difference finite-difference kernels
        ddx_k = torch.tensor([[0.0,  0.0, 0.0],
                              [-0.5, 0.0, 0.5],
                              [0.0,  0.0, 0.0]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer("ddx_kernel", ddx_k)

        if lat_decreasing:
            ddy_k = torch.tensor([[0.0,  0.5, 0.0],
                                  [0.0,  0.0, 0.0],
                                  [0.0, -0.5, 0.0]], dtype=torch.float32).reshape(1, 1, 3, 3)
        else:
            ddy_k = torch.tensor([[0.0, -0.5, 0.0],
                                  [0.0,  0.0, 0.0],
                                  [0.0,  0.5, 0.0]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer("ddy_kernel", ddy_k)

        lat_dir = "North -> South" if lat_decreasing else "South -> North"
        print(f"  [PhysicsLoss] Latitude orientation: {lat_dir}")
        print(f"  [PhysicsLoss] dy = {dy:.1f} m, dx range = [{dx.min():.1f}, {dx.max():.1f}] m")
        print(f"  [PhysicsLoss] geo_weight mode = {geo_weight_mode}, alpha = {geo_weight_alpha}, min = {geo_weight_min}")
        print(f"  [PhysicsLoss] geo_weight range = [{geo_w.min():.3f}, {geo_w.max():.3f}]")
        print(f"  [PhysicsLoss] cp = {cp_air:.1f}, Lv = {lv:.3e}")

    # ---------- Helper Functions ----------
    def inverse_std(self, data):
        return data * self.y_std + self.y_mean

    def _conv_fd(self, field_4d, kernel):
        padded = F.pad(field_4d, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(padded, kernel)

    def compute_ddx(self, field):
        raw = self._conv_fd(field.unsqueeze(1), self.ddx_kernel)
        return (raw * self.inv_dx).squeeze(1)

    def compute_ddy(self, field):
        raw = self._conv_fd(field.unsqueeze(1), self.ddy_kernel)
        return (raw * self.inv_dy).squeeze(1)

    # ---------- Physics Computations ----------
    def compute_divergence(self, u, v):
        du_dx = self.compute_ddx(u)
        dv_dy = self.compute_ddy(v)
        metric = v * self.tan_over_R.squeeze(0).squeeze(0)
        return du_dx + dv_dy - metric

    def compute_vorticity(self, u, v):
        dv_dx = self.compute_ddx(v)
        du_dy = self.compute_ddy(u)
        metric = u * self.tan_over_R.squeeze(0).squeeze(0)
        return dv_dx - du_dy + metric

    def compute_geostrophic_wind(self, z):
        dz_dy = self.compute_ddy(z)
        dz_dx = self.compute_ddx(z)
        inv_f = self.inv_f.squeeze(0).squeeze(0)  # (H, 1)
        ug = -inv_f * dz_dy
        vg =  inv_f * dz_dx
        return ug, vg

    def compute_mse(self, t, z, q):
        # MSE = cp*T + Phi + Lv*q ; z represents geopotential Phi
        return self.cp_air * t + z + self.lv * q

    def compute_mse_gradients(self, t, z, q):
        mse = self.compute_mse(t, z, q)
        mse_x = self.compute_ddx(mse)
        mse_y = self.compute_ddy(mse)
        return mse_x, mse_y

    # ---------- Global Relative Error (GRE) ----------
    def global_relative_error(self, pred_qty, target_qty, eps, weight=None):
        err = torch.abs(pred_qty - target_qty)
        ref = torch.abs(target_qty)
        if weight is not None:
            err = err * weight
            ref = ref * weight
        numer = err.sum()
        denom = ref.sum() + eps
        return numer / denom

    # ---------- Forward Pass ----------
    def forward(self, pred_std, target_std):
        pred = self.inverse_std(pred_std)
        target = self.inverse_std(target_std)

        z_p, q_p, t_p = pred[:, self.idx_z], pred[:, self.idx_q], pred[:, self.idx_t]
        u_p, v_p      = pred[:, self.idx_u], pred[:, self.idx_v]

        z_t, q_t, t_t = target[:, self.idx_z], target[:, self.idx_q], target[:, self.idx_t]
        u_t, v_t      = target[:, self.idx_u], target[:, self.idx_v]

        losses = {}

        # 1) Divergence
        div_p = self.compute_divergence(u_p, v_p)
        div_t = self.compute_divergence(u_t, v_t)
        losses["div"] = self.global_relative_error(div_p, div_t, self.eps_div)

        # 2) Vorticity
        vor_p = self.compute_vorticity(u_p, v_p)
        vor_t = self.compute_vorticity(u_t, v_t)
        losses["vor"] = self.global_relative_error(vor_p, vor_t, self.eps_vor)

        # 3) Geostrophic / Ageostrophic Residual
        ug_p, vg_p = self.compute_geostrophic_wind(z_p)
        ug_t, vg_t = self.compute_geostrophic_wind(z_t)

        ageo_u_p, ageo_v_p = u_p - ug_p, v_p - vg_p
        ageo_u_t, ageo_v_t = u_t - ug_t, v_t - vg_t

        w_geo = self.geo_weight  # (1, H, 1)
        loss_geo_u = self.global_relative_error(ageo_u_p, ageo_u_t, self.eps_geo, weight=w_geo)
        loss_geo_v = self.global_relative_error(ageo_v_p, ageo_v_t, self.eps_geo, weight=w_geo)
        losses["geo"] = 0.5 * (loss_geo_u + loss_geo_v)

        # 4) MSE Gradient
        mse_x_p, mse_y_p = self.compute_mse_gradients(t_p, z_p, q_p)
        mse_x_t, mse_y_t = self.compute_mse_gradients(t_t, z_t, q_t)
        loss_mse_x = self.global_relative_error(mse_x_p, mse_x_t, self.eps_mse)
        loss_mse_y = self.global_relative_error(mse_y_p, mse_y_t, self.eps_mse)
        losses["mse"] = 0.5 * (loss_mse_x + loss_mse_y)

        return losses


# =============================================================================
# 6) Combined Loss Function
# =============================================================================
class CombinedLoss(nn.Module):
    """Combined Objective Function: L1 + Physics Loss Terms."""
    def __init__(self, physics_module=None,
                 lam_l1=1.0, lam_div=0.0, lam_vor=0.0, lam_geo=0.0, lam_mse=0.0):
        super().__init__()
        self.l1_criterion = nn.L1Loss()
        self.physics = physics_module

        self.lam_l1  = float(lam_l1)
        self.lam_div = float(lam_div)
        self.lam_vor = float(lam_vor)
        self.lam_geo = float(lam_geo)
        self.lam_mse = float(lam_mse)

    def set_lambda_geo(self, new_lambda_geo: float):
        self.lam_geo = float(new_lambda_geo)

    def forward(self, pred, target):
        l1 = self.l1_criterion(pred, target)
        zero = l1.new_tensor(0.0)

        if self.physics is not None and (self.lam_div > 0 or self.lam_vor > 0 or self.lam_geo > 0 or self.lam_mse > 0):
            pl = self.physics(pred, target)
        else:
            pl = {"div": zero, "vor": zero, "geo": zero, "mse": zero}

        total = (self.lam_l1  * l1
               + self.lam_div * pl["div"]
               + self.lam_vor * pl["vor"]
               + self.lam_geo * pl["geo"]
               + self.lam_mse * pl["mse"])

        loss_dict = {
            "total": total.detach().item(),
            "l1":    l1.detach().item(),
            "div":   pl["div"].detach().item(),
            "vor":   pl["vor"].detach().item(),
            "geo":   pl["geo"].detach().item(),
            "mse":   pl["mse"].detach().item(),
        }
        return total, loss_dict


# =============================================================================
# 7) Evaluation Functions
# =============================================================================
def evaluate(model, dataloader, criterion, device, num_channels, scaler_Y):
    """
    Evaluates model performance over DataLoader.
    Returns:
      avg_total, avg_l1, avg_phys_dict, rmse_per_ch, overall_rmse
    Note: RMSE is calculated in physical units (unnormalized).
    """
    model.eval()
    sum_total = 0.0
    sum_l1 = 0.0
    sum_phys = {k: 0.0 for k in PHYSICS_KEYS}
    n_batches = 0

    total_se_std = np.zeros(num_channels, dtype=np.float64)
    total_pixels = np.zeros(num_channels, dtype=np.float64)

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            pred = model(X_batch)
            _, ld = criterion(pred, Y_batch)

            sum_total += ld["total"]
            sum_l1 += ld["l1"]
            for k in PHYSICS_KEYS:
                sum_phys[k] += ld[k]
            n_batches += 1

            for c in range(num_channels):
                diff = pred[:, c] - Y_batch[:, c]
                total_se_std[c] += (diff ** 2).sum().item()
                total_pixels[c] += diff.numel()

    nb = max(n_batches, 1)
    avg_total = sum_total / nb
    avg_l1 = sum_l1 / nb
    avg_phys = {k: sum_phys[k] / nb for k in PHYSICS_KEYS}

    std_Y = scaler_Y.std.astype(np.float64)
    rmse_std = np.sqrt(total_se_std / np.maximum(total_pixels, 1))
    rmse_per_ch = rmse_std * std_Y

    total_se_orig = np.sum(total_se_std * (std_Y ** 2))
    total_px_all = np.sum(total_pixels)
    overall_rmse = np.sqrt(total_se_orig / max(total_px_all, 1))

    return avg_total, avg_l1, avg_phys, rmse_per_ch, overall_rmse


def evaluate_bicubic_baseline(model, dataloader, device, num_channels, scaler_Y):
    """Evaluates pure bicubic baseline (without trained deep residual component)."""
    model.eval()
    l1_fn = nn.L1Loss()
    l1_sum = 0.0
    n_batches = 0

    total_se_std = np.zeros(num_channels, dtype=np.float64)
    total_pixels = np.zeros(num_channels, dtype=np.float64)

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            base_pred = model.bicubic_baseline_ystd(X_batch)
            l1_sum += l1_fn(base_pred, Y_batch).item()
            n_batches += 1

            for c in range(num_channels):
                diff = base_pred[:, c] - Y_batch[:, c]
                total_se_std[c] += (diff ** 2).sum().item()
                total_pixels[c] += diff.numel()

    avg_l1 = l1_sum / max(n_batches, 1)

    std_Y = scaler_Y.std.astype(np.float64)
    rmse_std = np.sqrt(total_se_std / np.maximum(total_pixels, 1))
    rmse_per_ch = rmse_std * std_Y

    total_se_orig = np.sum(total_se_std * (std_Y ** 2))
    total_px_all = np.sum(total_pixels)
    overall_rmse = np.sqrt(total_se_orig / max(total_px_all, 1))

    return avg_l1, rmse_per_ch, overall_rmse


# =============================================================================
# 8) Data Loading Utilities
# =============================================================================
def load_nc_data(x_path, y_path, variables, return_coords=False):
    """Load ERA5 predictor (X) and target (Y) datasets from NetCDF files."""
    print(f"  Loading X: {x_path}")
    ds_x = xr.open_dataset(x_path)
    X = np.stack([ds_x[v].values for v in variables], axis=1).astype(np.float32)
    ds_x.close()

    print(f"  Loading Y: {y_path}")
    ds_y = xr.open_dataset(y_path)
    Y = np.stack([ds_y[v].values for v in variables], axis=1).astype(np.float32)

    latitudes = ds_y["latitude"].values.astype(np.float64)
    longitudes = ds_y["longitude"].values.astype(np.float64)
    ds_y.close()

    assert X.shape[0] == Y.shape[0], f"Temporal mismatch between X ({X.shape[0]}) and Y ({Y.shape[0]})"
    print(f"       X shape: {X.shape} | Y shape: {Y.shape}")

    if return_coords:
        return X, Y, latitudes, longitudes
    return X, Y


def build_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size, num_workers, device):
    """Construct PyTorch DataLoaders for train, validation, and test sets."""
    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    return train_loader, valid_loader, test_loader


# =============================================================================
# 9) Visualization Utilities
# =============================================================================
def plot_losses(epoch, train_total, val_total, test_total,
                train_l1, val_l1, test_l1,
                val_phys_history, in_h, in_w, out_h, out_w,
                loss_plot_path, physics_plot_path,
                best_epoch=None):
    """Plot training curves for total, L1, and physical losses."""
    n = epoch + 1
    epochs_range = range(1, n + 1)

    # Plot 1: Total and L1 Loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(epochs_range, train_total[:n], label="Train Total", lw=1.5)
    ax1.plot(epochs_range, val_total[:n],   label="Valid Total", lw=1.5)
    ax1.plot(epochs_range, test_total[:n],  label="Test Total",  lw=1.5)
    if best_epoch is not None:
        ax1.axvline(x=best_epoch, color="red", ls="--", alpha=0.5, label=f"Best ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss (L1 + Physics)")
    ax1.set_title(f"Total Loss ({in_h}x{in_w} -> {out_h}x{out_w})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, train_l1[:n], label="Train L1", lw=1.5)
    ax2.plot(epochs_range, val_l1[:n],   label="Valid L1", lw=1.5)
    ax2.plot(epochs_range, test_l1[:n],  label="Test L1",  lw=1.5)
    if best_epoch is not None:
        ax2.axvline(x=best_epoch, color="red", ls="--", alpha=0.5, label=f"Best ({best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("L1 Loss")
    ax2.set_title("L1 Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()

    # Plot 2: Physics Losses (Validation)
    n_keys = len(PHYSICS_KEYS)
    ncols = 3 if n_keys <= 6 else 4
    nrows = math.ceil(n_keys / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
    axes = np.array(axes).reshape(-1)

    titles = {
        "div": "Divergence (comparison)",
        "vor": "Vorticity (comparison)",
        "geo": "Geostrophic balance (comparison, latitude-weighted)",
        "mse": "Moist static energy gradient (comparison)"
    }

    for i, k in enumerate(PHYSICS_KEYS):
        ax = axes[i]
        vals = val_phys_history[k][:n]
        ax.plot(epochs_range, vals, lw=1.5, color="tab:orange")
        if best_epoch is not None:
            ax.axvline(x=best_epoch, color="red", ls="--", alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Validation: {titles[k]}")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

    for j in range(n_keys, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(physics_plot_path, dpi=300)
    plt.close()


# =============================================================================
# 10) Single Experiment Runner
# =============================================================================
def train_one_experiment(exp_cfg,
                         train_dataset, valid_dataset, test_dataset,
                         scaler_X, scaler_Y, lat_hr, lon_hr,
                         input_h, input_w, output_h, output_w,
                         device):
    """Executes a single model training experiment pipeline."""
    exp_name = exp_cfg["name"]
    exp_dir = os.path.join(OUTPUT_ROOT, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    model_save_path   = os.path.join(exp_dir, "best_model.pth")
    loss_plot_path    = os.path.join(exp_dir, "loss_epoch.png")
    physics_plot_path = os.path.join(exp_dir, "physics_loss_epoch.png")
    history_csv_path  = os.path.join(exp_dir, "training_history.csv")
    model_info_path   = os.path.join(exp_dir, "model_info.json")

    seed_offset = int(exp_cfg.get("seed_offset", 0))
    set_seed(RANDOM_SEED + seed_offset)

    print("=" * 90)
    print(f"[EXPERIMENT START] {exp_name}")
    print("=" * 90)

    train_loader, valid_loader, test_loader = build_dataloaders(
        train_dataset, valid_dataset, test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device
    )

    # ---- Model Initialization ----
    model = VDSRSRModel(
        in_channels=NUM_CHANNELS,
        out_channels=NUM_CHANNELS,
        num_features=NUM_FEATURES,
        num_layers=NUM_VDSR_LAYERS,
        upscale_factor=UPSCALE_FACTOR,
        scaler_X_mean=scaler_X.mean,
        scaler_X_std=scaler_X.std,
        scaler_Y_mean=scaler_Y.mean,
        scaler_Y_std=scaler_Y.std,
        align_corners=BICUBIC_ALIGN_CORNERS,
        zero_init_last=ZERO_INIT_LAST
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model Architecture: VDSR")
    print(f"  Conv Layers          : {NUM_VDSR_LAYERS}")
    print(f"  Feature Channels     : {NUM_FEATURES}")
    print(f"  Upscale Factor       : {UPSCALE_FACTOR}x")
    print(f"  Bicubic Align Corners: {BICUBIC_ALIGN_CORNERS}")
    print(f"  Zero Init Last Conv  : {ZERO_INIT_LAST}")
    print(f"  Total Parameters     : {total_params:,}")
    print(f"  Trainable Parameters : {trainable_params:,}")

    # ---- Baseline Reference ----
    b_valid_l1, b_valid_rmse, b_valid_overall = evaluate_bicubic_baseline(
        model, valid_loader, device, NUM_CHANNELS, scaler_Y
    )
    b_test_l1, b_test_rmse, b_test_overall = evaluate_bicubic_baseline(
        model, test_loader, device, NUM_CHANNELS, scaler_Y
    )
    print(f"  [Valid Baseline] L1={b_valid_l1:.6f}, RMSE: {format_rmse_str(b_valid_rmse, VAR_NAMES, b_valid_overall)}")
    print(f"  [Test  Baseline] L1={b_test_l1:.6f}, RMSE: {format_rmse_str(b_test_rmse, VAR_NAMES, b_test_overall)}")

    # ---- Loss Setup ----
    physics_loss = None
    if exp_cfg.get("use_physics", False):
        print("Initializing PhysicsConstraintLoss...")
        physics_loss = PhysicsConstraintLoss(
            scaler_Y_mean=scaler_Y.mean,
            scaler_Y_std=scaler_Y.std,
            latitudes=lat_hr,
            longitudes=lon_hr,
            cp_air=CP_AIR,
            lv=LV_WATER,
            eps_div=EPS_DIV,
            eps_vor=EPS_VOR,
            eps_geo=EPS_GEO,
            eps_mse=EPS_MSE,
            geo_weight_mode=GEO_WEIGHT_MODE,
            geo_weight_alpha=GEO_WEIGHT_ALPHA,
            geo_weight_min=GEO_WEIGHT_MIN
        ).to(device)

    criterion = CombinedLoss(
        physics_module=physics_loss,
        lam_l1=exp_cfg["lam_l1"],
        lam_div=exp_cfg["lam_div"],
        lam_vor=exp_cfg["lam_vor"],
        lam_geo=exp_cfg["lam_geo"],
        lam_mse=exp_cfg["lam_mse"],
    ).to(device)

    print("Loss Weights:")
    print(f"  lambda_l1={criterion.lam_l1}, lambda_div={criterion.lam_div}, lambda_vor={criterion.lam_vor}, lambda_geo={criterion.lam_geo}, lambda_mse={criterion.lam_mse}")

    # Initial Batch Diagnostics
    model.eval()
    with torch.no_grad():
        X_sample, Y_sample = next(iter(train_loader))
        X_sample = X_sample.to(device)
        Y_sample = Y_sample.to(device)
        pred_sample = model(X_sample)
        _, diag_ld = criterion(pred_sample, Y_sample)

    print("Initial Batch Loss Diagnostics:")
    for k, v in diag_ld.items():
        print(f"  {k:>8s}: {v:.6e}")

    # Optional Auto-tune for lambda_geo
    if exp_cfg.get("use_physics", False) and AUTO_TUNE_LAMBDA_GEO and criterion.lam_geo > 0:
        raw_lam, tuned_lam = suggest_lambda_geo(
            diag_loss_dict=diag_ld,
            lam_l1=criterion.lam_l1,
            target_rel_to_l1=GEO_TARGET_REL_TO_L1,
            lam_min=GEO_LAMBDA_MIN,
            lam_max=GEO_LAMBDA_MAX
        )
        old = criterion.lam_geo
        criterion.set_lambda_geo(tuned_lam)
        print(f"[Auto lambda_geo] raw={raw_lam:.6e}, clip={tuned_lam:.6e}, {old:.6e}->{criterion.lam_geo:.6e}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )

    # ---- Training Loop ----
    best_val_loss = float("inf")
    history_records = []

    train_total_list, val_total_list, test_total_list = [], [], []
    train_l1_list, val_l1_list, test_l1_list = [], [], []
    val_phys_history = {k: [] for k in PHYSICS_KEYS}

    total_start = time.time()
    print(f"Training for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # Train Step
        model.train()
        train_loss_accum = {k: 0.0 for k in ["total", "l1"] + PHYSICS_KEYS}
        n_train_batches = 0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(X_batch)
            total_loss, ld = criterion(pred, Y_batch)
            total_loss.backward()
            optimizer.step()

            for k in train_loss_accum:
                train_loss_accum[k] += ld[k]
            n_train_batches += 1

        train_avg = {k: train_loss_accum[k] / max(n_train_batches, 1) for k in train_loss_accum}

        # Evaluation Step
        train_total, train_l1, train_phys, train_rmse, train_overall = evaluate(
            model, train_loader, criterion, device, NUM_CHANNELS, scaler_Y
        )
        val_total, val_l1, val_phys, val_rmse, val_overall = evaluate(
            model, valid_loader, criterion, device, NUM_CHANNELS, scaler_Y
        )
        test_total, test_l1, test_phys, test_rmse, test_overall = evaluate(
            model, test_loader, criterion, device, NUM_CHANNELS, scaler_Y
        )

        scheduler.step(val_total)
        new_lr = optimizer.param_groups[0]["lr"]

        model_status = "Not updated"
        if val_total <= best_val_loss:
            best_val_loss = val_total
            torch.save({
                "epoch": epoch + 1,
                "experiment_name": exp_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": float(best_val_loss),
                "model_type": "VDSRSRModel",
                "num_vdsr_layers": NUM_VDSR_LAYERS,
                "num_features": NUM_FEATURES,
                "upscale_factor": UPSCALE_FACTOR,
                "bicubic_align_corners": BICUBIC_ALIGN_CORNERS,
                "zero_init_last": ZERO_INIT_LAST,
                "scaler_X_mean": scaler_X.mean.tolist(),
                "scaler_X_std": scaler_X.std.tolist(),
                "scaler_Y_mean": scaler_Y.mean.tolist(),
                "scaler_Y_std": scaler_Y.std.tolist(),
                "latitudes": lat_hr.tolist(),
                "longitudes": lon_hr.tolist(),
                "physics_weights": {
                    "lam_l1": float(criterion.lam_l1),
                    "lam_div": float(criterion.lam_div),
                    "lam_vor": float(criterion.lam_vor),
                    "lam_geo": float(criterion.lam_geo),
                    "lam_mse": float(criterion.lam_mse),
                },
                "exp_cfg": exp_cfg
            }, model_save_path)
            model_status = "Saved (Best Valid)"

        epoch_time = time.time() - epoch_start
        remain = epoch_time * (NUM_EPOCHS - epoch - 1)

        train_total_list.append(train_total)
        val_total_list.append(val_total)
        test_total_list.append(test_total)

        train_l1_list.append(train_l1)
        val_l1_list.append(val_l1)
        test_l1_list.append(test_l1)

        for k in PHYSICS_KEYS:
            val_phys_history[k].append(val_phys[k])

        rec = {
            "Epoch": epoch + 1,
            "Train_Total": float(train_total),
            "Valid_Total": float(val_total),
            "Test_Total": float(test_total),
            "Train_L1": float(train_l1),
            "Valid_L1": float(val_l1),
            "Test_L1": float(test_l1),
            "Lambda_GEO": float(criterion.lam_geo),
            "LR": float(current_lr)
        }
        for k in PHYSICS_KEYS:
            rec[f"Train_P_{k}"] = float(train_phys[k])
            rec[f"Valid_P_{k}"] = float(val_phys[k])
            rec[f"Test_P_{k}"]  = float(test_phys[k])

        for i, name in enumerate(VAR_NAMES):
            rec[f"Train_RMSE_{name}"] = float(train_rmse[i])
            rec[f"Valid_RMSE_{name}"] = float(val_rmse[i])
            rec[f"Test_RMSE_{name}"]  = float(test_rmse[i])

        rec["Train_RMSE_Overall"] = float(train_overall)
        rec["Valid_RMSE_Overall"] = float(val_overall)
        rec["Test_RMSE_Overall"]  = float(test_overall)
        history_records.append(rec)

        print("-" * 90)
        print(
            f"[{exp_name}] Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Total train={train_total:.4f} valid={val_total:.4f} test={test_total:.4f} | "
            f"L1 train={train_l1:.4f} valid={val_l1:.4f} test={test_l1:.4f}"
        )
        print(f"  Model state: {model_status} | Time: {epoch_time:.1f}s | Est. Remaining: {remain:.0f}s")
        print(f"  [Valid Physics] {format_physics_str(val_phys)}")
        print(f"  [Valid RMSE] {format_rmse_str(val_rmse, VAR_NAMES, val_overall)}")
        print(f"  [Test  RMSE] {format_rmse_str(test_rmse, VAR_NAMES, test_overall)}")
        if new_lr != current_lr:
            print(f"  [LR Adjusted] {current_lr:.6f} -> {new_lr:.6f}")

        if (epoch + 1) % 5 == 0:
            plot_losses(
                epoch=epoch,
                train_total=train_total_list, val_total=val_total_list, test_total=test_total_list,
                train_l1=train_l1_list, val_l1=val_l1_list, test_l1=test_l1_list,
                val_phys_history=val_phys_history,
                in_h=input_h, in_w=input_w, out_h=output_h, out_w=output_w,
                loss_plot_path=loss_plot_path, physics_plot_path=physics_plot_path
            )

    # ---- Experiment Completion ----
    best_idx = int(np.argmin([r["Valid_Total"] for r in history_records]))
    best_epoch_num = int(history_records[best_idx]["Epoch"])
    best_rec = history_records[best_idx]

    plot_losses(
        epoch=NUM_EPOCHS - 1,
        train_total=train_total_list, val_total=val_total_list, test_total=test_total_list,
        train_l1=train_l1_list, val_l1=val_l1_list, test_l1=test_l1_list,
        val_phys_history=val_phys_history,
        in_h=input_h, in_w=input_w, out_h=output_h, out_w=output_w,
        loss_plot_path=loss_plot_path, physics_plot_path=physics_plot_path,
        best_epoch=best_epoch_num
    )

    df_hist = pd.DataFrame(history_records)
    df_hist.to_csv(history_csv_path, index=False)

    total_time = time.time() - total_start
    model_info = {
        "experiment_name": exp_name,
        "architecture": "VDSR (Bicubic baseline + Deep CNN residual + Global residual skip)",
        "num_vdsr_layers": NUM_VDSR_LAYERS,
        "num_features": NUM_FEATURES,
        "input_channels": NUM_CHANNELS,
        "output_channels": NUM_CHANNELS,
        "variables": VARIABLES,
        "variable_units": VAR_UNITS,
        "upscale_factor": UPSCALE_FACTOR,
        "bicubic_align_corners": BICUBIC_ALIGN_CORNERS,
        "zero_init_last": ZERO_INIT_LAST,
        "physics_enabled": bool(exp_cfg.get("use_physics", False)),
        "physics_weights": {
            "lam_l1": float(criterion.lam_l1),
            "lam_div": float(criterion.lam_div),
            "lam_vor": float(criterion.lam_vor),
            "lam_geo": float(criterion.lam_geo),
            "lam_mse": float(criterion.lam_mse),
        },
        "eps_values": {
            "eps_div": EPS_DIV, "eps_vor": EPS_VOR, "eps_geo": EPS_GEO, "eps_mse": EPS_MSE
        },
        "geo_weighting": {
            "mode": GEO_WEIGHT_MODE, "alpha": GEO_WEIGHT_ALPHA, "min": GEO_WEIGHT_MIN
        },
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "random_seed": RANDOM_SEED + seed_offset,
        "best_epoch": best_epoch_num,
        "best_valid_total": float(best_val_loss),
        "best_valid_l1": float(best_rec["Valid_L1"]),
        "best_test_total": float(best_rec["Test_Total"]),
        "best_test_l1": float(best_rec["Test_L1"]),
        "best_test_rmse_overall": float(best_rec["Test_RMSE_Overall"]),
        "bicubic_baseline_valid_l1": float(b_valid_l1),
        "bicubic_baseline_test_l1": float(b_test_l1),
        "bicubic_baseline_valid_rmse_overall": float(b_valid_overall),
        "bicubic_baseline_test_rmse_overall": float(b_test_overall),
        "training_time_sec": round(total_time, 1),
        "device": str(device),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False)

    print("=" * 90)
    print(f"[EXPERIMENT COMPLETED] {exp_name}")
    print(f"  Best Epoch: {best_epoch_num}")
    print(f"  Best Valid Total Loss: {best_val_loss:.6f}")
    print(f"  Best Test Overall RMSE: {best_rec['Test_RMSE_Overall']:.6f}")
    print(f"  Output Directory: {exp_dir}")
    print("=" * 90)

    # Cleanup
    del model, optimizer, scheduler, criterion, physics_loss
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "experiment": exp_name,
        "physics_enabled": bool(exp_cfg.get("use_physics", False)),
        "best_epoch": best_epoch_num,
        "best_valid_total": float(best_val_loss),
        "best_valid_l1": float(best_rec["Valid_L1"]),
        "best_test_total": float(best_rec["Test_Total"]),
        "best_test_l1": float(best_rec["Test_L1"]),
        "best_test_rmse_overall": float(best_rec["Test_RMSE_Overall"]),
        "baseline_test_l1": float(b_test_l1),
        "baseline_test_rmse_overall": float(b_test_overall),
    }


# =============================================================================
# 11) Main Entry Point
# =============================================================================
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 90)
    print("ERA5 Super-Resolution Training: VDSR + Physics Constraint Ablation")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print("=" * 90)

    # 1) Missing Value Checks
    if CHECK_MISSING:
        all_files = [X_TRAIN_PATH, Y_TRAIN_PATH, X_VALID_PATH, Y_VALID_PATH, X_TEST_PATH, Y_TEST_PATH]
        check_missing_values(all_files, VARIABLES)

    # 2) Load Datasets
    print("Loading datasets...")
    print("-" * 90)
    X_train_raw, Y_train_raw, lat_hr, lon_hr = load_nc_data(
        X_TRAIN_PATH, Y_TRAIN_PATH, VARIABLES, return_coords=True
    )
    X_valid_raw, Y_valid_raw = load_nc_data(X_VALID_PATH, Y_VALID_PATH, VARIABLES)
    X_test_raw, Y_test_raw   = load_nc_data(X_TEST_PATH, Y_TEST_PATH, VARIABLES)
    print("-" * 90)

    # Dimension and Upscaling Consistency Check
    input_h, input_w = X_train_raw.shape[2], X_train_raw.shape[3]
    output_h, output_w = Y_train_raw.shape[2], Y_train_raw.shape[3]

    if output_h % input_h != 0 or output_w % input_w != 0:
        raise ValueError(f"Dimensions not divisible: input={input_h}x{input_w}, output={output_h}x{output_w}")

    scale_h = output_h // input_h
    scale_w = output_w // input_w
    if scale_h != scale_w:
        raise ValueError(f"H/W scaling ratio mismatch: scale_h={scale_h}, scale_w={scale_w}")

    actual_scale = scale_h
    if UPSCALE_FACTOR != actual_scale:
        raise ValueError(
            f"UPSCALE_FACTOR={UPSCALE_FACTOR} does not match dataset resolution ratio ({actual_scale}x). "
            f"Current dimensions: {input_h}x{input_w} -> {output_h}x{output_w}"
        )

    print(f"Low-Resolution Input  : {input_h}x{input_w}")
    print(f"High-Resolution Output: {output_h}x{output_w}")
    print(f"Upscaling Factor      : {actual_scale}x")
    print(f"Channels ({NUM_CHANNELS})     : {VARIABLES}")
    print(f"Train samples: {X_train_raw.shape[0]:,}, Valid samples: {X_valid_raw.shape[0]:,}, Test samples: {X_test_raw.shape[0]:,}")
    print(f"Latitude Range        : [{lat_hr.min():.2f}, {lat_hr.max():.2f}]")
    print(f"Longitude Range       : [{lon_hr.min():.2f}, {lon_hr.max():.2f}]")
    print("=" * 90)

    # 3) Data Normalization
    print("Normalizing features (Independent fit for X and Y)...")
    scaler_X = ChannelScaler()
    scaler_Y = ChannelScaler()

    X_train_std = scaler_X.fit_transform(X_train_raw)
    X_valid_std = scaler_X.transform(X_valid_raw)
    X_test_std  = scaler_X.transform(X_test_raw)

    Y_train_std = scaler_Y.fit_transform(Y_train_raw)
    Y_valid_std = scaler_Y.transform(Y_valid_raw)
    Y_test_std  = scaler_Y.transform(Y_test_raw)

    print(f"X scaler: {scaler_X}")
    print(f"Y scaler: {scaler_Y}")
    for i, (n, u) in enumerate(zip(VAR_NAMES, VAR_UNITS)):
        print(f"  Y-{n} std={scaler_Y.std[i]:.6f} ({u})")

    joblib.dump({"scaler_X": scaler_X, "scaler_Y": scaler_Y}, SCALER_SAVE_PATH)
    print(f"Scalers saved to: {SCALER_SAVE_PATH}")
    print("=" * 90)

    # 4) PyTorch Dataset Preparation
    train_dataset = ERA5Dataset(X_train_std, Y_train_std)
    valid_dataset = ERA5Dataset(X_valid_std, Y_valid_std)
    test_dataset  = ERA5Dataset(X_test_std,  Y_test_std)

    # Free raw numpy arrays to release RAM
    del X_train_raw, Y_train_raw, X_valid_raw, Y_valid_raw, X_test_raw, Y_test_raw
    del X_train_std, Y_train_std, X_valid_std, Y_valid_std, X_test_std, Y_test_std
    gc.collect()

    # 5) Filter Experiments to Run
    if RUN_ONLY_EXPERIMENTS is None:
        exp_list = EXPERIMENTS
    else:
        wanted = set(RUN_ONLY_EXPERIMENTS)
        exp_list = [e for e in EXPERIMENTS if e["name"] in wanted]

    if len(exp_list) == 0:
        raise ValueError("No matching experiments found in RUN_ONLY_EXPERIMENTS.")

    print("Experiments to execute:")
    for e in exp_list:
        print(f"  - {e['name']} (physics={e['use_physics']})")
    print("=" * 90)

    # 6) Execute Ablation Suite
    summaries = []
    all_start = time.time()

    for i, e in enumerate(exp_list):
        cfg = dict(e)
        cfg["seed_offset"] = i * 1000
        summary = train_one_experiment(
            exp_cfg=cfg,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            scaler_X=scaler_X,
            scaler_Y=scaler_Y,
            lat_hr=lat_hr,
            lon_hr=lon_hr,
            input_h=input_h,
            input_w=input_w,
            output_h=output_h,
            output_w=output_w,
            device=device
        )
        summaries.append(summary)

    # 7) Save Global Summary
    df = pd.DataFrame(summaries)
    df = df.sort_values("best_test_rmse_overall", ascending=True).reset_index(drop=True)
    df.to_csv(SUMMARY_CSV_PATH, index=False)

    total_time = time.time() - all_start
    print("=" * 90)
    print("All experiments completed successfully.")
    print(f"Total Elapsed Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Ablation Summary Saved: {SUMMARY_CSV_PATH}")
    print("-" * 90)
    print(df)
    print("=" * 90)


if __name__ == "__main__":
    main()