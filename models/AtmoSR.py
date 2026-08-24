#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA5 Physics-Informed Super-Resolution Model Training Pipeline
==============================================================

Architecture:
  Input X_std (50x50)
    -> Bicubic upsampling to (100x100) in X_std space
    -> Affine mapping to Y_std space as baseline
    -> CNN residual branch outputs residual in Y_std space
        Residual branch: PixelShuffle -> 3x3 Conv sub-pixel fusion -> 3x3 Conv output
    -> Final output = Baseline + Residual (100x100 in Y_std space)

Loss Formulation:
  L1 Loss + Physics-Informed Constraints (Global Relative Error, GRE):
    1) Divergence
    2) Vorticity
    3) Geostrophic Balance (Ageostrophic wind, latitude-weighted)
    4) Moist Static Energy Gradient (x/y)

Notes:
  - Batch Normalization is excluded.
  - Early stopping, data augmentation, and gradient clipping are excluded.
"""

import os
import json
import time
import gc
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

# ============================================================================
# Hyperparameters and Configuration
# ============================================================================
NUM_EPOCHS         = 200
LEARNING_RATE      = 1e-4
BATCH_SIZE         = 32
DROPOUT_RATE       = 0.1
RANDOM_SEED        = 42
WEIGHT_DECAY       = 1e-4
SCHEDULER_FACTOR   = 0.1
SCHEDULER_PATIENCE = 3
NUM_FEATURES       = 64
NUM_RES_BLOCKS     = 8
UPSCALE_FACTOR     = 2
CHECK_MISSING      = False
NUM_WORKERS        = 2

# ---------- Bicubic + Residual Correction ----------
BICUBIC_ALIGN_CORNERS = False
RESIDUAL_SCALE        = 1.0
ZERO_INIT_LAST        = True  # Initializes last conv weights to zero so model starts at pure bicubic baseline

# ---------- Physics Constraint Loss Weights ----------
LAMBDA_L1          = 1.0
LAMBDA_DIV         = 0.1
LAMBDA_VOR         = 0.1
LAMBDA_GEO         = 0.1
LAMBDA_MSE         = 0.1

# ---------- Automatic Calibration for Lambda Geo ----------
AUTO_TUNE_LAMBDA_GEO = False
# Target: lambda_geo * L_geo ≈ GEO_TARGET_REL_TO_L1 * (lambda_l1 * L1)
GEO_TARGET_REL_TO_L1 = 0.10
GEO_LAMBDA_MIN       = 1e-3
GEO_LAMBDA_MAX       = 2.0

# ---------- Thermodynamic Constants ----------
CP_AIR             = 1004.0      # Specific heat capacity of air at constant pressure [J/(kg*K)]
LV_WATER           = 2.501e6     # Latent heat of vaporization for water [J/kg]

# ---------- Global Relative Error Epsilon (Eps) ----------
EPS_DIV   = 1e-8
EPS_VOR   = 1e-8
EPS_GEO   = 1e-4
EPS_MSE   = 1e-4

# ---------- Geostrophic Constraint Latitude Weighting ----------
GEO_WEIGHT_MODE  = 'sin_global'   # Options: ['f_ratio', 'sin_global', 'cos', 'one_minus_cos']
GEO_WEIGHT_ALPHA = 1.0
GEO_WEIGHT_MIN   = 0.10

# ---------- File Paths ----------
X_TRAIN_PATH = 'X2_train.nc'
Y_TRAIN_PATH = 'Y_train.nc'
X_VALID_PATH = 'X2_valid.nc'
Y_VALID_PATH = 'Y_valid.nc'
X_TEST_PATH  = 'X2_test.nc'
Y_TEST_PATH  = 'Y_test.nc'

VARIABLES    = ['z', 'q', 't', 'u', 'v']
VAR_NAMES    = ['Z', 'Q', 'T', 'U', 'V']
VAR_UNITS    = ['m²/s²', 'kg/kg', 'K', 'm/s', 'm/s']
NUM_CHANNELS = len(VARIABLES)

MODEL_SAVE_PATH    = 'best_model.pth'
LOSS_PLOT_PATH     = 'loss_epoch.png'
PHYSICS_PLOT_PATH  = 'physics_loss_epoch.png'
HISTORY_CSV_PATH   = 'training_history.csv'
MODEL_INFO_PATH    = 'model_info.json'
SCALER_SAVE_PATH   = 'scalers.joblib'

PHYSICS_KEYS = ['div', 'vor', 'geo', 'mse']


# ============================================================================
# Seed Initialization
# ============================================================================
def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('=' * 90)
print('ERA5 Super-Resolution Training Pipeline (Bicubic Baseline + Residual Correction + Physics)')
print(f'Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'Device: {device}')
print('=' * 90)


# ============================================================================
# 1. Missing Value Verification
# ============================================================================
def check_missing_values(file_paths, variables, missing_value=-999):
    """Verify if NetCDF datasets contain missing values (-999 or NaNs)."""
    print('Checking NetCDF datasets for missing values (-999) or NaNs...')
    print('-' * 90)
    found_missing = False
    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f'File not found: {path}')
        ds = xr.open_dataset(path)
        file_has_issue = False
        for var in variables:
            data = ds[var].values
            miss_count = int(np.sum(np.isclose(data, missing_value, atol=0.1)))
            nan_count  = int(np.sum(np.isnan(data)))
            if miss_count > 0:
                print(f'  [WARNING] {path} | Variable {var}: Found {miss_count:,} missing values (-999)')
                found_missing = True
                file_has_issue = True
            if nan_count > 0:
                print(f'  [WARNING] {path} | Variable {var}: Found {nan_count:,} NaN values')
                found_missing = True
                file_has_issue = True
        if not file_has_issue:
            print(f'  [PASSED] {path} - No missing values')
        ds.close()

    print('-' * 90)
    if found_missing:
        raise ValueError('Datasets contain missing values (-999 or NaN). Please clean the data before proceeding.')
    else:
        print('  [SUCCESS] All dataset checks passed successfully ✓')
    print('=' * 90)


# ============================================================================
# 2. Channel-wise Scaler (Z-score Normalization)
# ============================================================================
class ChannelScaler:
    """Per-channel Z-score standardization."""
    def __init__(self):
        self.mean = None
        self.std  = None
        self.n_channels = None

    def fit(self, data):
        # data shape: (N, C, H, W)
        self.n_channels = data.shape[1]
        self.mean = data.mean(axis=(0, 2, 3)).astype(np.float64)
        self.std  = data.std(axis=(0, 2, 3)).astype(np.float64)
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
        if self.mean is not None:
            mean_str = ', '.join([f'{m:.4f}' for m in self.mean])
            std_str  = ', '.join([f'{s:.4f}' for s in self.std])
            return f'ChannelScaler(mean=[{mean_str}], std=[{std_str}])'
        return 'ChannelScaler(unfitted)'


# ============================================================================
# 3. Dataset Wrapper
# ============================================================================
class ERA5Dataset(Dataset):
    """PyTorch Dataset wrapper for standardized ERA5 tensors."""
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)  # float32
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ============================================================================
# 4. Neural Network Architecture
# ============================================================================
class ResidualBlock(nn.Module):
    """Standard Residual Block without Batch Normalization."""
    def __init__(self, channels, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        return out + identity


class ResidualCorrectionBackbone(nn.Module):
    """
    CNN backbone predicting only the residual correction term (in Y_std space):
      x (50x50) -> Shallow Feature -> Residual Blocks -> Transition -> Skip Connection
                -> PixelShuffle (2x) -> 3x3 Fusion Conv -> 3x3 Output Conv
                -> Residual (100x100)
    """
    def __init__(self, in_channels=5, out_channels=5, num_features=64,
                 num_res_blocks=8, upscale_factor=2, dropout_rate=0.0,
                 residual_scale=1.0, zero_init_last=True):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.zero_init_last = bool(zero_init_last)

        self.shallow_feature = nn.Conv2d(in_channels, num_features, kernel_size=9, padding=4)

        self.deep_features = nn.Sequential(
            *[ResidualBlock(num_features, dropout_rate) for _ in range(num_res_blocks)]
        )
        self.transition = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )

        # Post-PixelShuffle sub-pixel fusion convolution
        self.post_shuffle_fuse = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.post_shuffle_act  = nn.ReLU(inplace=True)

        # Output mapping layer
        self.output_conv = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        # Kaiming Normal Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Zero-initialize last conv layer so initial output approximates pure bicubic baseline
        if self.zero_init_last:
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
    """
    Super-Resolution Model combining Bicubic Baseline and Neural Residual Correction:
      final_y_std = affine_map(bicubic(x_std)) + residual_y_std
    """
    def __init__(self,
                 in_channels, out_channels,
                 num_features, num_res_blocks, upscale_factor,
                 dropout_rate,
                 scaler_X_mean, scaler_X_std,
                 scaler_Y_mean, scaler_Y_std,
                 align_corners=False,
                 residual_scale=1.0,
                 zero_init_last=True):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.align_corners = align_corners

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

        # Affine mapping coefficients from X_std space to Y_std space
        x_mean = np.asarray(scaler_X_mean, dtype=np.float32)
        x_std  = np.asarray(scaler_X_std, dtype=np.float32)
        y_mean = np.asarray(scaler_Y_mean, dtype=np.float32)
        y_std  = np.asarray(scaler_Y_std, dtype=np.float32)

        alpha = x_std / y_std
        beta  = (x_mean - y_mean) / y_std

        self.register_buffer('x2y_alpha', torch.tensor(alpha, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer('x2y_beta',  torch.tensor(beta,  dtype=torch.float32).view(1, -1, 1, 1))

    def bicubic_baseline_ystd(self, x_std):
        base_x_std = F.interpolate(
            x_std,
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=self.align_corners
        )
        base_y_std = base_x_std * self.x2y_alpha + self.x2y_beta
        return base_y_std

    def forward(self, x_std):
        base_y_std = self.bicubic_baseline_ystd(x_std)
        residual_y_std = self.backbone(x_std)
        final_y_std = base_y_std + residual_y_std
        return final_y_std

    @torch.no_grad()
    def forward_with_components(self, x_std):
        self.eval()
        base = self.bicubic_baseline_ystd(x_std)
        res  = self.backbone(x_std)
        out  = base + res
        return out, base, res


# ============================================================================
# 5. Physics-Informed Constraint Losses
# ============================================================================
class PhysicsConstraintLoss(nn.Module):
    """
    Calculates physics-informed constraints in unnormalized physical space.
    Spatial differentiation: Central difference with replicate padding and spherical metric terms.

    Comparative Constraint Loss (Global Relative Error, GRE):
      L = Σ w|A_pred - A_target| / (Σ w|A_target| + ε)

    Evaluated Terms:
      1) Divergence
      2) Vorticity
      3) Geostrophic Balance (Ageostrophic wind components, latitude-weighted)
      4) Moist Static Energy (MSE) Gradient (x/y)
    """
    def __init__(self,
                 scaler_Y_mean, scaler_Y_std,
                 latitudes, longitudes,
                 cp_air=1004.0, lv=2.501e6,
                 eps_div=1e-8, eps_vor=1e-8, eps_geo=1e-4, eps_mse=1e-4,
                 geo_weight_mode='sin_global',
                 geo_weight_alpha=1.0,
                 geo_weight_min=0.10):
        super().__init__()

        # Variable indexing: z=0, q=1, t=2, u=3, v=4
        self.idx_z, self.idx_q, self.idx_t = 0, 1, 2
        self.idx_u, self.idx_v             = 3, 4

        # Epsilon values
        self.eps_div  = eps_div
        self.eps_vor  = eps_vor
        self.eps_geo  = eps_geo
        self.eps_mse  = eps_mse

        # Thermodynamic constants
        self.register_buffer('cp_air', torch.tensor(cp_air, dtype=torch.float32))
        self.register_buffer('lv',     torch.tensor(lv, dtype=torch.float32))

        # Standardization parameters
        self.register_buffer('y_mean', torch.tensor(scaler_Y_mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer('y_std',  torch.tensor(scaler_Y_std,  dtype=torch.float32).view(1, -1, 1, 1))

        # Physical constants
        R     = 6371000.0
        omega = 7.2921e-5
        self.R_earth = R

        # Spatial grid coordinates
        latitudes = latitudes.astype(np.float64)
        longitudes = longitudes.astype(np.float64)
        lat_rad = np.deg2rad(latitudes)
        lon_rad = np.deg2rad(longitudes)

        dlat = np.abs(np.mean(np.diff(lat_rad)))
        dlon = np.abs(np.mean(np.diff(lon_rad)))

        lat_decreasing = (latitudes[1] < latitudes[0])  # ERA5 latitude ordering (North to South)

        dy = R * dlat
        dx = R * np.cos(lat_rad) * dlon
        dx = np.maximum(dx, 100.0)

        inv_dx_arr = (1.0 / dx).reshape(1, 1, -1, 1)   # Shape: (1, 1, H, 1)
        self.register_buffer('inv_dx', torch.tensor(inv_dx_arr, dtype=torch.float32))
        self.register_buffer('inv_dy', torch.tensor(1.0 / dy, dtype=torch.float32))

        # Coriolis parameter (clipped near equator to prevent singularity in geostrophic wind calculation)
        f_raw = 2.0 * omega * np.sin(lat_rad)
        f_cor = f_raw.copy()
        f_min = 2.5e-5
        f_cor = np.where(np.abs(f_cor) < f_min, np.where(f_cor >= 0, f_min, -f_min), f_cor)
        inv_f_arr = (1.0 / f_cor).reshape(1, 1, -1, 1)
        self.register_buffer('inv_f', torch.tensor(inv_f_arr, dtype=torch.float32))

        # Latitude weighting for geostrophic constraint
        if geo_weight_mode == 'f_ratio':
            f_abs = np.abs(f_raw)
            denom = np.max(f_abs) + 1e-12
            geo_w = (f_abs / denom) ** float(geo_weight_alpha)
        elif geo_weight_mode == 'sin_global':
            geo_w = (np.abs(np.sin(lat_rad))) ** float(geo_weight_alpha)
        elif geo_weight_mode == 'cos':
            geo_w = (np.abs(np.cos(lat_rad))) ** float(geo_weight_alpha)
        elif geo_weight_mode == 'one_minus_cos':
            geo_w = (1.0 - np.abs(np.cos(lat_rad))) ** float(geo_weight_alpha)
            mx = np.max(geo_w)
            if mx > 0:
                geo_w = geo_w / mx
        else:
            raise ValueError(f'Unknown geo_weight_mode: {geo_weight_mode}')

        geo_w = np.clip(geo_w, float(geo_weight_min), 1.0)
        self.register_buffer('geo_weight', torch.tensor(geo_w.reshape(1, -1, 1), dtype=torch.float32))

        # Spherical metric correction term: tan(phi)/R
        tan_lat = np.tan(lat_rad)
        tan_lat = np.clip(tan_lat, -50.0, 50.0)
        tan_over_R = (tan_lat / R).reshape(1, 1, -1, 1)
        self.register_buffer('tan_over_R', torch.tensor(tan_over_R, dtype=torch.float32))

        # Finite difference convolution kernels
        ddx_k = torch.tensor([[0.0,  0.0, 0.0],
                              [-0.5, 0.0, 0.5],
                              [0.0,  0.0, 0.0]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer('ddx_kernel', ddx_k)

        if lat_decreasing:
            ddy_k = torch.tensor([[0.0,  0.5, 0.0],
                                  [0.0,  0.0, 0.0],
                                  [0.0, -0.5, 0.0]], dtype=torch.float32).reshape(1, 1, 3, 3)
        else:
            ddy_k = torch.tensor([[0.0, -0.5, 0.0],
                                  [0.0,  0.0, 0.0],
                                  [0.0,  0.5, 0.0]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer('ddy_kernel', ddy_k)

        # Logging configuration
        lat_dir = 'North -> South' if lat_decreasing else 'South -> North'
        print(f'  [PhysicsLoss] Latitude direction: {lat_dir}')
        print(f'  [PhysicsLoss] dy = {dy:.1f} m, dx range = [{dx.min():.1f}, {dx.max():.1f}] m')
        print(f'  [PhysicsLoss] |f_raw| range = [{np.abs(f_raw).min():.2e}, {np.abs(f_raw).max():.2e}] s^-1')
        if geo_weight_mode == 'sin_global':
            print('  [PhysicsLoss] geo_weight = clip(|sin(phi)|^alpha, w_min, 1) (domain-agnostic)')
        print(f'  [PhysicsLoss] geo_weight mode={geo_weight_mode}, alpha={geo_weight_alpha}, min={geo_weight_min}')
        print(f'  [PhysicsLoss] geo_weight range = [{geo_w.min():.3f}, {geo_w.max():.3f}]')
        print(f'  [PhysicsLoss] cp={cp_air:.1f} J/(kg*K), Lv={lv:.3e} J/kg')

    # -------------------------
    # Helper Operations
    # -------------------------
    def inverse_std(self, data):
        return data * self.y_std + self.y_mean

    def _conv_fd(self, field_4d, kernel):
        padded = F.pad(field_4d, (1, 1, 1, 1), mode='replicate')
        return F.conv2d(padded, kernel)

    def compute_ddx(self, field):
        raw = self._conv_fd(field.unsqueeze(1), self.ddx_kernel)
        return (raw * self.inv_dx).squeeze(1)

    def compute_ddy(self, field):
        raw = self._conv_fd(field.unsqueeze(1), self.ddy_kernel)
        return (raw * self.inv_dy).squeeze(1)

    # -------------------------
    # Physical Field Derivations
    # -------------------------
    def compute_divergence(self, u, v):
        du_dx  = self.compute_ddx(u)
        dv_dy  = self.compute_ddy(v)
        metric = v * self.tan_over_R.squeeze(0).squeeze(0)
        return du_dx + dv_dy - metric

    def compute_vorticity(self, u, v):
        dv_dx  = self.compute_ddx(v)
        du_dy  = self.compute_ddy(u)
        metric = u * self.tan_over_R.squeeze(0).squeeze(0)
        return dv_dx - du_dy + metric

    def compute_geostrophic_wind(self, z):
        dz_dy = self.compute_ddy(z)
        dz_dx = self.compute_ddx(z)
        inv_f = self.inv_f.squeeze(0).squeeze(0)
        ug = -inv_f * dz_dy
        vg =  inv_f * dz_dx
        return ug, vg

    def compute_mse(self, t, z, q):
        # MSE = cp*T + Φ + Lv*q ; z represents geopotential Φ
        return self.cp_air * t + z + self.lv * q

    def compute_mse_gradients(self, t, z, q):
        mse = self.compute_mse(t, z, q)
        mse_x = self.compute_ddx(mse)
        mse_y = self.compute_ddy(mse)
        return mse_x, mse_y

    # -------------------------
    # Loss Formulation
    # -------------------------
    def global_relative_error(self, pred_qty, target_qty, eps, weight=None):
        err = torch.abs(pred_qty - target_qty)
        ref = torch.abs(target_qty)
        if weight is not None:
            err = err * weight
            ref = ref * weight
        numer = err.sum()
        denom = ref.sum() + eps
        return numer / denom

    def forward(self, pred_std, target_std):
        """
        Input: Standardized Y space tensors (B, C, H, W)
        Internal: Unstandardizes tensors to compute physical metrics
        Returns: Dict containing individual physics losses
        """
        pred   = self.inverse_std(pred_std)
        target = self.inverse_std(target_std)

        z_p, q_p, t_p = pred[:, self.idx_z], pred[:, self.idx_q], pred[:, self.idx_t]
        u_p, v_p      = pred[:, self.idx_u], pred[:, self.idx_v]

        z_t, q_t, t_t = target[:, self.idx_z], target[:, self.idx_q], target[:, self.idx_t]
        u_t, v_t      = target[:, self.idx_u], target[:, self.idx_v]

        losses = {}

        # 1) Divergence
        div_p = self.compute_divergence(u_p, v_p)
        div_t = self.compute_divergence(u_t, v_t)
        losses['div'] = self.global_relative_error(div_p, div_t, self.eps_div)

        # 2) Vorticity
        vor_p = self.compute_vorticity(u_p, v_p)
        vor_t = self.compute_vorticity(u_t, v_t)
        losses['vor'] = self.global_relative_error(vor_p, vor_t, self.eps_vor)

        # 3) Geostrophic balance (ageostrophic components)
        ug_p, vg_p = self.compute_geostrophic_wind(z_p)
        ug_t, vg_t = self.compute_geostrophic_wind(z_t)

        ageo_u_p, ageo_v_p = u_p - ug_p, v_p - vg_p
        ageo_u_t, ageo_v_t = u_t - ug_t, v_t - vg_t

        w_geo = self.geo_weight
        loss_geo_u = self.global_relative_error(ageo_u_p, ageo_u_t, self.eps_geo, weight=w_geo)
        loss_geo_v = self.global_relative_error(ageo_v_p, ageo_v_t, self.eps_geo, weight=w_geo)
        losses['geo'] = 0.5 * (loss_geo_u + loss_geo_v)

        # 4) Moist static energy gradient
        mse_x_p, mse_y_p = self.compute_mse_gradients(t_p, z_p, q_p)
        mse_x_t, mse_y_t = self.compute_mse_gradients(t_t, z_t, q_t)
        loss_mse_x = self.global_relative_error(mse_x_p, mse_x_t, self.eps_mse)
        loss_mse_y = self.global_relative_error(mse_y_p, mse_y_t, self.eps_mse)
        losses['mse'] = 0.5 * (loss_mse_x + loss_mse_y)

        return losses


# ============================================================================
# 6. Combined Loss Module
# ============================================================================
class CombinedLoss(nn.Module):
    """Composite objective function: L1 Loss + Weighted Physics Constraints."""
    def __init__(self, physics_module,
                 lam_l1=1.0, lam_div=0.1, lam_vor=0.1,
                 lam_geo=0.1, lam_mse=0.1):
        super().__init__()
        self.l1_criterion = nn.L1Loss()
        self.physics = physics_module

        self.lam_l1   = float(lam_l1)
        self.lam_div  = float(lam_div)
        self.lam_vor  = float(lam_vor)
        self.lam_geo  = float(lam_geo)
        self.lam_mse  = float(lam_mse)

    def set_lambda_geo(self, new_lambda_geo: float):
        self.lam_geo = float(new_lambda_geo)

    def forward(self, pred, target):
        l1 = self.l1_criterion(pred, target)
        pl = self.physics(pred, target)

        total = (self.lam_l1   * l1
               + self.lam_div  * pl['div']
               + self.lam_vor  * pl['vor']
               + self.lam_geo  * pl['geo']
               + self.lam_mse  * pl['mse'])

        loss_dict = {
            'total': total.detach().item(),
            'l1':    l1.detach().item(),
            'div':   pl['div'].detach().item(),
            'vor':   pl['vor'].detach().item(),
            'geo':   pl['geo'].detach().item(),
            'mse':   pl['mse'].detach().item(),
        }
        return total, loss_dict


# ============================================================================
# 7. Model Evaluation Routines
# ============================================================================
def evaluate(model, dataloader, criterion, device, num_channels, scaler_Y):
    """
    Evaluates model performance over a dataloader.
    Returns: avg_total_loss, avg_l1, avg_physics_dict, rmse_per_channel, overall_rmse
    Note: RMSE is computed in unstandardized original physical space.
    """
    model.eval()
    sum_total   = 0.0
    sum_l1      = 0.0
    sum_phys    = {k: 0.0 for k in PHYSICS_KEYS}
    num_batches = 0

    total_se_std = np.zeros(num_channels, dtype=np.float64)
    total_pixels = np.zeros(num_channels, dtype=np.float64)

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            pred = model(X_batch)
            _, ld = criterion(pred, Y_batch)

            sum_total += ld['total']
            sum_l1    += ld['l1']
            for k in PHYSICS_KEYS:
                sum_phys[k] += ld[k]
            num_batches += 1

            for c in range(num_channels):
                diff = pred[:, c] - Y_batch[:, c]
                total_se_std[c] += (diff ** 2).sum().item()
                total_pixels[c] += diff.numel()

    nb = max(num_batches, 1)
    avg_total = sum_total / nb
    avg_l1    = sum_l1 / nb
    avg_phys  = {k: sum_phys[k] / nb for k in PHYSICS_KEYS}

    std_Y       = scaler_Y.std.astype(np.float64)
    rmse_std    = np.sqrt(total_se_std / np.maximum(total_pixels, 1))
    rmse_per_ch = rmse_std * std_Y

    total_se_orig = np.sum(total_se_std * (std_Y ** 2))
    total_px_all  = np.sum(total_pixels)
    overall_rmse  = np.sqrt(total_se_orig / max(total_px_all, 1))

    return avg_total, avg_l1, avg_phys, rmse_per_ch, overall_rmse


def evaluate_bicubic_baseline(model, dataloader, device, num_channels, scaler_Y):
    """Evaluates the pure bicubic baseline (excluding the neural residual branch)."""
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

    std_Y       = scaler_Y.std.astype(np.float64)
    rmse_std    = np.sqrt(total_se_std / np.maximum(total_pixels, 1))
    rmse_per_ch = rmse_std * std_Y

    total_se_orig = np.sum(total_se_std * (std_Y ** 2))
    total_px_all  = np.sum(total_pixels)
    overall_rmse  = np.sqrt(total_se_orig / max(total_px_all, 1))

    return avg_l1, rmse_per_ch, overall_rmse


# ============================================================================
# 8. Data Loading Utility
# ============================================================================
def load_nc_data(x_path, y_path, variables, return_coords=False):
    """Loads (X, Y) NetCDF datasets and stacks them into shape (time, C, H, W)."""
    print(f'  Loading X: {x_path}')
    ds_x = xr.open_dataset(x_path)
    X = np.stack([ds_x[v].values for v in variables], axis=1).astype(np.float32)
    ds_x.close()

    print(f'  Loading Y: {y_path}')
    ds_y = xr.open_dataset(y_path)
    Y = np.stack([ds_y[v].values for v in variables], axis=1).astype(np.float32)

    latitudes  = ds_y['latitude'].values.astype(np.float64)
    longitudes = ds_y['longitude'].values.astype(np.float64)
    ds_y.close()

    assert X.shape[0] == Y.shape[0], f'Temporal dimensions mismatch: X={X.shape[0]}, Y={Y.shape[0]}'
    print(f'       X shape: {X.shape}  |  Y shape: {Y.shape}')

    if return_coords:
        return X, Y, latitudes, longitudes
    return X, Y


# ============================================================================
# 9. Formatting and Tuning Helpers
# ============================================================================
def format_value(val):
    abs_val = abs(val)
    if abs_val >= 100:
        return f'{val:.2f}'
    elif abs_val >= 0.01:
        return f'{val:.4f}'
    else:
        return f'{val:.4e}'

def format_rmse_str(rmse_array, var_names, overall_rmse):
    parts = [f'{name}: {format_value(rmse_array[i])}' for i, name in enumerate(var_names)]
    return ', '.join(parts) + f', Overall: {format_value(overall_rmse)}'

def format_physics_str(phys_dict):
    items = []
    for k in PHYSICS_KEYS:
        v = phys_dict[k]
        if v >= 0.01:
            items.append(f'{k}={v:.4f}')
        else:
            items.append(f'{k}={v:.2e}')
    return ', '.join(items)

def suggest_lambda_geo(diag_loss_dict, lam_l1, target_rel_to_l1, lam_min, lam_max):
    """
    Suggests lambda_geo based on initial loss values:
      lambda_geo * geo ≈ target_rel_to_l1 * (lambda_l1 * l1)
    """
    l1  = max(float(diag_loss_dict['l1']),  1e-12)
    geo = max(float(diag_loss_dict['geo']), 1e-12)

    raw = (target_rel_to_l1 * float(lam_l1) * l1) / geo
    clipped = float(np.clip(raw, lam_min, lam_max))
    return raw, clipped


# ============================================================================
# 10. Main Execution Routine
# ============================================================================
def main():
    total_start_time = time.time()

    # 10-1 Missing value check
    if CHECK_MISSING:
        all_files = [X_TRAIN_PATH, Y_TRAIN_PATH, X_VALID_PATH, Y_VALID_PATH, X_TEST_PATH, Y_TEST_PATH]
        check_missing_values(all_files, VARIABLES)

    # 10-2 Load datasets
    print('Loading datasets...')
    print('-' * 90)
    X_train_raw, Y_train_raw, lat_hr, lon_hr = load_nc_data(
        X_TRAIN_PATH, Y_TRAIN_PATH, VARIABLES, return_coords=True
    )
    X_valid_raw, Y_valid_raw = load_nc_data(X_VALID_PATH, Y_VALID_PATH, VARIABLES)
    X_test_raw,  Y_test_raw  = load_nc_data(X_TEST_PATH,  Y_TEST_PATH,  VARIABLES)
    print('-' * 90)

    input_h, input_w = X_train_raw.shape[2], X_train_raw.shape[3]
    output_h, output_w = Y_train_raw.shape[2], Y_train_raw.shape[3]
    actual_scale = output_h // input_h

    if UPSCALE_FACTOR != actual_scale:
        raise ValueError(
            f'UPSCALE_FACTOR={UPSCALE_FACTOR} does not match dataset scale factor ({actual_scale}x)! '
            f'Current spatial dimensions: {input_h}x{input_w} -> {output_h}x{output_w}.'
        )

    print(f'  Low Resolution  (Input)  : {input_h} × {input_w}')
    print(f'  High Resolution (Output) : {output_h} × {output_w}')
    print(f'  Upscaling Factor         : {actual_scale}x')
    print(f'  Variables (Channels)     : {NUM_CHANNELS}  {VARIABLES}')
    print(f'  Dataset Sizes            : Train={X_train_raw.shape[0]:,} | Valid={X_valid_raw.shape[0]:,} | Test={X_test_raw.shape[0]:,}')
    print(f'  Latitude Range (HR)      : [{lat_hr.min():.2f}, {lat_hr.max():.2f}]°')
    print(f'  Longitude Range (HR)     : [{lon_hr.min():.2f}, {lon_hr.max():.2f}]°')
    print('=' * 90)

    # 10-3 Standardize datasets (fit independently on train X and Y)
    print('Applying channel-wise Z-score standardization...')
    scaler_X = ChannelScaler()
    scaler_Y = ChannelScaler()

    X_train_raw = scaler_X.fit_transform(X_train_raw)
    X_valid_raw = scaler_X.transform(X_valid_raw)
    X_test_raw  = scaler_X.transform(X_test_raw)

    Y_train_raw = scaler_Y.fit_transform(Y_train_raw)
    Y_valid_raw = scaler_Y.transform(Y_valid_raw)
    Y_test_raw  = scaler_Y.transform(Y_test_raw)

    print(f'  X Scaler: {scaler_X}')
    print(f'  Y Scaler: {scaler_Y}')
    print('  Y Channel Standard Deviations (RMSE scale factor):')
    for i, (name, unit) in enumerate(zip(VAR_NAMES, VAR_UNITS)):
        print(f'    {name}: std = {scaler_Y.std[i]:.6f} ({unit})')

    scalers_dict = {'scaler_X': scaler_X, 'scaler_Y': scaler_Y}
    joblib.dump(scalers_dict, SCALER_SAVE_PATH)
    print(f'  Scalers saved to: {SCALER_SAVE_PATH}')
    print('=' * 90)

    # 10-4 Instantiate DataLoaders
    print('Creating DataLoaders...')
    train_dataset = ERA5Dataset(X_train_raw, Y_train_raw)
    valid_dataset = ERA5Dataset(X_valid_raw, Y_valid_raw)
    test_dataset  = ERA5Dataset(X_test_raw,  Y_test_raw)

    del X_train_raw, Y_train_raw
    del X_valid_raw, Y_valid_raw
    del X_test_raw, Y_test_raw
    gc.collect()

    pin = (device.type == 'cuda')
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin
    )

    print(f'  Train Loader : {len(train_loader):,} batches ({len(train_dataset):,} samples, batch_size={BATCH_SIZE})')
    print(f'  Valid Loader : {len(valid_loader):,} batches ({len(valid_dataset):,} samples)')
    print(f'  Test Loader  : {len(test_loader):,} batches ({len(test_dataset):,} samples)')
    print('=' * 90)

    # 10-5 Model Initialization
    model = BicubicResidualSRModel(
        in_channels=NUM_CHANNELS,
        out_channels=NUM_CHANNELS,
        num_features=NUM_FEATURES,
        num_res_blocks=NUM_RES_BLOCKS,
        upscale_factor=UPSCALE_FACTOR,
        dropout_rate=DROPOUT_RATE,
        scaler_X_mean=scaler_X.mean,
        scaler_X_std=scaler_X.std,
        scaler_Y_mean=scaler_Y.mean,
        scaler_Y_std=scaler_Y.std,
        align_corners=BICUBIC_ALIGN_CORNERS,
        residual_scale=RESIDUAL_SCALE,
        zero_init_last=ZERO_INIT_LAST
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Model Architecture: BicubicResidualSRModel')
    print(f'  Feature Channels      : {NUM_FEATURES}')
    print(f'  Residual Blocks       : {NUM_RES_BLOCKS}')
    print(f'  Upscale Factor        : {UPSCALE_FACTOR}x (PixelShuffle in residual branch)')
    print(f'  Post-PixelShuffle Fuse: Conv3x3 + ReLU')
    print(f'  Output Layer          : Conv3x3 ({NUM_FEATURES} -> {NUM_CHANNELS})')
    print(f'  Dropout Rate          : {DROPOUT_RATE}')
    print(f'  Bicubic Align Corners : {BICUBIC_ALIGN_CORNERS}')
    print(f'  Residual Scale        : {RESIDUAL_SCALE}')
    print(f'  Zero Init Last Conv   : {ZERO_INIT_LAST}')
    print(f'  Total Parameters      : {total_params:,}')
    print(f'  Trainable Parameters  : {trainable_params:,}')
    print('=' * 90)

    # Evaluate pure bicubic baseline benchmark
    print('Evaluating pure Bicubic baseline benchmark (mapped to Y_std space)...')
    b_valid_l1, b_valid_rmse, b_valid_overall = evaluate_bicubic_baseline(
        model, valid_loader, device, NUM_CHANNELS, scaler_Y
    )
    b_test_l1, b_test_rmse, b_test_overall = evaluate_bicubic_baseline(
        model, test_loader, device, NUM_CHANNELS, scaler_Y
    )
    print(f'  [Valid Baseline] L1={b_valid_l1:.6f}, RMSE: {format_rmse_str(b_valid_rmse, VAR_NAMES, b_valid_overall)}')
    print(f'  [Test  Baseline] L1={b_test_l1:.6f}, RMSE: {format_rmse_str(b_test_rmse, VAR_NAMES, b_test_overall)}')
    print('=' * 90)

    # 10-6 Loss Function Initialization
    print('Initializing Physics Constraint Loss Module...')
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

    combined_criterion = CombinedLoss(
        physics_module=physics_loss,
        lam_l1=LAMBDA_L1,
        lam_div=LAMBDA_DIV,
        lam_vor=LAMBDA_VOR,
        lam_geo=LAMBDA_GEO,
        lam_mse=LAMBDA_MSE
    ).to(device)

    print('  Initial Loss Weights:')
    print(f'    λ_L1={combined_criterion.lam_l1}, λ_div={combined_criterion.lam_div}, '
          f'λ_vor={combined_criterion.lam_vor}, λ_geo={combined_criterion.lam_geo}, λ_mse={combined_criterion.lam_mse}')
    print(f'  Epsilon Settings: div={EPS_DIV}, vor={EPS_VOR}, geo={EPS_GEO}, mse={EPS_MSE}')
    print(f'  Geostrophic Latitude Weighting: mode={GEO_WEIGHT_MODE}, alpha={GEO_WEIGHT_ALPHA}, min={GEO_WEIGHT_MIN}')
    print('  Lambda Geo Auto-Tuning:')
    print(f'    Enabled={AUTO_TUNE_LAMBDA_GEO}, target_rel_to_l1={GEO_TARGET_REL_TO_L1}, '
          f'range=[{GEO_LAMBDA_MIN}, {GEO_LAMBDA_MAX}]')
    print('=' * 90)

    # Initial loss diagnostics
    print('Diagnostics: Evaluating initial loss components on first train batch...')
    model.eval()
    with torch.no_grad():
        X_sample, Y_sample = next(iter(train_loader))
        X_sample = X_sample.to(device)
        Y_sample = Y_sample.to(device)
        pred_sample = model(X_sample)
        _, diag_ld = combined_criterion(pred_sample, Y_sample)
        for k, v in diag_ld.items():
            print(f'  {k:>8s}: {v:.6e}')

    # Automatically tune lambda_geo if enabled
    if AUTO_TUNE_LAMBDA_GEO:
        raw_lam, tuned_lam = suggest_lambda_geo(
            diag_loss_dict=diag_ld,
            lam_l1=combined_criterion.lam_l1,
            target_rel_to_l1=GEO_TARGET_REL_TO_L1,
            lam_min=GEO_LAMBDA_MIN,
            lam_max=GEO_LAMBDA_MAX
        )
        old_lam = combined_criterion.lam_geo
        combined_criterion.set_lambda_geo(tuned_lam)

        print('-' * 90)
        print(f'  [Auto λ_geo] Suggested raw={raw_lam:.6e}, Applied (clipped)={tuned_lam:.6e}')
        if not np.isclose(raw_lam, tuned_lam):
            print('  [Auto λ_geo] Note: Suggested value exceeded boundaries; clipped to range limits.')
        print(f'  [Auto λ_geo] Updated λ_geo: {old_lam:.6e} -> {combined_criterion.lam_geo:.6e}')

        _, diag_after = combined_criterion(pred_sample, Y_sample)
        l1_contrib  = combined_criterion.lam_l1  * diag_after['l1']
        geo_contrib = combined_criterion.lam_geo * diag_after['geo']
        ratio = geo_contrib / (l1_contrib + 1e-12)
        print(f'  [Auto λ_geo] Calibrated weighted ratio geo/L1 ≈ {ratio:.4f} (Target {GEO_TARGET_REL_TO_L1:.4f})')

    print('-' * 90)
    print('  Final Loss Weights:')
    print(f'    λ_L1={combined_criterion.lam_l1:.6f}, λ_div={combined_criterion.lam_div:.6f}, '
          f'λ_vor={combined_criterion.lam_vor:.6f}, λ_geo={combined_criterion.lam_geo:.6f}, '
          f'λ_mse={combined_criterion.lam_mse:.6f}')
    print('=' * 90)

    # 10-7 Optimizer and Scheduler Initialization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )

    print('Loss Function : L1 + Physics (GRE)')
    print(f'Optimizer     : Adam (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})')
    print(f'LR Scheduler  : ReduceLROnPlateau (factor={SCHEDULER_FACTOR}, patience={SCHEDULER_PATIENCE})')
    print('RMSE Metric   : Unstandardized Original Physical Units')
    print('=' * 90)

    # 10-8 Training Loop
    print(f'Starting Training ({NUM_EPOCHS} Epochs)...')
    units_str = ', '.join([f'{n}({u})' for n, u in zip(VAR_NAMES, VAR_UNITS)])
    print(f'  RMSE Units: {units_str}')
    print('-' * 90)

    best_val_loss = float('inf')

    history_records  = []
    train_total_list = []
    val_total_list   = []
    test_total_list  = []
    train_l1_list    = []
    val_l1_list      = []
    test_l1_list     = []

    val_phys_history = {k: [] for k in PHYSICS_KEYS}

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # ------------------ Train Phase ------------------
        model.train()
        train_loss_accum = {k: 0.0 for k in ['total', 'l1'] + PHYSICS_KEYS}
        n_train_batches = 0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(X_batch)
            total_loss, ld = combined_criterion(pred, Y_batch)
            total_loss.backward()
            optimizer.step()

            for k in train_loss_accum:
                train_loss_accum[k] += ld[k]
            n_train_batches += 1

        train_avg = {k: train_loss_accum[k] / max(n_train_batches, 1) for k in train_loss_accum}

        # ------------------ Evaluation Phase ------------------
        _, _, _, train_rmse, train_overall = evaluate(
            model, train_loader, combined_criterion, device, NUM_CHANNELS, scaler_Y
        )
        val_total, val_l1, val_phys, val_rmse, val_overall = evaluate(
            model, valid_loader, combined_criterion, device, NUM_CHANNELS, scaler_Y
        )
        test_total, test_l1, test_phys, test_rmse, test_overall = evaluate(
            model, test_loader, combined_criterion, device, NUM_CHANNELS, scaler_Y
        )

        scheduler.step(val_total)
        new_lr = optimizer.param_groups[0]['lr']

        # Save Checkpoint if Validation Loss Improves
        model_status = 'Unchanged'
        if val_total <= best_val_loss:
            best_val_loss = val_total
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': float(best_val_loss),

                'model_type': 'BicubicResidualSRModel',
                'bicubic_align_corners': BICUBIC_ALIGN_CORNERS,
                'residual_scale': RESIDUAL_SCALE,
                'zero_init_last': ZERO_INIT_LAST,

                'scaler_X_mean': scaler_X.mean.tolist(),
                'scaler_X_std': scaler_X.std.tolist(),
                'scaler_Y_mean': scaler_Y.mean.tolist(),
                'scaler_Y_std': scaler_Y.std.tolist(),
                'latitudes': lat_hr.tolist(),
                'longitudes': lon_hr.tolist(),

                'physics_weights': {
                    'lam_l1':   float(combined_criterion.lam_l1),
                    'lam_div':  float(combined_criterion.lam_div),
                    'lam_vor':  float(combined_criterion.lam_vor),
                    'lam_geo':  float(combined_criterion.lam_geo),
                    'lam_mse':  float(combined_criterion.lam_mse),
                },
                'geo_lambda_auto_tune': {
                    'enabled': bool(AUTO_TUNE_LAMBDA_GEO),
                    'target_rel_to_l1': float(GEO_TARGET_REL_TO_L1),
                    'lam_min': float(GEO_LAMBDA_MIN),
                    'lam_max': float(GEO_LAMBDA_MAX),
                    'effective_lambda_geo': float(combined_criterion.lam_geo),
                },
                'geo_weighting': {
                    'mode':  GEO_WEIGHT_MODE,
                    'alpha': GEO_WEIGHT_ALPHA,
                    'min':   GEO_WEIGHT_MIN
                },
                'thermo_constants': {
                    'cp_air': CP_AIR,
                    'lv': LV_WATER
                },
                'eps_values': {
                    'eps_div': EPS_DIV,
                    'eps_vor': EPS_VOR,
                    'eps_geo': EPS_GEO,
                    'eps_mse': EPS_MSE
                }
            }, MODEL_SAVE_PATH)
            model_status = 'Saved (Best Valid)'

        epoch_time = time.time() - epoch_start
        remaining_time = epoch_time * (NUM_EPOCHS - epoch - 1)

        train_total_list.append(train_avg['total'])
        val_total_list.append(val_total)
        test_total_list.append(test_total)

        train_l1_list.append(train_avg['l1'])
        val_l1_list.append(val_l1)
        test_l1_list.append(test_l1)

        for k in PHYSICS_KEYS:
            val_phys_history[k].append(val_phys[k])

        record = {
            'Epoch': epoch + 1,
            'Train_Total': round(train_avg['total'], 6),
            'Valid_Total': round(val_total, 6),
            'Test_Total':  round(test_total, 6),
            'Train_L1':    round(train_avg['l1'], 6),
            'Valid_L1':    round(val_l1, 6),
            'Test_L1':     round(test_l1, 6),
            'Lambda_GEO':  float(combined_criterion.lam_geo),
            'LR':          current_lr,
        }

        for k in PHYSICS_KEYS:
            record[f'Train_P_{k}'] = float(train_avg.get(k, 0.0))
            record[f'Valid_P_{k}'] = float(val_phys[k])
            record[f'Test_P_{k}']  = float(test_phys[k])

        for i, name in enumerate(VAR_NAMES):
            record[f'Train_RMSE_{name}'] = float(train_rmse[i])
            record[f'Valid_RMSE_{name}'] = float(val_rmse[i])
            record[f'Test_RMSE_{name}']  = float(test_rmse[i])

        record['Train_RMSE_Overall'] = float(train_overall)
        record['Valid_RMSE_Overall'] = float(val_overall)
        record['Test_RMSE_Overall']  = float(test_overall)
        history_records.append(record)

        # Print Progress
        print('-' * 90)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]  '
              f'Total: train={train_avg["total"]:.4f} valid={val_total:.4f} test={test_total:.4f}  '
              f'L1: train={train_avg["l1"]:.4f} valid={val_l1:.4f} test={test_l1:.4f}')
        print(f'  Status: {model_status}  Time: {epoch_time:.1f}s  Remaining: {remaining_time:.0f}s')
        print(f'  [Valid Physics] {format_physics_str(val_phys)}')
        print(f'  [Train RMSE] {format_rmse_str(train_rmse, VAR_NAMES, train_overall)}')
        print(f'  [Valid RMSE] {format_rmse_str(val_rmse, VAR_NAMES, val_overall)}')
        print(f'  [Test  RMSE] {format_rmse_str(test_rmse, VAR_NAMES, test_overall)}')

        if new_lr != current_lr:
            print(f'  [LR Adjusted] {current_lr:.6f} → {new_lr:.6f}')

        if (epoch + 1) % 5 == 0:
            _plot_losses(
                epoch, train_total_list, val_total_list, test_total_list,
                train_l1_list, val_l1_list, test_l1_list,
                val_phys_history, input_h, input_w, output_h, output_w
            )

    # 10-9 Summary and Final Logging
    print('=' * 90)
    print('Training Complete!')

    best_idx = int(np.argmin([r['Valid_Total'] for r in history_records]))
    best_epoch_num = history_records[best_idx]['Epoch']
    print(f'  Best Epoch         : {best_epoch_num}')
    print(f'  Best Valid Loss    : {best_val_loss:.6f}')
    print(f'  Model Saved To     : {MODEL_SAVE_PATH}')

    best_rec = history_records[best_idx]
    print('  Best Epoch RMSE (Unstandardized Original Physical Units):')
    for prefix, label in [('Train', 'Train Set'), ('Valid', 'Valid Set'), ('Test', 'Test Set')]:
        rmse_vals = [best_rec[f'{prefix}_RMSE_{n}'] for n in VAR_NAMES]
        overall   = best_rec[f'{prefix}_RMSE_Overall']
        parts = [f'{n}: {format_value(v)}' for n, v in zip(VAR_NAMES, rmse_vals)]
        print(f'    [{label}] {", ".join(parts)}, Overall: {format_value(overall)}')

    print('  Best Epoch Physics Constraint Loss (Validation Set):')
    for k in PHYSICS_KEYS:
        v = best_rec[f'Valid_P_{k}']
        print(f'    {k}: {v:.6e}')

    _plot_losses(
        NUM_EPOCHS - 1, train_total_list, val_total_list, test_total_list,
        train_l1_list, val_l1_list, test_l1_list,
        val_phys_history, input_h, input_w, output_h, output_w,
        best_epoch=best_epoch_num
    )

    df_history = pd.DataFrame(history_records)
    df_history.to_csv(HISTORY_CSV_PATH, index=False)
    print(f'  Training history saved to: {HISTORY_CSV_PATH}')

    total_time = time.time() - total_start_time
    model_info = {
        'architecture': 'BicubicResidualSRModel + PhysicsConstraints (GRE)',
        'in_channels': NUM_CHANNELS,
        'out_channels': NUM_CHANNELS,
        'variables': VARIABLES,
        'variable_units': VAR_UNITS,
        'num_features': NUM_FEATURES,
        'num_res_blocks': NUM_RES_BLOCKS,
        'upscale_factor': UPSCALE_FACTOR,
        'dropout_rate': DROPOUT_RATE,

        'bicubic_skip': True,
        'bicubic_align_corners': BICUBIC_ALIGN_CORNERS,
        'residual_scale': RESIDUAL_SCALE,
        'zero_init_last': ZERO_INIT_LAST,
        'post_pixelshuffle_fuse_conv3x3': True,
        'output_conv_kernel': 3,
        'xstd_to_ystd_affine_mapping': True,

        'input_size': f'{input_h}x{input_w}',
        'output_size': f'{output_h}x{output_w}',
        'total_params': total_params,
        'trainable_params': trainable_params,

        'loss_function': 'L1 + Physics (GRE)',
        'physics_constraints': [
            'Divergence (comparison)',
            'Vorticity (comparison)',
            'Geostrophic balance - ageostrophic wind (comparison, latitude weighted)',
            'Moist static energy gradient MSE (comparison)'
        ],
        'physics_weights': {
            'lambda_l1':   float(combined_criterion.lam_l1),
            'lambda_div':  float(combined_criterion.lam_div),
            'lambda_vor':  float(combined_criterion.lam_vor),
            'lambda_geo':  float(combined_criterion.lam_geo),
            'lambda_mse':  float(combined_criterion.lam_mse)
        },
        'geo_lambda_auto_tune': {
            'enabled': bool(AUTO_TUNE_LAMBDA_GEO),
            'target_rel_to_l1': float(GEO_TARGET_REL_TO_L1),
            'lam_min': float(GEO_LAMBDA_MIN),
            'lam_max': float(GEO_LAMBDA_MAX),
            'effective_lambda_geo': float(combined_criterion.lam_geo),
        },
        'epsilon_values': {
            'eps_div': EPS_DIV,
            'eps_vor': EPS_VOR,
            'eps_geo': EPS_GEO,
            'eps_mse': EPS_MSE
        },
        'thermo_constants': {
            'cp_air': CP_AIR,
            'lv': LV_WATER
        },
        'geo_weighting': {
            'mode': GEO_WEIGHT_MODE,
            'alpha': GEO_WEIGHT_ALPHA,
            'min': GEO_WEIGHT_MIN
        },
        'spherical_metric': True,

        'optimizer': 'Adam',
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'random_seed': RANDOM_SEED,
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_factor': SCHEDULER_FACTOR,
        'scheduler_patience': SCHEDULER_PATIENCE,

        'bicubic_baseline_valid_l1': float(b_valid_l1),
        'bicubic_baseline_test_l1': float(b_test_l1),
        'bicubic_baseline_valid_rmse_overall': float(b_valid_overall),
        'bicubic_baseline_test_rmse_overall': float(b_test_overall),

        'best_epoch': int(best_epoch_num),
        'best_valid_loss': round(float(best_val_loss), 6),
        'rmse_space': 'inverse-standardized (original physical units)',
        'device': str(device),
        'training_time_sec': round(total_time, 1),
        'scaler_X': {
            'mean': scaler_X.mean.tolist(),
            'std':  scaler_X.std.tolist()
        },
        'scaler_Y': {
            'mean': scaler_Y.mean.tolist(),
            'std':  scaler_Y.std.tolist()
        },
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(MODEL_INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False)
    print(f'  Model metadata saved to: {MODEL_INFO_PATH}')

    print('=' * 90)
    print('Generated Output Files:')
    print(f'  Best Model Checkpoint : {MODEL_SAVE_PATH}')
    print(f'  Total Loss Plot       : {LOSS_PLOT_PATH}')
    print(f'  Physics Loss Plot     : {PHYSICS_PLOT_PATH}')
    print(f'  Training History CSV  : {HISTORY_CSV_PATH}')
    print(f'  Model Metadata JSON   : {MODEL_INFO_PATH}')
    print(f'  Data Scalers          : {SCALER_SAVE_PATH}')
    print(f'Total Runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)')
    print('=' * 90)


# ============================================================================
# 11. Plotting Utilities
# ============================================================================
def _plot_losses(epoch, train_total, val_total, test_total,
                 train_l1, val_l1, test_l1,
                 val_phys_history, in_h, in_w, out_h, out_w,
                 best_epoch=None):
    """Generates and updates loss progression plots."""
    n = epoch + 1
    epochs_range = range(1, n + 1)

    # Figure 1: Total and L1 Loss Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(epochs_range, train_total[:n], label='Train Total', lw=1.5)
    ax1.plot(epochs_range, val_total[:n],   label='Valid Total', lw=1.5)
    ax1.plot(epochs_range, test_total[:n],  label='Test Total',  lw=1.5)
    if best_epoch:
        ax1.axvline(x=best_epoch, color='red', ls='--', alpha=0.5, label=f'Best ({best_epoch})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss (L1 + Physics)')
    ax1.set_title(f'Total Loss ({in_h}x{in_w} -> {out_h}x{out_w})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, train_l1[:n], label='Train L1', lw=1.5)
    ax2.plot(epochs_range, val_l1[:n],   label='Valid L1', lw=1.5)
    ax2.plot(epochs_range, test_l1[:n],  label='Test L1',  lw=1.5)
    if best_epoch:
        ax2.axvline(x=best_epoch, color='red', ls='--', alpha=0.5, label=f'Best ({best_epoch})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('L1 Loss')
    ax2.set_title('L1 Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=600)
    plt.close()
    print(f'  [Plot] Loss curves saved to: {LOSS_PLOT_PATH}')

    # Figure 2: Validation Physics Losses
    n_keys = len(PHYSICS_KEYS)
    ncols = 3 if n_keys <= 6 else 4
    nrows = math.ceil(n_keys / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.8*nrows))
    axes = np.array(axes).reshape(-1)

    titles = {
        'div':   'Divergence (comparison)',
        'vor':   'Vorticity (comparison)',
        'geo':   'Geostrophic balance (comparison, latitude-weighted)',
        'mse':   'Moist static energy gradient MSE (comparison)',
    }

    for i, k in enumerate(PHYSICS_KEYS):
        ax = axes[i]
        vals = val_phys_history[k][:n]
        ax.plot(epochs_range, vals, lw=1.5, color='tab:orange')
        if best_epoch:
            ax.axvline(x=best_epoch, color='red', ls='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Validation: {titles[k]}')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))

    for j in range(n_keys, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(PHYSICS_PLOT_PATH, dpi=600)
    plt.close()
    print(f'  [Plot] Physics loss curves saved to: {PHYSICS_PLOT_PATH}')


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == '__main__':
    main()