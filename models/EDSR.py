#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA5 Super-Resolution Training Pipeline
Classic EDSR (ResBlock + No BN + PixelShuffle) with optional Physics Constraints (GRE)

Notes:
1) Classic EDSR residual blocks do not use Batch Normalization (BN) by default.
2) Set USE_BN_IN_RESBLOCK = True for comparative experiments with BN.
3) Physical constraints are calculated after inverse normalization using batch-level
   Global Relative Error (GRE): sum(|pred - target|) / (sum(|target|) + eps).
"""

import os
import gc
import math
import json
import time
import random
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# 0) Global Configurations
# ============================================================================
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42
NUM_WORKERS = 2

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 4

# --- Classic EDSR Settings ---
NUM_FEATURES = 64
NUM_RES_BLOCKS = 16
RES_BLOCK_SCALE = 1.0          # Default baseline scale = 1.0; reduce to 0.1 if unstable
UPSCALE_FACTOR = 2             # 2 or 4
USE_BN_IN_RESBLOCK = False     # Classic EDSR: False; set True for SRResNet-like setup

# --- Loss Weights ---
LAMBDA_L1 = 1.0
LAMBDA_DIV = 0.1
LAMBDA_VOR = 0.1
LAMBDA_GEO = 0.1
LAMBDA_MSE = 0.1

# --- Physical Constants ---
CP_AIR = 1004.0
LV_WATER = 2.501e6

# --- GRE Epsilon Values ---
EPS_DIV = 1e-8
EPS_VOR = 1e-8
EPS_GEO = 1e-4
EPS_MSE = 1e-4

# --- Geostrophic Latitude Weighting ---
GEO_WEIGHT_MODE = "sin_global"   # Options: ["f_ratio", "sin_global", "cos", "one_minus_cos"]
GEO_WEIGHT_ALPHA = 1.0
GEO_WEIGHT_MIN = 0.10

# --- File Paths ---
LOW_RES_PREFIX = "X2"  # Use "X2_*.nc" for 2x; use "X4_*.nc" + UPSCALE_FACTOR=4 for 4x

X_TRAIN_PATH = f"{LOW_RES_PREFIX}_train.nc"
X_VALID_PATH = f"{LOW_RES_PREFIX}_valid.nc"
X_TEST_PATH = f"{LOW_RES_PREFIX}_test.nc"

Y_TRAIN_PATH = "Y_train.nc"
Y_VALID_PATH = "Y_valid.nc"
Y_TEST_PATH = "Y_test.nc"

VARIABLES = ["z", "q", "t", "u", "v"]
VAR_NAMES = ["Z", "Q", "T", "U", "V"]
VAR_UNITS = ["m^2/s^2", "kg/kg", "K", "m/s", "m/s"]
NUM_CHANNELS = len(VARIABLES)
PHYSICS_KEYS = ["div", "vor", "geo", "mse"]

CHECK_MISSING = False

OUTPUT_ROOT = "runs_classic_edsr"
SCALER_SAVE_PATH = os.path.join(OUTPUT_ROOT, "scalers.joblib")
ABLATION_CSV_PATH = os.path.join(OUTPUT_ROOT, "ablation_summary.csv")

EXPERIMENTS = [
    {
        "name": "ClassicEDSR_L1_only",
        "use_physics": False,
        "use_bn": USE_BN_IN_RESBLOCK,
        "lambdas": {"l1": 1.0, "div": 0.0, "vor": 0.0, "geo": 0.0, "mse": 0.0}
    },
    {
        "name": "ClassicEDSR_L1_Physics_GRE",
        "use_physics": True,
        "use_bn": USE_BN_IN_RESBLOCK,
        "lambdas": {"l1": LAMBDA_L1, "div": LAMBDA_DIV, "vor": LAMBDA_VOR, "geo": LAMBDA_GEO, "mse": LAMBDA_MSE}
    },
    # Optional experiment to evaluate residual blocks with Batch Normalization
    # {
    #     "name": "SRResNetLike_BN_L1_Physics",
    #     "use_physics": True,
    #     "use_bn": True,
    #     "lambdas": {"l1": LAMBDA_L1, "div": LAMBDA_DIV, "vor": LAMBDA_VOR, "geo": LAMBDA_GEO, "mse": LAMBDA_MSE}
    # },
]


# ============================================================================
# 1) Utility Functions
# ============================================================================
def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def format_value(v: float) -> str:
    """Format floating point numbers for clean terminal reporting."""
    a = abs(v)
    if a >= 100:
        return f"{v:.2f}"
    if a >= 1e-2:
        return f"{v:.4f}"
    return f"{v:.3e}"


def format_rmse_str(rmse_array, var_names, overall_rmse):
    """Format RMSE metrics string for console output."""
    msg = [f"{n}:{format_value(rmse_array[i])}" for i, n in enumerate(var_names)]
    msg.append(f"Overall:{format_value(overall_rmse)}")
    return ", ".join(msg)


def format_physics_str(phys_dict):
    """Format physical loss metrics string for console output."""
    return ", ".join([f"{k}:{format_value(float(phys_dict[k]))}" for k in PHYSICS_KEYS])


def check_missing_values(file_paths, variables, missing_value=-999):
    """Verify dataset integrity by checking for NaN or sentinel missing values."""
    print("Checking missing values (-999 / NaN)...")
    bad = False
    for p in file_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")
        ds = xr.open_dataset(p)
        has_issue = False
        for v in variables:
            x = ds[v].values
            miss_cnt = int(np.sum(np.isclose(x, missing_value, atol=0.1)))
            nan_cnt = int(np.sum(np.isnan(x)))
            if miss_cnt > 0:
                print(f"[WARNING] {p} {v} contains sentinel value (-999): {miss_cnt:,}")
                bad = True
                has_issue = True
            if nan_cnt > 0:
                print(f"[WARNING] {p} {v} contains NaN values: {nan_cnt:,}")
                bad = True
                has_issue = True
        if not has_issue:
            print(f"[PASS] {p}")
        ds.close()
    if bad:
        raise ValueError("Dataset contains missing or invalid values. Clean data before training.")
    print("All file integrity checks passed.")


# ============================================================================
# 2) Data Standardization
# ============================================================================
class ChannelScaler:
    """Channel-wise Z-score standardization: (N, C, H, W)"""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = data.mean(axis=(0, 2, 3)).astype(np.float64)
        self.std = data.std(axis=(0, 2, 3)).astype(np.float64)
        self.std[self.std < 1e-12] = 1.0
        return self

    def transform(self, data):
        mean = self.mean.reshape(1, -1, 1, 1).astype(np.float32)
        std = self.std.reshape(1, -1, 1, 1).astype(np.float32)
        return ((data.astype(np.float32) - mean) / std).astype(np.float32)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform_numpy(self, data):
        mean = self.mean.reshape(1, -1, 1, 1).astype(np.float32)
        std = self.std.reshape(1, -1, 1, 1).astype(np.float32)
        return data * std + mean

    def __repr__(self):
        if self.mean is None:
            return "ChannelScaler(unfitted)"
        m = ", ".join([f"{x:.4f}" for x in self.mean])
        s = ", ".join([f"{x:.4f}" for x in self.std])
        return f"ChannelScaler(mean=[{m}], std=[{s}])"


# ============================================================================
# 3) Dataset / DataLoader
# ============================================================================
class ERA5Dataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def build_dataloaders(train_ds, valid_ds, test_ds, batch_size, num_workers, device, seed):
    pin = (device.type == "cuda")
    persistent = (num_workers > 0)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin, persistent_workers=persistent, generator=g
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin, persistent_workers=persistent
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin, persistent_workers=persistent
    )
    return train_loader, valid_loader, test_loader


# ============================================================================
# 4) Classic EDSR Architecture
# ============================================================================
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)


class ResBlock(nn.Module):
    """
    EDSR Residual Block:
      conv -> relu -> conv (+ optional BN) + residual scaling
    """
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
    """
    PixelShuffle Upsampler supporting scales 2, 3, and 4.
    """
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:
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
            raise ValueError(f"Unsupported upscale factor: {scale}")
        super().__init__(*m)


class ClassicEDSR(nn.Module):
    """
    Classic EDSR Model:
      head -> N ResBlocks -> body conv -> skip connection -> upsampler -> tail conv
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
        if scale not in [2, 4]:
            raise ValueError("This pipeline expects an upscale factor of 2 or 4.")
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


# ============================================================================
# 5) Physical Constraints Module
# ============================================================================
class PhysicsConstraintLoss(nn.Module):
    """
    Physics Constraint Loss evaluated on unstandardized variables:
      1) Divergence (div)
      2) Vorticity (vor)
      3) Geostrophic balance / ageostrophic component (geo)
      4) Moist Static Energy gradient (mse)

    Formulated using Global Relative Error (GRE):
      sum(weight * |pred - target|) / (sum(weight * |target|) + eps)
    """
    def __init__(
        self,
        scaler_y_mean, scaler_y_std,
        latitudes, longitudes,
        cp_air=1004.0, lv=2.501e6,
        eps_div=1e-8, eps_vor=1e-8, eps_geo=1e-4, eps_mse=1e-4,
        geo_weight_mode="sin_global",
        geo_weight_alpha=1.0,
        geo_weight_min=0.1
    ):
        super().__init__()

        # Channel indices: z, q, t, u, v
        self.idx_z, self.idx_q, self.idx_t = 0, 1, 2
        self.idx_u, self.idx_v = 3, 4

        self.eps_div = float(eps_div)
        self.eps_vor = float(eps_vor)
        self.eps_geo = float(eps_geo)
        self.eps_mse = float(eps_mse)

        self.register_buffer("cp_air", torch.tensor(cp_air, dtype=torch.float32))
        self.register_buffer("lv", torch.tensor(lv, dtype=torch.float32))

        self.register_buffer("y_mean", torch.tensor(scaler_y_mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("y_std", torch.tensor(scaler_y_std, dtype=torch.float32).view(1, -1, 1, 1))

        # Earth System Constants
        R = 6371000.0
        omega = 7.2921e-5
        self.R_earth = R

        lat = latitudes.astype(np.float64)
        lon = longitudes.astype(np.float64)
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        dlat = np.abs(np.mean(np.diff(lat_rad)))
        dlon = np.abs(np.mean(np.diff(lon_rad)))
        lat_decreasing = (lat[1] < lat[0])

        dy = R * dlat
        dx = R * np.cos(lat_rad) * dlon
        dx = np.maximum(dx, 100.0)

        inv_dx = (1.0 / dx).reshape(1, 1, -1, 1)
        self.register_buffer("inv_dx", torch.tensor(inv_dx, dtype=torch.float32))
        self.register_buffer("inv_dy", torch.tensor(1.0 / dy, dtype=torch.float32))

        f_raw = 2.0 * omega * np.sin(lat_rad)
        f_clamped = f_raw.copy()
        f_min = 2.5e-5
        f_clamped = np.where(np.abs(f_clamped) < f_min, np.where(f_clamped >= 0, f_min, -f_min), f_clamped)
        inv_f = (1.0 / f_clamped).reshape(1, 1, -1, 1)
        self.register_buffer("inv_f", torch.tensor(inv_f, dtype=torch.float32))

        # Geostrophic latitude weighting
        if geo_weight_mode == "f_ratio":
            f_abs = np.abs(f_raw)
            geo_w = (f_abs / (f_abs.max() + 1e-12)) ** float(geo_weight_alpha)
        elif geo_weight_mode == "sin_global":
            geo_w = (np.abs(np.sin(lat_rad))) ** float(geo_weight_alpha)
        elif geo_weight_mode == "cos":
            geo_w = (np.abs(np.cos(lat_rad))) ** float(geo_weight_alpha)
        elif geo_weight_mode == "one_minus_cos":
            geo_w = (1.0 - np.abs(np.cos(lat_rad))) ** float(geo_weight_alpha)
            m = geo_w.max()
            if m > 0:
                geo_w = geo_w / m
        else:
            raise ValueError(f"Unknown geo_weight_mode: {geo_weight_mode}")

        geo_w = np.clip(geo_w, float(geo_weight_min), 1.0)
        self.register_buffer("geo_weight", torch.tensor(geo_w.reshape(1, -1, 1), dtype=torch.float32))

        tan_lat = np.tan(lat_rad)
        tan_lat = np.clip(tan_lat, -50.0, 50.0)
        tan_over_r = (tan_lat / R).reshape(1, 1, -1, 1)
        self.register_buffer("tan_over_r", torch.tensor(tan_over_r, dtype=torch.float32))

        # Central Finite Difference Kernels
        ddx_k = torch.tensor(
            [[0., 0., 0.],
             [-0.5, 0., 0.5],
             [0., 0., 0.]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("ddx_kernel", ddx_k)

        if lat_decreasing:
            ddy_k = torch.tensor(
                [[0., 0.5, 0.],
                 [0., 0., 0.],
                 [0., -0.5, 0.]],
                dtype=torch.float32
            ).view(1, 1, 3, 3)
        else:
            ddy_k = torch.tensor(
                [[0., -0.5, 0.],
                 [0., 0., 0.],
                 [0., 0.5, 0.]],
                dtype=torch.float32
            ).view(1, 1, 3, 3)
        self.register_buffer("ddy_kernel", ddy_k)

        lat_dir = "North->South" if lat_decreasing else "South->North"
        print(f"[Physics] Latitude orientation: {lat_dir}, dy={dy:.1f}m, dx range=[{dx.min():.1f}, {dx.max():.1f}]m")
        print(f"[Physics] Geostrophic weight mode={geo_weight_mode}, range=[{geo_w.min():.3f}, {geo_w.max():.3f}]")

    def inverse_std(self, x):
        return x * self.y_std + self.y_mean

    def _conv_fd(self, field_4d, kernel):
        padded = F.pad(field_4d, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(padded, kernel)

    def compute_ddx(self, field):
        raw = self._conv_fd(field.unsqueeze(1), self.ddx_kernel)   # (B, 1, H, W)
        return (raw * self.inv_dx).squeeze(1)

    def compute_ddy(self, field):
        raw = self._conv_fd(field.unsqueeze(1), self.ddy_kernel)
        return (raw * self.inv_dy).squeeze(1)

    def compute_divergence(self, u, v):
        du_dx = self.compute_ddx(u)
        dv_dy = self.compute_ddy(v)
        metric = v * self.tan_over_r.squeeze(0).squeeze(0)   # Broadcast over (H, 1)
        return du_dx + dv_dy - metric

    def compute_vorticity(self, u, v):
        dv_dx = self.compute_ddx(v)
        du_dy = self.compute_ddy(u)
        metric = u * self.tan_over_r.squeeze(0).squeeze(0)
        return dv_dx - du_dy + metric

    def compute_geostrophic_wind(self, z):
        dz_dx = self.compute_ddx(z)
        dz_dy = self.compute_ddy(z)
        inv_f = self.inv_f.squeeze(0).squeeze(0)   # (H, 1)
        ug = -inv_f * dz_dy
        vg = inv_f * dz_dx
        return ug, vg

    def compute_mse(self, t, z, q):
        return self.cp_air * t + z + self.lv * q

    def compute_mse_gradients(self, t, z, q):
        mse = self.compute_mse(t, z, q)
        return self.compute_ddx(mse), self.compute_ddy(mse)

    @staticmethod
    def global_relative_error(pred_qty, target_qty, eps, weight=None):
        err = torch.abs(pred_qty - target_qty)
        ref = torch.abs(target_qty)
        if weight is not None:
            err = err * weight
            ref = ref * weight
        return err.sum() / (ref.sum() + eps)

    def forward(self, pred_std, target_std):
        pred = self.inverse_std(pred_std)
        target = self.inverse_std(target_std)

        z_p, q_p, t_p = pred[:, self.idx_z], pred[:, self.idx_q], pred[:, self.idx_t]
        u_p, v_p = pred[:, self.idx_u], pred[:, self.idx_v]

        z_t, q_t, t_t = target[:, self.idx_z], target[:, self.idx_q], target[:, self.idx_t]
        u_t, v_t = target[:, self.idx_u], target[:, self.idx_v]

        losses = {}

        # 1) Divergence
        div_p = self.compute_divergence(u_p, v_p)
        div_t = self.compute_divergence(u_t, v_t)
        losses["div"] = self.global_relative_error(div_p, div_t, self.eps_div)

        # 2) Vorticity
        vor_p = self.compute_vorticity(u_p, v_p)
        vor_t = self.compute_vorticity(u_t, v_t)
        losses["vor"] = self.global_relative_error(vor_p, vor_t, self.eps_vor)

        # 3) Geostrophic balance (ageostrophic components)
        ug_p, vg_p = self.compute_geostrophic_wind(z_p)
        ug_t, vg_t = self.compute_geostrophic_wind(z_t)
        ageo_u_p, ageo_v_p = u_p - ug_p, v_p - vg_p
        ageo_u_t, ageo_v_t = u_t - ug_t, v_t - vg_t

        w_geo = self.geo_weight
        geo_u = self.global_relative_error(ageo_u_p, ageo_u_t, self.eps_geo, weight=w_geo)
        geo_v = self.global_relative_error(ageo_v_p, ageo_v_t, self.eps_geo, weight=w_geo)
        losses["geo"] = 0.5 * (geo_u + geo_v)

        # 4) MSE gradients
        mse_x_p, mse_y_p = self.compute_mse_gradients(t_p, z_p, q_p)
        mse_x_t, mse_y_t = self.compute_mse_gradients(t_t, z_t, q_t)
        mse_x = self.global_relative_error(mse_x_p, mse_x_t, self.eps_mse)
        mse_y = self.global_relative_error(mse_y_p, mse_y_t, self.eps_mse)
        losses["mse"] = 0.5 * (mse_x + mse_y)

        return losses


# ============================================================================
# 6) Combined Loss Function
# ============================================================================
class CombinedLoss(nn.Module):
    def __init__(
        self,
        physics_module: PhysicsConstraintLoss,
        use_physics=True,
        lam_l1=1.0, lam_div=0.1, lam_vor=0.1, lam_geo=0.1, lam_mse=0.1
    ):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.physics = physics_module
        self.use_physics = bool(use_physics)

        self.lam_l1 = float(lam_l1)
        self.lam_div = float(lam_div)
        self.lam_vor = float(lam_vor)
        self.lam_geo = float(lam_geo)
        self.lam_mse = float(lam_mse)

    def forward(self, pred, target):
        l1_val = self.l1(pred, target)

        if self.use_physics:
            pl = self.physics(pred, target)
        else:
            zero = pred.new_tensor(0.0)
            pl = {k: zero for k in PHYSICS_KEYS}

        total = (
            self.lam_l1 * l1_val
            + self.lam_div * pl["div"]
            + self.lam_vor * pl["vor"]
            + self.lam_geo * pl["geo"]
            + self.lam_mse * pl["mse"]
        )

        ld = {
            "total": float(total.detach().item()),
            "l1": float(l1_val.detach().item()),
            "div": float(pl["div"].detach().item()),
            "vor": float(pl["vor"].detach().item()),
            "geo": float(pl["geo"].detach().item()),
            "mse": float(pl["mse"].detach().item()),
        }
        return total, ld


# ============================================================================
# 7) Evaluation Functions
# ============================================================================
@torch.no_grad()
def evaluate_model(model, dataloader, criterion, physics_eval_module, device, num_channels, scaler_y):
    model.eval()

    sum_total = 0.0
    sum_l1 = 0.0
    sum_phys = {k: 0.0 for k in PHYSICS_KEYS}
    n_batches = 0

    total_se_std = np.zeros(num_channels, dtype=np.float64)
    total_px = np.zeros(num_channels, dtype=np.float64)

    for xb, yb in dataloader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        pred = model(xb)
        _, ld = criterion(pred, yb)

        sum_total += ld["total"]
        sum_l1 += ld["l1"]

        if criterion.use_physics:
            for k in PHYSICS_KEYS:
                sum_phys[k] += ld[k]
        else:
            pl = physics_eval_module(pred, yb)
            for k in PHYSICS_KEYS:
                sum_phys[k] += float(pl[k].detach().item())

        diff = pred - yb
        se = (diff ** 2).sum(dim=(0, 2, 3)).detach().cpu().numpy().astype(np.float64)
        px = float(diff.shape[0] * diff.shape[2] * diff.shape[3])
        total_se_std += se
        total_px += px

        n_batches += 1

    nb = max(n_batches, 1)
    avg_total = sum_total / nb
    avg_l1 = sum_l1 / nb
    avg_phys = {k: sum_phys[k] / nb for k in PHYSICS_KEYS}

    std_y = scaler_y.std.astype(np.float64)
    rmse_std = np.sqrt(total_se_std / np.maximum(total_px, 1.0))
    rmse_per_ch = rmse_std * std_y

    total_se_orig = np.sum(total_se_std * (std_y ** 2))
    total_px_all = np.sum(total_px)
    overall_rmse = np.sqrt(total_se_orig / max(total_px_all, 1.0))

    return avg_total, avg_l1, avg_phys, rmse_per_ch, overall_rmse


@torch.no_grad()
def evaluate_bicubic_baseline(dataloader, device, num_channels, scaler_x, scaler_y, upscale_factor):
    l1_fn = nn.L1Loss()

    alpha = torch.tensor(
        (scaler_x.std / scaler_y.std).reshape(1, -1, 1, 1),
        dtype=torch.float32,
        device=device
    )
    beta = torch.tensor(
        ((scaler_x.mean - scaler_y.mean) / scaler_y.std).reshape(1, -1, 1, 1),
        dtype=torch.float32,
        device=device
    )

    l1_sum = 0.0
    n_batches = 0
    total_se_std = np.zeros(num_channels, dtype=np.float64)
    total_px = np.zeros(num_channels, dtype=np.float64)

    for xb, yb in dataloader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        out = F.interpolate(xb, scale_factor=upscale_factor, mode="bicubic", align_corners=False)
        out = out * alpha + beta   # Adjust standardized space from X to Y

        l1_sum += l1_fn(out, yb).item()
        n_batches += 1

        diff = out - yb
        se = (diff ** 2).sum(dim=(0, 2, 3)).detach().cpu().numpy().astype(np.float64)
        px = float(diff.shape[0] * diff.shape[2] * diff.shape[3])
        total_se_std += se
        total_px += px

    avg_l1 = l1_sum / max(n_batches, 1)

    std_y = scaler_y.std.astype(np.float64)
    rmse_std = np.sqrt(total_se_std / np.maximum(total_px, 1.0))
    rmse_per_ch = rmse_std * std_y

    total_se_orig = np.sum(total_se_std * (std_y ** 2))
    total_px_all = np.sum(total_px)
    overall_rmse = np.sqrt(total_se_orig / max(total_px_all, 1.0))

    return avg_l1, rmse_per_ch, overall_rmse


# ============================================================================
# 8) Data Ingestion
# ============================================================================
def load_nc_data(x_path, y_path, variables, return_coords=False):
    print(f"Loading X dataset: {x_path}")
    ds_x = xr.open_dataset(x_path)
    X = np.stack([ds_x[v].values for v in variables], axis=1).astype(np.float32)
    ds_x.close()

    print(f"Loading Y dataset: {y_path}")
    ds_y = xr.open_dataset(y_path)
    Y = np.stack([ds_y[v].values for v in variables], axis=1).astype(np.float32)

    lat = ds_y["latitude"].values.astype(np.float64)
    lon = ds_y["longitude"].values.astype(np.float64)
    ds_y.close()

    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Time dimension mismatch: X={X.shape[0]}, Y={Y.shape[0]}")
    print(f"X shape={X.shape}, Y shape={Y.shape}")

    if return_coords:
        return X, Y, lat, lon
    return X, Y


# ============================================================================
# 9) Visualization Utilities
# ============================================================================
def plot_training_curves(
    epoch_idx,
    train_total_hist, valid_total_hist, test_total_hist,
    train_l1_hist, valid_l1_hist, test_l1_hist,
    valid_phys_hist,
    in_h, in_w, out_h, out_w,
    loss_plot_path, physics_plot_path,
    best_epoch=None
):
    n = epoch_idx + 1
    epochs = range(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(epochs, train_total_hist[:n], label="Train Total")
    ax1.plot(epochs, valid_total_hist[:n], label="Valid Total")
    ax1.plot(epochs, test_total_hist[:n], label="Test Total")
    if best_epoch is not None:
        ax1.axvline(best_epoch, color="red", ls="--", alpha=0.5, label=f"Best {best_epoch}")
    ax1.set_title(f"Total Loss ({in_h}x{in_w} -> {out_h}x{out_w})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, train_l1_hist[:n], label="Train L1")
    ax2.plot(epochs, valid_l1_hist[:n], label="Valid L1")
    ax2.plot(epochs, test_l1_hist[:n], label="Test L1")
    if best_epoch is not None:
        ax2.axvline(best_epoch, color="red", ls="--", alpha=0.5, label=f"Best {best_epoch}")
    ax2.set_title("L1 Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=250)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    titles = {
        "div": "Divergence",
        "vor": "Vorticity",
        "geo": "Geostrophic Balance (Ageostrophic, Lat-Weighted)",
        "mse": "Moist Static Energy Gradient",
    }

    for i, k in enumerate(PHYSICS_KEYS):
        axes[i].plot(epochs, valid_phys_hist[k][:n], color="tab:orange")
        if best_epoch is not None:
            axes[i].axvline(best_epoch, color="red", ls="--", alpha=0.5)
        axes[i].set_title(f"Validation Physics: {titles[k]}")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("GRE")
        axes[i].grid(True, alpha=0.3)
        axes[i].ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

    plt.tight_layout()
    plt.savefig(physics_plot_path, dpi=250)
    plt.close()


# ============================================================================
# 10) Single Experiment Execution Loop
# ============================================================================
def run_single_experiment(
    exp_cfg,
    train_dataset, valid_dataset, test_dataset,
    scaler_x, scaler_y,
    lat_hr, lon_hr,
    input_h, input_w, output_h, output_w,
    baseline_valid, baseline_test,
    device
):
    exp_name = exp_cfg["name"]
    use_physics = bool(exp_cfg["use_physics"])
    use_bn = bool(exp_cfg.get("use_bn", False))
    lambdas = exp_cfg["lambdas"]

    set_seed(RANDOM_SEED)

    run_dir = os.path.join(OUTPUT_ROOT, exp_name)
    os.makedirs(run_dir, exist_ok=True)

    model_save_path = os.path.join(run_dir, "best_model.pth")
    history_csv_path = os.path.join(run_dir, "training_history.csv")
    model_info_path = os.path.join(run_dir, "model_info.json")
    loss_plot_path = os.path.join(run_dir, "loss_epoch.png")
    physics_plot_path = os.path.join(run_dir, "physics_epoch.png")

    train_loader, valid_loader, test_loader = build_dataloaders(
        train_dataset, valid_dataset, test_dataset, BATCH_SIZE, NUM_WORKERS, device, RANDOM_SEED
    )

    print("=" * 100)
    print(f"[Experiment] {exp_name}")
    print(f"use_physics={use_physics}, use_bn={use_bn}, lambdas={lambdas}")

    model = ClassicEDSR(
        in_channels=NUM_CHANNELS,
        out_channels=NUM_CHANNELS,
        n_feats=NUM_FEATURES,
        n_resblocks=NUM_RES_BLOCKS,
        scale=UPSCALE_FACTOR,
        res_scale=RES_BLOCK_SCALE,
        use_bn=use_bn
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    physics_module = PhysicsConstraintLoss(
        scaler_y_mean=scaler_y.mean,
        scaler_y_std=scaler_y.std,
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
        physics_module=physics_module,
        use_physics=use_physics,
        lam_l1=lambdas["l1"],
        lam_div=lambdas["div"],
        lam_vor=lambdas["vor"],
        lam_geo=lambdas["geo"],
        lam_mse=lambdas["mse"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )

    print(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")

    best_val = float("inf")
    train_total_hist, valid_total_hist, test_total_hist = [], [], []
    train_l1_hist, valid_l1_hist, test_l1_hist = [], [], []
    valid_phys_hist = {k: [] for k in PHYSICS_KEYS}
    records = []

    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]

        model.train()
        acc = {k: 0.0 for k in ["total", "l1"] + PHYSICS_KEYS}
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            total_loss, ld = criterion(pred, yb)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in acc:
                acc[k] += ld[k]
            n_batches += 1

        train_avg = {k: acc[k] / max(n_batches, 1) for k in acc}

        val_total, val_l1, val_phys, val_rmse, val_overall = evaluate_model(
            model, valid_loader, criterion, physics_module, device, NUM_CHANNELS, scaler_y
        )
        test_total, test_l1, test_phys, test_rmse, test_overall = evaluate_model(
            model, test_loader, criterion, physics_module, device, NUM_CHANNELS, scaler_y
        )

        scheduler.step(val_total)
        new_lr = optimizer.param_groups[0]["lr"]

        # Check for best validation result
        status = "no-update"
        if val_total <= best_val:
            best_val = val_total
            status = "best-updated"
            torch.save({
                "epoch": epoch,
                "best_val_total": float(best_val),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),

                "architecture": "ClassicEDSR",
                "use_bn": use_bn,
                "in_channels": NUM_CHANNELS,
                "out_channels": NUM_CHANNELS,
                "num_features": NUM_FEATURES,
                "num_res_blocks": NUM_RES_BLOCKS,
                "res_block_scale": RES_BLOCK_SCALE,
                "upscale_factor": UPSCALE_FACTOR,

                "use_physics": use_physics,
                "physics_weights": {
                    "lam_l1": criterion.lam_l1,
                    "lam_div": criterion.lam_div,
                    "lam_vor": criterion.lam_vor,
                    "lam_geo": criterion.lam_geo,
                    "lam_mse": criterion.lam_mse
                },
                "eps": {"div": EPS_DIV, "vor": EPS_VOR, "geo": EPS_GEO, "mse": EPS_MSE},
                "geo_weighting": {"mode": GEO_WEIGHT_MODE, "alpha": GEO_WEIGHT_ALPHA, "min": GEO_WEIGHT_MIN},
                "thermo_constants": {"cp_air": CP_AIR, "lv": LV_WATER},

                "scaler_x_mean": scaler_x.mean.tolist(),
                "scaler_x_std": scaler_x.std.tolist(),
                "scaler_y_mean": scaler_y.mean.tolist(),
                "scaler_y_std": scaler_y.std.tolist(),
                "latitudes": lat_hr.tolist(),
                "longitudes": lon_hr.tolist(),

                "experiment_name": exp_name
            }, model_save_path)

        train_total_hist.append(train_avg["total"])
        valid_total_hist.append(val_total)
        test_total_hist.append(test_total)
        train_l1_hist.append(train_avg["l1"])
        valid_l1_hist.append(val_l1)
        test_l1_hist.append(test_l1)
        for k in PHYSICS_KEYS:
            valid_phys_hist[k].append(val_phys[k])

        rec = {
            "epoch": epoch,
            "lr": float(lr),
            "train_total": float(train_avg["total"]),
            "valid_total": float(val_total),
            "test_total": float(test_total),
            "train_l1": float(train_avg["l1"]),
            "valid_l1": float(val_l1),
            "test_l1": float(test_l1),
            "valid_div": float(val_phys["div"]),
            "valid_vor": float(val_phys["vor"]),
            "valid_geo": float(val_phys["geo"]),
            "valid_mse": float(val_phys["mse"]),
            "test_div": float(test_phys["div"]),
            "test_vor": float(test_phys["vor"]),
            "test_geo": float(test_phys["geo"]),
            "test_mse": float(test_phys["mse"]),
            "valid_rmse_overall": float(val_overall),
            "test_rmse_overall": float(test_overall),
            "use_physics": int(use_physics),
            "use_bn": int(use_bn),
        }
        for i, name in enumerate(VAR_NAMES):
            rec[f"valid_rmse_{name}"] = float(val_rmse[i])
            rec[f"test_rmse_{name}"] = float(test_rmse[i])
        records.append(rec)

        epoch_sec = time.time() - epoch_t0
        remain_sec = epoch_sec * (NUM_EPOCHS - epoch)

        print("-" * 100)
        print(
            f"[{exp_name}] Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"train_total={train_avg['total']:.4f} valid_total={val_total:.4f} test_total={test_total:.4f} | "
            f"train_l1={train_avg['l1']:.4f} valid_l1={val_l1:.4f} test_l1={test_l1:.4f}"
        )
        print(f"status={status}, lr={lr:.6f}, epoch_time={epoch_sec:.1f}s, remaining~{remain_sec/60:.1f}m")
        print(f"valid physics: {format_physics_str(val_phys)}")
        print(f"valid RMSE: {format_rmse_str(val_rmse, VAR_NAMES, val_overall)}")
        print(f"test  RMSE: {format_rmse_str(test_rmse, VAR_NAMES, test_overall)}")
        if new_lr != lr:
            print(f"[LR Decay] {lr:.6f} -> {new_lr:.6f}")

        if epoch % 5 == 0:
            plot_training_curves(
                epoch_idx=epoch - 1,
                train_total_hist=train_total_hist,
                valid_total_hist=valid_total_hist,
                test_total_hist=test_total_hist,
                train_l1_hist=train_l1_hist,
                valid_l1_hist=valid_l1_hist,
                test_l1_hist=test_l1_hist,
                valid_phys_hist=valid_phys_hist,
                in_h=input_h, in_w=input_w, out_h=output_h, out_w=output_w,
                loss_plot_path=loss_plot_path,
                physics_plot_path=physics_plot_path,
                best_epoch=None
            )

    # Wrap up training
    elapsed = time.time() - t_start
    df_hist = pd.DataFrame(records)
    df_hist.to_csv(history_csv_path, index=False)

    best_idx = int(df_hist["valid_total"].values.argmin())
    best_row = df_hist.iloc[best_idx]
    best_epoch = int(best_row["epoch"])

    plot_training_curves(
        epoch_idx=NUM_EPOCHS - 1,
        train_total_hist=train_total_hist,
        valid_total_hist=valid_total_hist,
        test_total_hist=test_total_hist,
        train_l1_hist=train_l1_hist,
        valid_l1_hist=valid_l1_hist,
        test_l1_hist=test_l1_hist,
        valid_phys_hist=valid_phys_hist,
        in_h=input_h, in_w=input_w, out_h=output_h, out_w=output_w,
        loss_plot_path=loss_plot_path,
        physics_plot_path=physics_plot_path,
        best_epoch=best_epoch
    )

    b_valid_l1, b_valid_rmse, b_valid_overall = baseline_valid
    b_test_l1, b_test_rmse, b_test_overall = baseline_test

    model_info = {
        "experiment_name": exp_name,
        "architecture": "ClassicEDSR",
        "use_bn_resblock": bool(use_bn),
        "use_physics": bool(use_physics),

        "variables": VARIABLES,
        "variable_units": VAR_UNITS,

        "in_channels": NUM_CHANNELS,
        "out_channels": NUM_CHANNELS,
        "num_features": NUM_FEATURES,
        "num_res_blocks": NUM_RES_BLOCKS,
        "res_block_scale": RES_BLOCK_SCALE,
        "upscale_factor": UPSCALE_FACTOR,

        "input_size": f"{input_h}x{input_w}",
        "output_size": f"{output_h}x{output_w}",
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),

        "loss_function": "L1 only" if not use_physics else "L1 + Physics(GRE)",
        "physics_weights": {
            "lambda_l1": float(criterion.lam_l1),
            "lambda_div": float(criterion.lam_div),
            "lambda_vor": float(criterion.lam_vor),
            "lambda_geo": float(criterion.lam_geo),
            "lambda_mse": float(criterion.lam_mse),
        },
        "epsilon_values": {
            "eps_div": EPS_DIV, "eps_vor": EPS_VOR, "eps_geo": EPS_GEO, "eps_mse": EPS_MSE
        },
        "geo_weighting": {"mode": GEO_WEIGHT_MODE, "alpha": GEO_WEIGHT_ALPHA, "min": GEO_WEIGHT_MIN},

        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "random_seed": RANDOM_SEED,

        "best_epoch": best_epoch,
        "best_valid_total": float(best_row["valid_total"]),
        "best_valid_l1": float(best_row["valid_l1"]),
        "best_valid_rmse_overall": float(best_row["valid_rmse_overall"]),
        "best_test_rmse_overall": float(best_row["test_rmse_overall"]),

        "bicubic_baseline_valid_l1": float(b_valid_l1),
        "bicubic_baseline_test_l1": float(b_test_l1),
        "bicubic_baseline_valid_rmse_overall": float(b_valid_overall),
        "bicubic_baseline_test_rmse_overall": float(b_test_overall),

        "training_time_sec": float(round(elapsed, 1)),
        "device": str(device),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    result = {
        "experiment": exp_name,
        "use_physics": int(use_physics),
        "use_bn": int(use_bn),
        "best_epoch": best_epoch,
        "best_valid_total": float(best_row["valid_total"]),
        "best_valid_l1": float(best_row["valid_l1"]),
        "best_valid_rmse_overall": float(best_row["valid_rmse_overall"]),
        "best_test_rmse_overall": float(best_row["test_rmse_overall"]),
        "best_valid_div": float(best_row["valid_div"]),
        "best_valid_vor": float(best_row["valid_vor"]),
        "best_valid_geo": float(best_row["valid_geo"]),
        "best_valid_mse": float(best_row["valid_mse"]),
        "model_path": model_save_path,
        "history_csv": history_csv_path,
        "model_info_json": model_info_path,
    }

    print("=" * 100)
    print(f"[Completed] {exp_name} | best_epoch={best_epoch}, best_valid_total={best_row['valid_total']:.6f}")
    print(f"Output directory: {run_dir}")
    print("=" * 100)
    return result


# ============================================================================
# 11) Main Execution Function
# ============================================================================
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("ERA5 Super-Resolution | Classic EDSR + Physics GRE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print("=" * 100)

    if CHECK_MISSING:
        all_files = [X_TRAIN_PATH, Y_TRAIN_PATH, X_VALID_PATH, Y_VALID_PATH, X_TEST_PATH, Y_TEST_PATH]
        check_missing_values(all_files, VARIABLES)

    # 1) Load NetCDF datasets
    X_train_raw, Y_train_raw, lat_hr, lon_hr = load_nc_data(X_TRAIN_PATH, Y_TRAIN_PATH, VARIABLES, return_coords=True)
    X_valid_raw, Y_valid_raw = load_nc_data(X_VALID_PATH, Y_VALID_PATH, VARIABLES)
    X_test_raw, Y_test_raw = load_nc_data(X_TEST_PATH, Y_TEST_PATH, VARIABLES)

    in_h, in_w = X_train_raw.shape[2], X_train_raw.shape[3]
    out_h, out_w = Y_train_raw.shape[2], Y_train_raw.shape[3]
    actual_scale = out_h // in_h

    if UPSCALE_FACTOR != actual_scale:
        raise ValueError(
            f"UPSCALE_FACTOR ({UPSCALE_FACTOR}) does not match spatial scaling factor ({actual_scale}).\n"
            f"Current dimensions: {in_h}x{in_w} -> {out_h}x{out_w}\n"
            "For 4x upscaling, use 25x25 input files (e.g., X4_*.nc) and set LOW_RES_PREFIX='X4'."
        )

    print(f"Input size: {in_h}x{in_w} | Output size: {out_h}x{out_w} | Scale factor: {actual_scale}x")
    print(f"Variables: {VARIABLES}")
    print(f"Train/Valid/Test sizes: {X_train_raw.shape[0]:,}/{X_valid_raw.shape[0]:,}/{X_test_raw.shape[0]:,}")
    print(f"High-Res Latitude range: [{lat_hr.min():.2f}, {lat_hr.max():.2f}]")
    print(f"High-Res Longitude range: [{lon_hr.min():.2f}, {lon_hr.max():.2f}]")
    print("=" * 100)

    # 2) Channel-wise standardization
    scaler_x = ChannelScaler()
    scaler_y = ChannelScaler()

    X_train = scaler_x.fit_transform(X_train_raw)
    X_valid = scaler_x.transform(X_valid_raw)
    X_test = scaler_x.transform(X_test_raw)

    Y_train = scaler_y.fit_transform(Y_train_raw)
    Y_valid = scaler_y.transform(Y_valid_raw)
    Y_test = scaler_y.transform(Y_test_raw)

    print("Scaler X:", scaler_x)
    print("Scaler Y:", scaler_y)
    for i, (n, u) in enumerate(zip(VAR_NAMES, VAR_UNITS)):
        print(f"{n} std={scaler_y.std[i]:.6f} ({u})")

    joblib.dump({"scaler_X": scaler_x, "scaler_Y": scaler_y}, SCALER_SAVE_PATH)
    print(f"Scalers saved to: {SCALER_SAVE_PATH}")
    print("=" * 100)

    # 3) Build datasets
    train_ds = ERA5Dataset(X_train, Y_train)
    valid_ds = ERA5Dataset(X_valid, Y_valid)
    test_ds = ERA5Dataset(X_test, Y_test)

    # Free memory
    del X_train_raw, Y_train_raw, X_valid_raw, Y_valid_raw, X_test_raw, Y_test_raw
    del X_train, Y_train, X_valid, Y_valid, X_test, Y_test
    gc.collect()

    print(f"Datasets ready: train={len(train_ds):,}, valid={len(valid_ds):,}, test={len(test_ds):,}")
    print("=" * 100)

    # 4) Evaluate Bicubic Baseline
    _, valid_loader_base, test_loader_base = build_dataloaders(
        train_ds, valid_ds, test_ds, BATCH_SIZE, NUM_WORKERS, device, RANDOM_SEED
    )
    b_valid_l1, b_valid_rmse, b_valid_overall = evaluate_bicubic_baseline(
        valid_loader_base, device, NUM_CHANNELS, scaler_x, scaler_y, UPSCALE_FACTOR
    )
    b_test_l1, b_test_rmse, b_test_overall = evaluate_bicubic_baseline(
        test_loader_base, device, NUM_CHANNELS, scaler_x, scaler_y, UPSCALE_FACTOR
    )

    print(f"[Valid Bicubic] L1={b_valid_l1:.6f} | {format_rmse_str(b_valid_rmse, VAR_NAMES, b_valid_overall)}")
    print(f"[Test  Bicubic] L1={b_test_l1:.6f} | {format_rmse_str(b_test_rmse, VAR_NAMES, b_test_overall)}")
    print("=" * 100)

    baseline_valid = (b_valid_l1, b_valid_rmse, b_valid_overall)
    baseline_test = (b_test_l1, b_test_rmse, b_test_overall)

    # 5) Run defined experiments
    all_results = []
    t0 = time.time()
    for exp in EXPERIMENTS:
        r = run_single_experiment(
            exp_cfg=exp,
            train_dataset=train_ds,
            valid_dataset=valid_ds,
            test_dataset=test_ds,
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            lat_hr=lat_hr,
            lon_hr=lon_hr,
            input_h=in_h,
            input_w=in_w,
            output_h=out_h,
            output_w=out_w,
            baseline_valid=baseline_valid,
            baseline_test=baseline_test,
            device=device
        )
        all_results.append(r)

    # 6) Export ablation experiment summary
    df = pd.DataFrame(all_results).sort_values("best_valid_total", ascending=True).reset_index(drop=True)
    df.to_csv(ABLATION_CSV_PATH, index=False)

    elapsed = time.time() - t0
    print("=" * 100)
    print("All experiments completed.")
    print(f"Ablation summary saved to: {ABLATION_CSV_PATH}")
    print(df[[
        "experiment", "use_physics", "use_bn", "best_epoch",
        "best_valid_total", "best_valid_l1",
        "best_valid_rmse_overall", "best_test_rmse_overall",
        "best_valid_div", "best_valid_vor", "best_valid_geo", "best_valid_mse"
    ]].to_string(index=False))
    print(f"Total processing time: {elapsed/60:.1f} minutes")
    print("=" * 100)


if __name__ == "__main__":
    main()