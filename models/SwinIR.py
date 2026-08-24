#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA5 Super-Resolution Training Pipeline (SwinIR Fast + Physics Ablation)

Key Features:
1) Model Architecture: SwinIR (Swin Transformer backbone with shifted window mechanism).
2) Scale Factors: Supports x2 (50 -> 100) and x4 (25 -> 100) spatial upscaling.
3) Physics Ablation Study:
   - swinir_l1_only: Pure data-driven L1 loss baseline.
   - swinir_l1_physics: L1 loss constrained by Global Relative Error (GRE) physics losses.
4) Computational Efficiency Optimizations:
   - Lightweight SwinIR configurations for fast iteration.
   - Automatic Mixed Precision (AMP) support (fp16 / bf16).
   - TensorFloat-32 (TF32) support on Ampere+ GPUs.
   - Attention mask caching to prevent redundant tensor allocations.
   - Omission of full-dataset training evaluation per epoch.
   - Periodic test dataset evaluation (every N epochs and on best valid epoch).
"""

import os
import gc
import json
import time
import math
import random
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint as checkpoint
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 0) Configuration & Hyperparameters
# =============================================================================
RANDOM_SEED = 42
CHECK_MISSING = False

# -----------------------------
# Upscaling Factor (2 or 4)
# -----------------------------
UPSCALE_FACTOR = 2  # Options: 2 or 4

X_PREFIX = "X2" if UPSCALE_FACTOR == 2 else "X4"

X_TRAIN_PATH = f"{X_PREFIX}_train.nc"
Y_TRAIN_PATH = "Y_train.nc"
X_VALID_PATH = f"{X_PREFIX}_valid.nc"
Y_VALID_PATH = "Y_valid.nc"
X_TEST_PATH = f"{X_PREFIX}_test.nc"
Y_TEST_PATH = "Y_test.nc"

VARIABLES = ["z", "q", "t", "u", "v"]
VAR_NAMES = ["Z", "Q", "T", "U", "V"]
VAR_UNITS = ["m^2/s^2", "kg/kg", "K", "m/s", "m/s"]
NUM_CHANNELS = len(VARIABLES)
PHYSICS_KEYS = ["div", "vor", "geo", "mse"]

# -----------------------------
# Speed Preset Configuration
# -----------------------------
SPEED_PRESET = "balanced"  # Options: "fast" / "balanced"

if SPEED_PRESET == "fast":
    NUM_EPOCHS = 30
    BATCH_SIZE = 16
    NUM_WORKERS = 4

    # Lightweight SwinIR configuration
    SWINIR_EMBED_DIM = 48
    SWINIR_DEPTHS = [4, 4]
    SWINIR_NUM_HEADS = [4, 4]
    SWINIR_WINDOW_SIZE = 5
    SWINIR_MLP_RATIO = 2.0
    SWINIR_DROP_RATE = 0.0
    SWINIR_ATTN_DROP_RATE = 0.0
    SWINIR_DROP_PATH_RATE = 0.05

elif SPEED_PRESET == "balanced":
    NUM_EPOCHS = 60
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    SWINIR_EMBED_DIM = 60
    SWINIR_DEPTHS = [6, 6, 6, 6]
    SWINIR_NUM_HEADS = [6, 6, 6, 6]
    SWINIR_WINDOW_SIZE = 5
    SWINIR_MLP_RATIO = 2.0
    SWINIR_DROP_RATE = 0.0
    SWINIR_ATTN_DROP_RATE = 0.0
    SWINIR_DROP_PATH_RATE = 0.10
else:
    raise ValueError(f"Unknown SPEED_PRESET: {SPEED_PRESET}")

# -----------------------------
# Training Settings
# -----------------------------
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 3
PLOT_EVERY = 5
EVAL_TEST_EVERY = 5  # Evaluate on test set every N epochs and on best validation epoch

# -----------------------------
# SwinIR Architecture Settings
# -----------------------------
SWINIR_QKV_BIAS = True
SWINIR_PATCH_NORM = True
SWINIR_USE_CHECKPOINT = False
SWINIR_UPSAMPLER = "pixelshuffledirect"
SWINIR_RESI_CONNECTION = "1conv"
ZERO_INIT_LAST = True

USE_BICUBIC_BASELINE = True
BICUBIC_ALIGN_CORNERS = False
RESIDUAL_SCALE = 1.0

# -----------------------------
# Hardware & Acceleration Settings
# -----------------------------
USE_AMP = True                 # Enable Automatic Mixed Precision on CUDA
AMP_DTYPE = "fp16"             # Options: "fp16" or "bf16"
ENABLE_TF32 = True             # Recommended True for Ampere+ architectures
REPRODUCIBLE = False           # Set True for strict determinism (slower execution)
CUDNN_BENCHMARK = True
USE_ZERO_GRAD_SET_TO_NONE = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# -----------------------------
# Loss Function Weights
# -----------------------------
LAMBDA_L1 = 1.0
LAMBDA_DIV = 0.1
LAMBDA_VOR = 0.1
LAMBDA_GEO = 0.1
LAMBDA_MSE = 0.1

AUTO_TUNE_LAMBDA_GEO = False
GEO_TARGET_REL_TO_L1 = 0.10
GEO_LAMBDA_MIN = 1e-3
GEO_LAMBDA_MAX = 2.0

# -----------------------------
# Physical Constants & Tolerances
# -----------------------------
CP_AIR = 1004.0
LV_WATER = 2.501e6

EPS_DIV = 1e-8
EPS_VOR = 1e-8
EPS_GEO = 1e-4
EPS_MSE = 1e-4

GEO_WEIGHT_MODE = "sin_global"  # Options: ['f_ratio', 'sin_global', 'cos', 'one_minus_cos']
GEO_WEIGHT_ALPHA = 1.0
GEO_WEIGHT_MIN = 0.10

# -----------------------------
# Ablation Experiment Matrix
# -----------------------------
EXPERIMENTS = [
    # {
    #     "name": "swinir_l1_only",
    #     "use_physics": False,
    #     "lam_l1": 1.0,
    #     "lam_div": 0.0,
    #     "lam_vor": 0.0,
    #     "lam_geo": 0.0,
    #     "lam_mse": 0.0,
    # },
    {
        "name": "swinir_l1_physics",
        "use_physics": True,
        "lam_l1": LAMBDA_L1,
        "lam_div": LAMBDA_DIV,
        "lam_vor": LAMBDA_VOR,
        "lam_geo": LAMBDA_GEO,
        "lam_mse": LAMBDA_MSE,
    },
]

OUTPUT_ROOT = f"outputs_swinir_ablation_scale{UPSCALE_FACTOR}_{SPEED_PRESET}"


# =============================================================================
# 1) Utility Functions
# =============================================================================
def set_seed(seed: int, deterministic: bool = False):
    """Set random seed across python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = CUDNN_BENCHMARK


def setup_acceleration():
    """Configure hardware acceleration features (TF32, float32 precision)."""
    if torch.cuda.is_available() and ENABLE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


def get_amp_dtype(name: str):
    """Retrieve PyTorch precision dtype for AMP."""
    name = name.lower().strip()
    if name == "bf16":
        return torch.bfloat16
    return torch.float16


def autocast_ctx(amp_enabled: bool, amp_dtype: torch.dtype):
    """Context manager for automatic mixed precision."""
    if amp_enabled:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def format_value(val):
    """Format numerical values for logging output."""
    abs_val = abs(val)
    if abs_val >= 100:
        return f"{val:.2f}"
    elif abs_val >= 0.01:
        return f"{val:.4f}"
    else:
        return f"{val:.4e}"


def format_rmse_str(rmse_array, var_names, overall_rmse):
    """Format variable-wise and overall RMSE into a formatted string."""
    parts = [f"{name}: {format_value(rmse_array[i])}" for i, name in enumerate(var_names)]
    return ", ".join(parts) + f", Overall: {format_value(overall_rmse)}"


def format_physics_str(phys_dict):
    """Format physics loss components into a formatted string."""
    items = []
    for k in PHYSICS_KEYS:
        v = phys_dict[k]
        if np.isnan(v):
            items.append(f"{k}=nan")
        elif v >= 0.01:
            items.append(f"{k}={v:.4f}")
        else:
            items.append(f"{k}={v:.2e}")
    return ", ".join(items)


def suggest_lambda_geo(diag_loss_dict, lam_l1, target_rel_to_l1, lam_min, lam_max):
    """Suggest an adjusted geostrophic loss weight based on initial diagnostic step."""
    l1 = max(float(diag_loss_dict["l1"]), 1e-12)
    geo = max(float(diag_loss_dict["geo"]), 1e-12)
    raw = (target_rel_to_l1 * float(lam_l1) * l1) / geo
    clipped = float(np.clip(raw, lam_min, lam_max))
    return raw, clipped


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 2) Data Processing & Loading
# =============================================================================
def check_missing_values(file_paths, variables, missing_value=-999):
    """Check dataset files for missing values (-999 or NaN)."""
    print("Checking NetCDF files for missing values (-999 / NaN)...")
    found_missing = False

    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        ds = xr.open_dataset(path, decode_times=False)
        file_has_issue = False

        for var in variables:
            data = ds[var].values
            miss_count = int(np.sum(np.isclose(data, missing_value, atol=0.1)))
            nan_count = int(np.sum(np.isnan(data)))

            if miss_count > 0:
                print(f"  [Warning] {path} | {var}: -999 count={miss_count:,}")
                file_has_issue = True
                found_missing = True
            if nan_count > 0:
                print(f"  [Warning] {path} | {var}: NaN count={nan_count:,}")
                file_has_issue = True
                found_missing = True

        if not file_has_issue:
            print(f"  [Passed] {path}")
        ds.close()

    if found_missing:
        raise ValueError("Dataset contains missing values. Please clean data prior to training.")

    print("  [Result] All files passed missing value checks.")


class ChannelScaler:
    """Channel-wise standard scaler for multi-variable atmospheric data."""
    def __init__(self):
        self.mean = None
        self.std = None
        self.n_channels = None

    def fit(self, data):
        # Expects shape: (N, C, H, W)
        self.n_channels = data.shape[1]
        self.mean = data.mean(axis=(0, 2, 3)).astype(np.float64)
        self.std = data.std(axis=(0, 2, 3)).astype(np.float64)
        self.std[self.std < 1e-10] = 1.0
        return self

    def transform(self, data):
        mean = self.mean.reshape(1, -1, 1, 1).astype(np.float32)
        std = self.std.reshape(1, -1, 1, 1).astype(np.float32)
        data = data.astype(np.float32, copy=False)
        return ((data - mean) / std).astype(np.float32)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def __repr__(self):
        if self.mean is None:
            return "ChannelScaler(Unfitted)"
        mean_str = ", ".join([f"{m:.4f}" for m in self.mean])
        std_str = ", ".join([f"{s:.4f}" for s in self.std])
        return f"ChannelScaler(mean=[{mean_str}], std=[{std_str}])"


class ERA5Dataset(Dataset):
    """PyTorch Dataset wrapper for ERA5 standard tensor pairs."""
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)  # float32
        self.Y = torch.from_numpy(Y)  # float32

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_nc_data(x_path, y_path, variables, return_coords=False):
    """Load NetCDF data into numpy arrays."""
    print(f"  Loading X: {x_path}")
    ds_x = xr.open_dataset(x_path, decode_times=False)
    X = np.stack([ds_x[v].values for v in variables], axis=1).astype(np.float32)
    ds_x.close()

    print(f"  Loading Y: {y_path}")
    ds_y = xr.open_dataset(y_path, decode_times=False)
    Y = np.stack([ds_y[v].values for v in variables], axis=1).astype(np.float32)
    latitudes = ds_y["latitude"].values.astype(np.float64)
    longitudes = ds_y["longitude"].values.astype(np.float64)
    ds_y.close()

    assert X.shape[0] == Y.shape[0], f"Temporal mismatch: X={X.shape[0]}, Y={Y.shape[0]}"
    print(f"       X shape: {X.shape} | Y shape: {Y.shape}")

    if return_coords:
        return X, Y, latitudes, longitudes
    return X, Y


# =============================================================================
# 3) SwinIR Model Components
# =============================================================================
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or (not training):
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rnd = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    rnd.floor_()
    return x.div(keep_prob) * rnd


class DropPath(nn.Module):
    """DropPath module wrapper."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """Multilayer Perceptron for Swin Transformer."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


def window_partition(x, window_size):
    """Partition tensor into non-overlapping windows."""
    # x: (B, H, W, C)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reconstruct spatial tensor from partitioned windows."""
    # windows: (num_windows*B, ws, ws, C)
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based Multi-head Self-Attention (W-MSA) with Relative Position Bias."""
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or (head_dim ** -0.5)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1)
        relative_position_index = relative_coords.sum(-1)  # (N, N)

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        # x: (nW*B, N, C)
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, nH, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, nH, N, N)

        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rel_bias = rel_bias.view(N, N, -1).permute(2, 0, 1).contiguous()  # (nH, N, N)
        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path_prob=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.mlp_ratio = float(mlp_ratio)

        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * self.mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, act_layer=act_layer, drop=drop)

        # Cache dynamic attention mask to minimize compute overhead
        self._attn_mask = None
        self._attn_mask_size = None
        self._attn_mask_device = None

    def calculate_mask(self, x_size, device):
        H, W = x_size
        ws = self.window_size
        ss = self.shift_size

        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, ws)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, f"L mismatch: {L} vs {H*W}"
        assert H % self.window_size == 0 and W % self.window_size == 0, \
            f"x_size={x_size} must be divisible by window_size={self.window_size}"

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            need_refresh = (
                self._attn_mask is None
                or self._attn_mask_size != (H, W)
                or self._attn_mask_device != x.device
            )
            if need_refresh:
                self._attn_mask = self.calculate_mask((H, W), x.device)
                self._attn_mask_size = (H, W)
                self._attn_mask_device = x.device
            attn_mask = self._attn_mask
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, ws, ws, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    """Basic Swin Transformer Layer containing multiple Swin Transformer Blocks."""
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList()

        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path_prob=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer
                )
            )

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(lambda _x: blk(_x, x_size), x)
            else:
                x = blk(x, x_size)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, embed_dim, norm_layer=None):
        super().__init__()
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        # (B, C, H, W) -> (B, HW, C)
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """Patch to Image Unembedding."""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        # (B, HW, C) -> (B, C, H, W)
        B, HW, C = x.shape
        H, W = x_size
        return x.transpose(1, 2).contiguous().view(B, C, H, W)


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB)."""
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm, use_checkpoint=False,
                 resi_connection="1conv"):
        super().__init__()
        self.residual_group = BasicLayer(
            dim=dim, depth=depth, num_heads=num_heads, window_size=window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            norm_layer=norm_layer, use_checkpoint=use_checkpoint
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unsupported resi_connection: {resi_connection}")

        self.patch_embed = PatchEmbed(embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

    def forward(self, x, x_size):
        out = self.residual_group(x, x_size)
        out = self.patch_unembed(out, x_size)
        out = self.conv(out)
        out = self.patch_embed(out)
        return out + x


class Upsample(nn.Sequential):
    """Multi-stage PixelShuffle Upsampling Module."""
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported scale factor: {scale}")
        super().__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """Direct One-Step PixelShuffle Upsampling Module."""
    def __init__(self, scale, num_feat, num_out_ch):
        super().__init__(
            nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale),
        )


class SwinIRBackbone(nn.Module):
    """
    SwinIR Backbone Architecture.
    Processes low-resolution inputs and outputs high-resolution reconstructed fields/residuals.
    """
    def __init__(
        self,
        in_chans=5,
        out_chans=5,
        embed_dim=48,
        depths=(4, 4, 4),
        num_heads=(4, 4, 4),
        window_size=5,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        upscale=2,
        upsampler="pixelshuffledirect",
        resi_connection="1conv",
        zero_init_last=True
    ):
        super().__init__()
        assert len(depths) == len(num_heads), "Length of depths and num_heads must match."
        if upscale not in [2, 4]:
            raise ValueError("Supported upscale factors are 2 or 4.")

        self.upscale = int(upscale)
        self.window_size = int(window_size)
        self.upsampler = upsampler

        # 1) Shallow feature extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 2) Deep feature extraction
        self.patch_embed = PatchEmbed(embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)
        self.patch_unembed = PatchUnEmbed(embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist() if sum(depths) > 0 else []
        dpr_ptr = 0
        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = RSTB(
                dim=embed_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[dpr_ptr:dpr_ptr + depths[i]],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                resi_connection=resi_connection
            )
            self.layers.append(layer)
            dpr_ptr += depths[i]

        self.norm = norm_layer(embed_dim)

        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        else:
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        # 3) High-resolution reconstruction
        if upsampler == "pixelshuffle":
            num_feat = 64
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.upsample = Upsample(self.upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, out_chans, 3, 1, 1)
        elif upsampler == "pixelshuffledirect":
            self.upsample = UpsampleOneStep(self.upscale, embed_dim, out_chans)
        else:
            raise ValueError(f"Unsupported upsampler: {upsampler}")

        self.apply(self._init_weights)

        if zero_init_last:
            self._zero_init_last_layer()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _zero_init_last_layer(self):
        if self.upsampler == "pixelshuffle":
            nn.init.zeros_(self.conv_last.weight)
            if self.conv_last.bias is not None:
                nn.init.zeros_(self.conv_last.bias)
        elif self.upsampler == "pixelshuffledirect":
            last_conv = self.upsample[0]
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)

    def check_image_size(self, x):
        _, _, h, w = x.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        x = self.check_image_size(x)

        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x

        if self.upsampler == "pixelshuffle":
            x = self.conv_before_upsample(x)
            x = self.upsample(x)
            x = self.conv_last(x)
        else:
            x = self.upsample(x)

        return x[:, :, :H * self.upscale, :W * self.upscale]


class SwinIRSRModel(nn.Module):
    """
    SwinIR Super-Resolution Model in Standardized Space.
    If use_bicubic_baseline=True:
        output = bicubic_interpolated_base + learnable_residual
    Else:
        output = direct_network_prediction
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        upscale_factor,
        scaler_X_mean, scaler_X_std,
        scaler_Y_mean, scaler_Y_std,
        use_bicubic_baseline=True,
        bicubic_align_corners=False,
        residual_scale=1.0,
        embed_dim=48,
        depths=(4, 4, 4),
        num_heads=(4, 4, 4),
        window_size=5,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        patch_norm=True,
        use_checkpoint=False,
        upsampler="pixelshuffledirect",
        resi_connection="1conv",
        zero_init_last=True,
    ):
        super().__init__()
        self.upscale_factor = int(upscale_factor)
        self.use_bicubic_baseline = bool(use_bicubic_baseline)
        self.align_corners = bool(bicubic_align_corners)
        self.residual_scale = float(residual_scale)

        self.backbone = SwinIRBackbone(
            in_chans=in_channels,
            out_chans=out_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            upscale=upscale_factor,
            upsampler=upsampler,
            resi_connection=resi_connection,
            zero_init_last=zero_init_last,
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
        branch = self.backbone(x_std) * self.residual_scale
        if self.use_bicubic_baseline:
            return self.bicubic_baseline_ystd(x_std) + branch
        return branch

    @torch.no_grad()
    def forward_with_components(self, x_std):
        self.eval()
        branch = self.backbone(x_std) * self.residual_scale
        base = self.bicubic_baseline_ystd(x_std)
        if self.use_bicubic_baseline:
            out = base + branch
        else:
            out = branch
        return out, base, branch


# =============================================================================
# 4) Physics Loss Constraints
# =============================================================================
class PhysicsConstraintLoss(nn.Module):
    """
    Global Relative Error (GRE) Physics Loss calculated in physical domain:
      1) Horizontal divergence
      2) Relative vorticity
      3) Ageostrophic wind error (latitude-weighted)
      4) Moist Static Energy (MSE) spatial gradient
    """
    def __init__(
        self,
        scaler_Y_mean, scaler_Y_std,
        latitudes, longitudes,
        cp_air=1004.0, lv=2.501e6,
        eps_div=1e-8, eps_vor=1e-8, eps_geo=1e-4, eps_mse=1e-4,
        geo_weight_mode="sin_global",
        geo_weight_alpha=1.0,
        geo_weight_min=0.10,
    ):
        super().__init__()

        self.idx_z, self.idx_q, self.idx_t = 0, 1, 2
        self.idx_u, self.idx_v = 3, 4

        self.eps_div = float(eps_div)
        self.eps_vor = float(eps_vor)
        self.eps_geo = float(eps_geo)
        self.eps_mse = float(eps_mse)

        self.register_buffer("cp_air", torch.tensor(float(cp_air), dtype=torch.float32))
        self.register_buffer("lv", torch.tensor(float(lv), dtype=torch.float32))

        self.register_buffer("y_mean", torch.tensor(scaler_Y_mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("y_std", torch.tensor(scaler_Y_std, dtype=torch.float32).view(1, -1, 1, 1))

        R = 6371000.0
        omega = 7.2921e-5

        latitudes = latitudes.astype(np.float64)
        longitudes = longitudes.astype(np.float64)

        lat_rad = np.deg2rad(latitudes)
        lon_rad = np.deg2rad(longitudes)

        dlat = np.abs(np.mean(np.diff(lat_rad)))
        dlon = np.abs(np.mean(np.diff(lon_rad)))
        lat_decreasing = (latitudes[1] < latitudes[0])

        dy = R * dlat
        dx = R * np.cos(lat_rad) * dlon
        dx = np.maximum(dx, 100.0)

        inv_dx_arr = (1.0 / dx).reshape(1, 1, -1, 1)  # (1,1,H,1)
        self.register_buffer("inv_dx", torch.tensor(inv_dx_arr, dtype=torch.float32))
        self.register_buffer("inv_dy", torch.tensor(1.0 / dy, dtype=torch.float32))

        # Coriolis parameter calculation
        f_raw = 2.0 * omega * np.sin(lat_rad)
        f_cor = f_raw.copy()
        f_min = 2.5e-5
        f_cor = np.where(np.abs(f_cor) < f_min, np.where(f_cor >= 0, f_min, -f_min), f_cor)
        inv_f_arr = (1.0 / f_cor).reshape(1, 1, -1, 1)
        self.register_buffer("inv_f", torch.tensor(inv_f_arr, dtype=torch.float32))

        # Geostrophic latitude weighting
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

        tan_lat = np.tan(lat_rad)
        tan_lat = np.clip(tan_lat, -50.0, 50.0)
        tan_over_R = (tan_lat / R).reshape(1, 1, -1, 1)
        self.register_buffer("tan_over_R", torch.tensor(tan_over_R, dtype=torch.float32))

        ddx_k = torch.tensor(
            [[0.0, 0.0, 0.0],
             [-0.5, 0.0, 0.5],
             [0.0, 0.0, 0.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("ddx_kernel", ddx_k)

        if lat_decreasing:
            ddy_k = torch.tensor(
                [[0.0, 0.5, 0.0],
                 [0.0, 0.0, 0.0],
                 [0.0, -0.5, 0.0]], dtype=torch.float32
            ).view(1, 1, 3, 3)
        else:
            ddy_k = torch.tensor(
                [[0.0, -0.5, 0.0],
                 [0.0, 0.0, 0.0],
                 [0.0, 0.5, 0.0]], dtype=torch.float32
            ).view(1, 1, 3, 3)
        self.register_buffer("ddy_kernel", ddy_k)

        lat_dir = "North->South" if lat_decreasing else "South->North"
        print(f"  [Physics] Latitude orientation: {lat_dir}, dy={dy:.1f}m, dx=[{dx.min():.1f},{dx.max():.1f}]m")
        print(f"  [Physics] geo_weight mode={geo_weight_mode}, alpha={geo_weight_alpha}, min={geo_weight_min}")

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
        inv_f = self.inv_f.squeeze(0).squeeze(0)  # (H,1)
        ug = -inv_f * dz_dy
        vg = inv_f * dz_dx
        return ug, vg

    def compute_mse(self, t, z, q):
        return self.cp_air * t + z + self.lv * q

    def compute_mse_gradients(self, t, z, q):
        mse = self.compute_mse(t, z, q)
        return self.compute_ddx(mse), self.compute_ddy(mse)

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
        # Calculate physical quantities in FP32 space for numerical stability
        pred = self.inverse_std(pred_std.float())
        target = self.inverse_std(target_std.float())

        z_p, q_p, t_p = pred[:, self.idx_z], pred[:, self.idx_q], pred[:, self.idx_t]
        u_p, v_p = pred[:, self.idx_u], pred[:, self.idx_v]

        z_t, q_t, t_t = target[:, self.idx_z], target[:, self.idx_q], target[:, self.idx_t]
        u_t, v_t = target[:, self.idx_u], target[:, self.idx_v]

        losses = {}

        div_p = self.compute_divergence(u_p, v_p)
        div_t = self.compute_divergence(u_t, v_t)
        losses["div"] = self.global_relative_error(div_p, div_t, self.eps_div)

        vor_p = self.compute_vorticity(u_p, v_p)
        vor_t = self.compute_vorticity(u_t, v_t)
        losses["vor"] = self.global_relative_error(vor_p, vor_t, self.eps_vor)

        ug_p, vg_p = self.compute_geostrophic_wind(z_p)
        ug_t, vg_t = self.compute_geostrophic_wind(z_t)
        ageo_u_p, ageo_v_p = u_p - ug_p, v_p - vg_p
        ageo_u_t, ageo_v_t = u_t - ug_t, v_t - vg_t

        w_geo = self.geo_weight
        loss_geo_u = self.global_relative_error(ageo_u_p, ageo_u_t, self.eps_geo, weight=w_geo)
        loss_geo_v = self.global_relative_error(ageo_v_p, ageo_v_t, self.eps_geo, weight=w_geo)
        losses["geo"] = 0.5 * (loss_geo_u + loss_geo_v)

        mse_x_p, mse_y_p = self.compute_mse_gradients(t_p, z_p, q_p)
        mse_x_t, mse_y_t = self.compute_mse_gradients(t_t, z_t, q_t)
        loss_mse_x = self.global_relative_error(mse_x_p, mse_x_t, self.eps_mse)
        loss_mse_y = self.global_relative_error(mse_y_p, mse_y_t, self.eps_mse)
        losses["mse"] = 0.5 * (loss_mse_x + loss_mse_y)

        return losses


class CombinedLoss(nn.Module):
    """Joint Objective function combining standard L1 Loss and Physics Constraints."""
    def __init__(self, physics_module=None,
                 lam_l1=1.0, lam_div=0.0, lam_vor=0.0, lam_geo=0.0, lam_mse=0.0):
        super().__init__()
        self.l1_criterion = nn.L1Loss()
        self.physics = physics_module

        self.lam_l1 = float(lam_l1)
        self.lam_div = float(lam_div)
        self.lam_vor = float(lam_vor)
        self.lam_geo = float(lam_geo)
        self.lam_mse = float(lam_mse)

    def set_lambda_geo(self, new_lambda_geo: float):
        self.lam_geo = float(new_lambda_geo)

    @property
    def use_physics(self):
        if self.physics is None:
            return False
        return (self.lam_div > 0) or (self.lam_vor > 0) or (self.lam_geo > 0) or (self.lam_mse > 0)

    def forward(self, pred, target):
        l1 = self.l1_criterion(pred, target)

        if self.use_physics:
            pl = self.physics(pred, target)
        else:
            zero = pred.new_tensor(0.0)
            pl = {k: zero for k in PHYSICS_KEYS}

        total = (
            self.lam_l1 * l1
            + self.lam_div * pl["div"]
            + self.lam_vor * pl["vor"]
            + self.lam_geo * pl["geo"]
            + self.lam_mse * pl["mse"]
        )

        loss_dict = {
            "total": total.detach().item(),
            "l1": l1.detach().item(),
            "div": pl["div"].detach().item(),
            "vor": pl["vor"].detach().item(),
            "geo": pl["geo"].detach().item(),
            "mse": pl["mse"].detach().item(),
        }
        return total, loss_dict


# =============================================================================
# 5) Evaluation Procedures
# =============================================================================
def evaluate(model, dataloader, criterion, device, num_channels, scaler_Y, amp_enabled=False, amp_dtype=torch.float16):
    """Evaluate model performance on given dataset."""
    model.eval()

    sum_total = 0.0
    sum_l1 = 0.0
    sum_phys = {k: 0.0 for k in PHYSICS_KEYS}
    num_batches = 0

    total_se_std = np.zeros(num_channels, dtype=np.float64)
    total_pixels = np.zeros(num_channels, dtype=np.float64)

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            with autocast_ctx(amp_enabled, amp_dtype):
                pred = model(X_batch)

            pred_f = pred.float()
            y_f = Y_batch.float()

            _, ld = criterion(pred_f, y_f)

            sum_total += ld["total"]
            sum_l1 += ld["l1"]
            for k in PHYSICS_KEYS:
                sum_phys[k] += ld[k]
            num_batches += 1

            for c in range(num_channels):
                diff = pred_f[:, c] - y_f[:, c]
                total_se_std[c] += (diff ** 2).sum().item()
                total_pixels[c] += diff.numel()

    nb = max(num_batches, 1)
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


def evaluate_bicubic_baseline(model, dataloader, device, num_channels, scaler_Y, amp_enabled=False, amp_dtype=torch.float16):
    """Evaluate bicubic interpolation baseline."""
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

            with autocast_ctx(amp_enabled, amp_dtype):
                base_pred = model.bicubic_baseline_ystd(X_batch)

            base_f = base_pred.float()
            y_f = Y_batch.float()

            l1_sum += l1_fn(base_f, y_f).item()
            n_batches += 1

            for c in range(num_channels):
                diff = base_f[:, c] - y_f[:, c]
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
# 6) Visualization Functions
# =============================================================================
def plot_losses(
    epoch, train_total, val_total, test_total,
    train_l1, val_l1, test_l1,
    val_phys_history, in_h, in_w, out_h, out_w,
    loss_plot_path, physics_plot_path,
    best_epoch=None
):
    """Plot total loss, L1 loss, and physics constraint loss histories."""
    n = epoch + 1
    ep = range(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(ep, train_total[:n], label="Train Total", lw=1.5)
    ax1.plot(ep, val_total[:n], label="Valid Total", lw=1.5)
    ax1.plot(ep, test_total[:n], label="Test Total", lw=1.5)
    if best_epoch is not None:
        ax1.axvline(best_epoch, color="red", ls="--", alpha=0.5, label=f"Best({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss")
    ax1.set_title(f"Total Loss ({in_h}x{in_w} -> {out_h}x{out_w})")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(ep, train_l1[:n], label="Train L1", lw=1.5)
    ax2.plot(ep, val_l1[:n], label="Valid L1", lw=1.5)
    ax2.plot(ep, test_l1[:n], label="Test L1", lw=1.5)
    if best_epoch is not None:
        ax2.axvline(best_epoch, color="red", ls="--", alpha=0.5, label=f"Best({best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("L1")
    ax2.set_title("L1 Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()

    n_keys = len(PHYSICS_KEYS)
    ncols = 2
    nrows = math.ceil(n_keys / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    titles = {
        "div": "Divergence",
        "vor": "Vorticity",
        "geo": "Geostrophic (ageostrophic, lat-weighted)",
        "mse": "Moist Static Energy Gradient",
    }

    for i, k in enumerate(PHYSICS_KEYS):
        ax = axes[i]
        vals = val_phys_history[k][:n]
        ax.plot(ep, vals, lw=1.5, color="tab:orange")
        if best_epoch is not None:
            ax.axvline(best_epoch, color="red", ls="--", alpha=0.5)
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
# 7) Single Experiment Runner
# =============================================================================
def train_one_experiment(
    exp_cfg,
    train_loader, valid_loader, test_loader,
    scaler_X, scaler_Y,
    lat_hr, lon_hr,
    input_h, input_w, output_h, output_w
):
    exp_name = exp_cfg["name"]
    exp_dir = os.path.join(OUTPUT_ROOT, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    model_save_path = os.path.join(exp_dir, "best_model.pth")
    loss_plot_path = os.path.join(exp_dir, "loss_epoch.png")
    physics_plot_path = os.path.join(exp_dir, "physics_loss_epoch.png")
    history_csv_path = os.path.join(exp_dir, "training_history.csv")
    model_info_path = os.path.join(exp_dir, "model_info.json")

    amp_enabled = USE_AMP and (device.type == "cuda")
    amp_dtype = get_amp_dtype(AMP_DTYPE)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))

    print("=" * 100)
    print(f"Experiment: {exp_name}")
    print(f"use_physics={exp_cfg['use_physics']} | amp={amp_enabled}({AMP_DTYPE})")
    print("=" * 100)

    model = SwinIRSRModel(
        in_channels=NUM_CHANNELS,
        out_channels=NUM_CHANNELS,
        upscale_factor=UPSCALE_FACTOR,
        scaler_X_mean=scaler_X.mean,
        scaler_X_std=scaler_X.std,
        scaler_Y_mean=scaler_Y.mean,
        scaler_Y_std=scaler_Y.std,
        use_bicubic_baseline=USE_BICUBIC_BASELINE,
        bicubic_align_corners=BICUBIC_ALIGN_CORNERS,
        residual_scale=RESIDUAL_SCALE,
        embed_dim=SWINIR_EMBED_DIM,
        depths=SWINIR_DEPTHS,
        num_heads=SWINIR_NUM_HEADS,
        window_size=SWINIR_WINDOW_SIZE,
        mlp_ratio=SWINIR_MLP_RATIO,
        qkv_bias=SWINIR_QKV_BIAS,
        drop_rate=SWINIR_DROP_RATE,
        attn_drop_rate=SWINIR_ATTN_DROP_RATE,
        drop_path_rate=SWINIR_DROP_PATH_RATE,
        patch_norm=SWINIR_PATCH_NORM,
        use_checkpoint=SWINIR_USE_CHECKPOINT,
        upsampler=SWINIR_UPSAMPLER,
        resi_connection=SWINIR_RESI_CONNECTION,
        zero_init_last=ZERO_INIT_LAST,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")
    print(f"SwinIR config: embed={SWINIR_EMBED_DIM}, depths={SWINIR_DEPTHS}, heads={SWINIR_NUM_HEADS}, ws={SWINIR_WINDOW_SIZE}")

    # Baseline calculation
    b_valid_l1, b_valid_rmse, b_valid_overall = evaluate_bicubic_baseline(
        model, valid_loader, device, NUM_CHANNELS, scaler_Y, amp_enabled, amp_dtype
    )
    b_test_l1, b_test_rmse, b_test_overall = evaluate_bicubic_baseline(
        model, test_loader, device, NUM_CHANNELS, scaler_Y, amp_enabled, amp_dtype
    )
    print(f"[Valid baseline] L1={b_valid_l1:.6f} | RMSE: {format_rmse_str(b_valid_rmse, VAR_NAMES, b_valid_overall)}")
    print(f"[Test  baseline] L1={b_test_l1:.6f} | RMSE: {format_rmse_str(b_test_rmse, VAR_NAMES, b_test_overall)}")

    physics_loss = None
    if exp_cfg["use_physics"]:
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
            geo_weight_min=GEO_WEIGHT_MIN,
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
    print(f"  lambda_l1={criterion.lam_l1}, lambda_div={criterion.lam_div}, lambda_vor={criterion.lam_vor}, "
          f"lambda_geo={criterion.lam_geo}, lambda_mse={criterion.lam_mse}")

    # Diagnostic sample batch
    model.eval()
    with torch.no_grad():
        X_sample, Y_sample = next(iter(train_loader))
        X_sample = X_sample.to(device)
        Y_sample = Y_sample.to(device)
        with autocast_ctx(amp_enabled, amp_dtype):
            pred_sample = model(X_sample)
        _, diag_ld = criterion(pred_sample.float(), Y_sample.float())
        print("Initial batch loss diagnostics:")
        for k, v in diag_ld.items():
            print(f"  {k:>8s}: {v:.6e}")

    if AUTO_TUNE_LAMBDA_GEO and exp_cfg["use_physics"]:
        raw_lam, tuned_lam = suggest_lambda_geo(
            diag_loss_dict=diag_ld,
            lam_l1=criterion.lam_l1,
            target_rel_to_l1=GEO_TARGET_REL_TO_L1,
            lam_min=GEO_LAMBDA_MIN,
            lam_max=GEO_LAMBDA_MAX,
        )
        old = criterion.lam_geo
        criterion.set_lambda_geo(tuned_lam)
        print(f"[Auto lambda_geo] raw={raw_lam:.6e}, tuned={tuned_lam:.6e}, {old:.6e}->{criterion.lam_geo:.6e}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )

    best_val_loss = float("inf")
    best_epoch_num = -1
    best_valid_snapshot = None
    best_test_snapshot = None

    history_records = []

    train_total_list, val_total_list, test_total_list = [], [], []
    train_l1_list, val_l1_list, test_l1_list = [], [], []
    val_phys_history = {k: [] for k in PHYSICS_KEYS}

    print("-" * 100)
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # ---------------- Training Phase ----------------
        model.train()
        train_loss_accum = {k: 0.0 for k in ["total", "l1"] + PHYSICS_KEYS}
        n_train_batches = 0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=USE_ZERO_GRAD_SET_TO_NONE)

            with autocast_ctx(amp_enabled, amp_dtype):
                pred = model(X_batch)

            total_loss, ld = criterion(pred.float(), Y_batch.float())

            scaler_amp.scale(total_loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

            for k in train_loss_accum:
                train_loss_accum[k] += ld[k]
            n_train_batches += 1

        train_avg = {k: train_loss_accum[k] / max(n_train_batches, 1) for k in train_loss_accum}

        # ---------------- Validation Phase ----------------
        val_total, val_l1, val_phys, val_rmse, val_overall = evaluate(
            model, valid_loader, criterion, device, NUM_CHANNELS, scaler_Y, amp_enabled, amp_dtype
        )

        scheduler.step(val_total)
        new_lr = optimizer.param_groups[0]["lr"]

        need_test_eval = ((epoch + 1) % EVAL_TEST_EVERY == 0)

        is_best = val_total <= best_val_loss
        model_status = "Unchanged"

        if is_best:
            best_val_loss = val_total
            best_epoch_num = epoch + 1
            need_test_eval = True
            model_status = "Best Valid, Model Updated"

            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": float(best_val_loss),
                "experiment_name": exp_name,
                "use_physics": bool(exp_cfg["use_physics"]),
                "model_type": "SwinIRSRModel",
                "upscale_factor": UPSCALE_FACTOR,
                "swinir_config": {
                    "embed_dim": SWINIR_EMBED_DIM,
                    "depths": SWINIR_DEPTHS,
                    "num_heads": SWINIR_NUM_HEADS,
                    "window_size": SWINIR_WINDOW_SIZE,
                    "mlp_ratio": SWINIR_MLP_RATIO,
                    "drop_rate": SWINIR_DROP_RATE,
                    "attn_drop_rate": SWINIR_ATTN_DROP_RATE,
                    "drop_path_rate": SWINIR_DROP_PATH_RATE,
                    "qkv_bias": SWINIR_QKV_BIAS,
                    "patch_norm": SWINIR_PATCH_NORM,
                    "upsampler": SWINIR_UPSAMPLER,
                    "resi_connection": SWINIR_RESI_CONNECTION,
                    "zero_init_last": ZERO_INIT_LAST,
                },
                "bicubic_baseline": {
                    "enabled": USE_BICUBIC_BASELINE,
                    "align_corners": BICUBIC_ALIGN_CORNERS,
                    "residual_scale": RESIDUAL_SCALE,
                },
                "physics_weights": {
                    "lam_l1": float(criterion.lam_l1),
                    "lam_div": float(criterion.lam_div),
                    "lam_vor": float(criterion.lam_vor),
                    "lam_geo": float(criterion.lam_geo),
                    "lam_mse": float(criterion.lam_mse),
                },
                "eps_values": {
                    "eps_div": EPS_DIV,
                    "eps_vor": EPS_VOR,
                    "eps_geo": EPS_GEO,
                    "eps_mse": EPS_MSE,
                },
            }
            torch.save(ckpt, model_save_path)

            best_valid_snapshot = {
                "val_total": float(val_total),
                "val_l1": float(val_l1),
                "val_phys": {k: float(val_phys[k]) for k in PHYSICS_KEYS},
                "val_rmse": [float(x) for x in val_rmse],
                "val_overall": float(val_overall),
            }

        if need_test_eval:
            test_total, test_l1, test_phys, test_rmse, test_overall = evaluate(
                model, test_loader, criterion, device, NUM_CHANNELS, scaler_Y, amp_enabled, amp_dtype
            )
        else:
            test_total = np.nan
            test_l1 = np.nan
            test_phys = {k: np.nan for k in PHYSICS_KEYS}
            test_rmse = np.full(NUM_CHANNELS, np.nan, dtype=np.float64)
            test_overall = np.nan

        if is_best and need_test_eval:
            best_test_snapshot = {
                "test_total": float(test_total),
                "test_l1": float(test_l1),
                "test_phys": {k: float(test_phys[k]) for k in PHYSICS_KEYS},
                "test_rmse": [float(x) for x in test_rmse],
                "test_overall": float(test_overall),
            }

        epoch_time = time.time() - epoch_start
        remain = epoch_time * (NUM_EPOCHS - epoch - 1)

        train_total_list.append(train_avg["total"])
        val_total_list.append(val_total)
        test_total_list.append(test_total)

        train_l1_list.append(train_avg["l1"])
        val_l1_list.append(val_l1)
        test_l1_list.append(test_l1)

        for k in PHYSICS_KEYS:
            val_phys_history[k].append(val_phys[k])

        rec = {
            "Epoch": epoch + 1,
            "Train_Total": float(train_avg["total"]),
            "Valid_Total": float(val_total),
            "Test_Total": float(test_total),
            "Train_L1": float(train_avg["l1"]),
            "Valid_L1": float(val_l1),
            "Test_L1": float(test_l1),
            "Lambda_GEO": float(criterion.lam_geo),
            "LR": float(current_lr),
        }

        for k in PHYSICS_KEYS:
            rec[f"Train_P_{k}"] = float(train_avg.get(k, 0.0))
            rec[f"Valid_P_{k}"] = float(val_phys[k])
            rec[f"Test_P_{k}"] = float(test_phys[k])

        for i, name in enumerate(VAR_NAMES):
            rec[f"Train_RMSE_{name}"] = np.nan
            rec[f"Valid_RMSE_{name}"] = float(val_rmse[i])
            rec[f"Test_RMSE_{name}"] = float(test_rmse[i])

        rec["Train_RMSE_Overall"] = np.nan
        rec["Valid_RMSE_Overall"] = float(val_overall)
        rec["Test_RMSE_Overall"] = float(test_overall)
        history_records.append(rec)

        print("-" * 100)
        print(
            f"[{exp_name}] Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"train_total={train_avg['total']:.4f} | valid_total={val_total:.4f} | test_total={test_total:.4f}"
        )
        print(f"  L1 train={train_avg['l1']:.4f}, valid={val_l1:.4f}, test={test_l1:.4f}")
        print(f"  Model status: {model_status} | epoch_time={epoch_time:.1f}s | remaining={remain:.0f}s")
        print(f"  [Valid Physics] {format_physics_str(val_phys)}")
        print(f"  [Valid RMSE] {format_rmse_str(val_rmse, VAR_NAMES, val_overall)}")
        if not np.isnan(test_overall):
            print(f"  [Test  RMSE] {format_rmse_str(test_rmse, VAR_NAMES, test_overall)}")
        if new_lr != current_lr:
            print(f"  [LR Adjusted] {current_lr:.6f} -> {new_lr:.6f}")

        if (epoch + 1) % PLOT_EVERY == 0:
            plot_losses(
                epoch, train_total_list, val_total_list, test_total_list,
                train_l1_list, val_l1_list, test_l1_list,
                val_phys_history, input_h, input_w, output_h, output_w,
                loss_plot_path, physics_plot_path
            )

    print("=" * 100)
    print(f"Experiment completed: {exp_name}")

    if best_test_snapshot is None:
        ckpt = torch.load(model_save_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        test_total, test_l1, test_phys, test_rmse, test_overall = evaluate(
            model, test_loader, criterion, device, NUM_CHANNELS, scaler_Y, amp_enabled, amp_dtype
        )
        best_test_snapshot = {
            "test_total": float(test_total),
            "test_l1": float(test_l1),
            "test_phys": {k: float(test_phys[k]) for k in PHYSICS_KEYS},
            "test_rmse": [float(x) for x in test_rmse],
            "test_overall": float(test_overall),
        }

    if best_valid_snapshot is None:
        best_valid_snapshot = {
            "val_total": np.nan,
            "val_l1": np.nan,
            "val_phys": {k: np.nan for k in PHYSICS_KEYS},
            "val_rmse": [np.nan] * NUM_CHANNELS,
            "val_overall": np.nan,
        }

    plot_losses(
        NUM_EPOCHS - 1, train_total_list, val_total_list, test_total_list,
        train_l1_list, val_l1_list, test_l1_list,
        val_phys_history, input_h, input_w, output_h, output_w,
        loss_plot_path, physics_plot_path,
        best_epoch=best_epoch_num
    )

    df_hist = pd.DataFrame(history_records)
    df_hist.to_csv(history_csv_path, index=False)

    model_info = {
        "experiment": exp_name,
        "use_physics": bool(exp_cfg["use_physics"]),
        "architecture": "SwinIR + optional bicubic baseline + Physics(GRE)",
        "speed_preset": SPEED_PRESET,
        "variables": VARIABLES,
        "variable_units": VAR_UNITS,
        "input_size": f"{input_h}x{input_w}",
        "output_size": f"{output_h}x{output_w}",
        "upscale_factor": UPSCALE_FACTOR,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "acceleration": {
            "amp_enabled": bool(amp_enabled),
            "amp_dtype": AMP_DTYPE,
            "tf32": ENABLE_TF32,
            "cudnn_benchmark": CUDNN_BENCHMARK,
            "eval_test_every": EVAL_TEST_EVERY,
        },
        "swinir": {
            "embed_dim": SWINIR_EMBED_DIM,
            "depths": SWINIR_DEPTHS,
            "num_heads": SWINIR_NUM_HEADS,
            "window_size": SWINIR_WINDOW_SIZE,
            "mlp_ratio": SWINIR_MLP_RATIO,
            "drop_rate": SWINIR_DROP_RATE,
            "attn_drop_rate": SWINIR_ATTN_DROP_RATE,
            "drop_path_rate": SWINIR_DROP_PATH_RATE,
            "upsampler": SWINIR_UPSAMPLER,
            "resi_connection": SWINIR_RESI_CONNECTION,
            "patch_norm": SWINIR_PATCH_NORM,
            "zero_init_last": ZERO_INIT_LAST,
        },
        "bicubic_baseline": {
            "enabled": USE_BICUBIC_BASELINE,
            "align_corners": BICUBIC_ALIGN_CORNERS,
            "residual_scale": RESIDUAL_SCALE,
        },
        "loss_weights": {
            "lambda_l1": float(criterion.lam_l1),
            "lambda_div": float(criterion.lam_div),
            "lambda_vor": float(criterion.lam_vor),
            "lambda_geo": float(criterion.lam_geo),
            "lambda_mse": float(criterion.lam_mse),
        },
        "eps_values": {
            "eps_div": EPS_DIV,
            "eps_vor": EPS_VOR,
            "eps_geo": EPS_GEO,
            "eps_mse": EPS_MSE,
        },
        "geo_weighting": {
            "mode": GEO_WEIGHT_MODE,
            "alpha": GEO_WEIGHT_ALPHA,
            "min": GEO_WEIGHT_MIN,
        },
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "random_seed": RANDOM_SEED,
        "bicubic_baseline_valid_l1": float(b_valid_l1),
        "bicubic_baseline_test_l1": float(b_test_l1),
        "bicubic_baseline_valid_rmse_overall": float(b_valid_overall),
        "bicubic_baseline_test_rmse_overall": float(b_test_overall),
        "best_epoch": int(best_epoch_num),
        "best_valid_total": float(best_valid_snapshot["val_total"]),
        "best_valid_l1": float(best_valid_snapshot["val_l1"]),
        "best_valid_rmse_overall": float(best_valid_snapshot["val_overall"]),
        "best_valid_physics": {k: float(best_valid_snapshot["val_phys"][k]) for k in PHYSICS_KEYS},
        "best_test_total": float(best_test_snapshot["test_total"]),
        "best_test_l1": float(best_test_snapshot["test_l1"]),
        "best_test_rmse_overall": float(best_test_snapshot["test_overall"]),
        "best_test_physics": {k: float(best_test_snapshot["test_phys"][k]) for k in PHYSICS_KEYS},
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    print(f"  Best Epoch={best_epoch_num}, Best Valid Total={best_valid_snapshot['val_total']:.6f}")
    print(f"  Saved: {model_save_path}")
    print(f"  Saved: {history_csv_path}")
    print(f"  Saved: {loss_plot_path}")
    print(f"  Saved: {physics_plot_path}")
    print(f"  Saved: {model_info_path}")

    summary = {
        "experiment": exp_name,
        "use_physics": bool(exp_cfg["use_physics"]),
        "best_epoch": int(best_epoch_num),
        "best_valid_total": float(best_valid_snapshot["val_total"]),
        "valid_l1": float(best_valid_snapshot["val_l1"]),
        "test_l1": float(best_test_snapshot["test_l1"]),
        "valid_rmse_overall": float(best_valid_snapshot["val_overall"]),
        "test_rmse_overall": float(best_test_snapshot["test_overall"]),
        "valid_div": float(best_valid_snapshot["val_phys"]["div"]),
        "valid_vor": float(best_valid_snapshot["val_phys"]["vor"]),
        "valid_geo": float(best_valid_snapshot["val_phys"]["geo"]),
        "valid_mse": float(best_valid_snapshot["val_phys"]["mse"]),
        "output_dir": exp_dir,
    }
    return summary


# =============================================================================
# 8) Main Pipeline Execution
# =============================================================================
def main():
    if len(EXPERIMENTS) == 0:
        raise ValueError("EXPERIMENTS matrix is empty. Please define at least one experiment configuration.")

    set_seed(RANDOM_SEED, deterministic=REPRODUCIBLE)
    setup_acceleration()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    start_all = time.time()
    print("=" * 100)
    print("ERA5 Super-Resolution Training: SwinIR Fast + Physics Ablation")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Speed Preset: {SPEED_PRESET}")
    print(f"Upscale Factor: x{UPSCALE_FACTOR}")
    print("=" * 100)

    if CHECK_MISSING:
        all_files = [X_TRAIN_PATH, Y_TRAIN_PATH, X_VALID_PATH, Y_VALID_PATH, X_TEST_PATH, Y_TEST_PATH]
        check_missing_values(all_files, VARIABLES)

    print("Loading Datasets...")
    X_train_raw, Y_train_raw, lat_hr, lon_hr = load_nc_data(
        X_TRAIN_PATH, Y_TRAIN_PATH, VARIABLES, return_coords=True
    )
    X_valid_raw, Y_valid_raw = load_nc_data(X_VALID_PATH, Y_VALID_PATH, VARIABLES)
    X_test_raw, Y_test_raw = load_nc_data(X_TEST_PATH, Y_TEST_PATH, VARIABLES)

    input_h, input_w = X_train_raw.shape[2], X_train_raw.shape[3]
    output_h, output_w = Y_train_raw.shape[2], Y_train_raw.shape[3]
    actual_scale = output_h // input_h

    if UPSCALE_FACTOR != actual_scale:
        raise ValueError(
            f"UPSCALE_FACTOR={UPSCALE_FACTOR} conflicts with dataset scale ratio {actual_scale} "
            f"({input_h}x{input_w} -> {output_h}x{output_w})"
        )

    if input_h % SWINIR_WINDOW_SIZE != 0 or input_w % SWINIR_WINDOW_SIZE != 0:
        print(f"[Note] Input resolution {input_h}x{input_w} is not divisible by window_size={SWINIR_WINDOW_SIZE}. Reflection padding will be automatically applied.")

    print(f"LR size: {input_h}x{input_w}, HR size: {output_h}x{output_w}, scale={actual_scale}x")
    print(f"Train samples={X_train_raw.shape[0]:,}, Valid samples={X_valid_raw.shape[0]:,}, Test samples={X_test_raw.shape[0]:,}")
    print(f"Latitude range=[{lat_hr.min():.2f},{lat_hr.max():.2f}], Longitude range=[{lon_hr.min():.2f},{lon_hr.max():.2f}]")

    print("Standardizing Datasets...")
    scaler_X = ChannelScaler()
    scaler_Y = ChannelScaler()

    X_train_raw = scaler_X.fit_transform(X_train_raw)
    X_valid_raw = scaler_X.transform(X_valid_raw)
    X_test_raw = scaler_X.transform(X_test_raw)

    Y_train_raw = scaler_Y.fit_transform(Y_train_raw)
    Y_valid_raw = scaler_Y.transform(Y_valid_raw)
    Y_test_raw = scaler_Y.transform(Y_test_raw)

    print(f"X scaler: {scaler_X}")
    print(f"Y scaler: {scaler_Y}")

    scalers_path = os.path.join(OUTPUT_ROOT, "scalers.joblib")
    joblib.dump({"scaler_X": scaler_X, "scaler_Y": scaler_Y}, scalers_path)
    print(f"Scalers saved: {scalers_path}")

    train_dataset = ERA5Dataset(X_train_raw, Y_train_raw)
    valid_dataset = ERA5Dataset(X_valid_raw, Y_valid_raw)
    test_dataset = ERA5Dataset(X_test_raw, Y_test_raw)

    del X_train_raw, Y_train_raw, X_valid_raw, Y_valid_raw, X_test_raw, Y_test_raw
    gc.collect()

    pin = (device.type == "cuda")
    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": pin,
    }
    if NUM_WORKERS > 0 and PERSISTENT_WORKERS:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs
    )

    print(f"DataLoader ready: Train batches={len(train_loader):,}, Valid batches={len(valid_loader):,}, Test batches={len(test_loader):,}")

    summaries = []
    for exp_cfg in EXPERIMENTS:
        set_seed(RANDOM_SEED, deterministic=REPRODUCIBLE)

        summary = train_one_experiment(
            exp_cfg=exp_cfg,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            scaler_X=scaler_X,
            scaler_Y=scaler_Y,
            lat_hr=lat_hr,
            lon_hr=lon_hr,
            input_h=input_h,
            input_w=input_w,
            output_h=output_h,
            output_w=output_w,
        )
        summaries.append(summary)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summaries)
    summary_csv = os.path.join(OUTPUT_ROOT, "ablation_summary.csv")
    summary_json = os.path.join(OUTPUT_ROOT, "ablation_summary.json")
    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("Ablation Study Summary")
    print(summary_df.to_string(index=False))
    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_json}")

    if (summary_df["use_physics"] == False).any() and (summary_df["use_physics"] == True).any():
        base = summary_df[summary_df["use_physics"] == False].iloc[0]
        phys = summary_df[summary_df["use_physics"] == True].iloc[0]
        dv = float(base["valid_rmse_overall"] - phys["valid_rmse_overall"])
        dt = float(base["test_rmse_overall"] - phys["test_rmse_overall"])
        print("-" * 100)
        print("Physics vs. No-Physics Comparison (Positive values indicate Physics improvement: lower RMSE)")
        print(f"  Valid RMSE Improvement: {dv:.6f}")
        print(f"  Test  RMSE Improvement: {dt:.6f}")

    total_time = time.time() - start_all
    print("=" * 100)
    print(f"All experiments completed. Total elapsed time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Output Directory: {OUTPUT_ROOT}")
    print("=" * 100)


if __name__ == "__main__":
    main()