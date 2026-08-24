#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA5 SwinIR Super-Resolution Model Evaluation Script with Bootstrap 95% CI
-------------------------------------------------------------------------
Evaluated Targets:
  1) Cubic spline baseline
  2) One or multiple SwinIR models (best_model.pth)

Datasets:
  - Low-resolution inputs:  X2_test.nc or X4_test.nc
  - High-resolution truth:  Y_test.nc
  - Normalization scalers: scalers.joblib
  - Checkpoints:           best_model.pth

Evaluation Metrics:
  A) Raw variables (Z, Q, T, U, V):
     RMSE, MSE, MAE, MRE, Correlation, PSNR, SSIM
  B) Derived physical quantities:
     DIV, VOR, AGEO_MAG, MSE_GRAD_X, MSE_GRAD_Y
     RMSE, MSE, MAE, MRE, Correlation

Bootstrap Confidence Intervals (CI):
  - CI Methods: percentile / basic / BCa
  - Resampling schemes: iid / moving_block / circular_block / stationary
  - BCa Acceleration: Jackknife estimation via delete-1 or block delete-d
"""

import os
import math
import random
from collections import OrderedDict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import xarray as xr
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm
from skimage.metrics import structural_similarity as ssim_2d
from tqdm import tqdm


# =============================================================================
# 0) User Configuration
# =============================================================================
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# Set to an integer (e.g., 256) for quick debugging; set to None for full evaluation
EVAL_MAX_SAMPLES = None

# AMP Inference (CUDA only)
USE_AMP_INFERENCE = False
AMP_DTYPE = "bf16"    # "fp16" or "bf16"

# Cubic spline parameters
CUBIC_SPLINE_KX = 3
CUBIC_SPLINE_KY = 3

# MRE stabilization eps: max(MRE_EPS_ABS, MRE_EPS_REL * mean_abs_truth)
MRE_EPS_REL = 1e-3
MRE_EPS_ABS = 1e-8

# SSIM parameters
SSIM_WIN_SIZE = 7

# Physical constants
CP_AIR = 1004.0
LV_WATER = 2.501e6
EARTH_RADIUS = 6371000.0
OMEGA = 7.2921e-5
F_CORIOLIS_ABS_MIN = 2.5e-5
TAN_CLIP = 50.0

# -----------------------------
# Bootstrap Settings
# -----------------------------
ENABLE_BOOTSTRAP = True
BOOTSTRAP_N_RESAMPLES = 2000          # Recommended >= 2000 for BCa
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_CI_METHOD = "BCa"           # {"percentile", "basic", "BCa"}
BOOTSTRAP_SCHEME = "circular_block"
#   "iid"            : Independent and identically distributed
#   "moving_block"   : Moving block bootstrap (Künsch 1989)
#   "circular_block" : Circular block bootstrap (Politis & Romano 1992) [Recommended]
#   "stationary"     : Stationary bootstrap with block length ~ Geometric(1/L)
BOOTSTRAP_BLOCK_LENGTH = "auto"       # int / "auto" (N^(1/3)) / "sqrt" (N^(1/2))
BOOTSTRAP_JACKKNIFE = "auto"          # "auto" / "delete1" / "block"
BOOTSTRAP_SEED = 42
BOOTSTRAP_CHUNK = 256                 # Number of resamples per chunk (memory control)

# -----------------------------
# File Paths
# -----------------------------
X_TEST_PATH = "X2_test.nc"            # or X4_test.nc
Y_TEST_PATH = "Y_test.nc"
SCALER_PATH = "outputs_swinir_ablation_scale2_balanced/scalers.joblib"

MODEL_PATHS = OrderedDict([
    ("swinir_l1_only", "outputs_swinir_ablation_scale2_balanced/swinir_l1_only/best_model.pth"),
    ("swinir_l1_physics", "outputs_swinir_ablation_scale2_balanced/swinir_l1_physics/best_model.pth"),
])

CSV_RAW_METRICS_PATH = "test_metrics_raw.csv"
CSV_PHYS_METRICS_PATH = "test_metrics_physics.csv"
CSV_ALL_IN_ONE_PATH = "test_metrics_all_in_one.csv"

# Bootstrap CI outputs
CSV_RAW_CI_PATH = "test_metrics_raw_bootstrap_ci.csv"
CSV_PHYS_CI_PATH = "test_metrics_physics_bootstrap_ci.csv"
CSV_ALL_CI_PATH = "test_metrics_all_bootstrap_ci.csv"

VARIABLES = ["z", "q", "t", "u", "v"]
VAR_DISPLAY = ["Z", "Q", "T", "U", "V"]
PHYS_DERIVED_NAMES = ["DIV", "VOR", "AGEO_MAG", "MSE_GRAD_X", "MSE_GRAD_Y"]

RAW_METRIC_ORDER = ["RMSE", "MSE", "MAE", "MRE", "Correlation", "PSNR", "SSIM"]
PHYS_METRIC_ORDER = ["RMSE", "MSE", "MAE", "MRE", "Correlation"]

BASE_STAT_KEYS = [
    "sum_abs", "sum_sq", "sum_rel",
    "sum_pred", "sum_true", "sum_pred2", "sum_true2", "sum_pred_true"
]


# =============================================================================
# 1) Basic Utilities
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_torch_backend():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True


def get_amp_dtype(name: str):
    if str(name).lower().strip() == "bf16":
        return torch.bfloat16
    return torch.float16


def pick_dim_name(ds: xr.Dataset, candidates):
    for c in candidates:
        if c in ds.dims:
            return c
    raise KeyError(f"Dimension not found. Candidates: {candidates}, Available: {list(ds.dims)}")


def pick_coord_name(ds: xr.Dataset, candidates):
    for c in candidates:
        if c in ds.coords or c in ds.variables:
            return c
    raise KeyError(f"Coordinate not found. Candidates: {candidates}, Available coords: {list(ds.coords)}")


class ChannelScaler:
    """
    Dummy class for joblib compatibility during deserialization.
    """
    def __init__(self):
        self.mean = None
        self.std = None
        self.n_channels = None


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


def read_batch_from_ds(ds, variables, time_dim, start, end):
    arr = np.stack(
        [ds[v].isel({time_dim: slice(start, end)}).values.astype(np.float32) for v in variables],
        axis=1
    )  # (B, C, H, W)
    return arr


# =============================================================================
# 2) SwinIR Model Definition
# =============================================================================
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or (not training):
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rnd = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    rnd.floor_()
    return x.div(keep_prob) * rnd


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
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
    # x: (B, H, W, C)
    B, H, W, C = x.shape
    x = x.contiguous().view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    # windows: (num_windows*B, ws, ws, C)
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.contiguous().view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = int(dim)
        self.window_size = (int(window_size), int(window_size))
        self.num_heads = int(num_heads)
        head_dim = dim // num_heads
        self.scale = qk_scale or (head_dim ** -0.5)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1)
        relative_position_index = relative_coords.sum(-1)

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
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rel_bias = rel_bias.view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj_drop(self.proj(out))
        return out


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path_prob=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
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

        x_windows = window_partition(shifted_x, self.window_size)
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
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        self.blocks = nn.ModuleList()

        for i in range(int(depth)):
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
    def __init__(self, embed_dim, norm_layer=None):
        super().__init__()
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = int(embed_dim)

    def forward(self, x, x_size):
        B, HW, C = x.shape
        H, W = x_size
        return x.transpose(1, 2).contiguous().view(B, C, H, W)


class RSTB(nn.Module):
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
            raise ValueError(f"Unsupported resi_connection={resi_connection}")

        self.patch_embed = PatchEmbed(embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

    def forward(self, x, x_size):
        out = self.residual_group(x, x_size)
        out = self.patch_unembed(out, x_size)
        out = self.conv(out)
        out = self.patch_embed(out)
        return out + x


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported scale={scale}")
        super().__init__(*m)


class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch):
        super().__init__(
            nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale),
        )


class SwinIRBackbone(nn.Module):
    def __init__(
        self,
        in_chans=5,
        out_chans=5,
        embed_dim=60,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
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
        zero_init_last=False
    ):
        super().__init__()
        assert len(depths) == len(num_heads), "Length of depths and num_heads must match."

        self.upscale = int(upscale)
        self.window_size = int(window_size)
        self.upsampler = str(upsampler)

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

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
        elif resi_connection == "3conv":
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unsupported resi_connection={resi_connection}")

        if self.upsampler == "pixelshuffle":
            num_feat = 64
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.upsample = Upsample(self.upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, out_chans, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            self.upsample = UpsampleOneStep(self.upscale, embed_dim, out_chans)
        else:
            raise ValueError(f"Unsupported upsampler={self.upsampler}")

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
    Output space in Y_std:
      use_bicubic_baseline=True:
        y = bicubic(x_std -> Y_std) + branch
      False:
        y = branch
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
        embed_dim=60,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
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
        zero_init_last=False
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


# =============================================================================
# 3) Checkpoint Loading / Hyperparameter Inference
# =============================================================================
def strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def infer_swinir_hparams_from_state_dict(state_dict):
    keys = list(state_dict.keys())

    if "backbone.conv_first.weight" not in state_dict:
        raise KeyError("State dict is missing 'backbone.conv_first.weight'. Unrecognized SwinIR checkpoint.")

    in_channels = int(state_dict["backbone.conv_first.weight"].shape[1])
    embed_dim = int(state_dict["backbone.conv_first.weight"].shape[0])

    if "x2y_alpha" in state_dict:
        out_channels = int(state_dict["x2y_alpha"].shape[1])
    elif "backbone.conv_last.weight" in state_dict:
        out_channels = int(state_dict["backbone.conv_last.weight"].shape[0])
    else:
        out_channels = 5

    layer_ids = sorted({
        int(k.split(".")[2]) for k in keys
        if k.startswith("backbone.layers.") and ".residual_group.blocks." in k and k.endswith(".attn.qkv.weight")
    })
    if not layer_ids:
        layer_ids = sorted({
            int(k.split(".")[2]) for k in keys
            if k.startswith("backbone.layers.") and ".relative_position_bias_table" in k
        })

    depths = []
    num_heads = []
    for lid in layer_ids:
        prefix = f"backbone.layers.{lid}.residual_group.blocks."
        block_ids = sorted({
            int(k.split(".")[5]) for k in keys
            if k.startswith(prefix) and k.endswith(".attn.qkv.weight")
        })
        if not block_ids:
            block_ids = sorted({
                int(k.split(".")[5]) for k in keys
                if k.startswith(prefix) and k.endswith(".attn.relative_position_bias_table")
            })

        depth = (max(block_ids) + 1) if block_ids else 0
        depths.append(depth)

        head_key = f"{prefix}0.attn.relative_position_bias_table"
        if head_key in state_dict:
            num_heads.append(int(state_dict[head_key].shape[1]))
        else:
            qkv_key = f"{prefix}0.attn.qkv.weight"
            if qkv_key in state_dict:
                dim = int(state_dict[qkv_key].shape[1])
                cands = [h for h in [12, 10, 8, 6, 5, 4, 3, 2, 1] if dim % h == 0]
                num_heads.append(cands[0] if cands else 1)
            else:
                num_heads.append(1)

    if len(depths) == 0:
        depths = [6, 6, 6, 6]
    if len(num_heads) == 0:
        num_heads = [6] * len(depths)

    window_size = 5
    for k in keys:
        if k.endswith("attn.relative_position_bias_table"):
            L = int(state_dict[k].shape[0])  # (2*ws-1)^2
            s = int(round(math.sqrt(L)))
            ws = (s + 1) // 2
            if ws >= 1:
                window_size = int(ws)
            break

    mlp_ratio = 2.0
    for lid in layer_ids:
        fc1_key = f"backbone.layers.{lid}.residual_group.blocks.0.mlp.fc1.weight"
        n2_key = f"backbone.layers.{lid}.residual_group.blocks.0.norm2.weight"
        if fc1_key in state_dict and n2_key in state_dict:
            hidden = int(state_dict[fc1_key].shape[0])
            dim = int(state_dict[n2_key].shape[0])
            if dim > 0:
                mlp_ratio = float(hidden / dim)
            break

    qkv_bias = any(k.endswith(".attn.qkv.bias") for k in keys)
    patch_norm = any(k.startswith("backbone.patch_embed.norm.") for k in keys)

    if any(k.startswith("backbone.conv_last.") for k in keys):
        upsampler = "pixelshuffle"
    else:
        upsampler = "pixelshuffledirect"

    if "backbone.conv_after_body.weight" in state_dict:
        resi_connection = "1conv"
    else:
        resi_connection = "3conv"

    upscale_factor = 2
    if "backbone.upsample.0.weight" in state_dict:
        up_w = state_dict["backbone.upsample.0.weight"]
        if upsampler == "pixelshuffledirect":
            ratio = up_w.shape[0] / float(max(out_channels, 1))
            upscale_factor = int(round(math.sqrt(ratio)))
        else:
            up_conv_keys = sorted([
                k for k in keys
                if k.startswith("backbone.upsample.") and k.endswith(".weight")
            ])
            upscale_factor = 2 ** len(up_conv_keys)

    if upscale_factor <= 0:
        upscale_factor = 2

    return {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "embed_dim": embed_dim,
        "depths": tuple(depths),
        "num_heads": tuple(num_heads),
        "window_size": int(window_size),
        "mlp_ratio": float(mlp_ratio),
        "qkv_bias": bool(qkv_bias),
        "patch_norm": bool(patch_norm),
        "upsampler": str(upsampler),
        "resi_connection": str(resi_connection),
        "upscale_factor": int(upscale_factor),
    }


def load_one_swinir_model(ckpt_path, scaler_x_stats, scaler_y_stats, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = strip_module_prefix(state_dict)

    inferred = infer_swinir_hparams_from_state_dict(state_dict)
    swin_cfg = ckpt.get("swinir_config", {})
    bic_cfg = ckpt.get("bicubic_baseline", {})

    in_channels = int(inferred["in_channels"])
    out_channels = int(inferred["out_channels"])
    upscale_factor = int(ckpt.get("upscale_factor", inferred["upscale_factor"]))

    embed_dim = int(swin_cfg.get("embed_dim", inferred["embed_dim"]))
    depths = tuple(int(x) for x in swin_cfg.get("depths", inferred["depths"]))
    num_heads = tuple(int(x) for x in swin_cfg.get("num_heads", inferred["num_heads"]))
    window_size = int(swin_cfg.get("window_size", inferred["window_size"]))
    mlp_ratio = float(swin_cfg.get("mlp_ratio", inferred["mlp_ratio"]))
    qkv_bias = bool(swin_cfg.get("qkv_bias", inferred["qkv_bias"]))
    patch_norm = bool(swin_cfg.get("patch_norm", inferred["patch_norm"]))
    upsampler = str(swin_cfg.get("upsampler", inferred["upsampler"]))
    resi_connection = str(swin_cfg.get("resi_connection", inferred["resi_connection"]))
    zero_init_last = bool(swin_cfg.get("zero_init_last", False))

    drop_rate = float(swin_cfg.get("drop_rate", 0.0))
    attn_drop_rate = float(swin_cfg.get("attn_drop_rate", 0.0))
    drop_path_rate = float(swin_cfg.get("drop_path_rate", 0.0))

    use_bicubic_baseline = bool(bic_cfg.get("enabled", ckpt.get("use_bicubic_baseline", True)))
    bicubic_align_corners = bool(bic_cfg.get("align_corners", ckpt.get("bicubic_align_corners", False)))
    residual_scale = float(bic_cfg.get("residual_scale", ckpt.get("residual_scale", 1.0)))

    x_mean = np.asarray(ckpt.get("scaler_X_mean", scaler_x_stats["mean"]), dtype=np.float32)
    x_std = np.asarray(ckpt.get("scaler_X_std", scaler_x_stats["std"]), dtype=np.float32)
    y_mean = np.asarray(ckpt.get("scaler_Y_mean", scaler_y_stats["mean"]), dtype=np.float32)
    y_std = np.asarray(ckpt.get("scaler_Y_std", scaler_y_stats["std"]), dtype=np.float32)

    model = SwinIRSRModel(
        in_channels=in_channels,
        out_channels=out_channels,
        upscale_factor=upscale_factor,
        scaler_X_mean=x_mean,
        scaler_X_std=x_std,
        scaler_Y_mean=y_mean,
        scaler_Y_std=y_std,
        use_bicubic_baseline=use_bicubic_baseline,
        bicubic_align_corners=bicubic_align_corners,
        residual_scale=residual_scale,
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
        use_checkpoint=False,
        upsampler=upsampler,
        resi_connection=resi_connection,
        zero_init_last=zero_init_last,
    )

    strict_loaded = True
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"[Warning] strict=True loading failed, falling back to strict=False. Reason: {e}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        strict_loaded = False
        if len(missing) > 0:
            print(f"   Missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if len(unexpected) > 0:
            print(f"   Unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    model.to(device)
    model.eval()

    info = {
        "upscale_factor": upscale_factor,
        "embed_dim": embed_dim,
        "depths": depths,
        "num_heads": num_heads,
        "window_size": window_size,
        "mlp_ratio": mlp_ratio,
        "upsampler": upsampler,
        "resi_connection": resi_connection,
        "patch_norm": patch_norm,
        "use_bicubic_baseline": use_bicubic_baseline,
        "bicubic_align_corners": bicubic_align_corners,
        "residual_scale": residual_scale,
        "strict_loaded": strict_loaded,
    }
    return model, info


# =============================================================================
# 4) Cubic Spline and Spherical Physics Calculations
# =============================================================================
def to_increasing_axis(axis_1d):
    axis_1d = np.asarray(axis_1d, dtype=np.float64)
    if axis_1d[1] > axis_1d[0]:
        return axis_1d, False
    return axis_1d[::-1], True


class CubicSplineInterpolator2D:
    """
    Interpolates low-resolution grid to high-resolution grid using 2D cubic spline.
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
    """
    Spherical differentiation and atmospheric physical quantity calculation (NumPy implementation).
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
# 5) Metric Accumulator and Sample Sufficient Statistics
# =============================================================================
def _ensure_2d(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def init_sample_stat_buffer(n_samples, n_vars, enable_image_metrics=False):
    buf = {
        "sum_abs": np.zeros((n_samples, n_vars), dtype=np.float64),
        "sum_sq": np.zeros((n_samples, n_vars), dtype=np.float64),
        "sum_rel": np.zeros((n_samples, n_vars), dtype=np.float64),
        "sum_pred": np.zeros((n_samples, n_vars), dtype=np.float64),
        "sum_true": np.zeros((n_samples, n_vars), dtype=np.float64),
        "sum_pred2": np.zeros((n_samples, n_vars), dtype=np.float64),
        "sum_true2": np.zeros((n_samples, n_vars), dtype=np.float64),
        "sum_pred_true": np.zeros((n_samples, n_vars), dtype=np.float64),
        "count": np.zeros((n_samples,), dtype=np.float64),
    }
    if enable_image_metrics:
        buf["ssim_sum"] = np.zeros((n_samples, n_vars), dtype=np.float64)
        buf["ssim_count"] = np.zeros((n_samples, n_vars), dtype=np.float64)
    return buf


def compute_sample_contributions(pred, true, mre_eps, data_range=None, enable_image_metrics=False):
    """
    pred/true: (B, C, H, W)
    Returns sufficient statistics for each sample, shape (B, C), count shape (B,)
    """
    pred = pred.astype(np.float64, copy=False)
    true = true.astype(np.float64, copy=False)

    B, C, H, W = pred.shape
    err = pred - true

    denom = np.maximum(np.abs(true), np.asarray(mre_eps, dtype=np.float64).reshape(1, -1, 1, 1))

    sum_abs = np.sum(np.abs(err), axis=(2, 3))
    sum_sq = np.sum(err ** 2, axis=(2, 3))
    sum_rel = np.sum(np.abs(err) / denom, axis=(2, 3))

    pred_flat = pred.reshape(B, C, -1)
    true_flat = true.reshape(B, C, -1)

    sum_pred = pred_flat.sum(axis=2)
    sum_true = true_flat.sum(axis=2)
    sum_pred2 = (pred_flat ** 2).sum(axis=2)
    sum_true2 = (true_flat ** 2).sum(axis=2)
    sum_pred_true = (pred_flat * true_flat).sum(axis=2)

    out = {
        "sum_abs": sum_abs,
        "sum_sq": sum_sq,
        "sum_rel": sum_rel,
        "sum_pred": sum_pred,
        "sum_true": sum_true,
        "sum_pred2": sum_pred2,
        "sum_true2": sum_true2,
        "sum_pred_true": sum_pred_true,
        "count": np.full((B,), H * W, dtype=np.float64),
    }

    if enable_image_metrics:
        ssim_sum = np.zeros((B, C), dtype=np.float64)
        ssim_count = np.zeros((B, C), dtype=np.float64)

        for b in range(B):
            for c in range(C):
                dr = float(data_range[c]) if data_range is not None else None
                if dr is None or (not np.isfinite(dr)) or dr <= 0:
                    continue
                try:
                    val = ssim_2d(
                        true[b, c], pred[b, c],
                        data_range=dr,
                        win_size=SSIM_WIN_SIZE
                    )
                except Exception:
                    val = np.nan
                if np.isfinite(val):
                    ssim_sum[b, c] = float(val)
                    ssim_count[b, c] = 1.0

        out["ssim_sum"] = ssim_sum
        out["ssim_count"] = ssim_count

    return out


def store_sample_contributions(buf, start_idx, contrib):
    B = contrib["sum_abs"].shape[0]
    ed = start_idx + B
    for k in BASE_STAT_KEYS:
        buf[k][start_idx:ed] = contrib[k]
    buf["count"][start_idx:ed] = contrib["count"]

    if "ssim_sum" in buf:
        buf["ssim_sum"][start_idx:ed] = contrib["ssim_sum"]
        buf["ssim_count"][start_idx:ed] = contrib["ssim_count"]


def totals_from_buffer(buf):
    out = {}
    for k in BASE_STAT_KEYS:
        out[k] = np.sum(buf[k], axis=0)
    out["count"] = float(np.sum(buf["count"]))
    if "ssim_sum" in buf:
        out["ssim_sum"] = np.sum(buf["ssim_sum"], axis=0)
        out["ssim_count"] = np.sum(buf["ssim_count"], axis=0)
    return out


def aggregate_buffer_with_weights(buf, W):
    """
    W: (R, N) Integer weight matrix
    Returns agg dict:
      base keys -> (R, C)
      count     -> (R,)
    """
    agg = {}
    for k in BASE_STAT_KEYS:
        agg[k] = W @ buf[k]
    agg["count"] = W @ buf["count"]

    if "ssim_sum" in buf:
        agg["ssim_sum"] = W @ buf["ssim_sum"]
        agg["ssim_count"] = W @ buf["ssim_count"]

    return agg


def compute_metrics_from_aggregated(agg, data_range=None, enable_image_metrics=False):
    sum_abs = _ensure_2d(agg["sum_abs"])
    sum_sq = _ensure_2d(agg["sum_sq"])
    sum_rel = _ensure_2d(agg["sum_rel"])

    sum_pred = _ensure_2d(agg["sum_pred"])
    sum_true = _ensure_2d(agg["sum_true"])
    sum_pred2 = _ensure_2d(agg["sum_pred2"])
    sum_true2 = _ensure_2d(agg["sum_true2"])
    sum_pred_true = _ensure_2d(agg["sum_pred_true"])

    count = np.asarray(agg["count"], dtype=np.float64)
    if count.ndim == 0:
        count = count.reshape(1)
    n = np.maximum(count.reshape(-1, 1), 1.0)

    mse = sum_sq / n
    rmse = np.sqrt(mse)
    mae = sum_abs / n
    mre = sum_rel / n

    num = n * sum_pred_true - sum_pred * sum_true
    den = np.sqrt(
        np.maximum(n * sum_pred2 - sum_pred ** 2, 0.0) *
        np.maximum(n * sum_true2 - sum_true ** 2, 0.0)
    )
    corr = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    out = {
        "RMSE": rmse,
        "MSE": mse,
        "MAE": mae,
        "MRE": mre,
        "Correlation": corr
    }

    if enable_image_metrics:
        dr = np.asarray(data_range, dtype=np.float64).reshape(1, -1)
        dr = np.maximum(dr, 1e-12)
        psnr = 20.0 * np.log10(dr) - 10.0 * np.log10(np.maximum(mse, 1e-12))
        ssim_sum = _ensure_2d(agg["ssim_sum"])
        ssim_count = _ensure_2d(agg["ssim_count"])
        ssim = np.divide(
            ssim_sum, ssim_count,
            out=np.full_like(ssim_sum, np.nan),
            where=ssim_count > 0
        )
        out["PSNR"] = psnr
        out["SSIM"] = ssim

    return out


class MetricAccumulator:
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

    def update_from_contrib(self, contrib):
        self.sum_abs += np.sum(contrib["sum_abs"], axis=0)
        self.sum_sq += np.sum(contrib["sum_sq"], axis=0)
        self.sum_rel += np.sum(contrib["sum_rel"], axis=0)

        n_pix = np.sum(contrib["count"])
        self.count += n_pix

        self.sum_pred += np.sum(contrib["sum_pred"], axis=0)
        self.sum_true += np.sum(contrib["sum_true"], axis=0)
        self.sum_pred2 += np.sum(contrib["sum_pred2"], axis=0)
        self.sum_true2 += np.sum(contrib["sum_true2"], axis=0)
        self.sum_pred_true += np.sum(contrib["sum_pred_true"], axis=0)

        if self.enable_image_metrics and ("ssim_sum" in contrib):
            self.ssim_sum += np.sum(contrib["ssim_sum"], axis=0)
            self.ssim_count += np.sum(contrib["ssim_count"], axis=0)

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
                self.ssim_sum, self.ssim_count,
                out=np.full_like(self.ssim_sum, np.nan),
                where=self.ssim_count > 0
            )
            out["PSNR"] = psnr
            out["SSIM"] = ssim

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
# 6) Pre-scanning and Main Evaluation
# =============================================================================
def precompute_raw_target_stats(ds_y, variables, time_dim, batch_size, max_samples=None):
    n_total = int(ds_y.sizes[time_dim])
    if max_samples is not None:
        n_total = min(n_total, int(max_samples))

    vmin = np.full(len(variables), np.inf, dtype=np.float64)
    vmax = np.full(len(variables), -np.inf, dtype=np.float64)
    sum_abs = np.zeros(len(variables), dtype=np.float64)
    count = 0.0

    for st in tqdm(range(0, n_total, batch_size), desc="Pre-scanning Y_test (raw)"):
        ed = min(st + batch_size, n_total)
        y = read_batch_from_ds(ds_y, variables, time_dim, st, ed)

        vmin = np.minimum(vmin, y.min(axis=(0, 2, 3)))
        vmax = np.maximum(vmax, y.max(axis=(0, 2, 3)))

        sum_abs += np.sum(np.abs(y), axis=(0, 2, 3))
        count += y.shape[0] * y.shape[2] * y.shape[3]

    data_range = vmax - vmin
    mean_abs = sum_abs / max(count, 1.0)
    return data_range, mean_abs


def precompute_physics_target_mean_abs(ds_y, variables, time_dim, batch_size, max_samples, physics_ops):
    n_total = int(ds_y.sizes[time_dim])
    if max_samples is not None:
        n_total = min(n_total, int(max_samples))

    sum_abs = np.zeros(len(PHYS_DERIVED_NAMES), dtype=np.float64)
    count = 0.0

    for st in tqdm(range(0, n_total, batch_size), desc="Pre-scanning Y_test (physics)"):
        ed = min(st + batch_size, n_total)
        y = read_batch_from_ds(ds_y, variables, time_dim, st, ed)
        phy = compute_physics_derived(y, physics_ops)

        sum_abs += np.sum(np.abs(phy), axis=(0, 2, 3))
        count += phy.shape[0] * phy.shape[2] * phy.shape[3]

    return sum_abs / max(count, 1.0)


def evaluate_all_methods(
    ds_x, ds_y, variables,
    time_dim_x, time_dim_y,
    models, cubic_interpolator, physics_ops,
    scaler_x_stats, scaler_y_stats,
    raw_data_range, raw_mre_eps, phys_mre_eps,
    batch_size, max_samples, device,
    amp_enabled=False, amp_dtype=torch.float16,
    collect_bootstrap_stats=False
):
    method_names = ["cubic spline"] + list(models.keys())

    raw_acc = OrderedDict()
    phys_acc = OrderedDict()

    for m in method_names:
        raw_acc[m] = MetricAccumulator(
            var_names=VAR_DISPLAY,
            mre_eps=raw_mre_eps,
            data_range=raw_data_range,
            enable_image_metrics=True
        )
        phys_acc[m] = MetricAccumulator(
            var_names=PHYS_DERIVED_NAMES,
            mre_eps=phys_mre_eps,
            data_range=None,
            enable_image_metrics=False
        )

    x_mean = scaler_x_stats["mean"]
    x_std = scaler_x_stats["std"]
    y_mean = scaler_y_stats["mean"]
    y_std = scaler_y_stats["std"]

    n_total = int(ds_x.sizes[time_dim_x])
    if max_samples is not None:
        n_total = min(n_total, int(max_samples))

    raw_buffers = None
    phys_buffers = None
    if collect_bootstrap_stats:
        raw_buffers = OrderedDict()
        phys_buffers = OrderedDict()
        for m in method_names:
            raw_buffers[m] = init_sample_stat_buffer(n_total, len(VAR_DISPLAY), enable_image_metrics=True)
            phys_buffers[m] = init_sample_stat_buffer(n_total, len(PHYS_DERIVED_NAMES), enable_image_metrics=False)

    use_amp = bool(amp_enabled and (device.type == "cuda"))

    for st in tqdm(range(0, n_total, batch_size), desc="Main evaluation loop"):
        ed = min(st + batch_size, n_total)

        x_raw = read_batch_from_ds(ds_x, variables, time_dim_x, st, ed)
        y_raw = read_batch_from_ds(ds_y, variables, time_dim_y, st, ed)

        true_phys = compute_physics_derived(y_raw, physics_ops)

        # ---------- Cubic Spline Baseline ----------
        pred_cubic = cubic_interpolator(x_raw)

        raw_contrib = compute_sample_contributions(
            pred=pred_cubic,
            true=y_raw,
            mre_eps=raw_mre_eps,
            data_range=raw_data_range,
            enable_image_metrics=True
        )
        raw_acc["cubic spline"].update_from_contrib(raw_contrib)
        if collect_bootstrap_stats:
            store_sample_contributions(raw_buffers["cubic spline"], st, raw_contrib)

        pred_cubic_phys = compute_physics_derived(pred_cubic, physics_ops)
        phys_contrib = compute_sample_contributions(
            pred=pred_cubic_phys,
            true=true_phys,
            mre_eps=phys_mre_eps,
            data_range=None,
            enable_image_metrics=False
        )
        phys_acc["cubic spline"].update_from_contrib(phys_contrib)
        if collect_bootstrap_stats:
            store_sample_contributions(phys_buffers["cubic spline"], st, phys_contrib)

        # ---------- SwinIR Models ----------
        if len(models) > 0:
            x_std_batch = standardize_batch(x_raw, x_mean, x_std)
            x_tensor = torch.from_numpy(x_std_batch).to(device, non_blocking=True)

            with torch.no_grad():
                for model_name, model in models.items():
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            pred_std = model(x_tensor)
                    else:
                        pred_std = model(x_tensor)

                    pred_std_np = pred_std.float().cpu().numpy().astype(np.float32)
                    pred_raw = inverse_standardize_batch(pred_std_np, y_mean, y_std)

                    raw_contrib = compute_sample_contributions(
                        pred=pred_raw,
                        true=y_raw,
                        mre_eps=raw_mre_eps,
                        data_range=raw_data_range,
                        enable_image_metrics=True
                    )
                    raw_acc[model_name].update_from_contrib(raw_contrib)
                    if collect_bootstrap_stats:
                        store_sample_contributions(raw_buffers[model_name], st, raw_contrib)

                    pred_phys = compute_physics_derived(pred_raw, physics_ops)
                    phys_contrib = compute_sample_contributions(
                        pred=pred_phys,
                        true=true_phys,
                        mre_eps=phys_mre_eps,
                        data_range=None,
                        enable_image_metrics=False
                    )
                    phys_acc[model_name].update_from_contrib(phys_contrib)
                    if collect_bootstrap_stats:
                        store_sample_contributions(phys_buffers[model_name], st, phys_contrib)

    raw_result = OrderedDict((k, v.finalize()) for k, v in raw_acc.items())
    phys_result = OrderedDict((k, v.finalize()) for k, v in phys_acc.items())
    return raw_result, phys_result, raw_buffers, phys_buffers, n_total


# =============================================================================
# 7) Bootstrap / Jackknife / Confidence Intervals
# =============================================================================
def normalize_ci_method(name: str):
    s = str(name).strip().lower()
    if s in ["bca", "bca."]:
        return "bca"
    if s in ["percentile", "basic", "bca"]:
        return s
    raise ValueError(f"Unsupported BOOTSTRAP_CI_METHOD={name}")


def normalize_bootstrap_scheme(name: str):
    s = str(name).strip().lower()
    valid = {"iid", "moving_block", "circular_block", "stationary"}
    if s not in valid:
        raise ValueError(f"Unsupported BOOTSTRAP_SCHEME={name}. Options: {sorted(valid)}")
    return s


def normalize_jackknife_mode(name: str, scheme: str):
    s = str(name).strip().lower()
    if s == "auto":
        return "delete1" if scheme == "iid" else "block"
    if s not in {"delete1", "block"}:
        raise ValueError(f"Unsupported BOOTSTRAP_JACKKNIFE={name}. Options: auto/delete1/block")
    return s


def resolve_block_length(n_samples, spec):
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    if isinstance(spec, str):
        s = spec.strip().lower()
        if s == "auto":
            L = int(round(n_samples ** (1.0 / 3.0)))
        elif s == "sqrt":
            L = int(round(n_samples ** 0.5))
        elif s.isdigit():
            L = int(s)
        else:
            raise ValueError(f"Unsupported BOOTSTRAP_BLOCK_LENGTH={spec}")
    else:
        L = int(spec)

    L = max(1, min(n_samples, L))
    return L


def sample_indices_iid(n, rng):
    return rng.integers(0, n, size=n, dtype=np.int64)


def sample_indices_moving_block(n, L, rng):
    n_blocks = int(math.ceil(n / float(L)))
    max_start = max(n - L, 0)
    starts = rng.integers(0, max_start + 1, size=n_blocks, dtype=np.int64)

    offsets = np.arange(L, dtype=np.int64)
    idx = (starts[:, None] + offsets[None, :]).reshape(-1)
    idx = np.clip(idx, 0, n - 1)
    return idx[:n]


def sample_indices_circular_block(n, L, rng):
    n_blocks = int(math.ceil(n / float(L)))
    starts = rng.integers(0, n, size=n_blocks, dtype=np.int64)

    offsets = np.arange(L, dtype=np.int64)
    idx = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)
    return idx[:n]


def sample_indices_stationary(n, L, rng):
    p = 1.0 / float(max(L, 1))
    idx = np.empty(n, dtype=np.int64)

    idx[0] = rng.integers(0, n)
    new_starts = rng.integers(0, n, size=n - 1)
    u = rng.random(n - 1)

    for t in range(1, n):
        if u[t - 1] < p:
            idx[t] = new_starts[t - 1]
        else:
            idx[t] = (idx[t - 1] + 1) % n
    return idx


def sample_one_resample_indices(n, scheme, block_length, rng):
    if scheme == "iid":
        return sample_indices_iid(n, rng)
    if scheme == "moving_block":
        return sample_indices_moving_block(n, block_length, rng)
    if scheme == "circular_block":
        return sample_indices_circular_block(n, block_length, rng)
    if scheme == "stationary":
        return sample_indices_stationary(n, block_length, rng)
    raise ValueError(f"Unknown scheme: {scheme}")


def generate_bootstrap_weight_chunk(n_samples, n_resamples_chunk, scheme, block_length, rng):
    W = np.zeros((n_resamples_chunk, n_samples), dtype=np.float64)
    for r in range(n_resamples_chunk):
        idx = sample_one_resample_indices(n_samples, scheme, block_length, rng)
        W[r] = np.bincount(idx, minlength=n_samples).astype(np.float64)
    return W


def run_bootstrap_distributions(
    raw_buffers, phys_buffers,
    raw_data_range,
    n_resamples, scheme, block_length, chunk_size, seed
):
    method_names = list(raw_buffers.keys())
    n_samples = raw_buffers[method_names[0]]["count"].shape[0]

    nv_raw = raw_buffers[method_names[0]]["sum_abs"].shape[1]
    nv_phys = phys_buffers[method_names[0]]["sum_abs"].shape[1]

    boot_raw = OrderedDict()
    boot_phys = OrderedDict()
    for m in method_names:
        boot_raw[m] = {metric: np.empty((n_resamples, nv_raw), dtype=np.float64) for metric in RAW_METRIC_ORDER}
        boot_phys[m] = {metric: np.empty((n_resamples, nv_phys), dtype=np.float64) for metric in PHYS_METRIC_ORDER}

    rng = np.random.default_rng(seed)

    for st in tqdm(range(0, n_resamples, chunk_size), desc="Bootstrap resampling"):
        ed = min(st + chunk_size, n_resamples)
        k = ed - st

        W = generate_bootstrap_weight_chunk(
            n_samples=n_samples,
            n_resamples_chunk=k,
            scheme=scheme,
            block_length=block_length,
            rng=rng
        )

        for m in method_names:
            agg_raw = aggregate_buffer_with_weights(raw_buffers[m], W)
            met_raw = compute_metrics_from_aggregated(
                agg_raw,
                data_range=raw_data_range,
                enable_image_metrics=True
            )
            for metric in RAW_METRIC_ORDER:
                boot_raw[m][metric][st:ed, :] = met_raw[metric]

            agg_phys = aggregate_buffer_with_weights(phys_buffers[m], W)
            met_phys = compute_metrics_from_aggregated(
                agg_phys,
                data_range=None,
                enable_image_metrics=False
            )
            for metric in PHYS_METRIC_ORDER:
                boot_phys[m][metric][st:ed, :] = met_phys[metric]

    return boot_raw, boot_phys


def run_jackknife_distributions(
    raw_buffers, phys_buffers,
    raw_data_range,
    mode, block_length
):
    """
    Returns:
      jack_raw[model][metric]  : (J, C_raw)
      jack_phys[model][metric] : (J, C_phys)
      used_mode, J
    """
    method_names = list(raw_buffers.keys())
    n_samples = raw_buffers[method_names[0]]["count"].shape[0]

    nv_raw = raw_buffers[method_names[0]]["sum_abs"].shape[1]
    nv_phys = phys_buffers[method_names[0]]["sum_abs"].shape[1]

    jack_raw = OrderedDict()
    jack_phys = OrderedDict()

    if mode == "delete1":
        J = n_samples
        for m in method_names:
            jack_raw[m] = {metric: np.empty((J, nv_raw), dtype=np.float64) for metric in RAW_METRIC_ORDER}
            jack_phys[m] = {metric: np.empty((J, nv_phys), dtype=np.float64) for metric in PHYS_METRIC_ORDER}

        for m in tqdm(method_names, desc="Jackknife (delete-1)"):
            # Raw
            total_raw = totals_from_buffer(raw_buffers[m])
            keep_raw = {k: total_raw[k][None, :] - raw_buffers[m][k] for k in BASE_STAT_KEYS}
            keep_raw["count"] = total_raw["count"] - raw_buffers[m]["count"]
            keep_raw["ssim_sum"] = total_raw["ssim_sum"][None, :] - raw_buffers[m]["ssim_sum"]
            keep_raw["ssim_count"] = total_raw["ssim_count"][None, :] - raw_buffers[m]["ssim_count"]

            met_raw = compute_metrics_from_aggregated(
                keep_raw,
                data_range=raw_data_range,
                enable_image_metrics=True
            )
            for metric in RAW_METRIC_ORDER:
                jack_raw[m][metric][:, :] = met_raw[metric]

            # Physics
            total_phys = totals_from_buffer(phys_buffers[m])
            keep_phys = {k: total_phys[k][None, :] - phys_buffers[m][k] for k in BASE_STAT_KEYS}
            keep_phys["count"] = total_phys["count"] - phys_buffers[m]["count"]

            met_phys = compute_metrics_from_aggregated(
                keep_phys,
                data_range=None,
                enable_image_metrics=False
            )
            for metric in PHYS_METRIC_ORDER:
                jack_phys[m][metric][:, :] = met_phys[metric]

        return jack_raw, jack_phys, "delete1", J

    elif mode == "block":
        # Non-overlapping block-delete jackknife
        starts = np.arange(0, n_samples, block_length, dtype=np.int64)
        J = len(starts)
        offsets = np.arange(block_length, dtype=np.int64)
        idx_mat = (starts[:, None] + offsets[None, :]) % n_samples  # (J, L)

        for m in method_names:
            jack_raw[m] = {metric: np.empty((J, nv_raw), dtype=np.float64) for metric in RAW_METRIC_ORDER}
            jack_phys[m] = {metric: np.empty((J, nv_phys), dtype=np.float64) for metric in PHYS_METRIC_ORDER}

        for m in tqdm(method_names, desc="Jackknife (block-delete)"):
            # Raw
            total_raw = totals_from_buffer(raw_buffers[m])

            del_raw = {k: raw_buffers[m][k][idx_mat].sum(axis=1) for k in BASE_STAT_KEYS}
            del_count_raw = raw_buffers[m]["count"][idx_mat].sum(axis=1)
            del_ssim_sum = raw_buffers[m]["ssim_sum"][idx_mat].sum(axis=1)
            del_ssim_count = raw_buffers[m]["ssim_count"][idx_mat].sum(axis=1)

            keep_raw = {k: total_raw[k][None, :] - del_raw[k] for k in BASE_STAT_KEYS}
            keep_raw["count"] = total_raw["count"] - del_count_raw
            keep_raw["ssim_sum"] = total_raw["ssim_sum"][None, :] - del_ssim_sum
            keep_raw["ssim_count"] = total_raw["ssim_count"][None, :] - del_ssim_count

            met_raw = compute_metrics_from_aggregated(
                keep_raw,
                data_range=raw_data_range,
                enable_image_metrics=True
            )
            for metric in RAW_METRIC_ORDER:
                jack_raw[m][metric][:, :] = met_raw[metric]

            # Physics
            total_phys = totals_from_buffer(phys_buffers[m])
            del_phys = {k: phys_buffers[m][k][idx_mat].sum(axis=1) for k in BASE_STAT_KEYS}
            del_count_phys = phys_buffers[m]["count"][idx_mat].sum(axis=1)

            keep_phys = {k: total_phys[k][None, :] - del_phys[k] for k in BASE_STAT_KEYS}
            keep_phys["count"] = total_phys["count"] - del_count_phys

            met_phys = compute_metrics_from_aggregated(
                keep_phys,
                data_range=None,
                enable_image_metrics=False
            )
            for metric in PHYS_METRIC_ORDER:
                jack_phys[m][metric][:, :] = met_phys[metric]

        return jack_raw, jack_phys, "block", J

    else:
        raise ValueError(f"Unsupported jackknife mode={mode}")


def ci_percentile(boot, conf):
    alpha = (1.0 - conf) / 2.0
    lo = np.quantile(boot, alpha)
    hi = np.quantile(boot, 1.0 - alpha)
    return float(lo), float(hi)


def ci_basic(theta_hat, boot, conf):
    alpha = (1.0 - conf) / 2.0
    q_low = np.quantile(boot, alpha)
    q_high = np.quantile(boot, 1.0 - alpha)
    lo = 2.0 * theta_hat - q_high
    hi = 2.0 * theta_hat - q_low
    if lo > hi:
        lo, hi = hi, lo
    return float(lo), float(hi)


def ci_bca(theta_hat, boot, jack, conf):
    """
    BCa interval:
      z0 estimated from bootstrap bias
      a  estimated from jackknife acceleration
    """
    boot = np.asarray(boot, dtype=np.float64)
    jack = np.asarray(jack, dtype=np.float64)

    boot = boot[np.isfinite(boot)]
    jack = jack[np.isfinite(jack)]

    if (not np.isfinite(theta_hat)) or boot.size < 10 or jack.size < 3:
        return np.nan, np.nan

    prop = np.mean(boot < theta_hat)
    prop = np.clip(prop, 1e-8, 1.0 - 1e-8)
    z0 = norm.ppf(prop)

    jack_mean = np.mean(jack)
    dif = jack_mean - jack
    num = np.sum(dif ** 3)
    den = 6.0 * (np.sum(dif ** 2) ** 1.5)
    a = (num / den) if den > 0 else 0.0

    alpha = (1.0 - conf) / 2.0
    z_alpha1 = norm.ppf(alpha)
    z_alpha2 = norm.ppf(1.0 - alpha)

    def adjusted_alpha(z_alpha):
        denom = 1.0 - a * (z0 + z_alpha)
        if abs(denom) < 1e-12:
            denom = 1e-12 if denom >= 0 else -1e-12
        z_adj = z0 + (z0 + z_alpha) / denom
        return float(np.clip(norm.cdf(z_adj), 0.0, 1.0))

    a1 = adjusted_alpha(z_alpha1)
    a2 = adjusted_alpha(z_alpha2)

    lo = np.quantile(boot, a1)
    hi = np.quantile(boot, a2)
    if lo > hi:
        lo, hi = hi, lo
    return float(lo), float(hi)


def bootstrap_confidence_interval(theta_hat, boot_dist, conf, ci_method, jack_dist=None):
    boot = np.asarray(boot_dist, dtype=np.float64)
    boot = boot[np.isfinite(boot)]

    if boot.size < 2 or (not np.isfinite(theta_hat)):
        return np.nan, np.nan

    if ci_method == "percentile":
        return ci_percentile(boot, conf)
    elif ci_method == "basic":
        return ci_basic(theta_hat, boot, conf)
    elif ci_method == "bca":
        if jack_dist is None:
            # Fallback to percentile if jackknife distribution is missing
            return ci_percentile(boot, conf)
        return ci_bca(theta_hat, boot, jack_dist, conf)
    else:
        raise ValueError(f"Unknown ci_method={ci_method}")


def build_bootstrap_ci_dataframe(
    point_result,
    boot_result,
    jack_result,
    var_names,
    metric_order,
    domain_name,
    ci_method,
    confidence,
    n_resamples,
    scheme,
    block_length,
    jackknife_mode,
    jackknife_reps
):
    rows = []
    for model_name, metric_map in point_result.items():
        for metric in metric_order:
            point_vals = np.asarray(metric_map[metric], dtype=np.float64)
            boot_vals = np.asarray(boot_result[model_name][metric], dtype=np.float64)  # (B, C)

            if jack_result is not None:
                jack_vals_all = np.asarray(jack_result[model_name][metric], dtype=np.float64)  # (J, C)
            else:
                jack_vals_all = None

            for i, vname in enumerate(var_names):
                theta = float(point_vals[i])
                boot_i = boot_vals[:, i]
                jack_i = jack_vals_all[:, i] if jack_vals_all is not None else None

                lo, hi = bootstrap_confidence_interval(
                    theta_hat=theta,
                    boot_dist=boot_i,
                    conf=confidence,
                    ci_method=ci_method,
                    jack_dist=jack_i
                )

                rows.append({
                    "Domain": domain_name,
                    "Model": model_name,
                    "Variable": vname,
                    "Metric": metric,
                    "PointEstimate": theta,
                    "CI_Lower": lo,
                    "CI_Upper": hi,
                    "CI_Method": ci_method,
                    "Confidence": confidence,
                    "N_Resamples": int(n_resamples),
                    "Bootstrap_Scheme": scheme,
                    "Block_Length": int(block_length),
                    "Jackknife_Mode": jackknife_mode,
                    "Jackknife_Replicates": int(jackknife_reps),
                })

    return pd.DataFrame(rows)


# =============================================================================
# 8) Main Execution
# =============================================================================
def main():
    set_seed(RANDOM_SEED)
    setup_torch_backend()
    device = torch.device(DEVICE)
    amp_dtype = get_amp_dtype(AMP_DTYPE)

    print("=" * 110)
    print("Starting ERA5 SwinIR Super-Resolution Evaluation")
    print(f"Device: {device}")
    print("=" * 110)

    must_exist = [X_TEST_PATH, Y_TEST_PATH, SCALER_PATH] + list(MODEL_PATHS.values())
    for p in must_exist:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    ds_x = xr.open_dataset(X_TEST_PATH, decode_times=False)
    ds_y = xr.open_dataset(Y_TEST_PATH, decode_times=False)

    try:
        for v in VARIABLES:
            if v not in ds_x.variables:
                raise KeyError(f"{X_TEST_PATH} missing variable: {v}")
            if v not in ds_y.variables:
                raise KeyError(f"{Y_TEST_PATH} missing variable: {v}")

        time_dim_x = pick_dim_name(ds_x, ["time", "sample", "t"])
        time_dim_y = pick_dim_name(ds_y, ["time", "sample", "t"])

        lat_name_x = pick_coord_name(ds_x, ["latitude", "lat"])
        lon_name_x = pick_coord_name(ds_x, ["longitude", "lon"])
        lat_name_y = pick_coord_name(ds_y, ["latitude", "lat"])
        lon_name_y = pick_coord_name(ds_y, ["longitude", "lon"])

        n_x = int(ds_x.sizes[time_dim_x])
        n_y = int(ds_y.sizes[time_dim_y])
        if n_x != n_y:
            raise ValueError(f"Time dimension length mismatch: X={n_x}, Y={n_y}")

        lat_lr = ds_x[lat_name_x].values.astype(np.float64)
        lon_lr = ds_x[lon_name_x].values.astype(np.float64)
        lat_hr = ds_y[lat_name_y].values.astype(np.float64)
        lon_hr = ds_y[lon_name_y].values.astype(np.float64)

        print(f"X_test shape: time={n_x}, lat={len(lat_lr)}, lon={len(lon_lr)}")
        print(f"Y_test shape: time={n_y}, lat={len(lat_hr)}, lon={len(lon_hr)}")

        scale_h = len(lat_hr) / float(len(lat_lr))
        scale_w = len(lon_hr) / float(len(lon_lr))
        if abs(scale_h - round(scale_h)) > 1e-6 or abs(scale_w - round(scale_w)) > 1e-6:
            raise ValueError(f"Upscaling ratio is non-integer: scale_h={scale_h}, scale_w={scale_w}")
        if int(round(scale_h)) != int(round(scale_w)):
            raise ValueError(f"Lat/lon upscaling ratio mismatch: scale_h={scale_h}, scale_w={scale_w}")
        data_upscale = int(round(scale_h))
        print(f"Dataset scaling factor: x{data_upscale}")

        scaler_x_stats, scaler_y_stats = load_scaler_stats(SCALER_PATH)
        print("Loaded scalers.joblib successfully.")

        models = OrderedDict()
        print("\nLoading SwinIR models...")
        for name, path in MODEL_PATHS.items():
            model, info = load_one_swinir_model(path, scaler_x_stats, scaler_y_stats, device)

            if int(info["upscale_factor"]) != data_upscale:
                raise ValueError(
                    f"Model '{name}' upscale_factor ({info['upscale_factor']}) "
                    f"does not match dataset scale (x{data_upscale})."
                )

            models[name] = model
            n_params = sum(p.numel() for p in model.parameters())
            print(
                f"  - {name:18s} | scale=x{info['upscale_factor']} | "
                f"embed={info['embed_dim']} | depths={list(info['depths'])} | "
                f"heads={list(info['num_heads'])} | ws={info['window_size']} | "
                f"bicubic_base={info['use_bicubic_baseline']} | params={n_params:,}"
            )

        cubic_interpolator = CubicSplineInterpolator2D(
            lat_lr=lat_lr, lon_lr=lon_lr,
            lat_hr=lat_hr, lon_hr=lon_hr,
            kx=CUBIC_SPLINE_KX, ky=CUBIC_SPLINE_KY
        )
        print("\nCubic spline interpolator initialized.")

        physics_ops = SphericalPhysicsOps(latitudes=lat_hr, longitudes=lon_hr)
        print("Spherical physics operator initialized.")

        raw_data_range, raw_mean_abs = precompute_raw_target_stats(
            ds_y=ds_y,
            variables=VARIABLES,
            time_dim=time_dim_y,
            batch_size=BATCH_SIZE,
            max_samples=EVAL_MAX_SAMPLES
        )
        raw_mre_eps = np.maximum(MRE_EPS_ABS, MRE_EPS_REL * raw_mean_abs)

        print("\nRaw variable data_range and MRE epsilon:")
        for i, name in enumerate(VAR_DISPLAY):
            print(f"  {name}: data_range={raw_data_range[i]:.6e}, mre_eps={raw_mre_eps[i]:.6e}")

        phys_mean_abs = precompute_physics_target_mean_abs(
            ds_y=ds_y,
            variables=VARIABLES,
            time_dim=time_dim_y,
            batch_size=BATCH_SIZE,
            max_samples=EVAL_MAX_SAMPLES,
            physics_ops=physics_ops
        )
        phys_mre_eps = np.maximum(MRE_EPS_ABS, MRE_EPS_REL * phys_mean_abs)

        print("\nPhysics variable MRE epsilon:")
        for i, name in enumerate(PHYS_DERIVED_NAMES):
            print(f"  {name}: mre_eps={phys_mre_eps[i]:.6e}")

        raw_result, phys_result, raw_buffers, phys_buffers, n_eval = evaluate_all_methods(
            ds_x=ds_x,
            ds_y=ds_y,
            variables=VARIABLES,
            time_dim_x=time_dim_x,
            time_dim_y=time_dim_y,
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
            amp_enabled=USE_AMP_INFERENCE,
            amp_dtype=amp_dtype,
            collect_bootstrap_stats=ENABLE_BOOTSTRAP
        )

        # Baseline evaluation metrics output
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

        df_raw.to_csv(CSV_RAW_METRICS_PATH, index=False, float_format="%.6e")
        df_phys.to_csv(CSV_PHYS_METRICS_PATH, index=False, float_format="%.6e")

        df_all = pd.concat([df_raw.set_index("Model"), df_phys.set_index("Model")], axis=1).reset_index()
        df_all.to_csv(CSV_ALL_IN_ONE_PATH, index=False, float_format="%.6e")

        print("\n" + "=" * 110)
        print("Base evaluation completed. CSV files saved:")
        print(f"  1) {CSV_RAW_METRICS_PATH}")
        print(f"  2) {CSV_PHYS_METRICS_PATH}")
        print(f"  3) {CSV_ALL_IN_ONE_PATH}")
        print("=" * 110)

        # ----------------------------- Bootstrap CI -----------------------------
        if ENABLE_BOOTSTRAP:
            print("\nEnabling Bootstrap CI estimation...")
            ci_method = normalize_ci_method(BOOTSTRAP_CI_METHOD)
            scheme = normalize_bootstrap_scheme(BOOTSTRAP_SCHEME)
            block_len = resolve_block_length(n_eval, BOOTSTRAP_BLOCK_LENGTH)
            jk_mode = normalize_jackknife_mode(BOOTSTRAP_JACKKNIFE, scheme)

            print(f"  n_eval_samples       = {n_eval}")
            print(f"  CI method            = {ci_method}")
            print(f"  Confidence level     = {BOOTSTRAP_CONFIDENCE}")
            print(f"  n_resamples          = {BOOTSTRAP_N_RESAMPLES}")
            print(f"  Resampling scheme    = {scheme}")
            print(f"  Block length         = {block_len}")
            print(f"  Jackknife mode(auto) = {jk_mode if ci_method == 'bca' else 'not_used'}")
            print(f"  Chunk size           = {BOOTSTRAP_CHUNK}")
            print(f"  Bootstrap seed       = {BOOTSTRAP_SEED}")

            if raw_buffers is None or phys_buffers is None:
                raise RuntimeError("Bootstrap enabled, but sample statistics buffer is missing.")

            boot_raw, boot_phys = run_bootstrap_distributions(
                raw_buffers=raw_buffers,
                phys_buffers=phys_buffers,
                raw_data_range=raw_data_range,
                n_resamples=int(BOOTSTRAP_N_RESAMPLES),
                scheme=scheme,
                block_length=block_len,
                chunk_size=int(BOOTSTRAP_CHUNK),
                seed=int(BOOTSTRAP_SEED),
            )

            jack_raw = None
            jack_phys = None
            jk_used = "not_used"
            jk_reps = 0

            if ci_method == "bca":
                jack_raw, jack_phys, jk_used, jk_reps = run_jackknife_distributions(
                    raw_buffers=raw_buffers,
                    phys_buffers=phys_buffers,
                    raw_data_range=raw_data_range,
                    mode=jk_mode,
                    block_length=block_len
                )
                print(f"  BCa Jackknife completed: mode={jk_used}, replicates={jk_reps}")

            df_raw_ci = build_bootstrap_ci_dataframe(
                point_result=raw_result,
                boot_result=boot_raw,
                jack_result=jack_raw,
                var_names=VAR_DISPLAY,
                metric_order=RAW_METRIC_ORDER,
                domain_name="raw",
                ci_method=ci_method,
                confidence=float(BOOTSTRAP_CONFIDENCE),
                n_resamples=int(BOOTSTRAP_N_RESAMPLES),
                scheme=scheme,
                block_length=block_len,
                jackknife_mode=jk_used,
                jackknife_reps=jk_reps
            )

            df_phys_ci = build_bootstrap_ci_dataframe(
                point_result=phys_result,
                boot_result=boot_phys,
                jack_result=jack_phys,
                var_names=PHYS_DERIVED_NAMES,
                metric_order=PHYS_METRIC_ORDER,
                domain_name="physics",
                ci_method=ci_method,
                confidence=float(BOOTSTRAP_CONFIDENCE),
                n_resamples=int(BOOTSTRAP_N_RESAMPLES),
                scheme=scheme,
                block_length=block_len,
                jackknife_mode=jk_used,
                jackknife_reps=jk_reps
            )

            df_raw_ci.to_csv(CSV_RAW_CI_PATH, index=False, float_format="%.6e")
            df_phys_ci.to_csv(CSV_PHYS_CI_PATH, index=False, float_format="%.6e")
            df_all_ci = pd.concat([df_raw_ci, df_phys_ci], axis=0, ignore_index=True)
            df_all_ci.to_csv(CSV_ALL_CI_PATH, index=False, float_format="%.6e")

            print("\nBootstrap CI completed. CSV files saved:")
            print(f"  4) {CSV_RAW_CI_PATH}")
            print(f"  5) {CSV_PHYS_CI_PATH}")
            print(f"  6) {CSV_ALL_CI_PATH}")

        print("\n" + "=" * 110)
        print("All workflows completed successfully.")
        print("=" * 110)

    finally:
        ds_x.close()
        ds_y.close()


if __name__ == "__main__":
    main()