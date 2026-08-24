#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA5 Super-Resolution Model Evaluation and Publication-Grade Visualization Script
- Designed for GitHub repository publication and manuscript submission (e.g., AGU JAMES).
- Evaluates super-resolution models across three extreme weather events:
  1) Meiyu Front (Background field: |∇MSE| - Moist Static Energy gradient magnitude)
  2) Typhoon (Background field: Wind Speed)
  3) Cold Surge (Background field: |Vag| - Ageostrophic wind speed)
- Generates 2x4 publication-grade comparative panels (Top: Absolute fields; Bottom: Reconstruction error fields).
- Features full font size scaling, geopotential height contours, wind vectors, and dynamic quiver keys.
"""

import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# 1. Hyperparameters and Plotting Configurations
# ============================================================================
# --- File Paths ---
X_TEST_PATH = 'X2_test.nc'
Y_TEST_PATH = 'Y_test.nc'
SCALER_PATH = 'scalers.joblib'
MODEL_NOPHY_PATH = 'best_model_noPhy.pth'
MODEL_ALLPHY_PATH = 'best_model_AllPhy.pth'

# --- Model Parameters (Must match training configurations) ---
NUM_CHANNELS = 5
NUM_FEATURES = 64
NUM_RES_BLOCKS = 8
UPSCALE_FACTOR = 2
VARIABLES = ['z', 'q', 't', 'u', 'v']

# --- Weather Case Time Indices ---
CASE_INDICES = {
    'Meiyu': 3816,      # Meiyu Heavy Rainfall Case
    'Typhoon': 6348,    # Typhoon Case
    'ColdSurge': 1440   # Cold Surge Case
}

# --- Global Figure Settings ---
DPI = 600
QUIVER_SKIP_HR = 4
QUIVER_SKIP_LR = 2

# --- Reference Quiver Key Positions (Axes Coordinates) ---
# Top-right above subplot frame: X close to 1.0, Y > 1.0
QUIVERKEY_MAIN_POS = {'X': 0.98, 'Y': 1.08, 'labelpos': 'S'}  # Subplot (a)
QUIVERKEY_ERR_POS  = {'X': 0.98, 'Y': 1.08, 'labelpos': 'S'}  # Subplot (f)

# --- Typography and Font Size Configurations ---
FONT_PANEL_TITLE = 20
FONT_GRID_LABEL = 16
FONT_CONTOUR_LABEL = 14
FONT_CBAR_LABEL = 18
FONT_CBAR_TICK = 16
FONT_QUIVER_KEY = 14

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': FONT_PANEL_TITLE,
    'axes.labelsize': 14,
    'xtick.labelsize': FONT_CBAR_TICK,
    'ytick.labelsize': FONT_CBAR_TICK
})

# --- Physical Constants ---
EARTH_RADIUS = 6_371_000.0        # Earth radius (m)
OMEGA = 7.2921159e-5              # Angular velocity of Earth (rad/s)
CP_AIR = 1004.64                  # Specific heat capacity of air at constant pressure (J/(kg·K))
LV = 2.5e6                        # Latent heat of vaporization (J/kg)

# --- Custom Windy Color Palette ---
windy_colors = [
    '#313695', '#4575b4', '#74add1', '#abdda4', '#66bd63',
    '#ffffbf', '#fee08b', '#fdae61', '#f46d43', '#d73027',
    '#a50026', '#800080', '#49006a'
]
windy_cmap = mcolors.LinearSegmentedColormap.from_list("windy_wind", windy_colors)

# --- Case-Specific Visualization Configurations ---
PLOT_CONFIGS = {
    # Meiyu Case: Background variable set to |∇MSE|
    'Meiyu': {
        'bg_var': 'mse_grad_abs',
        'cmap': 'YlGnBu',
        'vmin': 0.0, 'vmax': 0.08, 'levels': 21,
        'auto_abs_range': False, 'force_abs_vmin_zero': True,

        'z_levels': np.arange(56000, 59000, 200),

        'err_vmin': -0.02, 'err_vmax': 0.02, 'err_cmap': 'BrBG',
        'auto_err_range': False,

        'quiver_scale': 500, 'quiver_err_scale': 15,
        'ref_wind': 30, 'ref_err_wind': 1
    },

    # Typhoon Case: Background variable set to Wind Speed
    'Typhoon': {
        'bg_var': 'wind_speed',
        'cmap': windy_cmap,
        'vmin': 0, 'vmax': 40, 'levels': 21,
        'auto_abs_range': False, 'force_abs_vmin_zero': True,

        'z_levels': np.arange(55000, 59000, 200),

        'err_vmin': -2, 'err_vmax': 2, 'err_cmap': 'RdBu_r',
        'auto_err_range': False,

        'quiver_scale': 500, 'quiver_err_scale': 25,
        'ref_wind': 30, 'ref_err_wind': 2
    },

    # Cold Surge Case: Background variable set to |Vag|
    'ColdSurge': {
        'bg_var': 'vag_speed',
        'cmap': 'YlOrRd',
        'vmin': 0.0, 'vmax': 20.0, 'levels': 21,
        'auto_abs_range': False, 'force_abs_vmin_zero': True,

        'z_levels': np.arange(50000, 58000, 200),

        'err_vmin': -10.0, 'err_vmax': 10.0, 'err_cmap': 'RdBu_r',
        'auto_err_range': False,

        'quiver_scale': 500, 'quiver_err_scale': 15,
        'ref_wind': 30, 'ref_err_wind': 1
    }
}

BG_VAR_LABELS = {
    'q': 'Specific Humidity (kg/kg)',
    't': 'Temperature (K)',
    'wind_speed': 'Wind Speed (m/s)',
    'mse_grad_abs': '|∇MSE| (J kg⁻¹ m⁻¹)',
    'vag_speed': '|Va| (m/s)'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# 2. Helper Classes and Model Architectures
# ============================================================================
class ChannelScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def transform(self, data):
        mean = self.mean.reshape(1, -1, 1, 1).astype(np.float32)
        std = self.std.reshape(1, -1, 1, 1).astype(np.float32)
        return (data - mean) / std

    def inverse_transform_numpy(self, data):
        mean = self.mean.reshape(1, -1, 1, 1).astype(np.float32)
        std = self.std.reshape(1, -1, 1, 1).astype(np.float32)
        return data * std + mean


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))) + x


class ResidualCorrectionBackbone(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_res_blocks, upscale_factor):
        super().__init__()
        self.shallow_feature = nn.Conv2d(in_channels, num_features, kernel_size=9, padding=4)
        self.deep_features = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_res_blocks)])
        self.transition = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )
        self.post_shuffle_fuse = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.post_shuffle_act = nn.ReLU(inplace=True)
        self.output_conv = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        shallow = self.shallow_feature(x)
        deep = self.transition(self.deep_features(shallow))
        up = self.post_shuffle_act(self.post_shuffle_fuse(self.upsample(deep + shallow)))
        return self.output_conv(up)


class BicubicResidualSRModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_res_blocks, upscale_factor, scaler_X, scaler_Y):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.backbone = ResidualCorrectionBackbone(
            in_channels, out_channels, num_features, num_res_blocks, upscale_factor
        )

        alpha = (scaler_X.std / scaler_Y.std).astype(np.float32)
        beta = ((scaler_X.mean - scaler_Y.mean) / scaler_Y.std).astype(np.float32)
        self.register_buffer('x2y_alpha', torch.tensor(alpha).view(1, -1, 1, 1))
        self.register_buffer('x2y_beta', torch.tensor(beta).view(1, -1, 1, 1))

    def forward(self, x_std):
        base_x_std = F.interpolate(x_std, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        base_y_std = base_x_std * self.x2y_alpha + self.x2y_beta
        return base_y_std + self.backbone(x_std)


# ============================================================================
# 3. Diagnostic Calculation Utilities
# ============================================================================
def load_model_state(model, model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)


def compute_spatial_gradients(field_2d, lats_deg, lons_deg):
    """
    Computes horizontal spatial gradients (dfield/dx, dfield/dy) on a spherical grid.
    Returns gradients in units per meter.
    """
    lat_rad = np.deg2rad(lats_deg)
    lon_rad = np.deg2rad(lons_deg)

    dfield_dlat, dfield_dlon = np.gradient(field_2d, lat_rad, lon_rad, edge_order=2)

    cos_lat = np.cos(lat_rad)[:, None]
    cos_lat = np.clip(cos_lat, 1e-6, None)

    dfield_dx = dfield_dlon / (EARTH_RADIUS * cos_lat)
    dfield_dy = dfield_dlat / EARTH_RADIUS
    return dfield_dx, dfield_dy


def compute_mse_gradient_abs(data_array, lats, lons):
    """
    Computes Moist Static Energy (MSE = CP_AIR * T + LV * q + z) and its gradient magnitude |∇MSE|.
    """
    z = data_array[0]
    q = data_array[1]
    t = data_array[2]

    mse = CP_AIR * t + LV * q + z
    d_mse_dx, d_mse_dy = compute_spatial_gradients(mse, lats, lons)
    return np.sqrt(d_mse_dx**2 + d_mse_dy**2)


def compute_ageostrophic_speed(data_array, lats, lons):
    """
    Computes Ageostrophic Wind Speed |Vag| = sqrt((u-ug)^2 + (v-vg)^2).
    ug = -(1/f) * dPhi/dy, vg = (1/f) * dPhi/dx
    where Phi uses geopotential (z) in ERA5 (m^2/s^2).
    """
    z = data_array[0]
    u = data_array[3]
    v = data_array[4]

    dphi_dx, dphi_dy = compute_spatial_gradients(z, lats, lons)

    lat_rad = np.deg2rad(lats)[:, None]
    f = 2.0 * OMEGA * np.sin(lat_rad)
    f_safe = np.where(np.abs(f) < 1e-5, np.nan, f)

    u_g = -dphi_dy / f_safe
    v_g =  dphi_dx / f_safe

    u_ag = u - u_g
    v_ag = v - v_g
    return np.sqrt(u_ag**2 + v_ag**2)


def compute_background_scalar(data_array, lats, lons, bg_var):
    """
    Extracts or computes the target scalar field for subplot background filling.
    """
    if bg_var == 'q':
        return data_array[1]
    if bg_var == 't':
        return data_array[2]
    if bg_var == 'wind_speed':
        u, v = data_array[3], data_array[4]
        return np.sqrt(u**2 + v**2)
    if bg_var == 'mse_grad_abs':
        return compute_mse_gradient_abs(data_array, lats, lons)
    if bg_var == 'vag_speed':
        return compute_ageostrophic_speed(data_array, lats, lons)

    raise ValueError(f"Unknown background variable (bg_var): {bg_var}")


def flatten_finite(arr_list):
    valid = []
    for arr in arr_list:
        arr = np.asarray(arr)
        mask = np.isfinite(arr)
        if np.any(mask):
            valid.append(arr[mask].ravel())
    if len(valid) == 0:
        return np.array([], dtype=np.float32)
    return np.concatenate(valid)


def resolve_color_ranges(panel_records, config):
    """
    Determines colorbar range dynamically or based on config presets.
    """
    # Absolute Value Plots
    abs_vmin, abs_vmax = config['vmin'], config['vmax']
    if config.get('auto_abs_range', False):
        abs_flat = flatten_finite([p['bg'] for p in panel_records if not p['is_err']])
        if abs_flat.size > 0:
            abs_vmax = float(np.nanpercentile(abs_flat, 99.0))
            if config.get('force_abs_vmin_zero', False):
                abs_vmin = 0.0
            else:
                abs_vmin = float(np.nanpercentile(abs_flat, 1.0))
            if (not np.isfinite(abs_vmax)) or (abs_vmax <= abs_vmin):
                abs_vmax = abs_vmin + 1e-6
        else:
            abs_vmin, abs_vmax = 0.0, 1.0

    # Error Plots
    err_vmin, err_vmax = config['err_vmin'], config['err_vmax']
    if config.get('auto_err_range', False):
        err_flat = flatten_finite([p['bg'] for p in panel_records if p['is_err']])
        if err_flat.size > 0:
            emax = float(np.nanpercentile(np.abs(err_flat), 99.0))
            emax = max(emax, 1e-8)
            err_vmin, err_vmax = -emax, emax
        else:
            err_vmin, err_vmax = -1.0, 1.0

    return abs_vmin, abs_vmax, err_vmin, err_vmax


# ============================================================================
# 4. Plotting Utilities
# ============================================================================
def plot_weather_case(case_name, data_dict, lats_hr, lons_hr, lats_lr, lons_lr, config):
    """
    Generates a 2x4 publication-grade atmospheric figure with vector fields and quiver keys.
    """
    fig, axes = plt.subplots(
        2, 4,
        figsize=(26, 13),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    plt.subplots_adjust(left=0.045, right=0.955, top=0.94, bottom=0.11, wspace=0.14, hspace=0.26)

    idx_z, idx_u, idx_v = 0, 3, 4

    def get_plot_vars(data_array, is_lr=False, is_err=False, gt_array=None):
        lats = lats_lr if is_lr else lats_hr
        lons = lons_lr if is_lr else lons_hr
        skip = QUIVER_SKIP_LR if is_lr else QUIVER_SKIP_HR

        z = data_array[idx_z]
        u = data_array[idx_u]
        v = data_array[idx_v]
        bg = compute_background_scalar(data_array, lats, lons, config['bg_var'])

        if is_err and gt_array is not None:
            z_gt = gt_array[idx_z]
            u_gt = gt_array[idx_u]
            v_gt = gt_array[idx_v]
            bg_gt = compute_background_scalar(gt_array, lats, lons, config['bg_var'])

            z = z_gt - z
            u = u_gt - u
            v = v_gt - v
            bg = bg_gt - bg

        return lons, lats, bg, z, u, v, skip

    # Layout and Subplot Titles
    plot_layout = [
        (0, 0, '(a) Low Resolution (Input)', 'LR', True,  False, None),
        (0, 1, '(b) Cubic Spline',           'Spline', False, False, None),
        (0, 2, '(c) AtmoSR-Base',            'noPhy',  False, False, None),
        (0, 3, '(d) AtmoSR',                 'AllPhy', False, False, None),

        (1, 0, '(e) Ground Truth (Target)',      'GT',     False, False, None),
        (1, 1, '(f) Error: GT - Cubic Spline',   'Spline', False, True,  'GT'),
        (1, 2, '(g) Error: GT - AtmoSR-Base',    'noPhy',  False, True,  'GT'),
        (1, 3, '(h) Error: GT - AtmoSR',         'AllPhy', False, True,  'GT'),
    ]

    # Pre-extract data across panels
    panel_records = []
    for panel_idx, (row, col, title, data_key, is_lr, is_err, gt_key) in enumerate(plot_layout, start=1):
        data_array = data_dict[data_key]
        gt_array = data_dict[gt_key] if gt_key is not None else None
        lons, lats, bg, z, u, v, skip = get_plot_vars(
            data_array, is_lr=is_lr, is_err=is_err, gt_array=gt_array
        )
        panel_records.append({
            'panel_idx': panel_idx,
            'row': row,
            'col': col,
            'title': title,
            'is_err': is_err,
            'lons': lons,
            'lats': lats,
            'bg': bg,
            'z': z,
            'u': u,
            'v': v,
            'skip': skip
        })

    # Resolve Colorbar Ranges
    abs_vmin, abs_vmax, err_vmin, err_vmax = resolve_color_ranges(panel_records, config)

    cf_abs, cf_err = None, None

    # Render Subplots
    for p in panel_records:
        ax = axes[p['row'], p['col']]

        # Cartopy Map Features
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.9, edgecolor='black')
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.6, linestyle=':', edgecolor='gray')

        # Contourf (Background Scalar Field)
        if p['is_err']:
            cmap, vmin, vmax = config['err_cmap'], err_vmin, err_vmax
        else:
            cmap, vmin, vmax = config['cmap'], abs_vmin, abs_vmax

        levels = np.linspace(vmin, vmax, config['levels'])
        cf = ax.contourf(
            p['lons'], p['lats'], p['bg'],
            levels=levels, cmap=cmap, extend='both',
            transform=ccrs.PlateCarree()
        )

        if p['is_err']:
            cf_err = cf
        else:
            cf_abs = cf

        # Contour Lines (Geopotential Height - Absolute plots only)
        if not p['is_err']:
            cs = ax.contour(
                p['lons'], p['lats'], p['z'],
                levels=config['z_levels'],
                colors='black', linewidths=0.9, alpha=0.75,
                transform=ccrs.PlateCarree()
            )
            ax.clabel(cs, inline=True, fontsize=FONT_CONTOUR_LABEL, fmt='%1.0f')

        # Quiver Vector Field
        q_scale = config['quiver_err_scale'] if p['is_err'] else config['quiver_scale']
        q_color = 'darkred' if p['is_err'] else 'black'
        q = ax.quiver(
            p['lons'][::p['skip']], p['lats'][::p['skip']],
            p['u'][::p['skip'], ::p['skip']], p['v'][::p['skip'], ::p['skip']],
            transform=ccrs.PlateCarree(),
            scale=q_scale, width=0.0032, headwidth=3.2, headlength=4.2,
            color=q_color, alpha=0.88
        )

        # Quiver Key - Subplot (a) [Main Absolute Field Reference Vector]
        if p['panel_idx'] == 1:
            qk = ax.quiverkey(
                q,
                X=QUIVERKEY_MAIN_POS['X'] + 0.07,
                Y=QUIVERKEY_MAIN_POS['Y'],
                U=config['ref_wind'],
                label=f"{config['ref_wind']} m/s",
                labelpos=QUIVERKEY_MAIN_POS['labelpos'],
                coordinates='axes',
                fontproperties={'size': FONT_QUIVER_KEY, 'weight': 'bold'}
            )
            qk.set_zorder(20)
            qk.text.set_clip_on(False)
            qk.text.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.72, pad=1.5))

        # Quiver Key - Subplot (f) [Reconstruction Error Reference Vector]
        elif p['panel_idx'] == 6:
            qk = ax.quiverkey(
                q,
                X=QUIVERKEY_ERR_POS['X'] + 0.1,
                Y=QUIVERKEY_ERR_POS['Y'],
                U=config['ref_err_wind'],
                label=f"Err: {config['ref_err_wind']} m/s",
                labelpos=QUIVERKEY_ERR_POS['labelpos'],
                coordinates='axes',
                color='darkred',
                fontproperties={'size': FONT_QUIVER_KEY, 'weight': 'bold'}
            )
            qk.set_zorder(20)
            qk.text.set_color('darkred')
            qk.text.set_clip_on(False)
            qk.text.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.72, pad=1.5))

        # Gridlines and Formatting
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.55, color='gray', alpha=0.55, linestyle='--'
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': FONT_GRID_LABEL, 'weight': 'bold'}
        gl.ylabel_style = {'size': FONT_GRID_LABEL, 'weight': 'bold'}

        ax.set_title(p['title'], fontsize=FONT_PANEL_TITLE, fontweight='bold', pad=16)

    # Colorbars
    var_label = BG_VAR_LABELS.get(config['bg_var'], config['bg_var'])

    cbar_ax_abs = fig.add_axes([0.055, 0.045, 0.39, 0.024])
    cb_abs = fig.colorbar(cf_abs, cax=cbar_ax_abs, orientation='horizontal')
    cb_abs.ax.tick_params(labelsize=FONT_CBAR_TICK)
    cb_abs.set_label(f'Absolute Value: {var_label}', fontsize=FONT_CBAR_LABEL, fontweight='bold')

    cbar_ax_err = fig.add_axes([0.555, 0.045, 0.39, 0.024])
    cb_err = fig.colorbar(cf_err, cax=cbar_ax_err, orientation='horizontal')
    cb_err.ax.tick_params(labelsize=FONT_CBAR_TICK)
    cb_err.set_label(f'Error (GT - Reconstruction): {var_label}', fontsize=FONT_CBAR_LABEL, fontweight='bold')

    # Save Output
    png_path = f'Evaluation_{case_name}.png'
    pdf_path = f'Evaluation_{case_name}.pdf'
    plt.savefig(png_path, dpi=DPI, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"✅ Successfully generated figures for {case_name}: {png_path}, {pdf_path}")


# ============================================================================
# 5. Main Execution Flow
# ============================================================================
def main():
    print("🚀 Starting AI super-resolution model evaluation and visualization...")

    # Verification of dependencies and target data files
    required_files = [X_TEST_PATH, Y_TEST_PATH, SCALER_PATH, MODEL_NOPHY_PATH, MODEL_ALLPHY_PATH]
    for fp in required_files:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Required file not found: {fp}")

    scalers = joblib.load(SCALER_PATH)
    scaler_X = scalers['scaler_X']
    scaler_Y = scalers['scaler_Y']

    print(f"📦 Loading dataset from: {X_TEST_PATH}, {Y_TEST_PATH}")
    ds_x = xr.open_dataset(X_TEST_PATH)
    ds_y = xr.open_dataset(Y_TEST_PATH)

    lats_lr = ds_x['latitude'].values
    lons_lr = ds_x['longitude'].values
    lats_hr = ds_y['latitude'].values
    lons_hr = ds_y['longitude'].values

    print("🧠 Initializing and loading neural network models...")
    model_noPhy = BicubicResidualSRModel(
        NUM_CHANNELS, NUM_CHANNELS, NUM_FEATURES, NUM_RES_BLOCKS,
        UPSCALE_FACTOR, scaler_X, scaler_Y
    ).to(device)
    load_model_state(model_noPhy, MODEL_NOPHY_PATH, device)
    model_noPhy.eval()

    model_AllPhy = BicubicResidualSRModel(
        NUM_CHANNELS, NUM_CHANNELS, NUM_FEATURES, NUM_RES_BLOCKS,
        UPSCALE_FACTOR, scaler_X, scaler_Y
    ).to(device)
    load_model_state(model_AllPhy, MODEL_ALLPHY_PATH, device)
    model_AllPhy.eval()

    # Pre-calculate baseline standardization scaling factors
    alpha = torch.tensor(
        scaler_X.std / scaler_Y.std, dtype=torch.float32, device=device
    ).view(1, -1, 1, 1)
    beta = torch.tensor(
        (scaler_X.mean - scaler_Y.mean) / scaler_Y.std, dtype=torch.float32, device=device
    ).view(1, -1, 1, 1)

    for case_name, t_idx in CASE_INDICES.items():
        print(f"\n🔍 Processing case study: {case_name} (Time Index: {t_idx})")

        x_np = np.stack([ds_x[v].isel(time=t_idx).values for v in VARIABLES], axis=0)[np.newaxis, ...]
        y_np = np.stack([ds_y[v].isel(time=t_idx).values for v in VARIABLES], axis=0)[np.newaxis, ...]

        x_std = scaler_X.transform(x_np)
        x_tensor = torch.tensor(x_std, dtype=torch.float32, device=device)

        with torch.no_grad():
            spline_std = F.interpolate(
                x_tensor, scale_factor=UPSCALE_FACTOR, mode='bicubic', align_corners=False
            )
            spline_y_std = spline_std * alpha + beta

            pred_noPhy_std = model_noPhy(x_tensor)
            pred_AllPhy_std = model_AllPhy(x_tensor)

        data_dict = {
            'LR': x_np[0],
            'Spline': scaler_Y.inverse_transform_numpy(spline_y_std.cpu().numpy())[0],
            'noPhy': scaler_Y.inverse_transform_numpy(pred_noPhy_std.cpu().numpy())[0],
            'AllPhy': scaler_Y.inverse_transform_numpy(pred_AllPhy_std.cpu().numpy())[0],
            'GT': y_np[0]
        }

        plot_weather_case(
            case_name, data_dict,
            lats_hr, lons_hr, lats_lr, lons_lr,
            PLOT_CONFIGS[case_name]
        )

    ds_x.close()
    ds_y.close()
    print("\n🎉 All publication-grade figures have been generated successfully!")


if __name__ == '__main__':
    main()