# -*- coding: utf-8 -*-
"""
ERA5 GRIB to Super-Resolution NetCDF Dataset Converter.

This script processes ERA5 GRIB files into high-resolution (Y) and low-resolution/coarsened
(X2, X4) datasets for super-resolution deep learning tasks. The processed data are split
into train, validation, and test sets and saved in NetCDF4 format.

Output Files:
    - X2_train.nc, X4_train.nc, Y_train.nc
    - X2_valid.nc, X4_valid.nc, Y_valid.nc
    - X2_test.nc,  X4_test.nc,  Y_test.nc

Variables:
    latitude, longitude, time, utc_date, z, q, t, u, v
"""

import gc
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset, date2num
from tqdm.auto import tqdm

try:
    import psutil
except ImportError:
    psutil = None


# ==============================================================================
# Global Configuration & Hyperparameters
# ==============================================================================
DATA_DIR = "/content"                           # Directory containing GRIB input files
OUTPUT_DIR = "/content"                         # Directory for output NetCDF files
FILE_TEMPLATE = "era5_{year}.grib"             # Filename pattern for GRIB files

SPLIT_YEARS = {
    "train": [2019, 2020, 2021, 2022, 2023],
    "valid": [2024],
    "test": [2025],
}

VARIABLES = ["z", "q", "t", "u", "v"]

GRIB_FILTER = {
    "typeOfLevel": "isobaricInhPa",
    "level": 500,
}

# Native ERA5 grid resolution (degrees)
NATIVE_RESOLUTION = 0.25

# Number of hours processed per batch (larger = faster execution, higher RAM usage)
READ_CHUNK_HOURS = 168

# NetCDF Compression & Storage Settings
COMPRESSION_LEVEL = 4
NC_TIME_CHUNK = 24
ENABLE_SYNC_EACH_CHUNK = False  # Set to True for higher safety, slightly lower writing speed

# Coordinate alignment tolerance
COORD_TOL = 1e-4

# Time specifications
TIME_UNITS = "hours since 1970-01-01 00:00:00"
TIME_CALENDAR = "proleptic_gregorian"
UTC_DATE_FORMAT = "%Y%m%d%H"

# Target domain and grid parameters
TARGETS = {
    "Y": {
        "lat_start": 15.0, "lat_end": 39.75,
        "lon_start": 100.0, "lon_end": 124.75,
        "step": 0.25,
        "kernel": 1,
        "expected_shape": (100, 100),
    },
    "X2": {
        "lat_start": 15.0, "lat_end": 39.5,
        "lon_start": 100.0, "lon_end": 124.5,
        "step": 0.5,
        "kernel": 3,    # 3x3 average pooling
        "expected_shape": (50, 50),
    },
    "X4": {
        "lat_start": 15.0, "lat_end": 39.0,
        "lon_start": 100.0, "lon_end": 124.0,
        "step": 1.0,
        "kernel": 5,    # 5x5 average pooling
        "expected_shape": (25, 25),
    },
}

# Variable physical units metadata
VAR_UNITS = {
    "z": "m**2 s**-2",
    "q": "kg kg**-1",
    "t": "K",
    "u": "m s**-1",
    "v": "m s**-1",
}


# ==============================================================================
# Helper & Utility Functions
# ==============================================================================
def get_ram_gb() -> float | None:
    """Return the memory usage of the current process in GB, or None if psutil is unavailable."""
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def release_ram():
    """Explicitly trigger garbage collection to free memory."""
    gc.collect()


def build_axis(start: float, end: float, step: float) -> np.ndarray:
    """Generate evenly spaced grid coordinate axis (inclusive of endpoint)."""
    vals = np.round(np.arange(start, end + step * 0.5, step), 6)
    return vals.astype(np.float32)


def prepare_target_axes() -> dict:
    """Construct and validate grid coordinates for all target configurations."""
    axes = {}
    for name, cfg in TARGETS.items():
        lat = build_axis(cfg["lat_start"], cfg["lat_end"], cfg["step"])
        lon = build_axis(cfg["lon_start"], cfg["lon_end"], cfg["step"])
        exp_lat, exp_lon = cfg["expected_shape"]

        if len(lat) != exp_lat or len(lon) != exp_lon:
            raise ValueError(
                f"Grid dimension mismatch for {name}: lat={len(lat)}, lon={len(lon)}. "
                f"Expected lat={exp_lat}, lon={exp_lon}"
            )
        axes[name] = {"lat": lat, "lon": lon}
    return axes


def find_nearest_indices(all_coords: np.ndarray, target_coords: np.ndarray, name: str, tol: float = COORD_TOL) -> np.ndarray:
    """
    Locate indices of nearest values in all_coords for each element in target_coords.
    
    Raises:
        ValueError: If maximum error exceeds tolerance.
    """
    all_coords = np.asarray(all_coords, dtype=np.float64)
    target_coords = np.asarray(target_coords, dtype=np.float64)

    diff = np.abs(all_coords[:, None] - target_coords[None, :])
    idx = diff.argmin(axis=0)
    max_err = np.max(np.abs(all_coords[idx] - target_coords))

    if max_err > tol:
        raise ValueError(f"Alignment failed for {name}: max error {max_err:.6f} > tolerance {tol}")
    return idx.astype(np.int64)


def convert_times(times_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert numpy datetime64 array into NetCDF numeric time and integer UTC date format."""
    dt = pd.to_datetime(times_np).to_pydatetime()
    time_num = date2num(dt, units=TIME_UNITS, calendar=TIME_CALENDAR)
    utc_date = np.array([int(pd.Timestamp(t).strftime(UTC_DATE_FORMAT)) for t in dt], dtype=np.int64)
    return np.asarray(time_num, dtype=np.float64), utc_date


def open_grib_dataset(grib_path: Path) -> xr.Dataset:
    """Open a single GRIB file and return a dataset containing target variables."""
    backend_kwargs = {
        "indexpath": "",
        "filter_by_keys": GRIB_FILTER,
    }

    ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs=backend_kwargs)

    # Standardize coordinate names
    rename_map = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename_map["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename_map["lon"] = "longitude"
    if rename_map:
        ds = ds.rename(rename_map)

    # Validate target variable availability
    missing = [v for v in VARIABLES if v not in ds.data_vars]
    if missing:
        ds.close()
        raise ValueError(
            f"Missing variables {missing} in {grib_path.name}. "
            "Please ensure GRIB file contains z, q, t, u, and v."
        )
    ds = ds[VARIABLES]

    # Drop redundant single-value dimensions
    for dim in ["step", "isobaricInhPa", "number"]:
        if dim in ds.dims and ds.sizes[dim] == 1:
            ds = ds.isel({dim: 0}, drop=True)

    if "time" not in ds.coords:
        ds.close()
        raise ValueError("Missing 'time' coordinate in dataset.")

    # Sort latitude and longitude in ascending order
    if ds["latitude"].values[0] > ds["latitude"].values[-1]:
        ds = ds.sortby("latitude")
    if ds["longitude"].values[0] > ds["longitude"].values[-1]:
        ds = ds.sortby("longitude")

    return ds


def check_spatial_coverage(lat_all: np.ndarray, lon_all: np.ndarray):
    """
    Verify spatial coverage of input data to ensure support for:
    - Y extraction
    - X2 (3x3 spatial pooling)
    - X4 (5x5 spatial pooling)
    """
    lat_min_data, lat_max_data = float(np.min(lat_all)), float(np.max(lat_all))
    lon_min_data, lon_max_data = float(np.min(lon_all)), float(np.max(lon_all))

    req_lat_min = min(cfg["lat_start"] - (cfg["kernel"] // 2) * NATIVE_RESOLUTION for cfg in TARGETS.values())
    req_lat_max = max(cfg["lat_end"] + (cfg["kernel"] // 2) * NATIVE_RESOLUTION for cfg in TARGETS.values())
    req_lon_min = min(cfg["lon_start"] - (cfg["kernel"] // 2) * NATIVE_RESOLUTION for cfg in TARGETS.values())
    req_lon_max = max(cfg["lon_end"] + (cfg["kernel"] // 2) * NATIVE_RESOLUTION for cfg in TARGETS.values())

    ok = (
        lat_min_data <= req_lat_min + COORD_TOL
        and lat_max_data >= req_lat_max - COORD_TOL
        and lon_min_data <= req_lon_min + COORD_TOL
        and lon_max_data >= req_lon_max - COORD_TOL
    )
    if not ok:
        raise ValueError(
            "Insufficient GRIB spatial coverage for target domain and convolutions.\n"
            f"Data bounds:   lat [{lat_min_data}, {lat_max_data}], lon [{lon_min_data}, {lon_max_data}]\n"
            f"Required bounds: lat [{req_lat_min}, {req_lat_max}], lon [{req_lon_min}, {req_lon_max}]"
        )


# ==============================================================================
# NetCDF File Writer
# ==============================================================================
class NetCDFAppender:
    """NetCDF dataset writer supporting chunk-wise appending along time dimension."""

    def __init__(self, out_path: Path, lat_vals: np.ndarray, lon_vals: np.ndarray, variables: list, split_name: str, target_name: str):
        self.out_path = Path(out_path)
        self.nc = Dataset(self.out_path, "w", format="NETCDF4")

        # Create dimensions
        self.nc.createDimension("time", None)
        self.nc.createDimension("latitude", len(lat_vals))
        self.nc.createDimension("longitude", len(lon_vals))

        # Create coordinate variables
        lat_var = self.nc.createVariable("latitude", "f4", ("latitude",))
        lon_var = self.nc.createVariable("longitude", "f4", ("longitude",))
        self.time_var = self.nc.createVariable("time", "f8", ("time",))
        self.utc_var = self.nc.createVariable("utc_date", "i8", ("time",))

        lat_var[:] = lat_vals.astype(np.float32)
        lon_var[:] = lon_vals.astype(np.float32)

        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        lat_var.long_name = "latitude"
        lon_var.long_name = "longitude"

        self.time_var.units = TIME_UNITS
        self.time_var.calendar = TIME_CALENDAR
        self.time_var.long_name = "time"

        self.utc_var.long_name = "UTC datetime as integer"
        self.utc_var.description = f"format: {UTC_DATE_FORMAT}"

        # Create data variables
        self.data_vars = {}
        for v in variables:
            var_obj = self.nc.createVariable(
                v,
                "f4",
                ("time", "latitude", "longitude"),
                zlib=True,
                complevel=COMPRESSION_LEVEL,
                shuffle=True,
                chunksizes=(NC_TIME_CHUNK, len(lat_vals), len(lon_vals)),
                fill_value=np.float32(np.nan),
            )
            var_obj.long_name = v
            if v in VAR_UNITS:
                var_obj.units = VAR_UNITS[v]
            self.data_vars[v] = var_obj

        # Global attributes metadata
        self.nc.Conventions = "CF-1.8"
        self.nc.title = f"ERA5 {target_name} dataset ({split_name})"
        self.nc.source = "ERA5 GRIB -> NetCDF"
        self.nc.processing = "Y=original; X2=3x3 rolling mean + 0.5deg sampling; X4=5x5 rolling mean + 1.0deg sampling"
        self.nc.history = f"Created on {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"

        self.t_cursor = 0

    def append(self, time_num: np.ndarray, utc_date: np.ndarray, data_dict: dict):
        """Append data along the unbounded time dimension."""
        n = len(time_num)
        sl = slice(self.t_cursor, self.t_cursor + n)

        self.time_var[sl] = time_num
        self.utc_var[sl] = utc_date

        for v, arr in data_dict.items():
            self.data_vars[v][sl, :, :] = arr.astype(np.float32, copy=False)

        self.t_cursor += n

    def sync(self):
        """Sync buffered data to disk."""
        self.nc.sync()

    def close(self):
        """Close dataset handle."""
        self.nc.close()


# ==============================================================================
# Data Processing Pipeline
# ==============================================================================
def process_split(split_name: str, years: list, target_axes: dict):
    """Process all years belonging to a given dataset split (train/valid/test)."""
    years = sorted(years)
    print("\n" + "=" * 90)
    print(f"Processing split = {split_name}, Years = {years}")

    out_paths = {
        "X2": Path(OUTPUT_DIR) / f"X2_{split_name}.nc",
        "X4": Path(OUTPUT_DIR) / f"X4_{split_name}.nc",
        "Y": Path(OUTPUT_DIR) / f"Y_{split_name}.nc",
    }

    writers = {
        name: NetCDFAppender(
            out_path=out_paths[name],
            lat_vals=target_axes[name]["lat"],
            lon_vals=target_axes[name]["lon"],
            variables=VARIABLES,
            split_name=split_name,
            target_name=name,
        )
        for name in ["X2", "X4", "Y"]
    }

    try:
        for year in years:
            grib_path = Path(DATA_DIR) / FILE_TEMPLATE.format(year=year)
            if not grib_path.exists():
                raise FileNotFoundError(f"File not found: {grib_path}")

            print(f"\n📂 Reading: {grib_path.name}")
            ds = open_grib_dataset(grib_path)

            lat_all = ds["latitude"].values
            lon_all = ds["longitude"].values
            check_spatial_coverage(lat_all, lon_all)

            # Build alignment indices for current year
            idx = {}
            for name in ["Y", "X2", "X4"]:
                idx[name] = {
                    "lat": find_nearest_indices(lat_all, target_axes[name]["lat"], f"{name} latitude"),
                    "lon": find_nearest_indices(lon_all, target_axes[name]["lon"], f"{name} longitude"),
                }

            n_time = ds.sizes["time"]
            n_chunks = math.ceil(n_time / READ_CHUNK_HOURS)

            pbar = tqdm(
                range(0, n_time, READ_CHUNK_HOURS),
                total=n_chunks,
                desc=f"{split_name}-{year}",
                unit="chunk",
                leave=True,
            )

            for t0 in pbar:
                t1 = min(t0 + READ_CHUNK_HOURS, n_time)

                # Extract time window slice
                dsc = ds[VARIABLES].isel(time=slice(t0, t1)).astype(np.float32)

                # Y: High-resolution direct crop
                y_ds = dsc.isel(latitude=idx["Y"]["lat"], longitude=idx["Y"]["lon"])

                # X2: 3x3 Average pooling + 0.5-degree sampling
                roll3 = dsc.rolling(
                    latitude=TARGETS["X2"]["kernel"],
                    longitude=TARGETS["X2"]["kernel"],
                    center=True
                ).mean()
                x2_ds = roll3.isel(latitude=idx["X2"]["lat"], longitude=idx["X2"]["lon"])

                # X4: 5x5 Average pooling + 1.0-degree sampling
                roll5 = dsc.rolling(
                    latitude=TARGETS["X4"]["kernel"],
                    longitude=TARGETS["X4"]["kernel"],
                    center=True
                ).mean()
                x4_ds = roll5.isel(latitude=idx["X4"]["lat"], longitude=idx["X4"]["lon"])

                # Time coordinate conversions
                time_num, utc_date = convert_times(dsc["time"].values)

                # Export to numpy arrays and write to NetCDF
                y_data = {v: np.asarray(y_ds[v].values, dtype=np.float32) for v in VARIABLES}
                x2_data = {v: np.asarray(x2_ds[v].values, dtype=np.float32) for v in VARIABLES}
                x4_data = {v: np.asarray(x4_ds[v].values, dtype=np.float32) for v in VARIABLES}

                writers["Y"].append(time_num, utc_date, y_data)
                writers["X2"].append(time_num, utc_date, x2_data)
                writers["X4"].append(time_num, utc_date, x4_data)

                if ENABLE_SYNC_EACH_CHUNK:
                    for w in writers.values():
                        w.sync()

                ram = get_ram_gb()
                if ram is None:
                    pbar.set_postfix_str(f"written time steps={writers['Y'].t_cursor}")
                else:
                    pbar.set_postfix_str(f"written time steps={writers['Y'].t_cursor}, RAM={ram:.2f} GB")

                # Garbage collection per chunk
                del dsc, y_ds, x2_ds, x4_ds, roll3, roll5
                del y_data, x2_data, x4_data, time_num, utc_date
                release_ram()

            # Close dataset handle and release memory for processed year
            ds.close()
            del ds, idx, lat_all, lon_all
            release_ram()
            print(f"✅ Year {year} completed. Memory released.")

        # Ensure split outputs are flushed to disk
        for w in writers.values():
            w.sync()

    finally:
        for w in writers.values():
            try:
                w.close()
            except Exception:
                pass
        release_ram()

    print(f"🎉 Split {split_name} successfully finished.")
    for name in ["X2", "X4", "Y"]:
        p = out_paths[name]
        size_gb = p.stat().st_size / (1024 ** 3) if p.exists() else 0
        print(f"  - {p.name}: {size_gb:.2f} GB")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Output directory:", OUTPUT_DIR)
    target_axes = prepare_target_axes()

    print("\nTarget Grid Configurations:")
    for name in ["Y", "X2", "X4"]:
        print(
            f"  {name}: "
            f"lat={len(target_axes[name]['lat'])}, "
            f"lon={len(target_axes[name]['lon'])}, "
            f"lat_range=({target_axes[name]['lat'][0]}, {target_axes[name]['lat'][-1]}), "
            f"lon_range=({target_axes[name]['lon'][0]}, {target_axes[name]['lon'][-1]})"
        )

    for split_name, years in SPLIT_YEARS.items():
        process_split(split_name, years, target_axes)
        release_ram()
        print(f"🧹 RAM cleaned after split = {split_name}.")

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()