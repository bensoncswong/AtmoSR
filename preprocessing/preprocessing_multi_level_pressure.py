# -*- coding: utf-8 -*-
"""
ERA5 Pressure Level GRIB to Super-Resolution Test Dataset Converter (NetCDF).

This script converts multi-level ERA5 pressure level GRIB datasets into 
super-resolution test datasets (X2, X4 low-resolution, and Y high-resolution ground truth) 
and exports them to compressed NetCDF-4 format.

Target Variables:
    latitude, longitude, time, utc_date, z, q, t, u, v
"""

import gc
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
# Global Configuration & Parameters
# ==============================================================================
DATA_DIR = "/content"    # Directory containing input GRIB files
OUTPUT_DIR = "/content"  # Directory for output NetCDF files

# Test evaluation cases across multiple pressure levels
TEST_CASES = [
    {"input_file": "era5_925hpa_2025.grib", "tag": "925hpa_2025", "level": 925},
    {"input_file": "era5_850hpa_2025.grib", "tag": "850hpa_2025", "level": 850},
    {"input_file": "era5_700hpa_2025.grib", "tag": "700hpa_2025", "level": 700},
    {"input_file": "era5_300hpa_2025.grib", "tag": "300hpa_2025", "level": 300},
    {"input_file": "era5_100hpa_2025.grib", "tag": "100hpa_2025", "level": 100},
]

VARIABLES = ["z", "q", "t", "u", "v"]

# Base cfgrib filtering keys
GRIB_BASE_FILTER = {
    "typeOfLevel": "isobaricInhPa",
}

# Native resolution of ERA5 dataset (in degrees)
NATIVE_RESOLUTION = 0.25

# Chunk size for time processing (hours)
READ_CHUNK_HOURS = 168

# NetCDF dataset output settings
COMPRESSION_LEVEL = 4
NC_TIME_CHUNK = 24
ENABLE_SYNC_EACH_CHUNK = False
OVERWRITE_OUTPUT = True

# Coordinate tolerance for alignment checks
COORD_TOL = 1e-4

# Datetime formatting configurations
TIME_UNITS = "hours since 1970-01-01 00:00:00"
TIME_CALENDAR = "proleptic_gregorian"
UTC_DATE_FORMAT = "%Y%m%d%H"

# Target spatial grid configurations
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
        "kernel": 3,  # 3x3 average pooling
        "expected_shape": (50, 50),
    },
    "X4": {
        "lat_start": 15.0, "lat_end": 39.0,
        "lon_start": 100.0, "lon_end": 124.0,
        "step": 1.0,
        "kernel": 5,  # 5x5 average pooling
        "expected_shape": (25, 25),
    },
}

VAR_UNITS = {
    "z": "m**2 s**-2",
    "q": "kg kg**-1",
    "t": "K",
    "u": "m s**-1",
    "v": "m s**-1",
}


# ==============================================================================
# Helper Functions
# ==============================================================================
def get_ram_gb() -> float | None:
    """Return the RSS memory usage of the current process in GB."""
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def release_ram() -> None:
    """Trigger explicit garbage collection."""
    gc.collect()


def build_axis(start: float, end: float, step: float) -> np.ndarray:
    """Generate a rounded floating-point coordinate array."""
    vals = np.round(np.arange(start, end + step * 0.5, step), 6)
    return vals.astype(np.float32)


def prepare_target_axes() -> Dict[str, Dict[str, np.ndarray]]:
    """Prepare and validate coordinate axes for target resolutions."""
    axes = {}
    for name, cfg in TARGETS.items():
        lat = build_axis(cfg["lat_start"], cfg["lat_end"], cfg["step"])
        lon = build_axis(cfg["lon_start"], cfg["lon_end"], cfg["step"])
        exp_lat, exp_lon = cfg["expected_shape"]

        if len(lat) != exp_lat or len(lon) != exp_lon:
            raise ValueError(
                f"Grid dimension mismatch for {name}: lat={len(lat)}, lon={len(lon)} "
                f"(expected lat={exp_lat}, lon={exp_lon})"
            )
        axes[name] = {"lat": lat, "lon": lon}
    return axes


def required_bounds() -> Dict[str, float]:
    """Calculate minimum spatial bounds required for filtering and convolution kernels."""
    req_lat_min = min(
        cfg["lat_start"] - (cfg["kernel"] // 2) * NATIVE_RESOLUTION
        for cfg in TARGETS.values()
    )
    req_lat_max = max(
        cfg["lat_end"] + (cfg["kernel"] // 2) * NATIVE_RESOLUTION
        for cfg in TARGETS.values()
    )
    req_lon_min = min(
        cfg["lon_start"] - (cfg["kernel"] // 2) * NATIVE_RESOLUTION
        for cfg in TARGETS.values()
    )
    req_lon_max = max(
        cfg["lon_end"] + (cfg["kernel"] // 2) * NATIVE_RESOLUTION
        for cfg in TARGETS.values()
    )
    return {
        "lat_min": float(req_lat_min),
        "lat_max": float(req_lat_max),
        "lon_min": float(req_lon_min),
        "lon_max": float(req_lon_max),
    }


def find_nearest_indices(
    all_coords: np.ndarray, target_coords: np.ndarray, name: str, tol: float = COORD_TOL
) -> np.ndarray:
    """Find index mappings from source coordinates to target coordinates within tolerance."""
    all_coords = np.asarray(all_coords, dtype=np.float64)
    target_coords = np.asarray(target_coords, dtype=np.float64)

    diff = np.abs(all_coords[:, None] - target_coords[None, :])
    idx = diff.argmin(axis=0)
    max_err = np.max(np.abs(all_coords[idx] - target_coords))

    if max_err > tol:
        raise ValueError(
            f"Coordinate alignment failed for {name}. Max error {max_err:.6f} exceeds tolerance {tol}"
        )
    return idx.astype(np.int64)


def convert_times(times_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert numpy datetime array into NetCDF numeric time and UTC integer date values."""
    dt = pd.to_datetime(times_np).to_pydatetime()
    time_num = date2num(dt, units=TIME_UNITS, calendar=TIME_CALENDAR)
    utc_date = np.array(
        [int(pd.Timestamp(t).strftime(UTC_DATE_FORMAT)) for t in dt],
        dtype=np.int64,
    )
    return np.asarray(time_num, dtype=np.float64), utc_date


def open_grib_dataset(grib_path: Path, level_hpa: int | float | None = None) -> xr.Dataset:
    """Open and preprocess GRIB file using xarray and cfgrib engine."""
    filter_by_keys = dict(GRIB_BASE_FILTER)
    if level_hpa is not None:
        filter_by_keys["level"] = int(level_hpa)

    backend_kwargs = {
        "indexpath": "",
        "filter_by_keys": filter_by_keys,
    }

    ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs=backend_kwargs)

    rename_map = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename_map["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename_map["lon"] = "longitude"
    if rename_map:
        ds = ds.rename(rename_map)

    missing = [v for v in VARIABLES if v not in ds.data_vars]
    if missing:
        ds.close()
        raise ValueError(
            f"Missing required variables in {grib_path.name}: {missing}. "
            "Ensure the GRIB file contains z, q, t, u, v with matching keys."
        )
    ds = ds[VARIABLES]

    for dim in ["step", "isobaricInhPa", "number"]:
        if dim in ds.dims and ds.sizes[dim] == 1:
            ds = ds.isel({dim: 0}, drop=True)

    if "time" not in ds.coords:
        ds.close()
        raise ValueError("Missing 'time' coordinate in dataset.")
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        ds.close()
        raise ValueError("Missing 'latitude' or 'longitude' coordinates in dataset.")

    # Ensure coordinates are monotonically increasing
    if ds["latitude"].values[0] > ds["latitude"].values[-1]:
        ds = ds.sortby("latitude")
    if ds["longitude"].values[0] > ds["longitude"].values[-1]:
        ds = ds.sortby("longitude")

    return ds


def check_spatial_coverage(lat_all: np.ndarray, lon_all: np.ndarray, bounds: Dict[str, float]) -> None:
    """Verify if source dataset covers the required bounding spatial domain."""
    lat_min_data, lat_max_data = float(np.min(lat_all)), float(np.max(lat_all))
    lon_min_data, lon_max_data = float(np.min(lon_all)), float(np.max(lon_all))

    req_lat_min, req_lat_max = bounds["lat_min"], bounds["lat_max"]
    req_lon_min, req_lon_max = bounds["lon_min"], bounds["lon_max"]

    ok = (
        lat_min_data <= req_lat_min + COORD_TOL
        and lat_max_data >= req_lat_max - COORD_TOL
        and lon_min_data <= req_lon_min + COORD_TOL
        and lon_max_data >= req_lon_max - COORD_TOL
    )
    if not ok:
        raise ValueError(
            "Spatial coverage of input GRIB is insufficient for target grids and convolution kernels.\n"
            f"Data bounds  : lat [{lat_min_data}, {lat_max_data}], lon [{lon_min_data}, {lon_max_data}]\n"
            f"Target bounds: lat [{req_lat_min}, {req_lat_max}], lon [{req_lon_min}, {req_lon_max}]"
        )


def select_crop_indices(all_coords: np.ndarray, vmin: float, vmax: float, name: str) -> np.ndarray:
    """Select array indices within specified coordinate boundaries."""
    all_coords = np.asarray(all_coords, dtype=np.float64)
    mask = (all_coords >= (vmin - COORD_TOL)) & (all_coords <= (vmax + COORD_TOL))
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(f"Failed to obtain crop indices for coordinate: {name}.")
    return idx.astype(np.int64)


# ==============================================================================
# NetCDF File Writer Class
# ==============================================================================
class NetCDFAppender:
    """Helper class for incrementally writing chunked meteorological data to NetCDF-4."""

    def __init__(
        self,
        out_path: Path | str,
        lat_vals: np.ndarray,
        lon_vals: np.ndarray,
        variables: List[str],
        split_name: str,
        target_name: str,
        global_attrs: Dict[str, Any] | None = None,
    ):
        self.out_path = Path(out_path)

        if self.out_path.exists() and (not OVERWRITE_OUTPUT):
            raise FileExistsError(f"Output file already exists and overwrite is disabled: {self.out_path}")

        self.nc = Dataset(self.out_path, "w", format="NETCDF4")

        # Dimensions
        self.nc.createDimension("time", None)
        self.nc.createDimension("latitude", len(lat_vals))
        self.nc.createDimension("longitude", len(lon_vals))

        # Coordinate Variables
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

        # Data Variables
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

        # Global Attributes
        self.nc.Conventions = "CF-1.8"
        self.nc.title = f"ERA5 {target_name} dataset ({split_name})"
        self.nc.source = "ERA5 GRIB -> NetCDF"
        self.nc.processing = (
            "Y=original; X2=3x3 rolling mean + 0.5deg sampling; "
            "X4=5x5 rolling mean + 1.0deg sampling"
        )
        self.nc.history = f"Created at {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"

        if global_attrs:
            for k, v in global_attrs.items():
                setattr(self.nc, k, v)

        self.t_cursor = 0

    def append(self, time_num: np.ndarray, utc_date: np.ndarray, data_dict: Dict[str, np.ndarray]) -> None:
        """Append chunk data to NetCDF file."""
        n = len(time_num)
        sl = slice(self.t_cursor, self.t_cursor + n)

        self.time_var[sl] = time_num
        self.utc_var[sl] = utc_date

        for v, arr in data_dict.items():
            self.data_vars[v][sl, :, :] = arr.astype(np.float32, copy=False)

        self.t_cursor += n

    def sync(self) -> None:
        """Sync buffered data to disk."""
        self.nc.sync()

    def close(self) -> None:
        """Close NetCDF dataset."""
        self.nc.close()


# ==============================================================================
# Pipeline Execution Functions
# ==============================================================================
def process_test_case(
    case: Dict[str, Any], target_axes: Dict[str, Dict[str, np.ndarray]], bounds: Dict[str, float]
) -> None:
    """Process a single ERA5 test case file, converting to X2, X4, and Y NetCDF datasets."""
    input_file = case["input_file"]
    tag = case["tag"]
    level_hpa = case.get("level", None)

    print("\n" + "=" * 90)
    print(f"Processing case tag: {tag}")
    print(f"  Input file : {input_file}")
    print(f"  Level (hPa): {level_hpa}")

    grib_path = Path(DATA_DIR) / input_file
    if not grib_path.exists():
        raise FileNotFoundError(f"Input file not found: {grib_path}")

    out_paths = {
        "X2": Path(OUTPUT_DIR) / f"X2_test_{tag}.nc",
        "X4": Path(OUTPUT_DIR) / f"X4_test_{tag}.nc",
        "Y": Path(OUTPUT_DIR) / f"Y_test_{tag}.nc",
    }

    global_attrs = {
        "dataset_split": "test",
        "case_tag": tag,
        "input_grib": input_file,
    }
    if level_hpa is not None:
        global_attrs["pressure_level_hPa"] = int(level_hpa)

    writers = {
        name: NetCDFAppender(
            out_path=out_paths[name],
            lat_vals=target_axes[name]["lat"],
            lon_vals=target_axes[name]["lon"],
            variables=VARIABLES,
            split_name="test",
            target_name=f"{name}_{tag}",
            global_attrs=global_attrs,
        )
        for name in ["X2", "X4", "Y"]
    }

    ds = None
    try:
        ds = open_grib_dataset(grib_path, level_hpa=level_hpa)

        lat_all = ds["latitude"].values
        lon_all = ds["longitude"].values
        check_spatial_coverage(lat_all, lon_all, bounds)

        # Crop domain prior to rolling convolutions to optimize memory usage
        crop_lat_idx = select_crop_indices(lat_all, bounds["lat_min"], bounds["lat_max"], "latitude")
        crop_lon_idx = select_crop_indices(lon_all, bounds["lon_min"], bounds["lon_max"], "longitude")

        lat_crop = lat_all[crop_lat_idx]
        lon_crop = lon_all[crop_lon_idx]

        idx = {}
        for name in ["Y", "X2", "X4"]:
            idx[name] = {
                "lat": find_nearest_indices(lat_crop, target_axes[name]["lat"], f"{name} latitude"),
                "lon": find_nearest_indices(lon_crop, target_axes[name]["lon"], f"{name} longitude"),
            }

        n_time = ds.sizes["time"]
        n_chunks = math.ceil(n_time / READ_CHUNK_HOURS)

        pbar = tqdm(
            range(0, n_time, READ_CHUNK_HOURS),
            total=n_chunks,
            desc=f"test-{tag}",
            unit="chunk",
            leave=True,
        )

        for t0 in pbar:
            t1 = min(t0 + READ_CHUNK_HOURS, n_time)

            dsc = ds[VARIABLES].isel(
                time=slice(t0, t1),
                latitude=crop_lat_idx,
                longitude=crop_lon_idx,
            ).astype(np.float32)

            # High resolution Ground Truth (Y): Direct cropping
            y_ds = dsc.isel(latitude=idx["Y"]["lat"], longitude=idx["Y"]["lon"])

            # Low resolution X2: 3x3 average convolution + downsampling
            roll3 = dsc.rolling(
                latitude=TARGETS["X2"]["kernel"],
                longitude=TARGETS["X2"]["kernel"],
                center=True
            ).mean()
            x2_ds = roll3.isel(latitude=idx["X2"]["lat"], longitude=idx["X2"]["lon"])

            # Low resolution X4: 5x5 average convolution + downsampling
            roll5 = dsc.rolling(
                latitude=TARGETS["X4"]["kernel"],
                longitude=TARGETS["X4"]["kernel"],
                center=True
            ).mean()
            x4_ds = roll5.isel(latitude=idx["X4"]["lat"], longitude=idx["X4"]["lon"])

            # Timestamp processing
            time_num, utc_date = convert_times(dsc["time"].values)

            # Convert to numpy arrays and output to NetCDF
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
                pbar.set_postfix_str(f"written time={writers['Y'].t_cursor}")
            else:
                pbar.set_postfix_str(f"written time={writers['Y'].t_cursor}, RAM={ram:.2f} GB")

            del dsc, y_ds, x2_ds, x4_ds, roll3, roll5
            del y_data, x2_data, x4_data, time_num, utc_date
            release_ram()

        for w in writers.values():
            w.sync()

    finally:
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass

        for w in writers.values():
            try:
                w.close()
            except Exception:
                pass

        release_ram()

    print(f"✅ Finished case tag: {tag}")
    for name in ["X2", "X4", "Y"]:
        p = out_paths[name]
        size_gb = p.stat().st_size / (1024 ** 3) if p.exists() else 0.0
        print(f"  - {p.name}: {size_gb:.3f} GB")


def validate_test_cases() -> None:
    """Validate test cases array configuration for potential duplicates or syntax errors."""
    tags = [c["tag"] for c in TEST_CASES]
    if len(tags) != len(set(tags)):
        raise ValueError(f"Duplicate 'tag' values found in TEST_CASES: {tags}")

    files = [c["input_file"] for c in TEST_CASES]
    if len(files) != len(set(files)):
        raise ValueError(f"Duplicate 'input_file' values found in TEST_CASES: {files}")

    for c in TEST_CASES:
        if "input_file" not in c or "tag" not in c:
            raise ValueError(f"Each case dictionary must contain 'input_file' and 'tag': {c}")


def main() -> None:
    """Main program execution entry point."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    validate_test_cases()

    print("DATA_DIR  :", DATA_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    target_axes = prepare_target_axes()
    bounds = required_bounds()

    print("\nTarget Grid Validation Summary:")
    for name in ["Y", "X2", "X4"]:
        print(
            f"  {name}: "
            f"lat={len(target_axes[name]['lat'])}, "
            f"lon={len(target_axes[name]['lon'])}, "
            f"lat_range=({target_axes[name]['lat'][0]}, {target_axes[name]['lat'][-1]}), "
            f"lon_range=({target_axes[name]['lon'][0]}, {target_axes[name]['lon'][-1]})"
        )

    print(
        "\nRequired Minimum Bounding Box for Convolution:\n"
        f"  lat[{bounds['lat_min']}, {bounds['lat_max']}], "
        f"lon[{bounds['lon_min']}, {bounds['lon_max']}]"
    )

    print("\nScheduled Test Cases:")
    for c in TEST_CASES:
        print(f"  - {c['input_file']}  ->  X2/X4/Y_test_{c['tag']}.nc")

    for case in TEST_CASES:
        process_test_case(case, target_axes, bounds)
        release_ram()
        print("🧹 RAM cleared after processing case.")

    print("\n🎉 All test datasets converted successfully!")


if __name__ == "__main__":
    main()