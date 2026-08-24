# -*- coding: utf-8 -*-
"""
ERA5 Regional GRIB to Super-Resolution Test Dataset Converter (NetCDF4).

This script processes regional ERA5 GRIB files (500 hPa pressure level)
and creates standardized Super-Resolution (SR) benchmarks (X2, X4, and Y) in NetCDF format.

Input files (one per region):
- era5_US_East_500hpa_2025.grib
- era5_US_West_500hpa_2025.grib
- era5_Alaska_500hpa_2025.grib
- era5_Europe_500hpa_2025.grib
- era5_Australia_500hpa_2025.grib
- era5_Africa_500hpa_2025.grib

Output files (three per region):
- X2_test_<REGION>.nc
- X4_test_<REGION>.nc
- Y_test_<REGION>.nc

NetCDF Variables:
- Coordinates: latitude, longitude, time, utc_date
- Data Variables: z, q, t, u, v
"""

import gc
import math
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

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
# Global Configurations & Hyperparameters
# ==============================================================================
DATA_DIR = "./data"        # Input directory containing GRIB files
OUTPUT_DIR = "./output"    # Output directory for generated NetCDF files

VARIABLES = ["z", "q", "t", "u", "v"]

GRIB_FILTER = {
    "typeOfLevel": "isobaricInhPa",
    "level": 500,
}

# Native ERA5 spatial resolution in degrees
NATIVE_RESOLUTION = 0.25

# Chunk size for temporal processing (in hours)
READ_CHUNK_HOURS = 1680

# NetCDF Compression and I/O Settings
COMPRESSION_LEVEL = 4
NC_TIME_CHUNK = 24
ENABLE_SYNC_EACH_CHUNK = False  # Synchronize to disk after every chunk write

# Crop dataset to minimum bounding box required to reduce memory consumption
SUBSET_TO_REQUIRED_BOUNDS = True

# Coordinate matching tolerance
COORD_TOL = 1e-4

# Time Encoding Standards
TIME_UNITS = "hours since 1970-01-01 00:00:00"
TIME_CALENDAR = "proleptic_gregorian"
UTC_DATE_FORMAT = "%Y%m%d%H"

VAR_UNITS = {
    "z": "m**2 s**-2",
    "q": "kg kg**-1",
    "t": "K",
    "u": "m s**-1",
    "v": "m s**-1",
}

# Regional domain definitions and target grid configurations
REGIONS = {
    "US_East_500hpa_2025": {
        "input_file": "era5_US_East_500hpa_2025.grib",
        "targets": {
            "Y":  {"lat_start": 25.0, "lat_end": 49.75, "lon_start": -85.0,  "lon_end": -60.25, "step": 0.25, "kernel": 1, "expected_shape": (100, 100)},
            "X2": {"lat_start": 25.0, "lat_end": 49.5,  "lon_start": -85.0,  "lon_end": -60.5,  "step": 0.5,  "kernel": 3, "expected_shape": (50, 50)},
            "X4": {"lat_start": 25.0, "lat_end": 49.0,  "lon_start": -85.0,  "lon_end": -61.0,  "step": 1.0,  "kernel": 5, "expected_shape": (25, 25)},
        },
    },
    "US_West_500hpa_2025": {
        "input_file": "era5_US_West_500hpa_2025.grib",
        "targets": {
            "Y":  {"lat_start": 30.0, "lat_end": 54.75, "lon_start": -130.0, "lon_end": -105.25, "step": 0.25, "kernel": 1, "expected_shape": (100, 100)},
            "X2": {"lat_start": 30.0, "lat_end": 54.5,  "lon_start": -130.0, "lon_end": -105.5,  "step": 0.5,  "kernel": 3, "expected_shape": (50, 50)},
            "X4": {"lat_start": 30.0, "lat_end": 54.0,  "lon_start": -130.0, "lon_end": -106.0,  "step": 1.0,  "kernel": 5, "expected_shape": (25, 25)},
        },
    },
    "Alaska_500hpa_2025": {
        "input_file": "era5_Alaska_500hpa_2025.grib",
        "targets": {
            "Y":  {"lat_start": 50.0, "lat_end": 74.75, "lon_start": -165.0, "lon_end": -140.25, "step": 0.25, "kernel": 1, "expected_shape": (100, 100)},
            "X2": {"lat_start": 50.0, "lat_end": 74.5,  "lon_start": -165.0, "lon_end": -140.5,  "step": 0.5,  "kernel": 3, "expected_shape": (50, 50)},
            "X4": {"lat_start": 50.0, "lat_end": 74.0,  "lon_start": -165.0, "lon_end": -141.0,  "step": 1.0,  "kernel": 5, "expected_shape": (25, 25)},
        },
    },
    "Europe_500hpa_2025": {
        "input_file": "era5_Europe_500hpa_2025.grib",
        "targets": {
            "Y":  {"lat_start": 35.0, "lat_end": 59.75, "lon_start": -10.0, "lon_end": 14.75, "step": 0.25, "kernel": 1, "expected_shape": (100, 100)},
            "X2": {"lat_start": 35.0, "lat_end": 59.5,  "lon_start": -10.0, "lon_end": 14.5,  "step": 0.5,  "kernel": 3, "expected_shape": (50, 50)},
            "X4": {"lat_start": 35.0, "lat_end": 59.0,  "lon_start": -10.0, "lon_end": 14.0,  "step": 1.0,  "kernel": 5, "expected_shape": (25, 25)},
        },
    },
    "Australia_500hpa_2025": {
        "input_file": "era5_Australia_500hpa_2025.grib",
        "targets": {
            "Y":  {"lat_start": -40.0, "lat_end": -15.25, "lon_start": 130.0, "lon_end": 154.75, "step": 0.25, "kernel": 1, "expected_shape": (100, 100)},
            "X2": {"lat_start": -40.0, "lat_end": -15.5,  "lon_start": 130.0, "lon_end": 154.5,  "step": 0.5,  "kernel": 3, "expected_shape": (50, 50)},
            "X4": {"lat_start": -40.0, "lat_end": -16.0,  "lon_start": 130.0, "lon_end": 154.0,  "step": 1.0,  "kernel": 5, "expected_shape": (25, 25)},
        },
    },
    "Africa_500hpa_2025": {
        "input_file": "era5_Africa_500hpa_2025.grib",
        "targets": {
            "Y":  {"lat_start": -12.0, "lat_end": 12.75, "lon_start": 10.0, "lon_end": 34.75, "step": 0.25, "kernel": 1, "expected_shape": (100, 100)},
            "X2": {"lat_start": -12.0, "lat_end": 12.5,  "lon_start": 10.0, "lon_end": 34.5,  "step": 0.5,  "kernel": 3, "expected_shape": (50, 50)},
            "X4": {"lat_start": -12.0, "lat_end": 12.0,  "lon_start": 10.0, "lon_end": 34.0,  "step": 1.0,  "kernel": 5, "expected_shape": (25, 25)},
        },
    },
}


# ==============================================================================
# Helper Functions
# ==============================================================================
def get_ram_gb() -> Optional[float]:
    """Return current process RAM usage in GB if psutil is installed."""
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def release_ram() -> None:
    """Trigger explicit garbage collection to clear unreferenced memory objects."""
    gc.collect()


def build_axis(start: float, end: float, step: float) -> np.ndarray:
    """Construct an evenly spaced 1D coordinate axis (inclusive of end bound)."""
    vals = np.round(np.arange(start, end + step * 0.5, step), 6)
    return vals.astype(np.float32)


def prepare_target_axes(targets: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    """Construct and validate grid coordinate vectors for all target scales."""
    axes = {}
    for name, cfg in targets.items():
        lat = build_axis(cfg["lat_start"], cfg["lat_end"], cfg["step"])
        lon = build_axis(cfg["lon_start"], cfg["lon_end"], cfg["step"])
        exp_lat, exp_lon = cfg["expected_shape"]
        if len(lat) != exp_lat or len(lon) != exp_lon:
            raise ValueError(
                f"Grid shape mismatch for '{name}': got lat={len(lat)}, lon={len(lon)}; "
                f"expected lat={exp_lat}, lon={exp_lon}"
            )
        axes[name] = {"lat": lat, "lon": lon}
    return axes


def compute_required_bounds(targets: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Calculate the minimal spatial extent required across all targets and convolution kernels."""
    req_lat_min = min(cfg["lat_start"] - (cfg["kernel"] // 2) * NATIVE_RESOLUTION for cfg in targets.values())
    req_lat_max = max(cfg["lat_end"]   + (cfg["kernel"] // 2) * NATIVE_RESOLUTION for cfg in targets.values())
    req_lon_min = min(cfg["lon_start"] - (cfg["kernel"] // 2) * NATIVE_RESOLUTION for cfg in targets.values())
    req_lon_max = max(cfg["lon_end"]   + (cfg["kernel"] // 2) * NATIVE_RESOLUTION for cfg in targets.values())
    return req_lat_min, req_lat_max, req_lon_min, req_lon_max


def find_nearest_indices(all_coords: np.ndarray, target_coords: np.ndarray, name: str, tol: float = COORD_TOL) -> np.ndarray:
    """
    Locate indices in `all_coords` closest to `target_coords`.
    
    Raises:
        ValueError: If maximum coordinate offset exceeds tolerance.
    """
    all_coords = np.asarray(all_coords, dtype=np.float64)
    target_coords = np.asarray(target_coords, dtype=np.float64)

    diff = np.abs(all_coords[:, None] - target_coords[None, :])
    idx = diff.argmin(axis=0)
    max_err = np.max(np.abs(all_coords[idx] - target_coords))
    if max_err > tol:
        raise ValueError(f"Alignment error for '{name}' exceeds tolerance: {max_err:.6f} > {tol}")
    return idx.astype(np.int64)


def find_nearest_index_1d(coords: np.ndarray, target: float, name: str, tol: float = COORD_TOL) -> int:
    """Find index of a single target coordinate value in a 1D coordinate array."""
    coords = np.asarray(coords, dtype=np.float64)
    idx = int(np.abs(coords - target).argmin())
    err = abs(coords[idx] - target)
    if err > tol:
        raise ValueError(f"Alignment error for '{name}' exceeds tolerance: {err:.6f} > {tol}")
    return idx


def convert_times(times_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert numpy datetime64 values into numerical NetCDF timestamps and UTC date integers."""
    dt = pd.to_datetime(times_np).to_pydatetime()
    time_num = date2num(dt, units=TIME_UNITS, calendar=TIME_CALENDAR)
    utc_date = np.array([int(pd.Timestamp(t).strftime(UTC_DATE_FORMAT)) for t in dt], dtype=np.int64)
    return np.asarray(time_num, dtype=np.float64), utc_date


def normalize_longitude_to_180(ds: xr.Dataset) -> xr.Dataset:
    """Convert longitude values from [0, 360) to [-180, 180) range if needed, sorted in ascending order."""
    lon = ds["longitude"].values.astype(np.float64)

    if np.nanmax(lon) > 180.0 + COORD_TOL:
        lon_new = ((lon + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords(longitude=lon_new)

    lon_vals = ds["longitude"].values
    if lon_vals[0] > lon_vals[-1] or np.any(np.diff(lon_vals) < 0):
        ds = ds.sortby("longitude")

    return ds


def open_grib_dataset(grib_path: Path) -> xr.Dataset:
    """Open a single GRIB file and return an xarray Dataset containing requested variables."""
    backend_kwargs = {
        "indexpath": "",
        "filter_by_keys": GRIB_FILTER,
    }
    ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs=backend_kwargs)

    # Standardize spatial coordinate names
    rename_map = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename_map["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename_map["lon"] = "longitude"
    if rename_map:
        ds = ds.rename(rename_map)

    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        ds.close()
        raise ValueError("GRIB dataset missing required latitude/longitude coordinates.")

    # Validate variable presence
    missing = [v for v in VARIABLES if v not in ds.data_vars]
    if missing:
        ds.close()
        raise ValueError(
            f"Missing required variables in {grib_path.name}: {missing}. "
            "Ensure GRIB file contains all target variables."
        )
    ds = ds[VARIABLES]

    # Squeeze non-essential single-value dimensions
    for dim in ["step", "isobaricInhPa", "number", "surface", "expver"]:
        if dim in ds.dims:
            if ds.sizes[dim] == 1:
                ds = ds.isel({dim: 0}, drop=True)
            else:
                ds.close()
                raise ValueError(
                    f"Unexpected dimension '{dim}' with size {ds.sizes[dim]}. "
                    "Pre-filter datasets to single-level slices."
                )

    if "time" not in ds.coords:
        if "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        else:
            ds.close()
            raise ValueError("Time coordinate not found in dataset.")

    # Ensure latitude and longitude coordinates are in ascending order
    if ds["latitude"].values[0] > ds["latitude"].values[-1]:
        ds = ds.sortby("latitude")

    ds = normalize_longitude_to_180(ds)
    return ds


def check_spatial_coverage(lat_all: np.ndarray, lon_all: np.ndarray, targets: Dict[str, Any]) -> None:
    """Verify that source dataset spatial bounds cover required targets and convolution boundaries."""
    lat_min_data, lat_max_data = float(np.min(lat_all)), float(np.max(lat_all))
    lon_min_data, lon_max_data = float(np.min(lon_all)), float(np.max(lon_all))

    req_lat_min, req_lat_max, req_lon_min, req_lon_max = compute_required_bounds(targets)

    ok = (
        lat_min_data <= req_lat_min + COORD_TOL
        and lat_max_data >= req_lat_max - COORD_TOL
        and lon_min_data <= req_lon_min + COORD_TOL
        and lon_max_data >= req_lon_max - COORD_TOL
    )
    if not ok:
        raise ValueError(
            "GRIB spatial extent is insufficient to cover target grids and convolution margins.\n"
            f"Data bounds:     lat [{lat_min_data}, {lat_max_data}], lon [{lon_min_data}, {lon_max_data}]\n"
            f"Required bounds: lat [{req_lat_min}, {req_lat_max}], lon [{req_lon_min}, {req_lon_max}]"
        )


def subset_to_required_bounds(ds: xr.Dataset, targets: Dict[str, Any]) -> xr.Dataset:
    """Crop dataset to minimal bounding box needed to reduce RAM usage."""
    req_lat_min, req_lat_max, req_lon_min, req_lon_max = compute_required_bounds(targets)

    lat_all = ds["latitude"].values
    lon_all = ds["longitude"].values

    i0 = find_nearest_index_1d(lat_all, req_lat_min, "subset lat_min")
    i1 = find_nearest_index_1d(lat_all, req_lat_max, "subset lat_max")
    j0 = find_nearest_index_1d(lon_all, req_lon_min, "subset lon_min")
    j1 = find_nearest_index_1d(lon_all, req_lon_max, "subset lon_max")

    if i0 > i1:
        i0, i1 = i1, i0
    if j0 > j1:
        j0, j1 = j1, j0

    return ds.isel(latitude=slice(i0, i1 + 1), longitude=slice(j0, j1 + 1))


# ==============================================================================
# NetCDF Appender Class
# ==============================================================================
class NetCDFAppender:
    """Incremental chunk-wise NetCDF4 writer supporting temporal appending."""

    def __init__(
        self,
        out_path: Path,
        lat_vals: np.ndarray,
        lon_vals: np.ndarray,
        variables: List[str],
        split_name: str,
        target_name: str,
        region_name: str,
        source_file: str,
    ):
        self.out_path = Path(out_path)
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
        self.nc.title = f"ERA5 {target_name} dataset ({split_name}, {region_name})"
        self.nc.source = f"{source_file} (GRIB) -> NetCDF"
        self.nc.processing = "Y=original; X2=3x3 rolling mean + 0.5deg sampling; X4=5x5 rolling mean + 1.0deg sampling"
        self.nc.history = f"Created at {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"

        self.t_cursor = 0

    def append(self, time_num: np.ndarray, utc_date: np.ndarray, data_dict: Dict[str, np.ndarray]) -> None:
        """Append a time-chunk of array data to the NetCDF file."""
        n = len(time_num)
        sl = slice(self.t_cursor, self.t_cursor + n)

        self.time_var[sl] = time_num
        self.utc_var[sl] = utc_date

        for v, arr in data_dict.items():
            self.data_vars[v][sl, :, :] = arr.astype(np.float32, copy=False)

        self.t_cursor += n

    def sync(self) -> None:
        """Synchronize NetCDF data to disk."""
        self.nc.sync()

    def close(self) -> None:
        """Close NetCDF dataset handle."""
        self.nc.close()


# ==============================================================================
# Execution Pipeline
# ==============================================================================
def process_region(region_name: str, region_cfg: Dict[str, Any]) -> None:
    """Process a single region GRIB file into X2, X4, and Y NetCDF datasets."""
    print("\n" + "=" * 100)
    print(f"Processing region: {region_name}")

    grib_path = Path(DATA_DIR) / region_cfg["input_file"]
    if not grib_path.exists():
        raise FileNotFoundError(f"Input file not found: {grib_path}")

    targets = region_cfg["targets"]
    target_axes = prepare_target_axes(targets)

    print("Target grid validation:")
    for name in ["Y", "X2", "X4"]:
        print(
            f"  {name}: "
            f"lat={len(target_axes[name]['lat'])}, lon={len(target_axes[name]['lon'])}, "
            f"lat_range=({target_axes[name]['lat'][0]}, {target_axes[name]['lat'][-1]}), "
            f"lon_range=({target_axes[name]['lon'][0]}, {target_axes[name]['lon'][-1]})"
        )

    out_paths = {
        "X2": Path(OUTPUT_DIR) / f"X2_test_{region_name}.nc",
        "X4": Path(OUTPUT_DIR) / f"X4_test_{region_name}.nc",
        "Y":  Path(OUTPUT_DIR) / f"Y_test_{region_name}.nc",
    }

    ds = None
    writers = {}

    try:
        print(f"\n📂 Reading: {grib_path.name}")
        ds = open_grib_dataset(grib_path)

        lat_all = ds["latitude"].values
        lon_all = ds["longitude"].values
        check_spatial_coverage(lat_all, lon_all, targets)

        if SUBSET_TO_REQUIRED_BOUNDS:
            ds = subset_to_required_bounds(ds, targets)
            lat_all = ds["latitude"].values
            lon_all = ds["longitude"].values
            print(
                "Cropped to minimal required bounds: "
                f"lat [{lat_all[0]}, {lat_all[-1]}] ({len(lat_all)} points), "
                f"lon [{lon_all[0]}, {lon_all[-1]}] ({len(lon_all)} points)"
            )

        # Coordinate index mapping
        idx = {}
        for name in ["Y", "X2", "X4"]:
            idx[name] = {
                "lat": find_nearest_indices(lat_all, target_axes[name]["lat"], f"{region_name}-{name} latitude"),
                "lon": find_nearest_indices(lon_all, target_axes[name]["lon"], f"{region_name}-{name} longitude"),
            }

        # Instantiate output file appenders
        writers = {
            name: NetCDFAppender(
                out_path=out_paths[name],
                lat_vals=target_axes[name]["lat"],
                lon_vals=target_axes[name]["lon"],
                variables=VARIABLES,
                split_name="test",
                target_name=name,
                region_name=region_name,
                source_file=grib_path.name,
            )
            for name in ["X2", "X4", "Y"]
        }

        n_time = ds.sizes["time"]
        n_chunks = math.ceil(n_time / READ_CHUNK_HOURS)

        pbar = tqdm(
            range(0, n_time, READ_CHUNK_HOURS),
            total=n_chunks,
            desc=region_name,
            unit="chunk",
            leave=True,
        )

        for t0 in pbar:
            t1 = min(t0 + READ_CHUNK_HOURS, n_time)

            # Extract time chunk
            dsc = ds[VARIABLES].isel(time=slice(t0, t1)).astype(np.float32)

            # Y Target: Direct extraction at native resolution
            y_ds = dsc.isel(latitude=idx["Y"]["lat"], longitude=idx["Y"]["lon"])

            # X2 Target: 3x3 Rolling mean filter + 0.5 deg subsampling
            roll3 = dsc.rolling(
                latitude=targets["X2"]["kernel"],
                longitude=targets["X2"]["kernel"],
                center=True,
            ).mean()
            x2_ds = roll3.isel(latitude=idx["X2"]["lat"], longitude=idx["X2"]["lon"])

            # X4 Target: 5x5 Rolling mean filter + 1.0 deg subsampling
            roll5 = dsc.rolling(
                latitude=targets["X4"]["kernel"],
                longitude=targets["X4"]["kernel"],
                center=True,
            ).mean()
            x4_ds = roll5.isel(latitude=idx["X4"]["lat"], longitude=idx["X4"]["lon"])

            # Temporal conversion
            time_num, utc_date = convert_times(dsc["time"].values)

            # Write arrays to disk
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
                pbar.set_postfix_str(f"Written time steps={writers['Y'].t_cursor}")
            else:
                pbar.set_postfix_str(f"Written time steps={writers['Y'].t_cursor}, RAM={ram:.2f} GB")

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

    print(f"🎉 Successfully completed region: {region_name}")
    for name in ["X2", "X4", "Y"]:
        p = out_paths[name]
        size_gb = p.stat().st_size / (1024 ** 3) if p.exists() else 0
        print(f"  - {p.name}: {size_gb:.2f} GB")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Input Directory:  {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Processing {len(REGIONS)} regions to generate Super-Resolution test benchmarks.")

    for region_name, region_cfg in REGIONS.items():
        process_region(region_name, region_cfg)
        release_ram()
        print(f"🧹 Freed memory state after completing region: {region_name}")

    print("\nAll Super-Resolution test datasets processed successfully!")


if __name__ == "__main__":
    main()