# AtmoSR

Physics-constrained dual-branch network for atmospheric spatial downscaling.
The repository contains preprocessing, training, evaluation, spectral and
extreme-value diagnostics, and figure scripts.

The recommended training environment is Google Colab with a single NVIDIA
A100 GPU (40 GB). The runs reported with this code used PyTorch 2.10.0
and CUDA 12.8.

Trained checkpoints, ERA5 GRIB extracts, and processed NetCDF are archived
on Zenodo (see [Data archive](#4-data)). Replace the DOI after publication:
`https://doi.org/10.5281/zenodo.22092919`.

## 1. Repository layout

| Path | Description |
| --- | --- |
| `preprocessing/` | ERA5 GRIB → NetCDF (East Asia 500 hPa, other regions, and other pressure levels) |
| `models/` | Training scripts: `AtmoSR.py`, `VDSR.py`, `EDSR.py`, `SwinIR.py` |
| `evaluate/` | Test-set metrics, physics diagnostics, bootstrap CIs |
| `extremes/` | Tail / extreme-value evaluation and plots |
| `spectral_analysis/` | Kinetic-energy spectra and spectral coherence |
| `plot/` | Case maps, loss curves, and summary figures |
| `requirements.txt` | Python packages (PyTorch is installed separately) |

## 2. Software requirements

- Python 3.10 or newer
- NVIDIA GPU with CUDA (A100 40 GB recommended for the full 200-epoch runs)
- PyTorch 2.10.0 with CUDA 12.8 (**do not** install `torch` from `requirements.txt`)

Python packages:

```bash
pip install -r requirements.txt
```

System libraries (needed only if you read ERA5 GRIB files or draw maps):

```bash
# Debian / Ubuntu / Colab
apt-get update
apt-get install -y libeccodes-dev libeccodes-tools \
                   libproj-dev proj-data proj-bin libgeos-dev
```

Check GRIB support after install:

```bash
python -m cfgrib selfcheck
```

## 3. Google Colab (A100)

**Runtime → Change runtime type → Hardware accelerator: A100 GPU.**
A High-RAM runtime is recommended for GRIB conversion.

Colab already provides a CUDA-enabled PyTorch. Install the remaining
packages from a notebook cell (adjust the clone URL after you publish
the repository):

```python
!apt-get update -qq
!apt-get install -y -qq libeccodes-dev libeccodes-tools \
    libproj-dev proj-data proj-bin libgeos-dev

!git clone https://github.com/<OWNER>/<REPO>.git
%cd <REPO>
!pip install -q -r requirements.txt
```

Training and evaluation read NetCDF files from the current working
directory. After cloning, either keep the notebook in the repository
folder and copy the `.nc` files there, or `%cd` to the folder that holds
the data and call the scripts by path, for example:

```python
!python /content/<REPO>/models/AtmoSR.py
```

Match the training stack (PyTorch 2.10.0, CUDA 12.8) if the runtime
ships a different version. Then restart the runtime once.

```python
import torch
print(torch.__version__, torch.cuda.get_device_name(0))

# Only if the printout is not 2.10.0 with CUDA:
# !pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

Confirm GPU before training:

```python
import torch
assert torch.cuda.is_available()
print(torch.__version__)
print(torch.cuda.get_device_name(0))
```

Colab disks are ephemeral. Mount Drive if you want checkpoints to persist:

```python
from google.colab import drive
drive.mount("/content/drive")
```

If you skip GRIB conversion and start from NetCDF files, the `libeccodes`
packages are not required. Cartopy is required only for `plot/PlotCase.py`.

## 4. Data

Source: ERA5 hourly pressure-level fields from the
[Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).
Registration and a CDS licence are required.

### Core experiment (East Asia, 500 hPa)

| Item | Setting |
| --- | --- |
| Domain | 15 N to 39.75 N, 100 E to 124.75 E |
| Variables | `z`, `q`, `t`, `u`, `v` |
| Native | 0.25 deg (`Y`, 100 × 100) |
| 2× input | 0.50 deg (`X2`, 50 × 50), 3 × 3 area average |
| 4× input | 1.00 deg (`X4`, 25 × 25), 5 × 5 area average |

Default GRIB names for the core converter (`preprocessing/preprocessing.py`):

```text
era5_{year}.grib
```

Year split is set by `SPLIT_YEARS` at the top of that script. The East Asia
500 hPa experiments use:

| Split | Years |
| --- | --- |
| train | 2022–2023 |
| valid | 2024 |
| test | 2025 |

Edit `SPLIT_YEARS` before running if the file currently lists a wider range.

Processed NetCDF files expected by training and evaluation (working directory):

```text
X2_train.nc  Y_train.nc
X2_valid.nc  Y_valid.nc
X2_test.nc   Y_test.nc
```

Optional 4× files: `X4_train.nc`, `X4_valid.nc`, `X4_test.nc`.

### Zenodo archive

The accompanying data record (CC-BY-4.0) holds three zip files plus a
README that lists every file. Upload `README.md` as the default preview.

| Zip file | Source folder | Contents |
| --- | --- | --- |
| `atmosr_trained_models_v1.0.zip` | `trained_model/` | Checkpoints, scalers, histories, test metrics |
| `atmosr_era5_raw_grib_v1.0.zip` | `ERA5_raw/` | Hourly ERA5 pressure-level GRIB extracts |
| `atmosr_processed_netcdf_v1.0.zip` | `dataset/` | Train / valid / test NetCDF (`Y`, `X2`, `X4`) |

Unzip so that `trained_model/`, `ERA5_raw/`, and `dataset/` appear as
top-level folders. The processed East Asia 500 hPa files sit in the root
of `dataset/` (`X2_train.nc`, `Y_train.nc`, …). Other levels are under
`dataset/multilevel/` and `dataset/925hpa/`; other regions under
`dataset/region/`.

Copy or rename checkpoints into the working directory used by the
evaluation scripts:

| File in `trained_model/` | Name expected by `evaluate/AtmoSR_ablation_models_eval.py` |
| --- | --- |
| `AtmoSRbase/best_model.pth` | `best_model_noPhy.pth` |
| `AtmoSR/best_model.pth` | `best_model_AllPhy.pth` |
| `Phy_DIV/best_model.pth` | `best_model_DIV.pth` |
| `Phy_VOR/best_model.pth` | `best_model_VOR.pth` |
| `Phy_GEO/best_model.pth` | `best_model_GEO.pth` |
| `Phy_MSE/best_model.pth` | `best_model_MSE.pth` |
| `AtmoSR/scalers.joblib` | `scalers.joblib` |

GRIB files in `ERA5_raw/ERA5_500hPa/` use `era5_{year}.grib` and match
`preprocessing/preprocessing.py`. Contains modified Copernicus Climate
Change Service information (2022–2025). Neither the European Commission
nor ECMWF is responsible for any use that may be made of the Copernicus
information or data it contains.

Full inventory, zip commands, and licence notes are in the Zenodo
`README.md`.

## 5. How paths are set

Scripts do not use a command-line interface. Paths, years, batch size,
and loss weights are constants at the top of each file. On Colab, several
preprocessing scripts default `DATA_DIR` and `OUTPUT_DIR` to `/content`.

Place NetCDF files, `scalers.joblib`, and checkpoints in the current
working directory, or edit the path constants before running. Training
and evaluation scripts look for NetCDF files in the process working
directory, not in the folder that contains the `.py` file.

## 6. Workflow

Commands below assume the repository root is the working directory and
that data files sit in that directory (or that you have edited the paths).

### 6.1 Preprocessing

```bash
python preprocessing/preprocessing.py
```

Other regions (500 hPa test sets):

```bash
python preprocessing/preprocessing_multi_region.py
```

Other pressure levels (East Asia test sets):

```bash
python preprocessing/preprocessing_multi_level_pressure.py
```

### 6.2 Train AtmoSR (physics-constrained default)

```bash
python models/AtmoSR.py
```

Defaults: Adam, lr 1e-4, weight decay 1e-4, batch size 32, 200 epochs,
ReduceLROnPlateau (factor 0.1, patience 3), no early stopping, no data
augmentation, no gradient clipping. Physics weights: `LAMBDA_DIV` =
`LAMBDA_VOR` = `LAMBDA_GEO` = `LAMBDA_MSE` = 0.1.

Outputs in the working directory:

```text
best_model.pth
scalers.joblib
training_history.csv
model_info.json
loss_epoch.png
physics_loss_epoch.png
```

AtmoSR-Base (L1 only): set `LAMBDA_DIV`, `LAMBDA_VOR`, `LAMBDA_GEO`, and
`LAMBDA_MSE` to `0.0` and run again. Single-constraint ablations: keep one
physics weight at `0.1` and set the others to `0.0`.

Rename each run so evaluation scripts can find them. Expected names in
`evaluate/AtmoSR_ablation_models_eval.py`:

```text
best_model_noPhy.pth
best_model_AllPhy.pth
best_model_DIV.pth
best_model_VOR.pth
best_model_GEO.pth
best_model_MSE.pth
```

`plot/PlotLoss.py` uses:

```text
training_history_noPhy.csv
training_history_AllPhy.csv
```

### 6.3 Train comparison architectures (optional)

```bash
python models/VDSR.py
python models/EDSR.py
python models/SwinIR.py
```

These scripts loop over L1-only and L1+GRE settings defined in
`EXPERIMENTS`. `VDSR.py` also has `RUN_ONLY_EXPERIMENTS` (set it to `None` to
run every entry). Confirm those lists before a long Colab session.
`SwinIR.py` may use automatic mixed precision.

### 6.4 Evaluation

```bash
python evaluate/AtmoSR_ablation_models_eval.py
python evaluate/AtmoSR_multi_region_eval.py
python evaluate/AtmoSR_multi_level_pressure_eval.py
python evaluate/VDSR_eval.py
python evaluate/EDSR_eval.py
python evaluate/SwinIR_eval.py
```

Check `MODEL_PATHS`, `X_TEST_PATH` / `Y_TEST_PATH`, and `SCALER_PATH` in each
file. The cubic-spline baseline is computed inside the evaluation
scripts; it is not a trained model.

### 6.5 Diagnostics and figures

```bash
python extremes/AtmoSR_extremes_eval.py
python extremes/AtmoSR_extremes_eval_plot.py
python spectral_analysis/spectral_analysis.py
python spectral_analysis/spectral_analysis_plot.py
python plot/PlotLoss.py
python plot/PlotCase.py
python plot/PlotAtmoSR_RMSE.py
python plot/cross_architecture_generalization.py
python plot/cross_architecture_generalization_two_panel.py
```

`PlotLoss.py` expects:

```text
/content/training_history_noPhy.csv
/content/training_history_AllPhy.csv
```

Point those constants at your history CSVs if they live elsewhere.

Spectral plotting reads CSV files under `./spectral_outputs/` (see
`INPUT_DIR` in `spectral_analysis/spectral_analysis_plot.py`). Some plot
scripts embed tabulated metrics and can be run without NetCDF files.

## 7. Hardware notes

Full AtmoSR training (batch 32, 50 × 50 → 100 × 100, 200 epochs) is
intended for one A100 40 GB GPU. A 200-epoch AtmoSR run is on the order
of one hour on that device; SwinIR is slower per epoch.

If GPU memory is tight, lower `BATCH_SIZE` and `NUM_WORKERS` at the top of
the training script. Evaluation can use `EVAL_MAX_SAMPLES` for a short
dry run (set it back to `None` for the full test year).

GRIB conversion is CPU- and RAM-bound. Use a High-RAM Colab runtime
and keep `READ_CHUNK_HOURS` at the script default unless memory is scarce.

## 8. Reproducibility

Random seed 42 is set in the training and evaluation scripts.
AtmoSR enables deterministic cuDNN. Bit-identical GPU results are not
guaranteed across driver or PyTorch versions; pin PyTorch 2.10.0 with
CUDA 12.8 to stay close to the reported configuration.

## 9. Citation

If you use this code, please cite:

> Wong, C., Kong, H., Zhong, C., Wang, Y., and Du, Y. (2026).
> AtmoSR: A physics-constrained dual-branch network for atmospheric
> downscaling.

If you use the archived checkpoints or NetCDF files, also cite the Zenodo
record (replace the DOI after publication):

> Wong, C., Kong, H., Zhong, C., Wang, Y., and Du, Y. (2026).
> AtmoSR trained models, ERA5 extracts, and processed NetCDF
> (Version 1.0) [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.22092919
