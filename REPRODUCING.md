# Reproducing AtmoSR v1.0

## 1. Archived resources

Software:
https://doi.org/10.5281/zenodo.24215574

Data and trained models:
https://doi.org/10.5281/zenodo.22092919

Use software version 1.0 together with the experiment settings and
artifact paths listed in docs/experiments.csv.

## 2. Environment

The documented training configuration uses Python 3.10 or newer,
PyTorch 2.10.0, CUDA 12.8, and one NVIDIA A100 GPU with 40 GB of memory.

Install PyTorch separately from requirements.txt. Install the remaining
Python dependencies with:

```bash
pip install -r requirements.txt

GRIB conversion additionally requires ecCodes and cfgrib. Map plotting
requires the mapping dependencies documented in README.md.

For an execution record, save the Python and package versions from the
environment in which the run is performed. Also record the PyTorch
version, PyTorch CUDA version, GPU model, and NVIDIA driver version.

3. Data layout
Extract the data archives so that the following directories are available:

assets/
├── dataset/
├── trained_model/
└── ERA5_raw/

The core East Asia 500 hPa NetCDF files are located in dataset/:
X2_train.nc
Y_train.nc
X2_valid.nc
Y_valid.nc
X2_test.nc
Y_test.nc

The 4x inputs are X4_train.nc, X4_valid.nc, and X4_test.nc.

The core temporal split is:
Training: 2022 and 2023
Validation: 2024
Test: 2025
The scripts read data from the process working directory unless their
path constants are changed.

4. Evaluate the archived 2x models
Create a separate evaluation working directory. Place X2_test.nc and
Y_test.nc in that directory, or set the evaluation script paths to their
actual locations.

Copy the following archived files into the evaluation working directory
using the expected names:
trained_model/AtmoSRbase/best_model.pth -> best_model_noPhy.pth
trained_model/AtmoSR/best_model.pth     -> best_model_AllPhy.pth
trained_model/Phy_DIV/best_model.pth   -> best_model_DIV.pth
trained_model/Phy_VOR/best_model.pth   -> best_model_VOR.pth
trained_model/Phy_GEO/best_model.pth   -> best_model_GEO.pth
trained_model/Phy_MSE/best_model.pth   -> best_model_MSE.pth
trained_model/AtmoSR/scalers.joblib    -> scalers.joblib

Check MODEL_PATHS, X_TEST_PATH, Y_TEST_PATH, and SCALER_PATH in
evaluate/AtmoSR_ablation_models_eval.py before execution.

From the evaluation working directory, run the script by its absolute
path:
python /absolute/path/to/AtmoSR/evaluate/AtmoSR_ablation_models_eval.py

If EVAL_MAX_SAMPLES is available, use a small value for an initial
execution check. Restore the full-test setting before calculating
manuscript results.

5. Train the core 2x model
Use a new training working directory. Place the six core training,
validation, and test NetCDF files there, or edit the path constants in
models/AtmoSR.py.

From that working directory, run:
python /absolute/path/to/AtmoSR/models/AtmoSR.py

The default model uses an L1 weight of 1.0 and weights of 0.1 for DIV,
VOR, GEO, and the moist static energy gradient constraint.

The script writes:
best_model.pth
scalers.joblib
training_history.csv
model_info.json
loss_epoch.png
physics_loss_epoch.png

Use separate working directories for different experiments because
these output names are reused.

For AtmoSR-Base, set all four physics weights to 0.0. For each
single-constraint experiment, set the selected physics weight to 0.1
and the remaining physics weights to 0.0.

The script evaluates the test set during training. Checkpoint selection
and learning-rate scheduling use the validation objective.

6. Additional experiments
The 4x experiments require the X4 input files and an upscale factor of 4.
Use the exact settings and checkpoints recorded in docs/experiments.csv.

Cross-architecture, cross-region, cross-level, and 925 hPa retraining
experiments also use the settings and artifact paths recorded in that
file.

Record every change to script-level constants. A command alone does
not specify an experiment when its configuration is stored in the
source file.

7. Spectral analysis and figures
Run the spectral computation script before running the coherence
summary and plotting scripts.

The spectral workflow must use the same coherence definition,
threshold, interpolation rule, and resolved-wavenumber range as the
manuscript.

The file docs/experiments.csv identifies the scripts, input artifacts,
and output files used for each manuscript figure and table.

Some plotting scripts contain tabulated values. Their corresponding
metric files and generating workflows are identified in the experiment
index.

8. Reproduction records
For each reproduced experiment, retain:

The software version and Git commit
The input-data version and relevant file paths
The experiment configuration
The execution environment
The execution log and generated outputs
The training seed is 42. Numerical results can vary across hardware,
drivers, and software versions.
