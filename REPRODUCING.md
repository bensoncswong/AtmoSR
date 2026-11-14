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
