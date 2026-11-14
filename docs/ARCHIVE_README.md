# AtmoSR v1.0 source-code archive

## Overview

This record archives the source code for AtmoSR v1.0, a dual-branch
model for atmospheric spatial downscaling with relative-error physics
constraints.

AtmoSR jointly reconstructs geopotential, specific humidity,
temperature, and horizontal wind. A fixed bicubic branch provides
the interpolated background, and a residual convolutional neural
network supplies the learned correction.

## Archived version

- Software version: 1.0
- GitHub tag: v1.0
- Software DOI: https://doi.org/<CODE_DOI>
- Repository: https://github.com/bensoncswong/AtmoSR
- Release:
  https://github.com/bensoncswong/AtmoSR/releases/tag/v1.0

PROVENANCE.txt identifies the exact Git commit associated with the
uploaded source archive.

## Files in this record

- atmosr_source_code_v1.0.zip:
  Source code and documentation for the archived release
- README.md:
  Overview of this software archive
- PROVENANCE.txt:
  Version and source-archive identification
- SHA256SUMS.txt:
  SHA-256 checksums for the uploaded archive and documentation

## Data and trained models

The accompanying data and trained-model artifacts are archived
separately at:

https://doi.org/10.5281/zenodo.22092919

That record contains ERA5 GRIB extracts, processed NetCDF files,
trained checkpoints, and associated model artifacts.

## Running the software

After extracting the source archive, read README.md and
REPRODUCING.md. The experiment index in docs/experiments.csv links
the manuscript analyses to their scripts, configurations, and
archived artifacts.

## Licence

The source code is released under the MIT License. See LICENSE
inside the source archive.

Data and model artifacts are governed by the terms documented in
the separate data record.

## Citation

Wong, C., Kong, H., Zhong, C., Wang, Y., and Du, Y. (2026).
Source code for AtmoSR v1.0: a dual-branch model for atmospheric spatial
downscaling with relative-error physics constraints (Version 1.0)
[Software]. Zenodo. https://doi.org/<CODE_DOI>
