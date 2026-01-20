# WaveDiff / WF-PSF Changelog

All notable changes to WaveDiff are documented in this file.

## [Unreleased]

- Changes in progress for the next release.

## [3.0.0] – 2026-01-20

Major update with PSF model refactoring, masked training, and CI/doc improvements.

## Breaking changes

- Removed the `--repodir` argument in CLI; scripts using it will now fail.

## New features

- Added physical layer model with improved modularity (`TFPhysicalPolychromatic` refactor) and configurable parameters.
- Introduced rotation of obscuration mask with configurable parameter.
- Updated `CentroidEstimator` to support mask-based estimation.
- Added options for user-configurable flags:
    - `use_prior` for Zernike prior
    - `correct_centroids` and `sigma_centroid_window` for centroid error correction
    - `add_ccd_misalignments` and `ccd_misalignments_input_path` for CCD misalignment correction
- Added option to randomize the data-driven part seed for reproducibility.
- Added phase retrieval projection algorithm considering obscurations.
- Masked training and evaluation: added `masked_mse` loss and `MaskedMeanSquaredErrorMetric` classes.
- Added configurable parameter for computing shape metrics for test datasets as optional.
- Added new sigmoid parameters to apply to the sample weights.
- Added masked datasets and corresponding generation notebooks.

## Bug fixes

- Fixed missing e₂ and R₂ shape metric plots
- Fixed broken contribution link in documentation
- Corrected bug in `MonochromaticMetricsPlotHandler` class regarding `eval_mono_metric` configuration.

## Performance improvements

- Improved numerical stability and reproducibility in training routines

## Internal changes

- Replaced Black with Ruff for linting and formatting.
- Updated TensorFlow to 2.11 (compatibility fixes for NumPy ≥ 1.26.4 and Astropy).
- Reorganized modules for clarity (e.g., `SimPSFToolkit.py` renamed).
- Improved PEP8 compliance across the codebase.
- Introduced Scriv-based changelog infrastructure.
- Configured `sphinx.ext.autosummary` to auto-generate stubs in `_autosummary/`.
- Added new documentation and templates: `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `DEV_WORKFLOW.md`.


