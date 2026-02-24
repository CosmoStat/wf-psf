# WaveDiff / WF-PSF Changelog

All notable changes to WaveDiff are documented in this file.

<!-- scriv-end-here -->

## [Unreleased]

- Changes in progress for the next release.
  
<a id='changelog-v3.1.0'></a>
# v3.1.0 — 2026-02-23

## New features

- Added PSF inference capabilities for generating broadband (polychromatic) PSFs from trained models given star positions and SEDs
- Introduced `PSFInferenceEngine` class to centralize training, simulation, metrics, and inference workflows
- Added `run_type` attribute to `DataHandler` supporting training, simulation, metrics, and inference modes
- Implemented `ZernikeInputsFactory` class for building `ZernikeInputs` instances based on run type
- Added `psf_model_loader.py` module for centralized model weights loading

- Added configurable optimizer selection system via new `optimizer.py` module with `get_optimizer` function
- Optimizer configuration now supports multiple input types: `RecursiveNamespace` from configs, dictionaries, or string names
- Added support for hyperparameter overrides (learning rate, beta1/beta2, epsilon, amsgrad) via YAML or programmatic configuration
- `RectifiedAdam` optimizer now dynamically imports TensorFlow Addons only when explicitly specified in configuration

## Bug fixes

- Fix logger formatting for relative RMSE metrics in `metrics.py` (values were not being displayed)

## Internal changes

- Added pre-commit hooks for code quality, formatting, and changelog enforcement
- Refactored `TFPhysicalPolychromatic` and related modules to separate training vs. inference behavior
- Enhanced `ZernikeInputs` data class with intelligent assembly based on run type and available data
- Implemented hybrid loading pattern with eager loading in constructors and lazy-loading via property decorators
- Centralized PSF data extraction in `data_handler` module
- Improved code organization with new `tf_utils.py` module in `psf_models` sub-package
- Updated configuration handling to support inference workflows via `inference_config.yaml`
- Fixed incorrect argument name in `DataHandler` that prevented proper TensorFlow data type conversion
- Removed deprecated `get_obs_positions` method
- Updated documentation to include inference package
- Refactored `build_PSF_model` to accept either Keras optimizer instances or configuration passed through `get_optimizer`
- Added `interpolation.py` and `types.py` modules with vendored code from TensorFlow Addons repository
- Replaced `tfa.image.interpolate_spline` with local `tfa_interpolate_spline_rbf` implementation
- Added comprehensive unit tests in `test_optimizer.py` and `test_interpolation.py`
- Updated README and added THIRD_PARTY_LICENSE directory with TensorFlow Addons license
- Training now runs on TensorFlow 2.11 without requiring TensorFlow Addons installation
- Removed TensorFlow Addons as a required dependency; RectifiedAdam optimizer now requires explicit TFA installation if needed
- Remove deprecated/optional import tensorflow-addons statement from tf_layers.py
- Fixed Sphinx autosummary import errors by removing core dependencies (tensorflow) from `autodoc_mock_imports` in `conf.py`.
- Updated pyproject.toml to include all wf_psf packages under `src/` and include config/yaml files.
- Updated example configuration files with clearer inline comments.
- Generated API documentation for new `inference` package in `api.rst`
- Generated API documentation for new `instrument` package in `api.rst`
- Inference Configuration section in `configuration.md` documenting `inference_config.yaml`
- Restructured Configuration documentation:
  - Split workflows into "CLI Tasks" and "Additional Components" sections
  - Added configuration file dependency table showing required vs optional files per task
  - Clarified configuration filename flexibility (filenames customizable, internal structure fixed)
  - Standardized section titles (Training Configuration, Metrics Configuration, etc.)
  - Improved markdown formatting and fixed broken anchor links
- Updated `dependencies.md` to document `tensorflow-addons` as optional dependency with manual installation instructions
- `tensorflow-addons` from core dependencies documentation (now documented as optional)
s
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


