# WaveDiff / WF-PSF Changelog

All notable changes to WaveDiff are documented in this file.

<!-- scriv-end-here -->

## [Unreleased]

- Changes in progress for the next release.

<a id='changelog-v3.2.0'></a>
# v3.2.0 — 2026-06-04

## Breaking changes

### Data configuration and data access
- `DataConfigHandler` is now responsible only for parsing, normalizing, and validating data configuration parameters from YAML files. It no longer loads, adapts, or returns datasets.
- Dataset loading, preprocessing and access are now handled by the `DataAdapter` system.
- Users must:
  1. Create a `DataConfigHandler` from a data configuration file.
  2. Pass the resulting configuration to the `DataAdapterFactory`.
  3. Use the returned `DataAdapter` instance to access data products.
- All training, metrics, and inference pipelines now use the `DataAdapterFactory` to provide a `DataAdapter` instance for dataset access.
- Code that previously relied on `DataConfigHandler` returning datasets must be updated to explicitly construct a `DataAdapter` via `DataAdapterFactory`.
- The deprecated `data_handler` module has been removed.

### Data configuration schema
- The `target_field` parameter is now required in `data_config.yaml`. It identifies the field containing the source images used to construct training and  test datasets replacing the prevous implicit use of `stars`/`noisy_stars` in the source code.
- Dataset fields must follow canonical names defined in `wf_psf.data.constants.CANONICAL_DATASET_KEYS`. Custom field names must be mapped via `canonical_keys` in the data configuration
- Previous configuration files are no longer compatible:
  - `training_config.yaml` and `metrics_config.yaml` must replace `n_bins_lda` with `n_bins_lambda`.
  - The previous `data_config.yaml` example has been removed. Users must update their data configuration files to match the schema exposed via `DataConfigHandler.params`.
- Example configuration files demonstrating the new supported formats are provided:
  - `data_complete_config.yaml` – configuration for loading a complete dataset and converting into train and test datasets.
  - `data_split_config.yaml` – configuration specifying separate train and test datasets.
  - `data_shallow_config.yaml` – configuration for shallow (without field "complete") dataset definitions and converting into train and test datasets.

### Import path changes
- Config handlers moved to their respective task packages. Import paths changed:
  - `TrainingConfigHandler` -> `wf_psf.training.training_config_handler`
  - `MetricsConfigHandler` -> `wf_psf.metrics.metrics_config_handler`
  - `PlottingConfigHandler` -> `wf_psf.plotting.plotting_config_handler`
  - `DataConfigHandler` -> `wf_psf.data.data_config_handler`
  - Backward compatibility maintained via `wf_psf.utils.configs_handler`

### API changes
- `tf_psf_field` now explicitly concatenates `data.train_data` and `data.test_data` when generating Zernike ground-truth inputs. These must be retrieved from the normalized `DatasetContainer`.
- `train.get_loss_metrics_monitor_and_outputs` has been removed. Its functionality is now partially replaced by the `TrainingDataAdapter` and the `get_loss_and_metrics` factory.

### Platform support
- WaveDiff now supports Python 3.10 and 3.11 with TensorFlow 2.15.
- Python 3.9 is no longer included in the CI test matrix.

## New features

### Configuration system
- Added `ConfigHandler` abstract base class (ABC) to provide type safety and enforce a common interface for configuration handlers.
- Training, metrics and plotting configuration handlers inherit from `ConfigHandler` and implement a standardised `run()` method
- Enhanced the configuration handler registry with ABC-based validation and interface enforcement.
- Added the SupportsParams protocol to allow external APIs to provide dataset parameters and metadata through generic objects.

### Dataset loading and adaptation
- Add `NpyDatasetLoader` class to load `.npy` files containing datasets, supporting both simulated and real datasets. Expected keys: `positions`, `seds`, `sources` (configurable `target_field`), and optional `masks` and `zernike_prior`.
- Introduced the `DataAdapter` and `DataAdapterFactory` architecture to provide a unified dataset access layer across training, evaluation, and inference pipelines.
- Added support for both complete datasets and pre-split train/test datasets through a common adapter interface.
- Implemented configurable splitting of complete datasets into training and test sets.
- Added logging throughout dataset resolution and adapter construction steps to improve traceability and debugging.

### Dataset normalization and containers
- Added `normalize_data_envelope` and dataset introspection utilities (`DatasetUtils`) for consistent handling of dictionaries, dataclasses, or objects with attributes.
- Introduced `DatasetContainer`, a dict-like structure for storing normalized datasets while supporting key and value transformations.
- Added default data configuration parameters in `wf_psf.data.constants`

### Training pipeline improvements
- The training pipeline now accepts complete datasets with normalized and canonicalized parameters.
- Added `TrainingDataAdapter` to simplify training-specific input and target preparation, including automatic stacking of source images and optional masks, positions, and SEDs for loss computation.
- Pipeline now supports configurable splitting of complete datasets into training and test sets.
- Metrics monitoring is now configurable via the `monitor` field in the training configuration.

### Data quality and filtering utilities
- Added `safe_batch` utilities for robust filtering of aligned datasets using sample-validity masks (e.g. handling NaNs or Infs in centroid arrays).
- Introduced `safe_batch_builder` to apply consistent filtering across multiple aligned arrays while preserving dataset integrity.
- Added support for filtering both NumPy arrays and sequence-based metadata (e.g. object identifiers) while maintaining alignment.
- Added logging utilities to facilitate auditing and debugging of filtered samples.


## Bug fixes

- Fixed an inference failure caused by strict matching against training `obs_pos` entries. Inference now correctly supports continuous focal-plane positions by using inference-time positions directly in the physical PSF model (replacing the previous discrete lookup/interpolation approach).

## Internal changes

### Dataset schema system (major architectural change)
- Introduced a schema-driven dataset conversion system enabling runtime selection of dataset contracts across workflows.
- Added support for runtime schema modes:
  - `TRAIN`
  - `EVALUATION`
  - `INFERENCE`
- Added dataset schema registry responsible for validating dataset structure based on the active mode.
- Introduced `ConversionContext` and `SEDContext` abstractions to decouple dataset schema validation from field-level conversion logic.
- Refactored SED processing to use domain-specific handler contexts instead of fixed pipeline logic.
- Removed hardcoded required-field assumptions from dataset validation logic.
- Added `schema_mode` configuration parameter to control dataset contract selection in inference workflows.

### Dataset constants and canonical structure
- Introduced centralized dataset constants in `constants.py`:
  - `CANONICAL_DATASET_KEYS = ("sources", "positions", "seds")`
  - `OPTIONAL_KEYS = ("masks", "zernike_prior")`
  - `DATASET_INDEX_KEY = "positions"`
  - `DEFAULT_TRAIN_FRACTION = 0.8`
  - `DEFAULT_SEED = 42`
  - `SED_DOMAIN = "seds"`
- Standardized dataset field naming and reduced ambiguity across training, evaluation, and inference pipelines.

### Data pipeline and adapter refactor (integration updates)
- Refactored dataset conversion pipeline to integrate schema-driven validation prior to conversion logic.
- Updated DataAdapter to load and convert datasets according to the active schema mode.
- Dataset adapters now expose training, evaluation, and inference-ready representations based on validated schema contracts.
- Improved handling of missing or optional fields based on schema context rather than static validation rules.
- Improved typing for lazily initialized adapters and internal pipeline state.
- Improved logging throughout dataset conversion and inference flows for better traceability.

### Training / evaluation / inference compatibility
- Enabled reuse of inference pipeline components for evaluation workflows through schema-aware dataset contracts.
- Inference workflows now explicitly require only `positions` and `seds`, while evaluation/training workflows may include additional fields such as `sources` and `masks`.
- Validated schema-driven execution across training, evaluation, inference, and mock Euclid SHE pipeline integration, ensuring reproducibility across modes.

### Testing and validation updates
- Refactored TensorFlow conversion tests into schema-based contract tests.
- Updated unit and integration tests to validate mode-dependent dataset behavior.
- Improved coverage for schema validation, conversion contexts, and handler dispatch logic.


<a id='changelog-v3.1.0'></a>
# v3.1.0 — 2026-02-23

## New features

- Added PSF inference capabilities for generating broadband (polychromatic) PSFs from trained models given star positions and SEDs
- Added `run_type` attribute to `DataHandler` supporting training, simulation, metrics, and inference modes
- Implemented `ZernikeInputsFactory` class for building `ZernikeInputs` instances based on run type
- Added `psf_model_loader.py` module for centralized model weights loading
- Added configurable optimizer selection system via new `optimizer.py` module with `get_optimizer` function
- Added support for hyperparameter overrides (learning rate, beta1/beta2, epsilon, amsgrad) via YAML or programmatic configuration
- `RectifiedAdam` optimizer now dynamically imports TensorFlow Addons only when explicitly specified in configuration

## Bug fixes

- Fix logger formatting for relative RMSE metrics in `metrics.py` (values were not being displayed)

## Internal changes

### Code quality and development workflow
- Added pre-commit hooks for code quality, formatting, and changelog enforcement.
- Fixed Sphinx autosummary import errors by removing core dependencies (tensorflow) from `autodoc_mock_imports` in `conf.py`.
- Updated `pyproject.toml` to include all `wf_psf` packages under `src/` when executing CI/CD execution.
- Added comprehensive unit tests in `test_optimizer.py` and `test_interpolation.py`

### PSF model and inference architecture
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

- Updated README and added THIRD_PARTY_LICENSE directory with TensorFlow Addons license
- Training now runs on TensorFlow 2.11 without requiring TensorFlow Addons installation
- Removed TensorFlow Addons as a required dependency; RectifiedAdam optimizer now requires explicit TFA installation if needed
- Remove deprecated/optional import tensorflow-addons statement from `tf_layers.py`


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


