# Configuration

WaveDiff uses a set of YAML and INI configuration files to control each pipeline task. This section provides a high-level overview of the configuration system, followed by detailed explanations of each file.

## Overview of Workflows

WaveDiff consists of three CLI tasks, configured by passing a configuration file to the `wavediff` command (e.g., `wavediff -c configs.yaml -o output/`):

| Task | Purpose |
|------|---------|
| `training` | Trains a PSF model using the provided dataset and hyperparameters. |
| `metrics` | Evaluates model performance using multiple metrics, optionally comparing against a ground-truth model. |
| `plotting` | Generates figures summarising the results from the metrics pipeline. |

WaveDiff also provides two standalone Python APIs used outside the `wavediff` CLI:

| Component | Purpose |
|-----------|---------|
| `sims` | Provides classes and methods for simulating monochromatic, polychromatic, and spatially-varying PSFs<br> for generating custom datasets. |
| `inference` | Provides classes and methods for inferring PSFs as a function of position and SED from a trained PSF<br> model. |

## Configuration File Structure

WaveDiff expects configuration files under the `config/` directory. **Configuration filenames are flexible** — you can name them as you wish (e.g., `training_euclid_v2.yaml`, `my_metrics.yaml`) as long as you reference them correctly in `configs.yaml` or command-line arguments. The filenames shown below are conventional defaults used in documentation examples.

The files required depend on which task or component you are running:

```
config/
├── configs.yaml          # Master configuration (all CLI tasks)
├── data_config.yaml      # Dataset paths (training task only)
├── logging.conf          # Logging configuration (all CLI tasks)
├── training_config.yaml  # Training task
├── metrics_config.yaml   # Metrics task
├── plotting_config.yaml  # Plotting task
└── inference_config.yaml # Inference API
```

| Task / Component | Required | Optional |
|---------------|--------------|---------|
| `training` | `configs.yaml`, `data_config.yaml` ,<br> `logging.conf`, `training_config.yaml` | `metrics_config.yaml` <br> (_triggers post-training metrics_) |
| `metrics` | `configs.yaml`, `logging.conf`, <br> `metrics_config.yaml` | `plotting_config.yaml` <br>(_triggers post-metrics plotting_) |
| `plotting` | `configs.yaml`, `logging.conf`, <br> `plotting_config.yaml`| — |
| `inference` | `inference_config.yaml` | `data_config.yaml` |

**Notes:**

- **Configuration filenames are flexible.** The names shown above (e.g., `training_config.yaml`) are conventional defaults. You may use any filename as long as you reference it correctly in `configs.yaml` or via command-line arguments.
- **Keys and section names within configuration files must be preserved.** While you can rename files, the internal YAML structure (keys like `model_params`, `training`, etc.) must remain unchanged, as the software depends on them.
- The metrics and plotting tasks retrieve dataset paths from the trained model's configuration and do not require `data_config.yaml`.
- When `metrics_config.yaml` is specified as optional for the `training` task, metrics evaluation runs automatically after training completes.
- When `plotting_config.yaml` is specified as optional for the `metrics` task, plots are generated automatically after metrics evaluation completes.
- `logging.conf` uses standard INI syntax and configures logging behaviour for all CLI tasks.

Each of the configuration files is described in detail below.

(data_config)=
## Data Configuration

### 1. Purpose
Specifies the training and test datasets used by the training CLI task.


### 2. Key Fields

Both `data.training` and `data.test` share the same structure:

| Field | Required | Description |
|-----------|--------------|--------------|
| `data_dir` | Yes | Path to the directory containing the dataset. |
| `file` | Yes     | Filename of the dataset (`.npy`). |


### 3. Notes

- The default dataset bundled with WaveDiff can be used by pointing `data_dir` to its installation directory.
- The `metrics` and `plotting` tasks retrieve dataset paths automatically from the trained model's configuration file and do not require this file.
- This file is optional for the `inference` API; see [inference_config.yaml](inference_config) if you need to supply prior information for inference.

### 4. Example

```yaml
data:
  training:
    data_dir: path/to/training/data
    file: train.npy
  test:
    data_dir: path/to/test/data
    file: test.npy
```

(training_config)=
## Training Configuration

### 1. Purpose

Controls the training pipeline, including model selection, hyperparameters, optional post-training metrics evaluation, and data loading behaviour.


### 2. General Notes

- **All required parameters must be specified.** There is currently no default configuration — missing values will prevent the model from being instantiated.
- **Optional fields:**
  - `metrics_config` — trigger metrics evaluation after training completes
  - `multi_cycle_params.save_all_cycles`— defaults to `False`
- Some parameters are specific to the physical PSF model and may be ignored by simpler model types.
- An example training configuration file is provided in the repository root (`config/training_config.yaml`). Copy and adapt this template for your own runs.
- **Fraction notation**: Fields like `reference_shifts` accept fraction strings (e.g., "`-1/3`") which are automatically converted to floats. You can also use decimal values directly (e.g., `-0.333`).
- Every field in the YAML file includes an inline comment. If any descriptions remain unclear or unexpected behavior occurs, please open a [GitHub issue](https://github.com/CosmoStat/wf-psf/issues/new).

**Note on example values**: The parameter values shown below correspond to a typical Euclid-like WaveDiff training run. Adapt `model_name`, telescope dimensions, pixel/field coordinates, and SED settings to match your instrument and dataset.

### 3. Top-Level Training Parameters

`training`
```yaml
training:
  # ID name for this run (used in output filenames and logs)
  id_name: run_001

  # Path to data configuration file
  data_config: data_config.yaml

  # Load dataset on initialization (True) or manually later (False)
  load_data_on_init: True

  # Optional: path to metrics configuration to run after training
  metrics_config:
```

### 4. Model Parameters

Controls PSF model type, geometry, oversampling, and physical corrections.

`training.model_params`
```yaml
model_params:
  # Model type. Options: 'poly', 'physical_poly'
  model_name: physical_poly

  # Number of wavelength bins for polychromatic reconstruction
  n_bins_lda: 8

  # Downsampling rate to match telescope pixel sampling
  output_Q: 3

  # Oversampling rate of OPD/WFE PSF model
  oversampling_rate: 3

  # Pixel PSF postage stamp size
  output_dim: 32

  # OPD/Wavefront space dimensions
  pupil_diameter: 256

  # Physical correction switches
  use_prior: False
  correct_centroids: True
  add_ccd_misalignments: False

  # Centroid correction parameters
  sigma_centroid_window: 2.5       # Std dev of centroiding window
  reference_shifts: [-0.333, -0.333]   # Reference pixel shifts (Euclid default: -1/3, -1/3)

  # Obscuration geometry
  obscuration_rotation_angle: 0    # Rotation in degrees (multiples of 90); counterclockwise

  # CCD misalignments input file path
  ccd_misalignments_input_path: /path/to/ccd_misalignments_file.txt

  # Sample weighting based on noise standard deviation
  use_sample_weights: True

  # Sample weight sigmoid function parameters
  sample_weights_sigmoid:
    apply_sigmoid: False           # Enable sigmoid weighting transform
    sigmoid_max_val: 5.0           # Maximum sample weight value
    sigmoid_power_k: 1.0           # Sigmoid steepness (higher = steeper)

  # Interpolation settings for physical-poly model
  interpolation_type: None
  interpolation_args: None

  # Spectral energy distribution (SED) parameters
  sed_interp_pts_per_bin: 0
  sed_extrapolate: True
  sed_interp_kind: linear
  sed_sigma: 0

  # Field and pixel coordinates
  x_lims: [0.0, 1000.0]
  y_lims: [0.0, 1000.0]
  pix_sampling: 12       # Pixel size in microns

  # Telescope parameters
  tel_diameter: 1.2      # Aperture diameter in meters
  tel_focal_length: 24.5 # Focal length in meters
  euclid_obsc: True      # Use Euclid-specific obscuration mask (set False for other instruments)
  LP_filter_length: 3    # Low-pass filter kernel size for obscurations
```

### 5. Parametric Model Hyperparameters

`training.model_params.param_hparams`
```yaml
param_hparams:
  random_seed: 3877572
  l2_param: 0.0             # L2 regularization weight for OPD/WFE
  n_zernikes: 15            # Number of Zernike polynomials
  d_max: 2                  # Maximum polynomial degree
  save_optim_history_param: True
```

### 6. Non-Parametric Model Hyperparameters

`training.model_params.nonparam_hparams`
```yaml
nonparam_hparams:
  d_max_nonparam: 5
  num_graph_features: 0
  l1_rate: 1.0e-8
  project_dd_features: False
  reset_dd_features: False
  save_optim_history_nonparam: True
```

### 7. Training Hyperparameters

Controls batch size, loss function, optimizer selection, and multi-cycle learning.

`training.training_hparams`
```yaml
training_hparams:
  batch_size: 32           # Number of samples per training batch
  loss: 'mask_mse'         # Loss function. Options: 'mask_mse', 'mse'
  optimizer:
    name: 'rectified_adam' # Options: 'adam', 'rectified_adam'

  multi_cycle_params:
    total_cycles: 2
    cycle_def: complete        # Options: 'parametric', 'non-parametric', 'complete'
    save_all_cycles: False     # If True, saves checkpoints for all cycles; otherwise only saved_cycle
    saved_cycle: cycle2        # Which cycle checkpoint to retain

    learning_rate_params: [1.0e-2, 1.0e-2]      # Per-cycle learning rate for parametric model
    learning_rate_non_params: [1.0e-1, 1.0e-1]  # Per-cycle learning rate for non-parametric model
    n_epochs_params: [20, 20]                   # Per-cycle epochs for parametric model
    n_epochs_non_params: [100, 120]             # Per-cycle epochs for non-parametric model
```

**Optimizer Notes:**
- `rectified_adam` requires tensorflow-addons to be installed manually.
- If TensorFlow Addons is not installed and `rectified_adam` is requested, WaveDiff will raise a runtime error with installation instructions.
- Standard workflows (`training`, `metrics`, `plotting`) run without TensorFlow Addons.

(metrics_config)=
## Metrics Configuration

### 1. Purpose
Defines how a trained PSF model is evaluated. This configuration specifies which metrics to compute, which model weights to use, and how ground truth stars are obtained. It allows you to:

- Select a fully trained PSF model or a checkpoint for evaluation.
- Specify which training cycle's weights to evaluate.
- Compute Polychromatic, Monochromatic, OPD, and Weak Lensing Shape metrics.
- Use precomputed ground truth stars from the dataset if available, or automatically generate them from the configured ground truth model.
- Optionally produce plots of the computed metrics via a plotting configuration file.

### 2. General Notes

- WaveDiff automatically searches the dataset used for training. If the dataset contains `stars`, `SR_stars`, or `super_res_stars` fields, these are used as the ground truth for metrics evaluation.
- If precomputed ground truth stars are not found in the dataset, WaveDiff regenerates them using the `ground_truth_model` parameters. **All required fields in `model_params` must be specified**; leaving them empty will prevent the metrics pipeline from running (see [Ground Truth Model Parameters](section-ground-truth-model) for details).
- Metrics evaluation can be run independently of training by specifying both `trained_model_path` and `trained_model_config` to point to a previously trained model.
- Metrics defined in [Metrics Overview table](metrics-table) are selectively computed according to their boolean flags. The Polychromatic Pixel Reconstruction metric is always computed.
- The `plotting_config` parameter triggers plotting of the metrics results if a valid configuration file is provided. If left empty, metrics are computed without generating plots (see [Plotting Configuration](section-plotting-config)).
- Batch size and other evaluation hyperparameters can be set under `metrics_hparams` (see [Evaluation Hyperparameters](section-evaluation-hyperparameters))

(section-metrics-overview)=
### 3. Metrics Overview

(metrics-table)=
| Metric type                        | Description                                                                                                                                                                    | Relevant v3.0 Flag                                              |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| Polychromatic Pixel Reconstruction | Computes absolute and relative RMSE of pixel residuals between the trained polychromatic PSF model and test data at low- and super-pixel resolution.                           | `eval_train_shape_results_dict`, `eval_test_shape_results_dict` |
| Monochromatic Pixel Reconstruction | Computes absolute and relative RMSE of pixel residuals as a function of wavelength between a monochromatic PSF model and test data.                                            | `eval_mono_metric`                                              |
| Optical Path Differences (OPD)     | Computes absolute and relative RMSE of residuals between predicted OPD maps and ground truth OPD.                                                                              | `eval_opd_metric`                                               |
| Weak Lensing Shape Metrics         | Second-order moments-based metrics for PSF ellipticity and size at super-pixel resolution using [GalSim HSM](https://galsim-developers.github.io/GalSim/_build/html/hsm.html). | `eval_train_shape_results_dict`, `eval_test_shape_results_dict` |

**Notes:**
- Metrics requiring ground truth (Monochromatic, OPD) are valid only on simulated datasets.
- RMSE may be less reliable for noisy stars (e.g., real data). Alternative formulations are in development.
- Super-resolution is required for Weak Lensing Shape metrics on undersampled PSFs (e.g., Euclid observations).

### 4. Top-Level Configuration Parameters

`metrics`

```yaml
metrics:
  model_save_path: <enter psf_model or checkpoint>
  saved_training_cycle: 2
  trained_model_path: </path/to/parent/directory/of/trained/model>
  trained_model_config: <enter name of trained model config file>
  eval_mono_metric: True
  eval_opd_metric: True
  eval_train_shape_results_dict: False
  eval_test_shape_results_dict: False
  plotting_config: <enter name of plotting_config.yaml or leave empty>
```

**Parameter descriptions:**

- `model_save_path`: Specifies which weights to load. Options: `psf_model` (final trained weights) or `checkpoint` (intermediate checkpoint).
- `saved_training_cycle`: Which training cycle to evaluate (e.g., `1`, `2`, ...).
- `trained_model_path`: Absolute path to the parent directory of a previously trained model. Leave empty if running `training` + `metrics` sequentially in the same workflow.
- `trained_model_config`: Filename of the training configuration (located in `<trained_model_path>/config/`).
- `eval_mono_metric`: If `True`, computes the monochromatic pixel reconstruction metric. Requires `ground_truth_model` to be configured (see [Ground Truth Model Parameters](section-ground-truth-model)).
- `eval_opd_metric`: If `True`, computes the optical path difference (OPD) metric. Requires `ground_truth_model` to be configured.
- `eval_train_shape_results_dict`: If `True`, computes Weak Lensing Shape metrics on the training dataset.
- `eval_test_shape_results_dict`: If `True`, computes Weak Lensing Shape metrics on the test dataset.
- `plotting_config`: Optional filename of a plotting configuration (e.g., `plotting_config.yaml`) to automatically generate plots after metrics evaluation. Leave empty to skip plotting.

**Notes:**

- The Polychromatic Pixel Reconstruction metric is **always computed** regardless of flag settings.
- All other metrics (`eval_mono_metric`, `eval_opd_metric`, `eval_train_shape_results_dict`, `eval_test_shape_results_dict`) are only computed when their respective flags are set to `True`.

(section-ground-truth-model)=
### 5. Ground Truth Model Parameters

Specifies parameters for generating ground truth PSFs when precomputed stars are not available in the dataset. This configuration includes a subset of the training parameters — only those needed to simulate ground truth PSFs for comparison.

`metrics.ground_truth_model`
```yaml
ground_truth_model:
  model_params:
    model_name: <ground_truth_poly or ground_truth_physical_poly>
    n_bins_lda: 20
    output_Q: 3
    oversampling_rate: 3
    output_dim: 32
    pupil_diameter: 256
    LP_filter_length: 2
    use_prior: False
    correct_centroids: False
    sigma_centroid_window: 2.5
    reference_shifts: [-0.333, -0.333]
    obscuration_rotation_angle: 0
    add_ccd_misalignments: False
    ccd_misalignments_input_path:
    interpolation_type: None
    sed_interp_pts_per_bin: 0
    sed_extrapolate: True
    sed_interp_kind: linear
    sed_sigma: 0
    x_lims: [0.0, 1000.0]
    y_lims: [0.0, 1000.0]
    param_hparams:
      random_seed: 3877572
      l2_param: 0.0
      n_zernikes: 45
      d_max: 2
      save_optim_history_param: True
    nonparam_hparams:
      d_max_nonparam: 5
      num_graph_features: 10
      l1_rate: 1.0e-8
      project_dd_features: False
      reset_dd_features: False
      save_optim_history_nonparam: True
```
**Notes:**
- **All fields shown above are required.**  Do not leave them empty. Even if the dataset contains precomputed ground truth stars, omitting these fields will prevent the metrics pipeline from running.
- This configuration uses a subset of the training parameters — telescope geometry (`tel_diameter`, `tel_focal_length`, `pix_sampling`) and sample weighting (`use_sample_weights`, `sample_weights_sigmoid`) are not required for metrics evaluation, as these are only needed during model training.
- Ground truth model parameters should match the simulation settings used to generate your dataset for meaningful comparison.
- Future releases may allow optional instantiation of `ground_truth_model` when precomputed stars are available in the dataset.

(section-evaluation-hyperparameters)=
### 6. Evaluation Hyperparameters

`metrics.metrics_hparams`
```yaml
metrics_hparams:
  batch_size: 16
  opt_stars_rel_pix_rmse: False
  l2_param: 0.0
  output_Q: 1
  output_dim: 64

  # Optimizer configuration for metrics evaluation
  optimizer:
    name: 'adam'           # Fixed to Adam for metrics evaluation
    learning_rate: 1.0e-2
    beta_1: 0.9
    beta_2: 0.999
    epsilon: 1.0e-7
    amsgrad: False
```

**Parameter descriptions:**
- `batch_size`: Number of samples processed per batch during evaluation.
- `opt_stars_rel_pix_rmse`: (_optional individual star RMSE_) If `True`, saves the relative pixel RMSE for each individual star in the test dataset in addition to the mean across the field of view.
- `l2_param`: L2 loss weight for the OPD metric.
- `output_Q`: Downsampling rate from the high-resolution pixel modeling space to the resolution at which PSF shapes are measured. Recommended value: `1`.
- `output_dim`: Pixel dimension of the PSF postage stamp. Should be large enough to contain most of the PSF signal. The required size depends on the `output_Q` value used. Recommended value: `64` or higher.
- `optimizer`: Optimizer configuration for metrics evaluation. Unlike training, metrics evaluation always uses the standard Adam optimizer.
  - `name`: Fixed to `'adam'` (no other optimizers supported for metrics).
  - `learning_rate`: Learning rate for optimizer.
  - `beta_1, beta_2`: Exponential decay rates for moment estimates.
  - `epsilon`: Small constant for numerical stability.
  - `amsgrad`: If `True`, uses AMSGrad variant of Adam.

(section-plotting-config)=
## Plotting Configuration

The `plotting_config.yaml` file defines how WaveDiff generates diagnostic plots from the metrics produced during model evaluation. While the plotting routines are mostly pre-configured internally, this file allows you to combine and compare metrics from multiple training runs, or simply visualize the results of the most recent `metrics` pipeline execution.

### 1. Purpose

This configuration controls how metric outputs from one or more WaveDiff runs are located and aggregated for plotting. It enables users to:

- Specify where metrics outputs are stored,
- Select which runs to include in joint plots,
- Associate each run with its corresponding `metrics_config.yaml`,
- Optionally display plots interactively during execution.

### 2. General Notes

- All plotting styles and figure settings are hard-coded and do not require user modification.
- If the plotting task is executed immediately after a metrics evaluation run, all fields except `plot_show` may be left empty—the pipeline will automatically locate the outputs of the active run.
- When plotting results from multiple runs, the entries in `metrics_dir` and `metrics_config` must appear **row-aligned**, with each position referring to the same run.
- If any descriptions are unclear, or if you encounter unexpected behavior, please open a [GitHub issue](<https://github.com/CosmoStat/wf-psf/issues/new>).

### 3. Configuration Structure

`plotting_params`

```yaml
plotting_params:
  # Path to the parent folder containing WaveDiff output directories
  metrics_output_path: /path/to/wf-outputs/

  # List of output directories whose metrics should be plotted
  # Leave commented/empty if plotting immediately after a metrics run
  metrics_dir:
  #   - wf-outputs-xxxxxxxxxxxxxxxxxxx1
  #   - wf-outputs-xxxxxxxxxxxxxxxxxxx2

  # List of metrics config filenames corresponding to each directory
  # Leave commented/empty if plotting immediately after a metrics run
  metrics_config:
  #   - metrics_config_1.yaml
  #   - metrics_config_2.yaml

  # If True, plots are shown interactively during execution
  plot_show: False
```

**Parameter descriptions:**

- `metrics_output_path`: Absolute path to the parent directory containing WaveDiff output folders (e.g., `/home/user/wf-outputs/`). Can be left as `<PATH>` placeholder if plotting immediately after a metrics run.
- `metrics_dir`: List of output directory names (e.g., `wf-outputs-xxxxxxxxxxxxxxxxxxx1`) whose metrics should be included in plots. **Leave empty or commented out if plotting immediately after a metrics run** — WaveDiff will automatically locate the current run's outputs.
- `metrics_config`: List of `metrics_config.yaml` filenames corresponding to each directory in `metrics_dir`. Each entry should match the config file in `<metrics_dir>/config/`. Must be row-aligned with `metrics_dir`. **Leave empty or commented out if plotting immediately after a metrics run.**
- `plot_show`: If `True`, displays plots interactively during execution. If `False`, plots are saved to disk without display.

### 4. Example Directory Structure
Below is an example of three WaveDiff runs stored under a single parent directory:

```
wf-outputs/
├── wf-outputs-xxxxxxxxxxxxxxxxxxx1
│   ├── config
│   │   ├── data_config.yaml
│   │   └── metrics_config_200.yaml
│   ├── metrics
│   │   └── metrics-poly-coherent_euclid_200stars.npy
├── wf-outputs-xxxxxxxxxxxxxxxxxxx2
│   ├── config
│   │   ├── data_config.yaml
│   │   └── metrics_config_500.yaml
│   ├── metrics
│   │   └── metrics-poly-coherent_euclid_500stars.npy
├── wf-outputs-xxxxxxxxxxxxxxxxxxx3
│   ├── config
│   │   ├── data_config.yaml
│   │   └── metrics_config_1000.yaml
│   ├── metrics
│   │   └── metrics-poly-coherent_euclid_1000stars.npy
```

### 5. Plotting Multiple Runs
To jointly plot metrics from the three runs shown above, the `plotting_config.yaml` would be:

```yaml
plotting_params:
  metrics_output_path: /path/to/wf-outputs/

  metrics_dir:
    - wf-outputs-xxxxxxxxxxxxxxxxxxx1
    - wf-outputs-xxxxxxxxxxxxxxxxxxx2
    - wf-outputs-xxxxxxxxxxxxxxxxxxx3

  metrics_config:
    - metrics_config_200.yaml
    - metrics_config_500.yaml
    - metrics_config_1000.yaml

  plot_show: False
```
This configuration instructs the plotting pipeline to load the metrics from each listed run and include them together in summary plots.

(inference_config)=
## Inference Configuration

### 1. Purpose
Configures the WaveDiff inference API for generating polychromatic PSFs from a trained model, given a set of source positions and SEDs. Unlike the CLI tasks, the inference API is designed for external use: users are expected to load their own positions and SEDs programmatically and interact with the API directly.

### 2. Key Fields

`inference`
| Field | Required | Description |
|---------------|--------------|---------|
| `batch_size` | Yes | Number of PSFs to process per batch. |
| `cycle` | Yes | Training cycle checkpoint to load (e.g. `2`). <br> WaveDiff training typically runs two cycles.|

`inference.configs`

| Field | Required | Description |
|---------------|--------------|---------|
| `trained_model_path` | Yes | Absolute path to the directory containing the trained <br> model. |
| `model_subdir` | Yes | Subdirectory name within `trained_model_path` <br> containing the model weights (e.g. model). |
|`trained_model_config_path` | Yes | Path to the training configuration file used to train the <br> model, relative to `trained_model_path`. |
| `data_config_path` | No. | Path to a data configuration file supplying prior <br> information (e.g. a Phase Diversity calibration prior)<br> relevant to the inference context. This may differ <br> from the data configuration used during training. Leave <br> blank if no external prior is required.

`inference.model_params`

These fields are optional. Any field left blank inherits its value from the trained model configuration file. Populated fields override the corresponding `model_params` values from the training config.

| Field | Required | Description |
|---------------|--------------|---------|
| `n_bins_lda` | inherited | Number of wavelength bins used to reconstruct polychromatic PSFs.|
| `output_Q` | inherited | Downsampling rate to match the oversampled model to the telescope's <br> native sampling. |
| `output_dim` | inherited | Pixel dimension of the output PSF postage stamp. |
| `correct_centroids` | False | If `True`, applies centroid error correction within the PSF model during inference.. |
| `add_ccd_misalignments` | False | If `True`, incorporates CCD misalignment corrections into <br> the PSF model during inference. Required data is retrieved <br> from the trained model configuration file. |

### 3. Example

```yaml
inference:
  batch_size: 16
  cycle: 2
  configs:
    trained_model_path: /path/to/trained/model/
    model_subdir: model
    trained_model_config_path: config/training_config.yaml
    data_config_path:
  model_params:
    n_bins_lda: 8
    output_Q: 1
    output_dim: 64
  correct_centroids: False
  add_ccd_misalignments: True
```

### 4. Notes

- `trained_model_config_path` is relative to `trained_model_path`, not to the working directory.
- All `model_params` fields are optional; omitting them inherits values from the training configuration. - Only populate fields where you explicitly want to override the trained model's parameters.
- `data_config_path` is intended for cases where inference is performed in a different data context than training, for example using an updated or alternative prior. Leave blank if the trained model's own configuration is sufficient.
- `correct_centroids` and `add_ccd_misalignments` are independent model behaviour flags that modify PSF model computation during inference. Both retrieve their required data from the trained model configuration file — no additional configuration is required to enable them.


(master_config_file)=
## Master Configuration

### 1. Purpose
The `configs.yaml` file is the master controller for WaveDiff CLI tasks. It defines **which pipeline tasks** should be executed (`training`, `metrics`, `plotting`) and in which order. Each task entry points to a dedicated YAML configuration file, allowing WaveDiff to run multiple jobs sequentially from a single entry point.

Each task points to a dedicated YAML configuration file—allowing WaveDiff to run multiple jobs sequentially using a single entry point.

### 2. General Notes

`configs.yaml` may contain any combination of the three CLI task types:

- `training`
- `metrics`
- `plotting`

-Tasks always execute **in the order they appear** in the file.
- The current release runs all jobs sequentially on a single GPU.
- Parallel multi-GPU execution is planned for a future version.
- For questions or feedback, please open a [GitHub issue](<https://github.com/CosmoStat/wf-psf/issues/new>).

### 3. Example: Multiple Training Runs

To launch a sequence of training runs (models 1…n), list each task and its corresponding configuration file:

`configs.yaml`
```yaml
---
  training_conf_1: training_config_1.yaml
  training_conf_2: training_config_2.yaml
  ...
  training_conf_n: training_config_n.yaml
```

WaveDiff will execute each training task sequentially and organize outputs as:

```
wf-outputs-xxxxxxxxxxxxxxxxxxx1/
├── checkpoint/
│   ├── checkpoint_callback_poly-coherent_euclid_200stars_1_cycle1.*
│   ├── ...
│   ├── checkpoint_callback_poly-coherent_euclid_200stars_n_cycle1.*
├── config/
│   ├── configs.yaml
│   ├── data_config.yaml
│   ├── training_config_1.yaml
│   ├── ...
│   └── training_config_n.yaml
├── optim-hist/
├── plots/
└── psf_model/
    ├── psf_model_poly-coherent_euclid_200stars_1_cycle1.*
    ├── ...
    └── psf_model_poly-coherent_euclid_200stars_n_cycle1.*
```

### 4. Example: Training + Metrics + Plotting
To evaluate metrics and generate plots after each training run, include metrics and plotting tasks in
`configs.yaml`:

```
training_conf_1: training_config_1.yaml
metrics_conf_1: metrics_config_1.yaml
plotting_conf_1: plotting_config_1.yaml
training_conf_2: training_config_2.yaml
metrics_conf_2: metrics_config_2.yaml
plotting_conf_2: plotting_config_2.yaml
...
```

Required configuration files:

```
config/
├── configs.yaml
├── data_config.yaml
├── training_config_1.yaml
├── metrics_config_1.yaml
├── plotting_config_1.yaml
├── training_config_2.yaml
├── metrics_config_2.yaml
├── plotting_config_2.yaml
└── ...
```

**Note:** Current WaveDiff versions generate one plot per metric per model. Creating combined comparison plots across multiple runs requires a separate plotting-only run (see [Plot Configuration](section-plotting-config)). Automatic combined plots may be supported in a future release.

