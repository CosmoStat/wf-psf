# Configuration 

WaveDiff uses a set of YAML and INI configuration files to control each pipeline task.
This section provides a high-level overview of the configuration system, followed by detailed explanations of each file.

## Overview of Pipeline Tasks

WaveDiff consists of four main pipeline tasks:

| Pipeline | Purpose |
|---------|---------|
| `training` | Trains a PSF model using the provided dataset and hyperparameters. |
| `metrics` |Evaluates model performance using multiple metrics, optionally comparing against a ground-truth model. |
| `plotting` | Generates figures summarizing the results from the metrics pipeline. |
| `sims` | Simulates stellar PSFs used as training/test datasets.<br>*(Currently executed via a standalone script, not via `wavediff`.)* |

You configure these tasks by passing a configuration file to the `wavediff` command (e.g., `--config configs.yaml`).

## Configuration File Structure

WaveDiff expects the following configuration files under the `config/` directory:

You configure these tasks by passing a configuration file to the `wavediff` command (e.g., `--config configs.yaml`).

## Configuration File Structure

WaveDiff expects the following configuration files under the `config/` directory:

```arduino
config
├── configs.yaml
├── data_config.yaml
├── logging.conf
├── metrics_config.yaml
├── plotting_config.yaml
└── training_config.yaml
```

- All `.yaml` files use standard **YAML** syntax and are loaded as nested dictionaries of key–value pairs.
- `logging.conf` uses standard **INI** syntax and configures logging behavior.
- Users may modify values but should **not rename keys or section names**, as the software depends on them.

Each of the configuration files is described in detail below.

(data_config)=
## `data_config.yaml` — Data Configuration

### 1. Purpose
Specifies where WaveDiff loads (or later versions may generate) the training and test datasets.  
All training, evaluation, and metrics pipelines depend on this file for consistent dataset paths.

### 2. Key Fields
- `data.training.data_dir` _(required)_ — directory containing training data
- `data.training.file` _(required)_ — filename of the training dataset
- `data.test.*` — same structure as `training`, for the test dataset
- **Simulation-related fields** — reserved for future releases

**Notes**
- The simulation options are placeholders; WaveDiff v3.x does **not yet** auto-generate datasets.
- The default dataset bundled with WaveDiff can be used by simply pointing to its directory.

**Example (minimal)**
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
## `training_config.yaml` — Training Pipeline Configuration

### 1. Purpose

Controls the training pipeline, including model selection, hyperparameters, optional metrics evaluation, and data loading behavior.

---

### 2. General Notes

- Every field has an inline comment in the YAML file.
- **All required parameters must be specified.** Missing values will prevent the model from being instantiated, as there is currently no default configuration provided.
- **Optional fields:**
  - `metrics_config` (run metrics after training)
  - `param_hparams`, `nonparam_hparams` 
  - `multi_cycle_params.save_all_cycles`
- Some parameters are specific to physical or polychromatic PSF models.
- Example training configuration file is provided in the top-level root directory of the repository (`training_config.yaml`). Users can copy and adapt this template for their own runs.
- If any descriptions are unclear, or unexpected behaviour occurs, please open a [GitHub issue](https://github.com/CosmoStat/wf-psf/issues/new).

**Note:** The values in the examples shown below correspond to a typical WaveDiff training run. Users should adapt parameters such as `model_name`, telescope dimensions, pixel/field coordinates, and SED settings to match their own instrument or dataset. All required fields must still be specified.

### 3. Top-Level Training Parameters

```yaml
training:
  # ID name for this run (used in output files)
  id_name: run__001

  # Path to Data Configuration file (required)
  data_config: data_config.yaml 

  # Load dataset on initialization (True) or manually later (False)
  load_data_on_init: True

  # Optional: metrics configuration to run after training
  metrics_config:
```

### 4. Model Parameters (`model_params`)

Controls PSF model type, geometry, oversampling, and preprocessing:

```yaml
model_params:
  # Model type. Options: 'mccd', 'graph', 'poly', 'param', 'physical_poly'
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

  # Flags for physical corrections
  use_prior: False
  correct_centroids: True
  sigma_centroid_window: 2.5       # Standard deviation of window for centroid computation
  reference_shifts: [-1/3, -1/3]   # Euclid-like default shifts
  obscuration_rotation_angle: 0     # Obscuration mask rotation (degrees, multiple of 90)
  add_ccd_misalignments: False
  ccd_misalignments_input_path: /path/to/tiles.npy

  # Sample weighting
  use_sample_weights: True
  sample_weights_sigmoid:
    apply_sigmoid: False
    sigmoid_max_val: 5.0
    sigmoid_power_k: 1.0

  # Interpolation settings for physical-poly model
  interpolation_type: None
  interpolation_args: None

  # Spectral energy distribution (SED) parameters
  sed_interp_pts_per_bin: 0
  sed_extrapolate: True
  sed_interp_kind: linear
  sed_sigma: 0

  # Field and pixel coordinates
  x_lims: [0.0, 1.0e3]
  y_lims: [0.0, 1.0e3]
  pix_sampling: 12       # in [um]

  # Telescope parameters
  tel_diameter: 1.2      # [m]
  tel_focal_length: 24.5 # [m]
  euclid_obsc: True.     # Use Euclid-specific obscuration mask. Set to False for other instruments or custom masks.
  LP_filter_length: 3    # Low-pass filter for obscurations
```

### 5. Parametric Model Hyperparameters (`param_hparams`)

```yaml
param_hparams:
  random_seed: 3877572
  l2_param: 0.0             # L2 loss for OPD/WFE
  n_zernikes: 15
  d_max: 2                  # Max polynomial degree
  save_optim_history_param: true
```

### 6. Non-Parametric Model Hyperparameters (`nonparam_hparams`)

```yaml
nonparam_hparams:
  d_max_nonparam: 5
  num_graph_features: 0
  l1_rate: 1.0e-8
  project_dd_features: False
  reset_dd_features: False
  save_optim_history_nonparam: true
```

### 7. Training Hyperparameters (`training_hparams`)

Controls batches, loss, and multi-cycle learning:

```yaml
training_hparams:
  batch_size: 32           # Number of samples per batch
  loss: 'mask_mse'         # Options: 'mask_mse', 'mse'

  multi_cycle_params:
    total_cycles: 2
    cycle_def: complete        # Options: 'parametric', 'non-parametric', 'complete', etc.
    save_all_cycles: False
    saved_cycle: cycle2

    learning_rate_params: [1.0e-2, 1.0e-2]
    learning_rate_non_params: [1.0e-1, 1.0e-1]
    n_epochs_params: [20, 20]
    n_epochs_non_params: [100, 120]
```

Note
(metrics_config)=
## `metrics_config.yaml`  — Metrics Configuration
 
### 1. Purpose
Defines how a trained PSF model is evaluated. This configuration specifies which metrics to compute, which model weights to use, and how ground truth stars are obtained. It allows you to:
- Select a fully trained PSF model or a checkpoint for evaluation.
- Specify which training cycle’s weights to evaluate.
- Compute Polychromatic, Monochromatic, OPD, and Weak Lensing Shape metrics.
- Use precomputed ground truth stars from the dataset if available, or automatically generate them from the configured ground truth model.
- Optionally produce plots of the computed metrics via a plotting configuration file.

### 2. General Notes

- WaveDiff automatically searches the dataset used for training. If the dataset contains `stars`, `SR_stars`, or `super_res_stars` fields, these are used as the ground truth for metrics evaluation.
- If precomputed ground truth stars are not found in the dataset, WaveDiff regenerates them using the `ground_truth_model` parameters. **All required fields in `model_params` must be specified**; leaving them empty will prevent the metrics pipeline from running (see [Ground Truth Model Parameters](section-ground-truth-model) for details).
- The metrics evaluation can be run independently of training by specifying trained_model_path and `trained_model_config`.
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

**Notes:**
- `model_save_path`: Load final PSF model weights (`psf_model`) or checkpoint weights (`checkpoint`).
- `saved_training_cycle`: Choose which training cycle to evaluate (1, 2, …).
- `trained_model_path`: Absolute path to parent directory of previously trained model. Leave empty for training + metrics in serial.
- `trained_model_config`: Name of training config file in `trained_model_path/config/`.
- `eval_mono_metric`: If True, computes the monochromatic pixel reconstruction metric. Requires a `ground_truth_model` (see).
- `eval_opd_metric`: If True, computes the optical path difference (OPD) metric. Requires a `ground_truth_model`.
- `eval_train_shape_results_dict` / `eval_test_shape_results_dict`: Compute Weak Lensing Shape metrics on the training and/or test dataset.
- `plotting_config:` Optionally provide a `plotting_config.yaml` file to generate plots after metrics evaluation.
- **Behaviour notes:**
  - Metrics controlled by flags (`eval_mono_metric`, `eval_opd_metric`, `eval_train_shape_results_dict`, `eval_test_shape_results_dict`) are only computed if their respective flags are True.  
  - The Polychromatic Pixel Reconstruction metric is always computed, regardless of flags.
  - Future releases may allow optional `ground_truth_model` instantiation if the dataset already contains precomputed stars.

(section-ground-truth-model)=
### 5. Ground Truth Model Parameters 

Mirrors training parameters for consistency:

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
    use_sample_weights: True
    use_prior: False
    correct_centroids: False
    sigma_centroid_window: 2.5
    reference_shifts: [-1/3, -1/3]
    obscuration_rotation_angle: 0
    add_ccd_misalignments: False
    ccd_misalignments_input_path: 
    interpolation_type: None
    sed_interp_pts_per_bin: 0
    sed_extrapolate: True
    sed_interp_kind: linear
    sed_sigma: 0
    x_lims: [0.0, 1.0e+3]
    y_lims: [0.0, 1.0e+3]
    param_hparams:
      random_seed: 3877572
      l2_param: 0.
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
- **All fields in `model_params` are required.** Do not leave them empty. Even if the dataset contains precomputed ground truth stars, omitting `model_params` will prevent the metrics pipeline from running.
- Parameters mirror `training_config.yaml` for consistency.
- Future releases may allow optional instantiation of `ground_truth_model` when precomputed stars are available in the dataset.

(section-evaluation-hyperparameters)=
### 6. Evaluation Hyperparameters

```yaml
metrics_hparams:
  batch_size: 16
  opt_stars_rel_pix_rmse: False
  l2_param: 0.
  output_Q: 1
  output_dim: 64
```

**Parameter explanations:**
- `batch_size`: Number of samples processed per batch during evaluation.
- `opt_stars_rel_pix_rmse`: If `True`, saves RMSE for each individual star in addition to mean across FOV.
- `l2_param`: L2 loss weight for OPD.
- `output_Q`: Downsampling rate from high-resolution pixel modeling space.
- `output_dim`: Size of the PSF postage stamp for evaluation.


(section-plotting-config)=
## `plotting_config.yaml — Plot Configuration

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
- If any descriptions are unclear, or if you encounter unexpected behavior, please open a GitHub issue (<https://github.com/CosmoStat/wf-psf/issues/new>).

### 3. Basic Structure

An example `plotting_config.yaml` is shown below:

```yaml
plotting_params:
  # Path to the parent folder containing wf-psf output directories (e.g. $WORK/wf-outputs/)
  metrics_output_path: <PATH>

  # List of output directories (e.g. wf-outputs-xxxxxxxxxxx) whose metrics should be plotted
  metrics_dir:
    # - wf-outputs-xxxxxxxxxxx1
    # - wf-outputs-xxxxxxxxxxx2

  # List of the metric config filenames corresponding to each listed directory
  metrics_config:
    # - metrics_config_1.yaml
    # - metrics_config_2.yaml

  # If True, plots are shown interactively during execution
  plot_show: False
```

### 4. Example Directory Structure
Below is an example of three WaveDiff runs stored under a single parent directory:

**Example Directory Structure**
Below is an example of three WaveDiff runs stored under a single parent directory:

```arduino
wf-outputs/
├── wf-outputs-202305271829
│   ├── config
│   │   ├── data_config.yaml
│   │   └── metrics_config_200.yaml
│   ├── metrics
│   │   └── metrics-poly-coherent_euclid_200stars.npy
├── wf-outputs-202305271845
│   ├── config
│   │   ├── data_config.yaml
│   │   └── metrics_config_500.yaml
│   ├── metrics
│   │   └── metrics-poly-coherent_euclid_500stars.npy
├── wf-outputs-202305271918
│   ├── config
│   │   ├── data_config.yaml
│   │   └── metrics_config_1000.yaml
│   ├── metrics
│   │   └── metrics-poly-coherent_euclid_1000stars.npy
```

**Example: Plotting Multiple Runs**
To jointly plot metrics from the three runs shown above, the `plotting_config.yaml` would be:

```yaml
plotting_params:
  metrics_output_path: $WORK/wf-outputs/

  metrics_dir:
    - wf-outputs-202305271829
    - wf-outputs-202305271845
    - wf-outputs-202305271918

  metrics_config:
    - metrics_config_200.yaml
    - metrics_config_500.yaml
    - metrics_config_1000.yaml

  plot_show: False
```
This configuration instructs the plotting pipeline to load the metrics from each listed run and include them together in summary plots.

(master_config_file)=
## Master Configuration

### 1. Purpose
The `configs.yaml` file is the _master controller_ for WaveDiff.
It defines **which pipeline tasks** should be executed (training, metrics evaluation, plotting) and in which order.

Each task points to a dedicated YAML configuration file—allowing WaveDiff to run multiple jobs sequentially using a single entry point.

### 2. Example: Multiple Training Runs
To launch a sequence of training runs (models 1…n), list each task and its corresponding configuration file:

```yaml
---
  training_conf_1: training_config_1.yaml
  training_conf_2: training_config_2.yaml
  ...
  training_conf_n: training_config_n.yaml
```
Outputs will be organized as:

```
wf-outputs-20231119151932213823/
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

### 3 Example: Training + Metrics + Plotting
To evaluate metrics and generate plots for each trained model, include the corresponding configuration files:


```arduino
config/
├── configs.yaml
├── data_config.yaml
├── metrics_config.yaml
├── plotting_config.yaml
├── training_config_1.yaml
├── ...
└── training_config_n.yaml
```

Note: current WaveDiff versions generate one plot per metric per model. Creating combined plots requires a separate run [Plot Configuration](section-plotting-config). . A future update will support automatic combined plots.

### 4 General Notes
- `configs.yaml` may contain **any combination** of the three task types:
  - `training`
  - `metrics`
  - `plotting`
- Tasks always execute **in the order they appear** in the file.
- The current release runs all jobs on a single GPU, sequentially.
- Parallel multi-GPU execution is planned for a future version.
- For questions or feedback, please open a [GitHub issue](https://github.com/CosmoStat/wf-psf/issues/new).
