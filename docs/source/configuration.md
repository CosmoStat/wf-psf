# Configuration 

The WaveDiff pipeline features four main packages for executing four pipeline tasks: 

| Pipeline Task | Description |
| --- | ----------- |
|`training` | This pipeline task is used to train a PSF model. |
|`metrics` | This pipeline task performs metrics evaluations of the trained PSF models.|
|`plotting`| This pipeline task is a utility feature for generating plots for the various metrics.|
|`simPSF`| This pipeline task is used to simulate stellar PSFs to use as training and test data for the training procedure.<br> (Currently, it runs as a separate code and is not triggered by the command `wavediff`).|

Configuring WaveDiff to execute one or more of the pipeline tasks (e.g. `training`, `metrics`, or `plotting`) is done by providing a configuration file with specific parameter settings as input into the software.

The directory tree below shows the various configuration files which contain their own unique settings for executing a particular task in WaveDiff:

```
config
├── configs.yaml
├── data_config.yaml
├── logging.conf
├── metrics_config.yaml
├── plotting_config.yaml
└── training_config.yaml
```

Most of the input configuration files (ending in .yaml) are constructed using `YAML` (Yet Another Markup Language).   The contents of the yaml file are read in as a nested dictionary with key:value pairs.  The `logging.conf` file contains configuration settings for storing a log of the run, and in this case we use the `ini` file syntax, which has a section-based structure.  Each section contains one or more key=value pairs, called properties. As a user, you should not modify the names of the keys or sections.  You can modify the value entries.

Next, we shall describe each configuration file.

(data_config)=
## Data Configuration

The file [data_config.yaml](https://github.com/CosmoStat/wf-psf/blob/main/config/data_config.yaml) stores the metadata for generating training and test data sets or retrieving existing ones.  A set of training and test data is provided in the `data/coherent_euclid_dataset` directory.  `Wavediff` will automatically retrieve the dataset within its directory tree.  In the field `data_dir`, the user should specify the sub-path to the data directory as in the example below.  The actual name of the dataset file is provided as an entry to the field `file`. However, new training and test data sets can be produced with the parameters in the file, which *should be* provided to the `simPSF` code although not at present (implementation upgrade pending).  

```
# Training and test data sets for training and/or metrics evaluation
data:
  training:
    # Specify directory path to data; Default setting is /path/to/repo/data
    data_dir: data/coherent_euclid_dataset/
    file: train_Euclid_res_200_TrainStars_id_001.npy
    # if training data set file does not exist, generate a simulated one by setting values below
    .
    . <params to generate training data set>
    .
  test:
    data_dir: data/coherent_euclid_dataset/
    file: test_Euclid_res_id_001.npy
    # If test data set file not provided produce a new one
    .
    . <params to generate test data set>
    .
```
(training_config)=
## Training Configuration

The file [training_config.yaml](https://github.com/CosmoStat/wf-psf/blob/main/config/training_config.yaml) is used to configure the settings for the training pipeline task. The first line contains the parent key `training`. All of the following child keys are treated as values of the `training` key. Above each child key a description is provided. Below is an abridged example of this:

```
training:
  # Run ID name
  id_name: -coherent_euclid_200stars

  # Name of Data Config file
  data_config: data_config.yaml 
  
  # Metrics Config file - Enter file to run metrics evaluation else if empty run train only
  metrics_config: /path/to/metrics_config.yaml

  # PSF model parameters
  model_params:
    # Model type.  Options are: 'mccd', 'graph', 'poly, 'param', 'poly_physical'."
     model_name:
     .
     .
     .
  # Training hyperparameters
  training_hparams:
     .
     .
     .

```
The key `id_name` is used to apply an identifier to the run, which the user may define to their preference or leave the field blank. The next parameter `data_config` stores the name of the [data_configuration](data_config) file, which will be parsed by WaveDiff to retrieve the training and test data sets for `training`. The `metrics_config` key is used to trigger the `metrics` pipeline task after the completion of training.  The options are to enter the filename of the [metrics configuration file](metrics_config), which contains the metrics configuration parameters, or to leave the field empty. A non-empty `metrics_config` field will prompt WaveDiff to launch the `metrics` evaluation of the trained PSF model. If the field is left empty, WaveDiff will run only the `training` pipeline task. 
The key `model_params` stores the model parameters for the type of PSF model to apply during training.  The identifier for the PSF model type to train is stored in `model_name`.  While the model options are listed just above in the key description, for now only the `poly` model is implemented. 

Training hyperparameters, defined by the parent key: `training_hparams`, include the learning rates, the number of epochs and the number of multi-cycles, etc.  These parameters can modified by the user. Setting the key [save_all_cycles](https://github.com/CosmoStat/wf-psf/blob/425cee776808eb230674103bdb317991dc0922b6/config/training_config.yaml#L105) to `True` will save the weights and models of all training cycles.  Otherwise, if it is set to `False`, only the last training cycle is saved.


(metrics_config)=
## Metrics Configuration

The [metrics_config.yaml](https://github.com/CosmoStat/wf-psf/blob/main/config/metrics_config.yaml) file stores the configuration parameters for the WaveDiff pipeline to perform computations of the four metrics listed in the table on a trained PSF model, as applied in {cite:t}`Liaudat:23`.

| Metric type | Description |
| --- | ----------- |
| Polychromatic Pixel Reconstruction | Pixel-based metric that computes the absolute and relative Root Mean <br> Square Error (RMSE) of the pixel reconstruction residuals between the <br> trained polychromatic PSF model and a test data set at low-pixel and <br> super-pixel resolutions.|
| Monochromatic Pixel Reconstruction | Pixel-based metric that computes the absolute and relative Root Mean <br> Square Error (RMSE) of the pixel reconstruction residuals as a function of <br> wavelength between a monochromatic PSF model and the test data set.|
| Optical Path Differences Reconstruction (OPD) | Metric that evaluates the absolute and relative RMSE of the residuals <br> between the predicted OPD (Wavefront Error) maps and the ground <br> truth OPD test data set.|
| Weak Lensing Shape Metrics | Second-order moments-based metrics that compute the shape (ellipticity) <br> and size residuals for a PSF at super-pixel resolution (i.e. well-sampled) <br> using [GalSim's HSM module](https://galsim-developers.github.io/GalSim/_build/html/hsm.html).  |

The test data set referenced in the table for each metric can be composed of noiseless or noisy stars. In the case of noisy stars (such as real data), we caution that the RMSE is not an adequate metric to use to assess the performance of the PSF model.  Alternative formulations are a work-in-progress.  Similarly, both the Monochromatic Pixel Reconstruction and OPD Reconstruction metrics can only be applied to simulated data for which a ground truth model is known. Finally, to apply the Weak Lensing Shape metrics for undersampled PSF observations typical of space experiments like *Euclid* requires super-resolving the PSF model.

Below is an example of some of the parameters contained in the metrics configuration file.  

```
metrics:
   # Specify the type of model weights to load by entering "psf_model" to load weights of final psf model or "checkpoint" to load weights from a checkpoint callback.
  model_save_path: <enter psf_model or checkpoint>

  # Choose the training cycle for which to evaluate the psf_model. Can be: 1, 2, ...
  saved_training_cycle: 2
  
  # Metrics-only run: Specify model_params for a pre-trained model else leave blank if running training + metrics
  # Specify path to Parent Directory of Trained Model 
  trained_model_path: </path/to/parent/directory/of/trained/model>

  # Name of the Trained Model Config file stored in config sub-directory in the trained_model_path parent directory
  trained_model_config: <enter name of trained model config file>
  
  #Evaluate the monchromatic RMSE metric.
  eval_mono_metric_rmse: True
  
  #Evaluate the OPD RMSE metric.
  eval_opd_metric_rmse: True
  
  #Evaluate the shape RMSE metrics at super-resolution (sr) for the training dataset.
  eval_train_shape_sr_metric_rmse: True

  # Name of Plotting Config file - Enter name of yaml file to run plot metrics else if empty run metrics evaluation only
  plotting_config: <enter name of plotting_config .yaml file or leave empty>

  ground_truth_model:
    model_params:
      .
      .
      .
  metrics_hparams:
      .
      .
      .

```
The metrics key `model_save_path` enables a choice of running the metrics evaluation for a fully trained PSF model or the weights of a given checkpoint cycle. 
The parameter `saved_training_cycle` specifies the cycle at which to run metrics evaluation.

As stated in the previous section, the `metrics` evaluation pipeline can be executed subsequently after the completion of the `training` routine to evaluate the trained PSF model.  It can also be launched independently to compute the metrics of a previously trained model.  This is done by setting the value of the parameter `trained_model_path` to the absolute path of the parent directory containing the output files of the model.  This is the directory with the naming convention: `wf-outputs-timestamp` (see this {ref}`example of the run output directory<wf-outputs>`).  The user must then provide as an entry for the key: `trained_model_config` the subdirectory path to the training configuration file, e.g. `config/train_config.yaml`. Below we show an example of this for the case where a user wants to run metrics evaluation of a pretrained full PSF model saved in the directory `wf-outputs-202310161536`. 

```
WaveDiff Pre-trained Model
--------------------------

wf-outputs-202310161536
├── checkpoint
│   ├── checkpoint
│   ├── checkpoint_callback_poly-coherent_euclid_200stars_cycle1.data-00000-of-00001
│   ├── checkpoint_callback_poly-coherent_euclid_200stars_cycle1.index
│   ├── checkpoint_callback_poly-coherent_euclid_200stars_cycle2.data-00000-of-00001
│   └── checkpoint_callback_poly-coherent_euclid_200stars_cycle2.index
├── config
│   ├── configs.yaml
│   ├── data_config.yaml
│   ├── metrics_config.yaml
│   └── training_config.yaml
├── log-files
│   └── wf-psf_202310161536.log
├── metrics
│   └── metrics-poly-coherent_euclid_200stars.npy
├── optim-hist
│   └── optim_hist_poly-coherent_euclid_200stars.npy
├── plots
└── psf_model
    ├── checkpoint
    ├── psf_model_poly-coherent_euclid_200stars_cycle1.data-00000-of-00001
    ├── psf_model_poly-coherent_euclid_200stars_cycle1.index
    ├── psf_model_poly-coherent_euclid_200stars_cycle2.data-00000-of-00001
    └── psf_model_poly-coherent_euclid_200stars_cycle2.index

metrics_config.yaml
-------------------

metrics:
   # Specify the type of model weights to load by entering "psf_model" to load weights of final psf model or "checkpoint" to load weights from a checkpoint callback.
  model_save_path: psf_model
  # Choose the training cycle for which to evaluate the psf_model. Can be: 1, 2, ...
  saved_training_cycle: 2
  # Metrics-only run: Specify model_params for a pre-trained model else leave blank if running training + metrics
  # Specify path to Parent Directory of Trained Model 
  trained_model_path: /path/to/wf-outputs-202310161536
  # Name of Trained Model Config file inside /trained_model_path/ parent directory
  trained_model_config: training_config.yaml
```
The results of the metrics evaluation will be saved in the new output directory created at runtime (not in the pretrained model directory created previously).

When the trained_model fields are left empty as stated in the commented line, WaveDiff will run the `training` and `metrics` pipelines in serial.  At the start of the `metrics` evaluation task, it will automatically retrieve the model weights at the specific cycle defined by `model_save_path` and `saved_training_cycle` from the `wf-outputs-<timestamp>` sub-directories generated at runtime just before the training task.

The WaveDiff `metrics` pipeline is programmed to automatically evaluate the Polychromatic Pixel reconstruction metrics for both the test (at low- and super-pixel resolution) and training data sets (at low-pixel resolution).  The Monochromatic Pixel Reconstruction and OPD Reconstruction metrics are both optional and can be selected by setting the Boolean flags for `eval_{metric_type}_metric_rmse` to `True` to compute the metric or `False` to disable.  Finally, the Weak Lensing Shape Metrics are computed by default for the test data set at super-pixel resolution and as an option for the training data set by setting the parameter `eval_train_shape_sr_metric_rmse` to `True` or `False` (Note: setting this option to `True` will also trigger WaveDiff to compute the Polychromatric Pixel Reconstruction metrics at super-pixel resolution for the training data set).  The table below provides a summary of these different settings. 

(metrics_settings)=
| Metric type | Metric Identifier | Test Data Set | Training Data Set |
|  ----------- | ------- | ------- | ------- |
| Polychromatic Pixel Reconstruction | `poly` | Default | Default (low-res), Optional (super-res)   |
| Monochromatic Pixel Reconstruction | `mono` | Optional | Optional   |
| Optical Path Differences Reconstruction (OPD) | `opd` | Optional | Optional |
| Weak Lensing Shape Metrics (super-res only) | `shape_sr` | Default  |  Optional  |

The option to generate plots of the metric evaluation results is provided by setting the value of the parameter `plotting_config` to the name of the [plotting configuration](plotting_config) file, e.g. `plotting_config.yaml`.  This will trigger WaveDiff's plotting pipeline to produce plots after completion of the metrics evaluation pipeline.  If the field is left empty, no plots are generated. 

To compute the errors of the trained PSF model, the `metrics` package can retrieve a ground truth data set if it exists in the dataset files listed in the [data_configuration](data_config) file. If they do exist, WaveDiff can generate at runtime a `ground truth model` using the parameters in the metrics configuration file associated to the key: `ground_truth_model`.  The parameter settings for the ground truth model are similar to those contained in the [training configuration](training_config) file.  Currently, the choice of model, which is indicated by the key `model_name`, is currently limited to the polychromatic PSF model, referenced by the short name `poly`.

The `metrics` package is run using [TensorFlow](https://www.tensorflow.org) to reconstruct the PSF model and to evaluate the various metrics. The `metrics_hparams` key contains some standard machine learning hyperparameters such as the `batch_size` as well as additional parameters like `output_dim`, which sets the dimension of the output pixel postage stamp, etc.  

(plotting_config)=
## Plot Configuration

The [plotting_config.yaml](https://github.com/CosmoStat/wf-psf/blob/main/config/plotting_config.yaml) file stores the configuration parameters for the WaveDiff pipeline to generate plots for the metrics listed in the {ref}`metrics settings table <metrics_settings>` for each data set.

An example of the contents of the `plotting_config.yaml` file is shown below.

```
plotting_params:
  # Specify path to parent folder containing wf-psf metrics outputs for all runs, e.g. $WORK/wf-outputs/
  metrics_output_path: <PATH>
  # List all of the parent output directories (i.e. wf-outputs-xxxxxxxxxxx) that contain metrics results to be included in the plot 
  metrics_dir: 
    #     - wf-outputs-xxxxxxxxxxx1 
    #     - wf-outputs-xxxxxxxxxxx2  
  # List the corresponding names of the metric config file in each of the parent output directories (would like to change such that code goes and finds them in the metrics_dir)
  metrics_config: 
    #     - metrics_config_1.yaml
    #     - metrics_config_2.yaml
  # Show plots flag
  plot_show: False
```
As nearly all of the specific plotting parameters are pre-coded by default, the `plotting_config` file parameters enable the option to plot jointly the metrics for other trained PSF models. Consider the example below where the user would like to plot the metrics from three output runs in the directories labelled: `wf-outputs-<timestamp>`. 

```
wf-outputs/
├── wf-outputs-202305271829
│   ├── config
│   │   ├── data_config.yaml
│   │   └── metrics_config_200.yaml
│   ├── metrics
│   │   └── metrics-poly-coherent_euclid_200stars.npy
├── wf-outputs-202305271845
│   ├── config
│   │   ├── data_config.yaml
│   │   └── metrics_config_500.yaml
│   ├── metrics
│   │   └── metrics-poly-coherent_euclid_500stars.npy
├── wf-outputs-202305271918
│   ├── config
│   │   ├── data_config.yaml
│   │   └── metrics_config_1000.yaml
│   ├── metrics
│   │   └── metrics-poly-coherent_euclid_1000stars.npy

```

Below is the following `plotting_config.yaml` file that would generate plots including each of the three metrics outputs in the example above:

```
plotting_params:
  # Specify path to parent folder containing wf-psf metrics outputs for all runs, ex: $WORK/wf-outputs/
  metrics_output_path: $WORK/wf-outputs/
  # List all of the parent output directories (i.e. wf-outputs-xxxxxxxxxxx) that contain metrics results to be included in the plot 
  metrics_dir: 
      - wf-outputs-202305271829 
      - wf-outputs-202305271845 
      - wf-outputs-202305271918

  # List the corresponding names of the metric config file in each of the parent output directories (would like to change such that code goes and finds them in the metrics_dir)
  metrics_config: 
      - metrics_config_200.yaml
      - metrics_config_500.yaml
      - metrics_config_1000.yaml

  # Show plots flag
  plot_show: False
```
 In the field for the key: `metrics_output_path`, the user provides the path to the parent directory containing the subdirectories of the runs to be plotted.  Then under the `metrics_dir` key, the user lists row-by-row the names of the parent directories for each run.  Similarly, for the `metrics_config` key, the user can list the names of the `metrics_config` files used for the additional runs row-by-row. Note, if the user only wants to plot the metric from an active metrics evaluation run, i.e. the `plotting` pipeline task is run subsequently after the `metrics` pipeline, these fields can be left empty. The only plotting parameter `plot_show` is a Boolean used to trigger a display of the plot at runtime (as in an interactive session).  If False, no plot is displayed. 

(master_config_file)=
## Master Configuration

The [configs.yaml](https://github.com/CosmoStat/wf-psf/blob/main/config/configs.yaml) file is the master configuration file that is used to define all of the pipeline tasks to be submitted and executed by `WaveDiff` during runtime. In this file, the user lists the processing tasks (one or more) to be performed by setting the values of the associated configuration variables `{pipeline_task}_conf` and the name of the configuration file `{pipeline_task}_config.yaml`.  See an example below to configure `WaveDiff` to launch a sequence of runs to train models 1...n with their respective configurations given in the files `training_config_{id}.yaml`.

```
---
  training_conf_1: training_config_1.yaml
  training_conf_2: training_config_2.yaml
  ...
  training_conf_n: training_config_n.yaml
```
Each training task is run sequentially and independently of the others.  All of the results are stored in the same `wf-outputs-<timestamp>` directory as shown in the example below.

```
├── wf-outputs-202310131055
│   ├── checkpoint
│   │   ├── checkpoint
│   │   ├── checkpoint_callback_poly-coherent_euclid_200stars_1_cycle1.data-00000-of-00001
│   │   ├── checkpoint_callback_poly-coherent_euclid_200stars_1_cycle1.index
│   ├── ...
│   │   ├── checkpoint_callback_poly-coherent_euclid_200stars_n_cycle1.data-00000-of-00001
│   │   ├── checkpoint_callback_poly-coherent_euclid_200stars_n_cycle1.index
│   ├── config
│   │   ├── configs.yaml
│   │   ├── data_config.yaml
│   │   ├── training_config_1.yaml
│   │   ├── ...
│   │   └── training_config_n.yaml
│   ├── log-files
│   │   └── wf-psf_202310131055.log
│   ├── optim-hist
│   │   ├── optim_hist_poly-coherent_euclid_200stars_1.npy
│   │   ├── ...
│   │   └── optim_hist_poly-coherent_euclid_200stars_n.npy
│   ├── plots
│   └── psf_model
│       ├── checkpoint
│       ├── psf_model_poly-coherent_euclid_200stars_1_cycle1.data-00000-of-00001
│       ├── psf_model_poly-coherent_euclid_200stars_1_cycle1.index
│       ├── ...
│       ├── psf_model_poly-coherent_euclid_200stars_n_cycle1.data-00000-of-00001
│       ├── psf_model_poly-coherent_euclid_200stars_n_cycle1.index
```

Likewise, to perform a metrics evaluation and generate plots for each training run (as in the example above), the corresponding names of the `metrics_config.yaml` and `plotting_config.yaml` files need to be provided as values to the `{metrics}_config` and `{plotting}_config` parameters in `training_config_{id}.yaml` and `metrics_config.yaml` files, respectively. The same `metrics_config.yaml` and `plotting_config.yaml` files can be used for each `training_config_{id}.yaml` file.  Below is an example of the `config` tree structure for a `training` + `metrics` + `plotting` run:

```
config/
├── configs.yaml
├── data_config.yaml
├── metrics_config.yaml
├── plotting_config.yaml
├── training_config_1.yaml
├── ...
└── training_config_n.yaml
```

Note, in this version of WaveDiff produces a single plot per each metric per trained model.  To display all of the metrics results for each trained model in a single plot, the user must do so in a different run following the steps defined in the section [Plot Configuration](plotting_config). The next upgrade to WaveDiff will feature options to produce independent metrics plots per trained model or a single master plot comparing the respective metric results for all trained models.

The master configuration file can include a combination of the three pipeline tasks, i.e. training, metrics and plotting.  This will prompt WaveDiff to do independent tasks like train a new PSF model, compute the metrics of a  pre-trained PSF model, or produce plots for a selection of pre-computed metrics. While currently WaveDiff executes these jobs sequentially on a single GPU, the future plan is to distribute these tasks in parallel across multiple GPUs to accelerate the computation.

If you have any questions or feedback, please don't hesitate to open a [Github issue](https://github.com/CosmoStat/wf-psf/issues).