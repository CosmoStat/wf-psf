# Use of the WaveDiff model

The model should be installed as a python package as detailed in the main [README](https://github.com/tobias-liaudat/wf-psf) file. The `wf-psf` package includes functions to train, evaluate and plot the results of the model. In this file we detail how to make use of this functions.

For more information on the input datasets see [Input_data.md](https://github.com/tobias-liaudat/wf-psf/blob/main/long-runs/Input_data.md).

## Auxiliary functions

The package has two main functions to perform test with the WaveDiff model. One function for running the model's optimisation process and one function for performing evaluation and saving the results. There is a third function that generates some plots when the evaluation is finished. This functions can be found in the [`script_utils.py`](https://github.com/tobias-liaudat/wf-psf/blob/main/wf_psf/script_utils.py) file. 

- `train_model(**args)` : trains the PSF model.
- `evaluate_model(**args)` : evaluate the trained PSF model. 
- `plot_metrics(**args)`: Plot multiple models results.

The arguments of these functions are detailed below. 

## Training script
For running the previous functions one can use the [`train_eval_plot_script_click.py`](https://github.com/tobias-liaudat/wf-psf/blob/main/long-runs/train_eval_plot_script_click.py) script. This script calls each of the above functions and serves as an interface to collect the `**args` options from command line arguments using the [`click`](https://click.palletsprojects.com/en/8.1.x/) package. 

To run the script one should input the desired parameters as command line arguments as in the following example:

```
./train_eval_plot_script_click.py \
    --train_opt True \
    --eval_opt True \
    --plot_opt True \
    --model poly \
    --id_name your_id \
    --suffix_id_name _1k_stars --suffix_id_name _2k_stars \
    --id_name your_id_1k_stars --id_name your_id_2k_stars \
    --star_numbers 1000 --stars_numbers 2000 \
    --base_path your/base/path/wf-outputs/ \
    ...
    --pupil_diameter 256 \
    --n_epochs_param_multi_cycle "15 15" \
    --n_epochs_non_param_multi_cycle "100 50" \
    --l_rate_non_param_multi_cycle "0.1 0.06" \
    --l_rate_param_multi_cycle "0.01 0.004" \
    ...
```

The options that remain unset will take the default values defined in the [`train_eval_plot_script_click.py`](https://github.com/tobias-liaudat/wf-psf/blob/main/long-runs/train_eval_plot_script_click.py) script.

# Running model on SLURM clusters
In the folder `./examples` a job scheduler example can be found. That script lets us run several experiments (model training and evaluation) in parallel on a computing cluster using SLURM directives. Each model can use different parameters, datasets or optimisation strategies. 

Note:
- The `--star_numbers` option is for the final plot's x-axis. It does not always represents the number of stars but it needs to be an integer.
- `--id_name` = `--base_id_name` + `--suffix_id_name`

# Output folders

For the auxiliary functions to work properly several folders must be created before running the experiments. The output file structure is shown below:

```
wf-outputs
├── chkp
|	├── model_1
|	├── ...
|	└── model_N
├── log-files
├── metrics
├── optim-hist
└── plots
```

One checkpoint folder is recommended for each model in a multi-model parallel training. This simplifies the organisation of multiple cycle model checkpoints. This folders should coincide with their corresponding script option.

# Script options 
Here we detail every option or argument used in the WaveDiff model auxiliary script. 

## Script options

- `--train_opt`, default=`True`, type: `bool`
	- Train the model.

- `--eval_opt`, default=`True`, type: `bool`
	- Evaluate the model.

- `--plot_opt`, default=`True`, type: `bool`
	- Plot the model results
## Training options
### Model definition

- `--model`, default="poly", type: str
	- Model type. Options are: 'mccd', 'graph', 'poly, 'param', 'poly_physical'.

- `--id_name`, default="-coherent_euclid_200stars", type: str
    - Model saving id.
### Saving paths

- `--base_path`, default="/gpfswork/rech/ynx/ulx23va/wf-outputs/", type: str
	- Base path for saving files.

- `--log_folder`, default="log-files/", type: str
	- Folder name to save log files.

- `--model_folder`, default="chkp/", type: str
	- Folder name to save trained models.

- `--optim_hist_folder`, default="optim-hist/", type: str
	- Folder name to save optimisation history files.

- `--chkp_save_path`, default="/gpfsscratch/rech/ynx/ulx23va/wf-outputs/chkp/", type: str
	- Path to save model checkpoints during training.

- `--plots_folder`, default="plots/", type: str
	- Folder name to save the generated plots.
### Input dataset paths

- `--dataset_folder`, default="/gpfswork/rech/ynx/ulx23va/repo/wf-psf/data/coherent_euclid_dataset/", type: str
	- Folder path of datasets.

- `--train_dataset_file`, default="train_Euclid_res_200_TrainStars_id_001.npy", type: str
	- Train dataset file name.

- `--test_dataset_file`, default="test_Euclid_res_id_001.npy", type: str
	- Test dataset file name.
## Model parameters

- `--n_zernikes`, default=15, type: int
	- Zernike polynomial modes to use on the parametric part.

- `--pupil_diameter`, default=256, type: int
	- Dimension of the OPD/Wavefront space.

- `--n_bins_lda`, default=20, type: int
	- Number of wavelength bins to use to reconstruct polychromatic objects.

- `--output_q`, default=3., type: float
	- Downsampling rate to match the specified telescope's sampling from the oversampling rate used in the model.

- `--oversampling_rate`, default=3., type: float
	- Oversampling rate used for the OPD/WFE PSF model.

- `--output_dim`, default=32, type: int
	- Dimension of the pixel PSF postage stamp.

- `--d_max`, default=2, type: int
	- Max polynomial degree of the parametric part.

- `--d_max_nonparam`, default=3, type: int
	- Max polynomial degree of the non-parametric part.

- `--x_lims`
nargs=2,, default=[0, 1e3], type: float
	- Limits of the PSF field coordinates for the x axis.

- `--y_lims`
nargs=2,, default=[0, 1e3], type: float
	- Limits of the PSF field coordinates for the y axis.

- `--graph_features`, default=10, type: int
	- Number of graph-constrained features of the non-parametric part.

- `--l1_rate`, default=`1e-8`, type: float
	- L1 regularisation parameter for the non-parametric part.

- `--use_sample_weights`, default=`False`, type: `bool`
	- Boolean to define if we use sample weights based on the noise standard deviation estimation.

- `--interpolation_type`, default="none", type: str
	- The interpolation type for the physical poly model. Options are: 'none', 'all', 'top_K', 'independent_Zk'.
## Training parameters

- `--batch_size`, default=32, type: int
	- Batch size used for the training in the stochastic gradient descend type of algorithm.
### Old multicycle parameters for backwards compatibility. 

- `--l_rate_param`, default=None, type: float
	- Learning rates for the parametric parts.

- `--l_rate_non_param`, default=None, type: float
	- Learning rates for the non-parametric parts.

- `--n_epochs_param`, default=None, type: int
	- Number of training epochs of the parametric parts.

- `--n_epochs_non_param`, default=None, type: int
	- Number of training epochs of the non-parametric parts.
### New multicycle parameters

- `--l_rate_param_multi_cycle`, default="`1e-2` `1e-2`", type: str
	- Learning rates for the parametric parts. It should be a string where numeric values are separated by spaces.

- `--l_rate_non_param_multi_cycle`, default="`1e-1` `1e-1`", type: str
	- Learning rates for the non-parametric parts. It should be a string where numeric values are separated by spaces.

- `--n_epochs_param_multi_cycle`, default="20 20", type: str
	- Number of training epochs of the parametric parts. It should be a string where numeric values are separated by spaces.

- `--n_epochs_non_param_multi_cycle`, default="100 120", type: str
	- Number of training epochs of the non-parametric parts. It should be a string where numeric values are separated by spaces.

- `--save_all_cycles`, default=`False`, type: `bool`
	- Make checkpoint at every cycle or just save the checkpoint at the end of the training.

- `--total_cycles`, default=2, type: int
	- Total amount of cycles to perform. For the moment the only available options are '1' or '2'.

- `--cycle_def`, default="complete", type: str
	- Train cycle definition. It can be: 'parametric', 'non-parametric', 'complete', 'only-non-parametric' and 'only-parametric'.
## Evaluation flags
### Saving paths

- `--model_eval`, default="poly", type: str
	- Model used as ground truth for the evaluation. Options are: 'poly', 'physical'.

- `--metric_base_path`, default="/gpfswork/rech/ynx/ulx23va/wf-outputs/metrics/", type: str
	- Base path for saving metric files.

- `--saved_model_type`, default="final", type: str
	- Type of saved model to use for the evaluation. Can be 'final' for a full model, 'checkpoint' for model weights or 'external' for a different model not saved under the same base_id as the current one..

- `--saved_cycle`, default="cycle2", type: str
	- Saved cycle to use for the evaluation. Can be 'cycle1', 'cycle2', ..., 'cycleN'.
### Evaluation parameters

- `--gt_n_zernikes`, default=45, type: int
	- Zernike polynomial modes to use on the ground truth model parametric part.

- `--eval_batch_size`, default=16, type: int
	- Batch size to use for the evaluation.

- `--n_bins_gt`, default=20, type: int,
help="Number of bins used for the ground truth model poly PSF generation."
)

- `--opt_stars_rel_pix_rmse`, default=`False`, type: `bool`
	- Option to get SR pixel PSF RMSE for each individual test star.
## Specific parameters

- `--l2_param`, default=0., type: float
	- Parameter for the l2 loss of the OPD.
## Plot parameters

- `--base_id_name`, default="-coherent_euclid_", type: str
	- Plot parameter. Base id_name before dataset suffix are added.

- `--suffix_id_name`, default=["2c", "5c"],
multiple=`True`, type: str
	- Plot parameter. Suffix needed to recreate the different id names.

- `--star_numbers`, default=[200, 500],
multiple=`True`, type: int
	- Plot parameter. Training star number of the different models evaluated. Needs to correspond with the `suffix_id_name`.
## WaveDiff new features
### Feature: SED interp

- `--interp_pts_per_bin`, default=0, type: int
	- Number of points per bin to add during the interpolation process. It can take values {0,1,2,3}, where 0 means no interpolation.

- `--extrapolate`, default=`True`, type: `bool`
	- Whether extrapolation is performed or not on the borders of the SED.

- `--SED_interp_kind`, default="linear", type: str
	- Type of interpolation for the SED. It can be "linear", "cubic", "quadratic", etc. Check [all available options](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html). 

- `--SED_sigma`, default=0, type: float
	- Standard deviation of the multiplicative SED Gaussian noise.
### Feature: project parameters

- `--project_dd_features`, default=`False`, type: `bool`
	- Project NP DD features onto parametric model.

- `--eval_only_param`, default=`False`, type: `bool`
	- Use only the parametric model for evaluation.

- `--reset_dd_features`, default=`False`, type: `bool`
	- Reset to random initialisation the non-parametric model after projecting the DD features.

- `--pretrained_model`, default=`None`, type: str
	- Path to pretrained model checkpoint callback.