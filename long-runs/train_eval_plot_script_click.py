#!/usr/bin/env python
# coding: utf-8

# PSF modelling and evaluation

import wf_psf as wf
import click

# from absl import app
# from absl import flags

@click.command()

## Script options
@click.option(
    "--train_opt",
    default=True,
    type=bool,
    help="Train the model.")
@click.option(
    "--eval_opt",
    default=True,
    type=bool,
    help="Evaluate the model.")
@click.option(
    "--plot_opt",
    default=True,
    type=bool,
    help="Plot the model results")
## Training options
# Model definition
@click.option(
    "--model",
    default="poly",
    type=str,
    help="Model type. Options are: 'mccd', 'graph', 'poly, 'param', 'poly_physical'.")
@click.option(
    "--id_name",
    default="-coherent_euclid_200stars",
    type=str,
    help="Model saving id.")
# Saving paths
@click.option(
    "--base_path",
    default="/gpfswork/rech/ynx/ulx23va/wf-outputs/",
    type=str,
    help="Base path for saving files.")
@click.option(
    "--log_folder",
    default="log-files/",
    type=str,
    help="Folder name to save log files.")
@click.option(
    "--model_folder",
    default="chkp/",
    type=str,
    help="Folder name to save trained models.")
@click.option(
    "--optim_hist_folder",
    default="optim-hist/",
    type=str,
    help="Folder name to save optimisation history files.")
@click.option(
    "--chkp_save_path",
    default="/gpfsscratch/rech/ynx/ulx23va/wf-outputs/chkp/",
    type=str,
    help="Path to save model checkpoints during training.")
@click.option(
    "--plots_folder",
    default="plots/",
    type=str,
    help="Folder name to save the generated plots.")
# Input dataset paths
@click.option(
    "--dataset_folder",
    default="/gpfswork/rech/ynx/ulx23va/repo/wf-psf/data/coherent_euclid_dataset/",
    type=str,
    help="Folder path of datasets.")
@click.option(
    "--train_dataset_file",
    default="train_Euclid_res_200_TrainStars_id_001.npy",
    type=str,
    help="Train dataset file name.")
@click.option(
    "--test_dataset_file",
    default="test_Euclid_res_id_001.npy",
    type=str,
    help="Test dataset file name.")
# Model parameters
@click.option(
    "--n_zernikes",
    default=15,
    type=int,
    help="Zernike polynomial modes to use on the parametric part.")
@click.option(
    "--pupil_diameter",
    default=256,
    type=int,
    help="Dimension of the OPD/Wavefront space.")
@click.option(
    "--n_bins_lda",
    default=20,
    type=int,
    help="Number of wavelength bins to use to reconstruct polychromatic objects.")
@click.option(
    "--output_q",
    default=3.,
    type=float,
    help="Downsampling rate to match the specified telescope's sampling from the oversampling rate used in the model.")
@click.option(
    "--oversampling_rate",
    default=3.,
    type=float,
    help="Oversampling rate used for the OPD/WFE PSF model.")
@click.option(
    "--output_dim",
    default=32,
    type=int,
    help="Dimension of the pixel PSF postage stamp.")
@click.option(
    "--d_max",
    default=2,
    type=int,
    help="Max polynomial degree of the parametric part.")
@click.option(
    "--d_max_nonparam",
    default=3,
    type=int,
    help="Max polynomial degree of the non-parametric part.")
@click.option(
    "--x_lims",
    nargs=2,
    default=[0, 1e3],
    type=float,
    help="Limits of the PSF field coordinates for the x axis.")
@click.option(
    "--y_lims",
    nargs=2,
    default=[0, 1e3],
    type=float,
    help="Limits of the PSF field coordinates for the y axis.")
@click.option(
    "--graph_features",
    default=10,
    type=int,
    help="Number of graph-constrained features of the non-parametric part.")
@click.option(
    "--l1_rate",
    default=1e-8,
    type=float,
    help="L1 regularisation parameter for the non-parametric part.")
@click.option(
    "--use_sample_weights",
    default=False,
    type=bool,
    help="Boolean to define if we use sample weights based on the noise standard deviation estimation.")
# Training parameters
@click.option(
    "--batch_size",
    default=32,
    type=int,
    help="Batch size used for the trainingin the stochastic gradient descend type of algorithm.")
@click.option(
    "--l_rate_param",
    nargs=2,
    default=[1e-2, 1e-2],
    type=float,
    help="Learning rates for the parametric parts.")
@click.option(
    "--l_rate_non_param",
    nargs=2,
    default=[1e-1, 1e-1],
    type=float,
    help="Learning rates for the non-parametric parts.")
@click.option(
    "--n_epochs_param",
    nargs=2,
    default=[20, 20],
    type=int,
    help="Number of training epochs of the parametric parts.")
@click.option(
    "--n_epochs_non_param",
    nargs=2,
    default=[100, 120],
    type=int,
    help="Number of training epochs of the non-parametric parts.")
@click.option(
    "--total_cycles",
    default=2,
    type=int,
    help="Total amount of cycles to perform. For the moment the only available options are '1' or '2'.")
@click.option(
    "--cycle_def",
    default="complete",
    type=str,
    help="Train cycle definition. It can be: 'parametric', 'non-parametric', 'complete'.")
## Evaluation flags
# Saving paths
@click.option(
    "--model_eval",
    default="poly",
    type=str,
    help="Model used as ground truth for the evaluation. Options are: 'poly', 'physical'.")
@click.option(
    "--metric_base_path",
    default="/gpfswork/rech/ynx/ulx23va/wf-outputs/metrics/",
    type=str,
    help="Base path for saving metric files.")
@click.option(
    "--saved_model_type",
    default="final",
    type=str,
    help="Type of saved model to use for the evaluation. Can be 'final' or 'checkpoint'.")
@click.option(
    "--saved_cycle",
    default="cycle2",
    type=str,
    help="Saved cycle to use for the evaluation. Can be 'cycle1' or 'cycle2'.")
# Evaluation parameters
@click.option(
    "--gt_n_zernikes",
    default=45,
    type=int,
    help="Zernike polynomial modes to use on the ground truth model parametric part.")
@click.option(
    "--eval_batch_size",
    default=16,
    type=int,
    help="Batch size to use for the evaluation.")
## Specific parameters
@click.option(
    "--l2_param",
    default=0.,
    type=float,
    help="Parameter for the l2 loss of the OPD.")
## Plot parameters
@click.option(
    "--base_id_name",
    default="-coherent_euclid_",
    type=str,
    help="Plot parameter. Base id_name before dataset suffix are added.")
@click.option(
    "--suffix_id_name",
    default=["2c", "5c"],
    multiple=True,
    type=str,
    help="Plot parameter. Suffix needed to recreate the different id names.")
@click.option(
    "--star_numbers",
    default=[200, 500],
    multiple=True,
    type=int,
    help="Plot parameter. Training star number of the different models evaluated. Needs to correspond with the `suffix_id_name`.")


def main(**args):
    print(args)
    if args['train_opt']:
        wf.script_utils.train_model(**args)
    if args['eval_opt']:
        wf.script_utils.evaluate_model(**args)
    if args['plot_opt']:
        wf.script_utils.plot_metrics(**args)
        wf.script_utils.plot_optimisation_metrics(**args)

if __name__ == "__main__":
  main()
