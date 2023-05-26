"""UNIT TESTS FOR PACKAGE MODULE: Metrics.

This module contains unit tests for the wf_psf.metrics module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""
import pytest
from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.training import train
from wf_psf.metrics.metrics_interface import MetricsParamsHandler, evaluate_model
import tensorflow as tf


import numpy as np


metrics_params = RecursiveNamespace(
    use_callback=False,
    saved_training_cycle="cycle2",
    chkp_save_path="checkpoint",
    id_name="-coherent_euclid_200stars",
    model_params=RecursiveNamespace(
        model_name="poly",
        use_callback=False,
        sed_interp_pts_per_bin=0,
        sed_extrapolate=True,
        sed_interp_kind="linear",
        sed_sigma=0,
        n_bins_lda=8,
        output_Q=3,
        oversampling_rate=3,
        output_dim=32,
        pupil_diameter=256,
        use_sample_weights=True,
        interpolation_type="None",
        x_lims=[0.0, 1000.0],
        y_lims=[0.0, 1000.0],
        param_hparams=RecursiveNamespace(
            l2_param=0.0, n_zernikes=15, d_max=2, save_optim_history_param=True
        ),
        nonparam_hparams=RecursiveNamespace(
            d_max_nonparam=5,
            num_graph_features=10,
            l1_rate=1e-08,
            project_dd_features=False,
            reset_dd_features=False,
            save_optim_history_nonparam=True,
        ),
    ),
    ground_truth_model=RecursiveNamespace(
        model_params=RecursiveNamespace(
            model_name="poly",
            sed_interp_pts_per_bin=0,
            sed_extrapolate=True,
            sed_interp_kind="linear",
            sed_sigma=0,
            n_bins_lda=20,
            output_Q=3,
            oversampling_rate=3,
            output_dim=32,
            pupil_diameter=256,
            use_sample_weights=True,
            interpolation_type="None",
            x_lims=[0.0, 1000.0],
            y_lims=[0.0, 1000.0],
            param_hparams=RecursiveNamespace(
                l2_param=0.0, n_zernikes=45, d_max=2, save_optim_history_param=True
            ),
            nonparam_hparams=RecursiveNamespace(
                d_max_nonparam=5,
                num_graph_features=10,
                l1_rate=1e-08,
                project_dd_features=False,
                reset_dd_features=False,
                save_optim_history_nonparam=True,
            ),
        )
    ),
    metrics_hparams=RecursiveNamespace(
        batch_size=16,
        opt_stars_rel_pix_rmse=False,
        eval_mono_metric_rmse=True,
        eval_opd_metric_rmse=True,
        eval_train_shape_sr_metric_rmse=True,
        l2_param=0.0,
        output_Q=1,
        output_dim=64,
    ),
)

chkp_dir = "../wf-outputs/checkpoint"
optim_dir = "../wf-outputs/optim-hist"
metrics_output = "../wf-outputs/metrics"


@pytest.fixture(scope="module", params=[metrics_params])
def metrics():
    return metrics_params


def test_metrics_params(metrics: RecursiveNamespace):
    metric_params = metrics
    print(metric_params)


def test_evaluate_metrics_opd(training_params, training_data, test_dataset, psf_model):
    metrics_handler = MetricsParamsHandler(metrics_params, training_params)
    cycle = 1

    ## Prepare models
    # Prepare np input
    simPSF_np = training_data.simPSF

    ## Load the model's weights
    psf_model.load_weights(
        train.filepath_chkp_callback(
            chkp_dir,
            training_params.model_params.model_name,
            training_params.id_name,
            cycle,
        )
    )

    mono_metric = metrics_handler.evaluate_metrics_opd(
        psf_model, simPSF_np, test_dataset
    )


def test_eval_metrics_mono_rmse(
    training_params, training_data, test_dataset, psf_model
):
    metrics_handler = MetricsParamsHandler(metrics_params, training_params)
    cycle = 1

    ## Prepare models
    # Prepare np input
    simPSF_np = training_data.simPSF

    ## Load the model's weights
    psf_model.load_weights(
        train.filepath_chkp_callback(
            chkp_dir,
            training_params.model_params.model_name,
            training_params.id_name,
            cycle,
        )
    )

    mono_metric = metrics_handler.evaluate_metrics_mono_rmse(
        psf_model, simPSF_np, test_dataset
    )


def test_eval_metrics_polychromatic_lowres(
    training_params, training_data, test_dataset, psf_model
):
    metrics_handler = MetricsParamsHandler(metrics_params, training_params)
    cycle = 1

    ## Prepare models
    # Prepare np input
    simPSF_np = training_data.simPSF

    # Load the model's weights
    psf_model.load_weights(
        train.filepath_chkp_callback(
            chkp_dir,
            training_params.model_params.model_name,
            training_params.id_name,
            cycle,
        )
    )

    poly_metric = metrics_handler.evaluate_metrics_polychromatic_lowres(
        psf_model, simPSF_np, test_dataset
    )


def test_evaluate_metrics_shape(
    training_params, training_data, test_dataset, psf_model
):
    metrics_handler = MetricsParamsHandler(metrics_params, training_params)
    cycle = 1

    ## Prepare models
    # Prepare np input
    simPSF_np = training_data.simPSF

    ## Load the model's weights
    psf_model.load_weights(
        train.filepath_chkp_callback(
            chkp_dir,
            training_params.model_params.model_name,
            training_params.id_name,
            cycle,
        )
    )

    mono_metric = metrics_handler.evaluate_metrics_shape(
        psf_model, simPSF_np, test_dataset
    )


def test_evaluate_model(
    training_params: RecursiveNamespace, training_data, test_data, psf_model
):
    cycle = 1
    evaluate_model(
        metrics_params,
        training_params,
        training_data,
        test_data,
        psf_model,
        train.filepath_chkp_callback(
            chkp_dir,
            training_params.model_params.model_name,
            training_params.id_name,
            cycle,
        ),
        metrics_output,
    )
