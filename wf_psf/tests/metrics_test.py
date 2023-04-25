"""UNIT TESTS FOR PACKAGE MODULE: Metrics.

This module contains unit tests for the wf_psf.metrics module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""
import pytest
from pyparsing import Any
from wf_psf.utils.read_config import RecursiveNamespace

# import wf_psf.training.train
# from wf_psf.training.train import TrainingParamsHandler
import tensorflow as tf

# import tensorflow_addons as tfa

import numpy as np


metrics_params = RecursiveNamespace(
    training_run=RecursiveNamespace(
        use_train_params=True, use_callback=False, saved_training_cycle="cycle2"
    ),
    compute_metrics_only_params=RecursiveNamespace(
        model="poly",
        use_callback=False,
        saved_training_cycle="cycle2",
        chkp_save_path="checkpoint",
        sed_interp_pts_per_bin=0,
        sed_extrapolate=True,
        sed_interp_kind="linear",
        sed_sigma=0,
    ),
    ground_truth_model=RecursiveNamespace(
        model="poly",
        gt_n_zernikes=45,
        eval_batch_size=16,
        n_bins_gt=20,
        opt_stars_rel_pix_rmse=False,
        eval_mono_metric_rmse=True,
        eval_opd_metric_rmse=True,
        eval_train_shape_sr_metric_rmse=True,
        l2_param=0.0,
    ),
    data=RecursiveNamespace(
        training=RecursiveNamespace(
            file="data/coherent_euclid_dataset/train_Euclid_res_200_TrainStars_id_001.npy",
            stars=None,
            positions=None,
            SEDS=None,
            zernike_coef=None,
            C_poly=None,
            params=RecursiveNamespace(
                d_max=2,
                max_order=45,
                x_lims=[0, 1000.0],
                y_lims=[0, 1000.0],
                grid_points=[4, 4],
                n_bins=20,
                max_wfe_rms=0.1,
                oversampling_rate=3.0,
                output_Q=3.0,
                output_dim=32,
                LP_filter_length=2,
                pupil_diameter=256,
                euclid_obsc=True,
                n_stars=200,
            ),
        ),
        test=RecursiveNamespace(
            file="data/coherent_euclid_dataset/test_Euclid_res_id_001.npy",
            stars=None,
            noisy_stars=None,
            positions=None,
            SEDS=None,
            zernike_coef=None,
            C_poly=None,
            parameters=RecursiveNamespace(
                d_max=2,
                max_order=45,
                x_lims=[0, 1000.0],
                y_lims=[0, 1000.0],
                grid_points=[4, 4],
                max_wfe_rms=0.1,
            ),
        ),
    ),
)

chkp_dir = "/wf-outputs/checkpoint/"


@pytest.fixture(scope="module", params=[metrics_params])
def metrics():
    return metrics_params


# @pytest.fixture(scope="module", params=[training_param_set, chkp_dir])
# def training_params():
#    return TrainingParamsHandler(training_params_set, chkp_dir)


# @pytest.fixture(scope="module")
# def training_handler():


def test_metrics_params(metrics: RecursiveNamespace):
    metric_params = metrics
    print(metric_params)


# def test_training_params(training_params: Any):
#    x = training_params
#    print(x)
