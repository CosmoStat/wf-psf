"""UNIT TESTS FOR PACKAGE MODULE: Metrics.

This module contains unit tests for the wf_psf.metrics module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""
import pytest
from wf_psf.utils.read_config import RecursiveNamespace
import tensorflow as tf

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
)



@pytest.fixture(scope="module", params=[metrics_params])
def metrics():
    return metrics_params


#@pytest.fixture(scope="module")
#def training_handler():
#    t_handler = 


def test_metrics_params(metrics: RecursiveNamespace):
    metric_params = metrics
    print(metric_params)


def test_training_params(training_params: RecursiveNamespace):
    x = training_params
    print(x)
