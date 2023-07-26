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
    id_name="_sample_w_bis1_2k",
    trained_model_path="/Users/jenniferpollack/Projects/wf-outputs/Archive/wf-outputs-202307041437/",
    trained_model_config="config/training_config.yaml",
    plotting_config=None,
    eval_mono_metric_rmse=False,
    eval_opd_metric_rmse= True,
    eval_train_shape_sr_metric_rmse=False,
    ground_truth_model=RecursiveNamespace(
        model_params=RecursiveNamespace(
            model_name="poly",
            dataset_type="C_poly",
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
        l2_param=0.0,
        output_Q=1,
        output_dim=64,
    ),
)

# paper_metrics = {"poly_metric":{
#         "rmse": 6.379096e-05,
#         "rel_rmse": 0.8615310303866863,
#         "std_rmse": 9.822091e-06,
#         "std_rel_rmse": 0.21410740446299314,
#     },
#     "opd_metrics": {
#         "rmse_opd": 0.10096897128873078,
#         "rel_rmse_opd": 128.57721760031515,
#         "rmse_std_opd": 0.019779305712904472,
#         "rel_rmse_std_opd": 13.43655258447226,
#     },
#     "shape_metrics": {
#         "rmse_e1": 0.0023064037656687175,
#         "std_rmse_e1": 0.0023053241656404403,
#         "rel_rmse_e1": 265.3203356146387,
#         "std_rel_rmse_e1": 264.71100899066596,
#         "rmse_e2": 0.001648851481731486,
#         "std_rmse_e2": 0.0013749469782346232,
#         "rel_rmse_e2": 335.0328737049857,
#         "std_rel_rmse_e2": 334.37154808696175,
#         "rmse_R2_meanR2": 0.013229994217695357,
#         "std_rmse_R2_meanR2": 0.0038716948775583057,
#         "pix_rmse": 1.9327308e-05,
#         "pix_rmse_std": 2.7280478e-06,
#         "rel_pix_rmse": 1.2878789566457272,
#         "rel_pix_rmse_std": 0.32910651061683893,
#     }}

chkp_dir = "wf_psf/tests/data/validation/checkpoint_paper"
# optim_dir = "../wf-outputs/optim-hist"
metrics_output = "wf_psf/tests/data/wf-outputs/metrics"
filename="/papers/article_IOP/data/metrics/wavediff-original/metrics-poly_sample_w_bis1_2k.npy"
paper_metrics = np.load(filename, allow_pickle=True)[()]


@pytest.fixture(scope="module", params=[metrics_params])
def metrics():
    # Load paper results
    return metrics_params


def test_metrics_params(metrics: RecursiveNamespace):
    metric_params = metrics
    print(metric_params)


def test_eval_metrics_polychromatic_lowres(
    training_params, training_data, test_dataset, psf_model
):
    metrics_handler = MetricsParamsHandler(metrics_params, training_params)

    cycle = 2

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
    print(poly_metric)
    print(paper_metrics["poly_metric"])


def test_evaluate_metrics_opd(training_params, training_data, test_dataset, psf_model):
    metrics_handler = MetricsParamsHandler(metrics_params, training_params)
    cycle = 2

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

    opd_metric = metrics_handler.evaluate_metrics_opd(
        psf_model, simPSF_np, test_dataset
    )
    #print(opd_metric)
    #print(paper_opd_metrics)
    tol = 1.e-5
    ratio_rmse_opd = abs(1-paper_metrics["rmse_opd"]/opd_metric["rmse_opd"])
    ratio_rmse_rel_rmse_opd = abs(1.-paper_metrics["rel_rmse_opd"]/opd_metric["rel_rmse_opd"])
    ratio_rmse_std_opd = abs(1.-paper_metrics["rmse_std_opd"]/opd_metric["rmse_std_opd"])
    ratio_rel_rmse_std_opd = abs(1.-paper_metrics["rel_rmse_std_opd"]/opd_metric["rel_rmse_std_opd"])

    assert ratio_rmse_opd < tol
    assert ratio_rel_rmse_std_opd < tol
    assert ratio_rmse_std_opd < tol
    assert ratio_rel_rmse_std_opd < tol

def test_eval_metrics_mono_rmse(
    training_params, training_data, test_dataset, psf_model
):
    metrics_handler = MetricsParamsHandler(metrics_params, training_params)
    cycle = 2

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


def test_evaluate_metrics_shape(
    training_params, training_data, test_dataset, psf_model
):
    metrics_handler = MetricsParamsHandler(metrics_params, training_params)
    cycle = 2

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

    shape_metric = metrics_handler.evaluate_metrics_shape(
        psf_model, simPSF_np, test_dataset
    )
    
    tol = 1.e-5
    ratio_rmse_e1 = abs(1.-paper_metrics["rmse_e1"]/shape_metric["rmse_e1"])
    print('ratio rmse e1', ratio_rmse_e1)
    ratio_std_rmse_e1 = abs(1.-paper_metrics["std_rmse_e1"]/shape_metric["std_rmse_e1"])
    print('ratio std rmse e1', ratio_std_rmse_e1)
    ratio_rel_rmse_e1 = abs(1.-paper_metrics["rel_rmse_e1"]/shape_metric["rel_rmse_e1"])
    print('ratio rel_rmse_e1', ratio_rel_rmse_e1)
    ratio_std_rel_rmse_e1 = abs(1.-paper_metrics["std_rel_rmse_e1"]/shape_metric["std_rel_rmse_e1"])
    print('ratio std_rel_rmse_e1', ratio_std_rel_rmse_e1)
    ratio_rmse_e2 = abs(1.-paper_metrics["rmse_e2"]/shape_metric["rmse_e2"])
    print('ratio rmse_e2', ratio_rmse_e2)
    ratio_std_rmse_e2 = abs(1.-paper_metrics["std_rmse_e2"]/shape_metric["std_rmse_e2"])
    print('ratio std_rmse_e2', ratio_std_rmse_e2)
    ratio_rmse_R2_meanR2=abs(1.-paper_metrics["rmse_R2_meanR2"]/shape_metric["rmse_R2_meanR2"])
    print('ratio rmse R2 mean R2', ratio_rmse_R2_meanR2)

    assert ratio_rmse_e1 < tol
    assert ratio_std_rmse_e1 < tol
    assert ratio_rel_rmse_e1 < tol
    assert ratio_std_rel_rmse_e1 < tol
    assert ratio_rmse_e2 < tol
    assert ratio_std_rmse_e2 < tol
    assert ratio_rmse_R2_meanR2 < tol

