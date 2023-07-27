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
import os


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

chkp_dir = "wf_psf/tests/data/validation/checkpoint_paper"
# optim_dir = "../wf-outputs/optim-hist"
metrics_output = "wf_psf/tests/data/wf-outputs/metrics"
paper_dir="/gpfswork/rech/ynx/uuu68hq/wf-psf/papers/article_IOP/data/metrics/wavediff-original"
main_dir="/gpfswork/rech/ynx/uuu68hq/wf-outputs/wf-outputs-bug/metrics"
filename="metrics-poly_sample_w_bis1_2k.npy"

paper_metrics = np.load(os.path.join(paper_dir,filename), allow_pickle=True)[()]
main_metrics = np.load(os.path.join(main_dir,filename), allow_pickle=True)[()] 

@pytest.fixture(scope="module", params=[metrics_params])
def metrics():
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
    print(paper_metrics["test_metrics"]["poly_metric"])
    
    tol=1.e-5
    ratio_rmse = abs(1.-paper_metrics["test_metrics"]["poly_metric"]["rmse"]/poly_metric["rmse"])
    ratio_rel_rmse = abs(1.-paper_metrics["test_metrics"]["poly_metric"]["rel_rmse"]/poly_metric["rel_rmse"])
    ratio_std_rmse = abs(1.-paper_metrics["test_metrics"]["poly_metric"]["std_rmse"]/poly_metric["std_rmse"])
    ratio_rel_std_rmse = abs(1.-paper_metrics["test_metrics"]["poly_metric"]["std_rel_rmse"]/poly_metric["std_rel_rmse"])

    print("ratio_rmse", ratio_rmse)
    print("ratio_rel_rmse", ratio_rel_rmse)
    print("ratio_std_rmse", ratio_std_rmse)
    print("ratio_rel_std_rmse", ratio_rel_std_rmse)

    m_ratio_rmse = abs(1.-paper_metrics["test_metrics"]["poly_metric"]["rmse"]/main_metrics["test_metrics"]["poly_metric"]["rmse"])
    m_ratio_rel_rmse = abs(1.-paper_metrics["test_metrics"]["poly_metric"]["rel_rmse"]/main_metrics["test_metrics"]["poly_metric"]["rel_rmse"])
    m_ratio_std_rmse = abs(1.-paper_metrics["test_metrics"]["poly_metric"]["std_rmse"]/main_metrics["test_metrics"]["poly_metric"]["std_rmse"])
    m_ratio_rel_std_rmse = abs(1.-paper_metrics["test_metrics"]["poly_metric"]["std_rel_rmse"]/main_metrics["test_metrics"]["poly_metric"]["std_rel_rmse"])

    print(" ")
    print("Main Branch vs. Tag 1.2.0")
    print("ratio_rmse", m_ratio_rmse)
    print("ratio_rel_rmse", m_ratio_rel_rmse)
    print("ratio_std_rmse", m_ratio_std_rmse)
    print("ratio_rel_std_rmse", m_ratio_rel_std_rmse)

    assert ratio_rmse < tol
    assert ratio_rel_rmse < tol
    assert ratio_std_rmse < tol
    assert ratio_rel_std_rmse < tol

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
    ratio_rmse_opd = abs(1-paper_metrics["test_metrics"]["opd_metric"]["rmse_opd"]/opd_metric["rmse_opd"])
    ratio_rel_rmse_opd = abs(1.-paper_metrics["test_metrics"]["opd_metric"]["rel_rmse_opd"]/opd_metric["rel_rmse_opd"])
    ratio_rmse_std_opd = abs(1.-paper_metrics["test_metrics"]["opd_metric"]["rmse_std_opd"]/opd_metric["rmse_std_opd"])
    ratio_rel_rmse_std_opd = abs(1.-paper_metrics["test_metrics"]["opd_metric"]["rel_rmse_std_opd"]/opd_metric["rel_rmse_std_opd"])

    m_ratio_rmse_opd = abs(1-paper_metrics["test_metrics"]["opd_metric"]["rmse_opd"]/main_metrics["test_metrics"]["opd_metric"]["rmse_opd"])
    m_ratio_rel_rmse_opd = abs(1.-paper_metrics["test_metrics"]["opd_metric"]["rel_rmse_opd"]/main_metrics["test_metrics"]["opd_metric"]["rel_rmse_opd"])
    m_ratio_rmse_std_opd = abs(1.-paper_metrics["test_metrics"]["opd_metric"]["rmse_std_opd"]/main_metrics["test_metrics"]["opd_metric"]["rmse_std_opd"])
    m_ratio_rel_rmse_std_opd = abs(1.-paper_metrics["test_metrics"]["opd_metric"]["rel_rmse_std_opd"]/main_metrics["test_metrics"]["opd_metric"]["rel_rmse_std_opd"])

    print(" ")
    print("ratio_rmse_opd", ratio_rmse_opd)
    print("ratio_rel_rmse_opd", ratio_rel_rmse_opd)
    print("ratio_rmse_std_opd", ratio_rmse_std_opd)
    print("ratio_rel_rmse_std_opd", ratio_rel_rmse_std_opd)

    print(" ")

    print("Main Branch vs. Tag 1.2.0.")
    print("ratio_rmse_opd", m_ratio_rmse_opd)
    print("ratio_rel_rmse_opd", m_ratio_rel_rmse_opd)
    print("ratio_rmse_std_opd", m_ratio_rmse_std_opd)
    print("ratio_rel_rmse_std_opd", m_ratio_rel_rmse_std_opd)

    assert ratio_rmse_opd < tol
    assert ratio_rel_rmse_opd < tol
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
    ratio_rmse_e1 = abs(1.-paper_metrics["test_metrics"]["shape_results_dict"]["rmse_e1"]/shape_metric["rmse_e1"])
    print('ratio rmse e1', ratio_rmse_e1)
    ratio_std_rmse_e1 = abs(1.-paper_metrics["test_metrics"]["shape_results_dict"]["std_rmse_e1"]/shape_metric["std_rmse_e1"])
    print('ratio std rmse e1', ratio_std_rmse_e1)
    ratio_rel_rmse_e1 = abs(1.-paper_metrics["test_metrics"]["shape_results_dict"]["rel_rmse_e1"]/shape_metric["rel_rmse_e1"])
    print('ratio rel_rmse_e1', ratio_rel_rmse_e1)
    ratio_std_rel_rmse_e1 = abs(1.-paper_metrics["test_metrics"]["shape_results_dict"]["std_rel_rmse_e1"]/shape_metric["std_rel_rmse_e1"])
    print('ratio std_rel_rmse_e1', ratio_std_rel_rmse_e1)
    ratio_rmse_e2 = abs(1.-paper_metrics["test_metrics"]["shape_results_dict"]["rmse_e2"]/shape_metric["rmse_e2"])
    print('ratio rmse_e2', ratio_rmse_e2)
    ratio_std_rmse_e2 = abs(1.-paper_metrics["test_metrics"]["shape_results_dict"]["std_rmse_e2"]/shape_metric["std_rmse_e2"])
    print('ratio std_rmse_e2', ratio_std_rmse_e2)
    ratio_rmse_R2_meanR2=abs(1.-paper_metrics["test_metrics"]["shape_results_dict"]["rmse_R2_meanR2"]/shape_metric["rmse_R2_meanR2"])
    print('ratio rmse R2 mean R2', ratio_rmse_R2_meanR2)

    print(" ")
    print("Main branch vs. Tag 1.2.0. (paper)")
    m_ratio_rmse_e1 = abs(1.-paper_metrics["test_metrics"]["shape_results_dict"]["rmse_e1"]/main_metrics["test_metrics"]["shape_results_dict"]["rmse_e1"])
    print('ratio rmse e1', m_ratio_rmse_e1)
    m_ratio_std_rmse_e1 = abs(1.-paper_metrics["test_metrics"]["shape_results_dict"]["std_rmse_e1"]/main_metrics["test_metrics"]["shape_results_dict"]["std_rmse_e1"])
    print('ratio std rmse e1', m_ratio_std_rmse_e1)
    m_ratio_rel_rmse_e1 = abs(1.-paper_metrics["test_metrics"]["shape_results_dict"]["rel_rmse_e1"]/main_metrics["test_metrics"]["shape_results_dict"]["rel_rmse_e1"])
    print('ratio rel_rmse_e1', m_ratio_rel_rmse_e1)
    m_ratio_std_rel_rmse_e1 = abs(1.-paper_metrics["test_metrics"]["shape_results_dict"]["std_rel_rmse_e1"]/main_metrics["test_metrics"]["shape_results_dict"]["std_rel_rmse_e1"])
    print('ratio std_rel_rmse_e1', ratio_std_rel_rmse_e1)
    m_ratio_rmse_e2 = abs(1.-paper_metrics["test_metrics"]["shape_results_dict"]["rmse_e2"]/main_metrics["test_metrics"]["shape_results_dict"]["rmse_e2"])
    print('ratio rmse_e2', m_ratio_rmse_e2)
    m_ratio_std_rmse_e2 = abs(1.-paper_metrics["test_metrics"]["shape_results_dict"]["std_rmse_e2"]/main_metrics["test_metrics"]["shape_results_dict"]["std_rmse_e2"])
    print('ratio std_rmse_e2', m_ratio_std_rmse_e2)
    m_ratio_rmse_R2_meanR2=abs(1.-paper_metrics["test_metrics"]["shape_results_dict"]["rmse_R2_meanR2"]/main_metrics["test_metrics"]["shape_results_dict"]["rmse_R2_meanR2"])
    print('ratio rmse R2 mean R2', m_ratio_rmse_R2_meanR2)
    
    assert ratio_rmse_e1 < tol
    assert ratio_std_rmse_e1 < tol
    assert ratio_rel_rmse_e1 < tol
    assert ratio_std_rel_rmse_e1 < tol
    assert ratio_rmse_e2 < tol
    assert ratio_std_rmse_e2 < tol
    assert ratio_rmse_R2_meanR2 < tol

