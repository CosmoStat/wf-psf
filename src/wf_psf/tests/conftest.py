"""FIXTURES FOR GENERATING TESTS FOR WF-PSF MODULES: CONFTEST.

This module contains fixtures to use in unit tests for 
various wf_psf packages.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""
import pytest
from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.training.train import TrainingParamsHandler
from wf_psf.psf_models import psf_models
from wf_psf.data.training_preprocessing import TrainingDataHandler, TestDataHandler

training_config = RecursiveNamespace(
    id_name="_sample_w_bis1_2k",
    data_config="data_config.yaml",
    metrics_config="metrics_config.yaml",
    model_params=RecursiveNamespace(
        model_name="poly",
        n_bins_lda=8,
        output_Q=3,
        oversampling_rate=3,
        output_dim=32,
        pupil_diameter=256,
        use_sample_weights=True,
        interpolation_type="None",
        sed_interp_pts_per_bin=0,
        sed_extrapolate=True,
        sed_interp_kind="linear",
        sed_sigma=0,
        x_lims=[0.0, 1000.0],
        y_lims=[0.0, 1000.0],
        param_hparams=RecursiveNamespace(
            random_seed=3877572,
            l2_param=0.0,
            n_zernikes=15,
            d_max=2,
            save_optim_history_param=True,
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
    training_hparams=RecursiveNamespace(
        n_epochs_params=[2, 2],
        n_epochs_non_params=[2, 2],
        batch_size=32,
        multi_cycle_params=RecursiveNamespace(
            total_cycles=2,
            cycle_def="complete",
            save_all_cycles=True,
            saved_cycle="cycle2",
            learning_rate_params=[1.0e-2, 1.0e-2],
            learning_rate_non_params=[1.0e-1, 1.0e-1],
            n_epochs_params=[2, 2],
            n_epochs_non_params=[2, 2],
        ),
    ),
)

data = RecursiveNamespace(
    training=RecursiveNamespace(
        data_dir="data",
        file="coherent_euclid_dataset/train_Euclid_res_200_TrainStars_id_001.npy",
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
        data_dir="data",
        file="coherent_euclid_dataset/test_Euclid_res_id_001.npy",
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
)


@pytest.fixture(scope="module", params=[training_config])
def training_params():
    return TrainingParamsHandler(training_config)


@pytest.fixture(scope="module")
def training_data():
    return TrainingDataHandler(
        data.training,
        psf_models.simPSF(training_config.model_params),
        training_config.model_params.n_bins_lda,
    )


@pytest.fixture(scope="module")
def test_data():
    return TestDataHandler(
        data.test,
        psf_models.simPSF(training_config.model_params),
        training_config.model_params.n_bins_lda,
    )


@pytest.fixture(scope="module")
def test_dataset(test_data):
    return test_data.test_dataset


@pytest.fixture(scope="module")
def psf_model():
    return psf_models.get_psf_model(
        training_config.model_params,
        training_config.training_hparams,
    )
