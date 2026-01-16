"""FIXTURES FOR GENERATING TESTS FOR WF-PSF MODULES: CONFTEST.

This module contains fixtures to use in unit tests for
various wf_psf packages.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""

import pytest
from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.training.train import TrainingParamsHandler
from wf_psf.utils.configs_handler import DataConfigHandler
from wf_psf.psf_models import psf_models
from wf_psf.data.training_preprocessing import DataHandler

training_config = RecursiveNamespace(
    id_name="-coherent_euclid_200stars",
    data_config="data_config.yaml",
    metrics_config="metrics_config.yaml",
    model_params=RecursiveNamespace(
        model_name="poly",
        n_bins_lda=8,
        output_Q=3,
        oversampling_rate=3,
        output_dim=32,
        pupil_diameter=256,
        use_prior=True,
        correct_centroids=False,
        sigma_centroid_window=2.5,
        obscuration_rotation_angle=0,
        add_ccd_misalignments=False,
        ccd_misalignments_input_path="/Users/tl255879/Documents/research/Euclid/real_data/CCD_missalignments/tiles.npy",
        use_sample_weights=True,
        sample_weights_sigmoid=RecursiveNamespace(
            apply_sigmoid=True,
            sigmoid_max_val=5.0,
            sigmoid_power_k=1.0,
        ),
        interpolation_type=None,
        sed_interp_pts_per_bin=0,
        sed_extrapolate=True,
        sed_interp_kind="linear",
        sed_sigma=0,
        x_lims=[0.0, 1000.0],
        y_lims=[0.0, 1000.0],
        pix_sampling=12,
        tel_diameter=1.2,
        tel_focal_length=24.5,
        euclid_obsc=True,
        LP_filter_length=3,
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
        loss="mse",
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

data_conf_object = RecursiveNamespace(
    data=RecursiveNamespace(
        training=RecursiveNamespace(
            data_dir="data",
            file="coherent_euclid_dataset/train_Euclid_res_200_TrainStars_id_001.npy",
        ),
        test=RecursiveNamespace(
            data_dir="data",
            file="coherent_euclid_dataset/test_Euclid_res_id_001.npy",
        ),
    ),
)


@pytest.fixture(scope="module", params=[training_config])
def training_params():
    return TrainingParamsHandler(training_config)


@pytest.fixture(scope="module")
def training_data():
    return DataHandler(
        "training",
        data_conf_object.data,
        psf_models.simPSF(training_config.model_params),
        training_config.model_params.n_bins_lda,
    )


@pytest.fixture(scope="module")
def test_data():
    return DataHandler(
        "test",
        data_conf_object.data,
        psf_models.simPSF(training_config.model_params),
        training_config.model_params.n_bins_lda,
    )


@pytest.fixture(scope="module")
def test_dataset(test_data):
    return test_data.dataset


@pytest.fixture(scope="function")
def data_conf(mocker, training_data, test_data):
    # Patch the DataConfigHandler.__init__() method
    mocker.patch(
        "wf_psf.utils.configs_handler.DataConfigHandler.__init__", return_value=None
    )
    mock_data_conf = DataConfigHandler(None, None)
    mock_data_conf = mocker.Mock()
    mock_data_conf.training_data = training_data
    mock_data_conf.test_data = test_data

    return mock_data_conf


@pytest.fixture(scope="module")
def psf_model():
    return psf_models.get_psf_model(
        training_config.model_params,
        training_config.training_hparams,
    )
