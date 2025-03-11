"""UNIT TESTS FOR PACKAGE MODULE: CONFIGS_HANDLER.

This module contains unit tests for the wf_psf.utils configs_handler module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""

import pytest
from wf_psf.utils import configs_handler
from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.utils.io import FileIOHandler
from wf_psf.utils.configs_handler import TrainingConfigHandler, DataConfigHandler
from wf_psf.data.training_preprocessing import DataHandler
import os


@pytest.fixture
def mock_training_model_params():
    return RecursiveNamespace(n_bins_lda=10)


@pytest.fixture
def mock_data_read_conf(mocker):
    return mocker.patch(
        "wf_psf.utils.configs_handler.read_conf",
        return_value=RecursiveNamespace(
            data=RecursiveNamespace(
                training=RecursiveNamespace(
                    data_dir="/path/to/train_data", file="train_data.npy"
                ),
                test=RecursiveNamespace(
                    data_dir="/path/to/test_data",
                    file="test_data.npy",
                ),
            ),
        ),
    )


@pytest.fixture
def mock_training_conf(mocker):
    return RecursiveNamespace(
        training=RecursiveNamespace(
            id_name="_test_",
            data_config="data_config.yaml",
            metrics_config=None,
            model_params=RecursiveNamespace(
                model_name="poly",
                param_hparams=RecursiveNamespace(
                    random_seed=3877572,
                ),
                nonparam_hparams=RecursiveNamespace(
                    d_max_nonparam=5,
                ),
            ),
            training_hparams=RecursiveNamespace(n_epochs_params=[2, 2]),
        ),
    )


@pytest.fixture
def mock_data_conf(mocker):
    # Create a mock object
    data_conf = mocker.Mock()

    # Set attributes on the mock object
    data_conf.training_data = "value1"
    data_conf.test_data = "value2"

    return data_conf


@pytest.fixture
def mock_training_config_handler(mocker, mock_training_conf, mock_data_conf):
    # Create a mock instance of TrainingConfigHandler
    mock_instance = mocker.Mock(spec=TrainingConfigHandler)

    # Set attributes of the mock instance as needed
    mock_instance.training_conf = mock_training_conf
    mock_instance.data_conf = mock_data_conf
    mock_instance.checkpoint_dir = "/mock/checkpoint/dir"
    mock_instance.optimizer_dir = "/mock/optimizer/dir"
    mock_instance.psf_model_dir = "/mock/psf/model/dir"

    return mock_instance


@configs_handler.register_configclass
class RegisterConfigClass:
    ids = ("test_conf",)

    def __init__(self, config_params, file_handler):
        self.config_param = config_params
        self.file_handler = file_handler


def test_register_configclass():
    assert configs_handler.CONFIG_CLASS["test_conf"] == RegisterConfigClass


def test_set_run_config():
    config_class = configs_handler.set_run_config("test_conf")
    assert config_class == RegisterConfigClass

    config_class = configs_handler.set_run_config("training_conf")
    assert config_class == configs_handler.TrainingConfigHandler

    config_class = configs_handler.set_run_config("metrics_conf")
    assert config_class == configs_handler.MetricsConfigHandler

    config_class = configs_handler.set_run_config("plotting_conf")
    assert config_class == configs_handler.PlottingConfigHandler


def test_get_run_config(path_to_repo_dir, path_to_tmp_output_dir, path_to_config_dir):
    test_file_handler = FileIOHandler(
        path_to_repo_dir, path_to_tmp_output_dir, path_to_config_dir
    )

    config_class = configs_handler.get_run_config(
        "test_conf", "fake_config.yaml", test_file_handler
    )

    assert type(config_class) is RegisterConfigClass


def test_data_config_handler_init(
    mock_training_model_params, mock_data_read_conf, mocker
):
    # Mock read_conf function
    mock_data_read_conf()

    # Mock SimPSF instance
    mock_simPSF_instance = mocker.Mock(name="SimPSFToolkit")
    mocker.patch(
        "wf_psf.psf_models.psf_models.simPSF", return_value=mock_simPSF_instance
    )

    # Patch the initialize and load_dataset methods inside DataHandler
    mocker.patch.object(DataHandler, "initialize")

    # Create DataConfigHandler instance
    data_config_handler = DataConfigHandler(
        "/path/to/data_config.yaml", mock_training_model_params
    )

    # Check that attributes are set correctly
    assert isinstance(data_config_handler.data_conf, RecursiveNamespace)
    assert isinstance(data_config_handler.simPSF, object)
    assert (
        data_config_handler.training_data.n_bins_lambda
        == mock_training_model_params.n_bins_lda
    )
    assert (
        data_config_handler.test_data.n_bins_lambda
        == mock_training_model_params.n_bins_lda
    )


def test_training_config_handler_init(mocker, mock_training_conf, mock_file_handler):
    # Mock read_conf function
    mocker.patch(
        "wf_psf.utils.configs_handler.read_conf", return_value=mock_training_conf
    )

    # Mock data_conf instance
    mock_data_conf = mocker.patch("wf_psf.utils.configs_handler.DataConfigHandler")

    # Mock SimPSF instance
    mock_simPSF_instance = mocker.Mock(name="SimPSFToolkit")
    mocker.patch(
        "wf_psf.psf_models.psf_models.simPSF", return_value=mock_simPSF_instance
    )

    # Initialize TrainingConfigHandler with the mock_file_handler
    training_config_handler = TrainingConfigHandler(
        "/path/to/training_config.yaml", mock_file_handler
    )

    # Assertions
    mock_file_handler.copy_conffile_to_output_dir.assert_called_once_with(
        training_config_handler.training_conf.training.data_config
    )
    mock_file_handler.get_checkpoint_dir.assert_called_once_with(
        mock_file_handler._run_output_dir
    )
    mock_file_handler.get_optimizer_dir.assert_called_once_with(
        mock_file_handler._run_output_dir
    )
    mock_file_handler.get_psf_model_dir.assert_called_once_with(
        mock_file_handler._run_output_dir
    )
    assert training_config_handler.training_conf == mock_training_conf
    assert training_config_handler.file_handler == mock_file_handler
    assert (
        training_config_handler.file_handler.repodir_path
        == mock_file_handler.repodir_path
    )

    mock_data_conf.assert_called_once_with(
        os.path.join(
            mock_file_handler.config_path,
            training_config_handler.training_conf.training.data_config,
        ),
        training_config_handler.training_conf.training.model_params,
    )
    assert training_config_handler.data_conf == mock_data_conf.return_value


def test_run_method_calls_train_with_correct_arguments(
    mocker, mock_training_conf, mock_data_conf
):
    # Patch the TrainingConfigHandler.__init__() method
    mocker.patch(
        "wf_psf.utils.configs_handler.TrainingConfigHandler.__init__", return_value=None
    )
    mock_th = TrainingConfigHandler(None, None)
    # Set attributes of the mock_th
    mock_th.training_conf = mock_training_conf
    mock_th.data_conf = mock_data_conf
    mock_th.data_conf.training_data = mock_data_conf.training_data
    mock_th.data_conf.test_data = mock_data_conf.test_data
    mock_th.checkpoint_dir = "/mock/checkpoint/dir"
    mock_th.optimizer_dir = "/mock/optimizer/dir"
    mock_th.psf_model_dir = "/mock/psf/model/dir"

    # Patch the train.train() function
    mock_train_function = mocker.patch("wf_psf.training.train.train")

    # Create a spy for the run method
    spy = mocker.spy(mock_th, "run")

    # Call the run method
    mock_th.run()

    # Assert that run() is called once
    spy.assert_called_once()

    # Assert that train.train() is called with the correct arguments
    mock_train_function.assert_called_once_with(
        mock_th.training_conf.training,
        mock_th.data_conf,
        mock_th.checkpoint_dir,
        mock_th.optimizer_dir,
        mock_th.psf_model_dir,
    )


def test_MetricsConfigHandler_weights_basename_filepath(
    path_to_repo_dir, path_to_tmp_output_dir, path_to_config_dir
):
    test_file_handler = FileIOHandler(
        path_to_repo_dir, path_to_tmp_output_dir, path_to_config_dir
    )

    metrics_config_file = "validation/main_random_seed/config/metrics_config.yaml"

    metrics_object = configs_handler.MetricsConfigHandler(
        os.path.join(path_to_config_dir, metrics_config_file), test_file_handler
    )
    weights_filepath = metrics_object.weights_basename_filepath

    assert (
        weights_filepath
        == "src/wf_psf/tests/data/validation/main_random_seed/checkpoint/checkpoint*_poly*_sample_w_bis1_2k_cycle2*"
    )
