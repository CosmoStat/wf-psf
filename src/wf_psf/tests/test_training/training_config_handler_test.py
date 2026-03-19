"""Unit tests for the TrainingConfigHandler class in training_config_handler.py"""

import pytest
from wf_psf.data.data_config_handler import DataConfigHandler
from wf_psf.utils.configs_handler import ConfigHandler
from wf_psf.training.training_config_handler import TrainingConfigHandler
from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.utils.io import FileIOHandler
import os


@pytest.fixture
def mock_training_conf(mocker):
    return RecursiveNamespace(
        training=RecursiveNamespace(
            id_name="_test_",
            data_config="data_config.yaml",
            load_data_on_init=True,
            metrics_config=None,
            model_params=RecursiveNamespace(
                model_name="poly",
                n_bins_lambda=10,
                param_hparams=RecursiveNamespace(
                    random_seed=3877572,
                ),
                nonparam_hparams=RecursiveNamespace(
                    d_max_nonparam=5,
                ),
            ),
            training_hparams=RecursiveNamespace(batch_size=32),
        ),
    )


@pytest.fixture
def mock_data_params(mocker):
    # Create a mock object
    data_params = mocker.Mock()

    return data_params


@pytest.fixture
def mock_file_handler(mocker, tmp_path):
    # Create a temporary directory
    temp_dir = tmp_path / "temp_dir"
    temp_dir.mkdir()

    # Create a mock FileIOHandler instance
    mock_fh = FileIOHandler(
        output_path="/path/to/output",
        config_path=str(temp_dir),
    )

    # Mock the methods of FileIOHandler as needed
    mocker.patch.object(
        mock_fh, "get_checkpoint_dir", return_value="/path/to/checkpoints"
    )
    mocker.patch.object(mock_fh, "get_optimizer_dir", return_value="/path/to/optimizer")
    mocker.patch.object(mock_fh, "get_psf_model_dir", return_value="/path/to/psf_model")
    mocker.patch.object(mock_fh, "copy_conffile_to_output_dir")

    return mock_fh


def test_training_handler_inherits_from_base():
    """Test TrainingConfigHandler inherits from ConfigHandler."""
    assert issubclass(TrainingConfigHandler, ConfigHandler)


def test_training_handler_has_run():
    """Test TrainingConfigHandler implements run()."""
    assert hasattr(TrainingConfigHandler, "run")
    assert callable(TrainingConfigHandler.run)


def test_model_params_renamed(mock_training_conf):
    """Test training_model_params renamed to model_params."""

    # Should have model_params
    assert hasattr(mock_training_conf.training, "model_params")

    # Should NOT have old name
    assert not hasattr(mock_training_conf.training, "training_model_params")


def test_training_config_handler_init(
    mocker, mock_training_conf, mock_file_handler, mock_data_params
):
    # Mock read_conf function
    mocker.patch(
        "wf_psf.training.training_config_handler.read_conf",
        return_value=mock_training_conf,
    )

    # Patch DataConfigHandler to return an object with .params
    mock_data_conf_instance = mocker.Mock()
    mock_data_conf_instance.params = mock_data_params
    data_config_handler_patch = mocker.patch(
        "wf_psf.training.training_config_handler.DataConfigHandler",
        return_value=mock_data_conf_instance,
    )

    # Mock SimPSF instance
    mock_simPSF_instance = mocker.Mock(name="SimPSFToolkist")
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

    # Assert that DataConfigHandler was instantiated with the correct path
    data_config_handler_patch.assert_called_once_with(
        os.path.join(
            mock_file_handler.config_path,
            training_config_handler.training_conf.training.data_config,
        )
    )

    # Assert that .params was correctly read
    assert training_config_handler.data_params == mock_data_conf_instance
