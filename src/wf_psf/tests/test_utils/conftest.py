"""FIXTURES FOR GENERATING TESTS FOR WF-PSF UTILS PACKAGE: CONFTEST.

This module contains fixtures to use in unit tests for the 
wf_psf utils package.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""

import pytest
import os
from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.utils.io import FileIOHandler

cwd = os.getcwd()

training_config = RecursiveNamespace(
    id_name="_sample_w_bis1_2k",
    data_config="data_config.yaml",
    metrics_config="metrics_config.yaml",
)


@pytest.fixture(scope="class")
def path_to_repo_dir():
    return cwd


@pytest.fixture
def path_to_test_dir(path_to_repo_dir):
    return os.path.join(path_to_repo_dir, "src", "wf_psf", "tests")


@pytest.fixture
def path_to_tmp_output_dir(tmp_path):
    return tmp_path


@pytest.fixture
def path_to_config_dir(path_to_test_dir):
    return os.path.join(path_to_test_dir, "data")


@pytest.fixture
def mock_file_handler(mocker, tmp_path):
    # Create a temporary directory
    temp_dir = tmp_path / "temp_dir"
    temp_dir.mkdir()

    # Create a mock FileIOHandler instance
    mock_fh = FileIOHandler(
        repodir_path="/path/to/repo",
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


@pytest.fixture()
def mock_config_dir(tmp_path):
    # Use os.path.join to construct the file path
    mock_data_conf_dir = tmp_path / "tmp_config_dir"
    mock_data_conf_dir.mkdir()
    return mock_data_conf_dir


@pytest.fixture(scope="function")
def mock_data_config(mock_config_dir):
    # Create a mock data configuration
    mock_data_conf_content = """
    data:
       training:
          data_dir: data/mock_dataset/
          file: train_data.npy
       test:
          data_dir: data/mock_dataset/
          file: test_data.npy
    """

    mock_data_conf_path = mock_config_dir / "data_config.yaml"

    # Write the mock training configuration to a file
    mock_data_conf_path.write_text(mock_data_conf_content)

    return mock_data_conf_path
