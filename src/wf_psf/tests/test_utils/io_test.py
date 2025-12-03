"""UNIT TESTS FOR PACKAGE MODULE: IO.

This module contains unit tests for the wf_psf.utils io module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""

import pytest
from wf_psf.utils.io import FileIOHandler
import os


@pytest.fixture
def test_file_handler(path_to_tmp_output_dir, path_to_config_dir):
    test_file_handler = FileIOHandler(path_to_tmp_output_dir, path_to_config_dir)
    test_file_handler._make_output_dir()
    test_file_handler._make_run_dir()
    test_file_handler._setup_dirs()
    return test_file_handler


def test_make_output_dir(test_file_handler, path_to_tmp_output_dir):
    assert os.path.exists(
        os.path.join(path_to_tmp_output_dir, test_file_handler.parent_output_dir)
    )


def test_make_run_dir(test_file_handler):
    assert os.path.exists(test_file_handler._run_output_dir)


def test_setup_dirs(test_file_handler):
    wf_outdirs = [
        "_config",
        "_checkpoint",
        "_log_files",
        "_metrics",
        "_optimizer",
        "_plots",
        "_psf_model",
    ]

    for odir in wf_outdirs:
        assert os.path.exists(
            os.path.join(
                test_file_handler._run_output_dir, test_file_handler.__dict__[odir]
            )
        )


def test_setup_logging_uses_files_and_fileconfig(mocker, path_to_tmp_output_dir):
    """
    Ensure that _setup_logging correctly loads logging.conf from package resources
    and calls logging.config.fileConfig with the expected arguments.
    """
    # Create handler
    fh = FileIOHandler(
        output_path=str(path_to_tmp_output_dir),
        config_path="/unused",
    )
    fh._make_output_dir()
    fh._make_run_dir()
    fh._setup_dirs()

    # --- Mock importlib.resources.files() and as_file() ---
    mock_files = mocker.patch("importlib.resources.files")
    mock_as_file = mocker.patch("importlib.resources.as_file")

    mock_config_files = mocker.MagicMock()
    mock_files.return_value = mock_config_files

    mock_conf_resource = mocker.MagicMock()
    mock_config_files.joinpath.return_value = mock_conf_resource

    mock_conf_path = mocker.MagicMock()
    mock_as_file.return_value.__enter__.return_value = mock_conf_path

    # --- Mock logging.config.fileConfig ---
    mock_fileconfig = mocker.patch("logging.config.fileConfig")

    # Run method
    fh._setup_logging()

    # --- Assertions ---
    mock_files.assert_called_once_with("wf_psf.config")
    mock_config_files.joinpath.assert_called_once_with("logging.conf")
    mock_as_file.assert_called_once_with(mock_conf_resource)

    mock_fileconfig.assert_called_once()
    args, kwargs = mock_fileconfig.call_args

    assert args[0] is mock_conf_path
    assert "defaults" in kwargs
    assert "filename" in kwargs["defaults"]
    assert kwargs["disable_existing_loggers"] is False

    logfile = kwargs["defaults"]["filename"]
    assert logfile.startswith(str(fh._run_output_dir))
    assert logfile.endswith(".log")
