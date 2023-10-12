"""UNIT TESTS FOR PACKAGE MODULE: IO.

This module contains unit tests for the wf_psf.utils io module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""

import pytest
from wf_psf.utils.io import FileIOHandler
import os


@pytest.fixture
def test_file_handler(path_to_repo_dir, path_to_tmp_output_dir, path_to_config_dir):
    test_file_handler = FileIOHandler(
        path_to_repo_dir, path_to_tmp_output_dir, path_to_config_dir
    )
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
