"""FIXTURES FOR GENERATING TESTS FOR WF-PSF UTILS PACKAGE: CONFTEST.

This module contains fixtures to use in unit tests for the 
wf_psf utils package.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""
import pytest
import os

cwd = os.getcwd()


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
