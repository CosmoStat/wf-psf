"""UNIT TESTS FOR PACKAGE MODULE: CONFIGS_HANDLER.

This module contains unit tests for the wf_psf.utils configs_handler module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""

import pytest
from wf_psf.utils import configs_handler
from wf_psf.utils.io import FileIOHandler
import os


@configs_handler.register_configclass
class TestClass:
    ids = ("test_conf",)

    def __init__(self, config_params, file_handler):
        self.config_param = config_params
        self.file_handler = file_handler


def test_register_configclass():
    assert configs_handler.CONFIG_CLASS["test_conf"] == TestClass


def test_set_run_config():
    config_class = configs_handler.set_run_config("test_conf")
    assert config_class == TestClass

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

    assert type(config_class) is TestClass


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
