"""UNIT TESTS FOR PACKAGE MODULE: Quality Control Configuration

This module contains unit tests for the quality control configuration module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from pathlib import Path
from wf_psf.quality_control.config import QualityControlConfigHandler

def get_quality_control_config_object(config_file):
    fixture_path = (
        Path(__file__).parent
        / "data"
        / config_file
    )

    handler = QualityControlConfigHandler(fixture_path)
    config = handler.load()
    return config

def test_quality_control_config_loading():

    config = get_quality_control_config_object(
        "quality_control.yaml"
    )

    assert "mask_obscuration" in config.metrics
    assert config.metrics["mask_obscuration"].enabled is True
    assert config.metrics["goodness_of_fit"].params["inference_config"] == (
        "inference_config.yaml"
    )
    assert config.rejection["mask_obscuration"].threshold == 0.25
    assert config.reporting.save_metrics is True

def test_metrics_minimal():

    config = get_quality_control_config_object(
        "metrics_minimal.yaml"
    )

    assert config.metrics["mask_obscuration"].enabled is True
    assert config.rejection == {}
    assert config.reporting.save_metrics is False
    assert config.reporting.log_statistics is False


def test_goodness_of_fit_extended():
    config = get_quality_control_config_object(
        "goodness_of_fit_extended.yaml"
    )

    assert config.metrics["goodness_of_fit"].enabled is True
    assert config.metrics["goodness_of_fit"].params["inference_config"] == (
        "inference_config.yaml"
    )
    assert config.metrics["goodness_of_fit"].params["model_cache"] == (
        True
    )