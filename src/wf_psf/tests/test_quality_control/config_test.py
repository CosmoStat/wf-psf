"""UNIT TESTS FOR PACKAGE MODULE: Quality Control Configuration

This module contains unit tests for the quality control configuration module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from pathlib import Path
import pytest
from wf_psf.quality_control.config import (
    QualityControlConfig,
    QualityControlConfigHandler,
    QualityMetricConfig,
    RejectionPolicyConfig,
    ReportingConfig,
)


def load_config(config_file: str) -> QualityControlConfig:
    handler = QualityControlConfigHandler(Path(__file__).parent / "data" / config_file)
    return handler.load()


def test_quality_control_config_loading():
    config = load_config("valid/quality_control.yaml")

    assert "mask_obscuration" in config.metrics
    assert isinstance(config.metrics["mask_obscuration"], QualityMetricConfig)
    assert config.metrics["mask_obscuration"].enabled is True

    assert isinstance(config.metrics["goodness_of_fit"], QualityMetricConfig)
    assert config.metrics["goodness_of_fit"].params["inference_config"] == (
        "inference_config.yaml"
    )

    assert isinstance(config.rejection["mask_obscuration"], RejectionPolicyConfig)
    assert config.rejection["mask_obscuration"].threshold == 0.25

    assert isinstance(config.reporting, ReportingConfig)
    assert config.reporting.save_metrics is True


def test_metrics_minimal():
    config = load_config("valid/metric_minimal.yaml")

    assert config.metrics["mask_obscuration"].enabled is True
    assert config.metrics["mask_obscuration"].params == {}
    assert config.rejection == {}
    assert config.reporting.save_metrics is False
    assert config.reporting.log_statistics is False


def test_metric_enabled_must_be_boolean():
    with pytest.raises(
        TypeError,
        match="Metric enabled flag for 'mask_obscuration' must be boolean",
    ):
        load_config("invalid/metric_invalid_enabled.yaml")


def test_metrics_configuration_must_be_mapping():
    with pytest.raises(TypeError, match="Metrics configuration must be a mapping"):
        load_config("invalid/metric_invalid_type.yaml")


def test_rejection_configuration_must_be_mapping():
    with pytest.raises(
        TypeError, match="Rejection policy configuration must be a mapping"
    ):
        load_config("invalid/rejection_invalid_type.yaml")


def test_reporting_configuration_must_be_mapping():
    with pytest.raises(
        TypeError,
        match="Reporting configuration must be a mapping",
    ):
        load_config("invalid/reporting_invalid_type.yaml")
