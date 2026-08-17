"""UNIT TESTS FOR PACKAGE MODULE: Quality Control Configuration

This module contains unit tests for the quality control configuration module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from contextlib import nullcontext as does_not_raise
from pathlib import Path
import pytest
from wf_psf.quality_control.config import (
    QualityControlConfig,
    QualityControlConfigHandler,
    QualityMetricConfig,
    RejectionPolicyConfig,
    ReportingConfig,
    ResourcesConfig,
)
from wf_psf.quality_control.config import (
    parse_resources_config,
    validate_metric_resources,
    validate_rejection_policy_metrics,
)


def load_config(config_file: str) -> QualityControlConfig:
    handler = QualityControlConfigHandler(Path(__file__).parent / "data" / config_file)
    return handler.load()


@pytest.fixture
def qc_config_factory():
    def factory(
        *,
        required_resources=None,
        rejection_metric=None,
        resources=None,
        metrics=None,
        rejection=None,
    ):
        metric_default = {
            "goodness_of_fit": QualityMetricConfig(
                enabled=True,
                required_resources=required_resources or [],
            )
        }

        resources_default = ResourcesConfig(
            available={
                "psf_models": {
                    "standard": {
                        "inference_config": "inference_standard.yaml",
                    }
                }
            }
        )

        rejection_default = {
            rejection_metric or "goodness_of_fit": RejectionPolicyConfig(
                enabled=True,
                threshold=0.25,
            )
        }

        return QualityControlConfig(
            metrics=metric_default if metrics is None else metrics,
            resources=resources_default if resources is None else resources,
            rejection=rejection_default if rejection is None else rejection,
        )

    return factory


# Test for config loading and parsers
def test_quality_control_config_loading():
    config = load_config("valid/quality_control.yaml")

    assert isinstance(config.resources, ResourcesConfig)
    assert "standard" in config.resources.available["psf_models"]
    assert "oversampled" in config.resources.available["psf_models"]
    assert (
        config.resources.available["psf_models"]["standard"]["inference_config"]
        == "inference_standard.yaml"
    )
    assert (
        config.resources.available["psf_models"]["oversampled"]["inference_config"]
        == "inference_oversampled.yaml"
    )

    assert "mask_obscuration" in config.metrics
    assert isinstance(config.metrics["mask_obscuration"], QualityMetricConfig)
    assert config.metrics["mask_obscuration"].enabled is True
    assert config.metrics["mask_obscuration"].required_resources == []

    assert isinstance(config.metrics["goodness_of_fit"], QualityMetricConfig)
    assert config.metrics["goodness_of_fit"].required_resources == (
        ["psf_models.standard"]
    )

    assert isinstance(config.rejection["mask_obscuration"], RejectionPolicyConfig)
    assert config.rejection["mask_obscuration"].threshold == 0.25

    assert isinstance(config.reporting, ReportingConfig)
    assert config.reporting.save_metrics is True


def test_parse_resources_config():
    config = {
        "psf_models": {"standard": {"inference_config": "inference_standard.yaml"}}
    }

    resources = parse_resources_config(config)

    assert resources.available["psf_models"]["standard"]["inference_config"] == (
        "inference_standard.yaml"
    )


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


# Tests for validation methods
def test_validate_metric_resources_all_valid(qc_config_factory):
    with does_not_raise():
        validate_metric_resources(qc_config_factory())


@pytest.mark.parametrize(
    "required_resource",
    [
        "psf_model_standard",
        "psf_models.standard.foo",
        "psf_models.",
        ".standard",
    ],
)
def test_validate_metric_resources_invalid_identifier(
    qc_config_factory,
    required_resource,
):
    config = qc_config_factory(required_resources=[required_resource])

    with pytest.raises(
        ValueError,
        match=(
            f"Resource identifier '{required_resource}' must have the form "
            "'<resource_type>.<resource_name>'."
        ),
    ):
        validate_metric_resources(config)


@pytest.mark.parametrize(
    "required_resource",
    [
        "images.segmentation_maps",
        "psf_models.imaginary",
    ],
)
def test_validate_metric_resources_unknown_resource(
    qc_config_factory,
    required_resource,
):
    config = qc_config_factory(required_resources=[required_resource])

    with pytest.raises(
        ValueError,
        match=(
            f"Metric 'goodness_of_fit' requires unknown resource '{required_resource}'."
        ),
    ):
        validate_metric_resources(config)


def test_validate_rejection_policy_metrics_all_valid(qc_config_factory):
    with does_not_raise():
        validate_rejection_policy_metrics(qc_config_factory())


def test_validate_rejection_policy_metrics_metric_not_found(qc_config_factory):
    config = qc_config_factory(rejection_metric="mask_obscuration")

    with pytest.raises(
        ValueError,
        match="Rejection policy configured for unknown metric 'mask_obscuration'.",
    ):
        validate_rejection_policy_metrics(config)


def test_validate_rejection_policy_metrics_metric_not_enabled(qc_config_factory):
    metric = {
        "goodness_of_fit": QualityMetricConfig(
            enabled=False,
            required_resources=[],
        )
    }

    config = qc_config_factory(metrics=metric)

    with pytest.raises(
        ValueError,
        match="Rejection policy cannot be enabled because metric 'goodness_of_fit' is disabled.",
    ):
        validate_rejection_policy_metrics(config)


# Integration tests


def test_load_config_validates_configuration_pass():
    with does_not_raise():
        load_config("valid/quality_control.yaml")


def test_load_config_validates_configuration_raise_unknown_identifier():
    with pytest.raises(
        ValueError,
        match="Metric 'goodness_of_fit' requires unknown resource",
    ):
        load_config("invalid/metric_resource_identifier_unknown.yaml")
