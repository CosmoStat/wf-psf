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
    ResourcesConfig,
)
from wf_psf.quality_control.config import (
    parse_resources_config,
    parse_rejection_policy_config,
)
from wf_psf.quality_control.resource_identifier import ResourceIdentifier


def load_config(config_file: str) -> QualityControlConfig:
    handler = QualityControlConfigHandler(Path(__file__).parent / "data" / config_file)
    return handler.load()


# Test for config loading and parsers
def test_quality_control_config_loading():
    config = load_config("valid/quality_control.yaml")

    pixel_mask_params = {
        "aperture": {
            "type": "circular",
            "centre": "stamp_centre",
            "radius": 2.6,
            "unit": "sigma",
        },
    }

    assert isinstance(config, QualityControlConfig)
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

    assert "pixel_mask" in config.metrics
    assert isinstance(config.metrics["pixel_mask"], QualityMetricConfig)
    assert config.metrics["pixel_mask"].enabled is True
    assert config.metrics["pixel_mask"].params == pixel_mask_params
    assert config.metrics["pixel_mask"].required_resources == []

    assert isinstance(config.metrics["goodness_of_fit"], QualityMetricConfig)
    assert config.metrics["goodness_of_fit"].required_resources == (
        [ResourceIdentifier.from_string("psf_models.standard")]
    )
    assert config.metrics["goodness_of_fit"].params == {
        "normalize_residuals": True,
    }

    assert isinstance(config.rejection["pixel_mask"], RejectionPolicyConfig)
    assert config.rejection["pixel_mask"].policy == {
        "threshold": {
            "value": 3.0,
        },
    }

    assert isinstance(config.reporting, ReportingConfig)
    assert config.reporting.save_metrics is True


## Tests for parsing resource configurations
def test_parse_resources_config():
    config = {
        "psf_models": {"standard": {"inference_config": "inference_standard.yaml"}}
    }

    resources = parse_resources_config(config)

    assert resources.available["psf_models"]["standard"]["inference_config"] == (
        "inference_standard.yaml"
    )


def test_required_resources_must_be_a_list():
    with pytest.raises(
        TypeError,
        match="Required resources for metric 'goodness_of_fit' must be a list.",
    ):
        load_config("invalid/metric_required_resources_invalid_type.yaml")


def test_required_resources_element_must_be_a_str():
    with pytest.raises(
        TypeError,
        match="Required resources for metric 'goodness_of_fit' must contain only strings.",
    ):
        load_config("invalid/metric_required_resources_invalid_element.yaml")


## Tests for parsing metrics configurations
def test_metrics_minimal():
    config = load_config("valid/metric_minimal.yaml")

    assert config.metrics["pixel_mask"].enabled is True
    assert config.metrics["pixel_mask"].params == {}
    assert config.rejection == {}
    assert config.reporting.save_metrics is False
    assert config.reporting.log_statistics is False


def test_metric_enabled_must_be_boolean():
    with pytest.raises(
        TypeError,
        match="Metric `enabled` flag for 'pixel_mask' must be boolean",
    ):
        load_config("invalid/metric_invalid_enabled.yaml")


def test_metrics_configuration_must_be_mapping():
    with pytest.raises(TypeError, match="Metrics configuration must be a mapping"):
        load_config("invalid/metric_invalid_type.yaml")


## Tests for parsing rejection policy configurations
def test_rejection_policy_configuration_valid():
    result = parse_rejection_policy_config(
        {
            "goodness_of_fit": {
                "enabled": True,
                "diagnostic": "reduced_chi_square",
                "policy": {
                    "threshold": {
                        "value": 0.25,
                    }
                },
            }
        }
    )

    assert result["goodness_of_fit"] == RejectionPolicyConfig(
        enabled=True,
        diagnostic="reduced_chi_square",
        policy={
            "threshold": {
                "value": 0.25,
            }
        },
    )


def test_rejection_policy_configuration_must_be_mapping():
    with pytest.raises(
        TypeError, match="Rejection policy configuration must be a mapping"
    ):
        load_config("invalid/rejection_invalid_type.yaml")


def test_rejection_policy_configuration_metric_must_be_mapping():
    with pytest.raises(
        TypeError,
        match="Rejection policy configuration 'goodness_of_fit' must be a mapping",
    ):
        parse_rejection_policy_config({"goodness_of_fit": 0.25})


def test_rejection_policy_enabled_defaults_to_false():
    policies = parse_rejection_policy_config(
        {
            "goodness_of_fit": {
                "diagnostic": "reduced_chi_square",
                "policy": {"threshold": {"value": 0.25}},
            }
        }
    )

    assert policies["goodness_of_fit"] == RejectionPolicyConfig(enabled=False)


def test_rejection_policy_metric_enabled_flag_must_be_boolean():
    with pytest.raises(
        TypeError,
        match="Rejection policy `enabled` flag for 'goodness_of_fit' must be boolean.",
    ):
        parse_rejection_policy_config(
            {
                "goodness_of_fit": {
                    "enabled": "foo",
                    "diagnostic": "reduced_chi_square",
                    "policy": "threshold",
                }
            }
        )


def test_rejection_policy_disabled_policies_are_skipped():
    rejection_policy = {
        "goodness_of_fit": {
            "enabled": False,
            "diagnostic": "None",
            "policy": "not a mapping",
        },
    }

    policies = parse_rejection_policy_config(rejection_policy)

    assert policies["goodness_of_fit"].enabled is False
    assert policies["goodness_of_fit"].diagnostic is None
    assert policies["goodness_of_fit"].policy == {}


def test_rejection_policy_must_specify_non_empty_diagnostic():
    rejection_policy = {
        "goodness_of_fit": {
            "enabled": True,
            "diagnostic": None,
            "policy": "not a mapping",
        },
    }

    with pytest.raises(
        ValueError,
        match="Rejection policy configuration for 'goodness_of_fit' must specify a non-empty `diagnostic`.",
    ):
        parse_rejection_policy_config(rejection_policy)


def test_rejection_policy_must_specify_policy():
    with pytest.raises(
        ValueError,
        match="must specify a `policy`",
    ):
        parse_rejection_policy_config(
            {"goodness_of_fit": {"enabled": True, "diagnostic": "reduced_chi_square"}}
        )


def test_rejection_policy_must_be_mapping():
    with pytest.raises(
        TypeError,
        match="Rejection policy `policy` field for 'goodness_of_fit' must be a mapping",
    ):
        parse_rejection_policy_config(
            {
                "goodness_of_fit": {
                    "enabled": True,
                    "diagnostic": "reduced_chi_square",
                    "policy": "threshold",
                }
            }
        )


@pytest.mark.parametrize(
    "policy",
    [
        {},
        {
            "threshold": {"value": 0.25},
            "quantile": {"value": 0.95},
        },
    ],
)
def test_rejection_policy_must_specify_exactly_one_policy(policy):
    with pytest.raises(
        ValueError,
        match="must specify exactly one policy type",
    ):
        parse_rejection_policy_config(
            {
                "goodness_of_fit": {
                    "enabled": True,
                    "diagnostic": "reduced_chi_square",
                    "policy": policy,
                }
            }
        )


def test_rejection_policy_identifier_must_be_string():
    rejection_policy = {
        "goodness_of_fit": {
            "enabled": True,
            "diagnostic": "reduced_chi_square",
            "policy": {123: {}},
        }
    }

    with pytest.raises(
        TypeError,
        match="Rejection policy identifier '123' for 'goodness_of_fit' must be a string.",
    ):
        parse_rejection_policy_config(rejection_policy)


def test_rejection_policy_params_must_be_mapping():
    rejection_policy = {
        "goodness_of_fit": {
            "enabled": True,
            "diagnostic": "reduced_chi_square",
            "policy": {"threshold": 3},
        }
    }

    with pytest.raises(
        TypeError,
        match="Rejection policy parameters for 'goodness_of_fit' must be a mapping.",
    ):
        parse_rejection_policy_config(rejection_policy)


## Tests for parsing reporting configurations
def test_reporting_configuration_must_be_mapping():
    with pytest.raises(
        TypeError,
        match="Reporting configuration must be a mapping",
    ):
        load_config("invalid/reporting_invalid_type.yaml")
