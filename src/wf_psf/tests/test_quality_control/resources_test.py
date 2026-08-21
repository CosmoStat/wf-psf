"""UNIT TESTS FOR PACKAGE MODULE: Quality Control Resources

This module contains unit tests for the quality control resources module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import pytest
from wf_psf.quality_control.config import QualityMetricConfig
from wf_psf.quality_control.resources import (
    get_required_resources,
    resolve_resources,
    resolve_required_resources,
)


def test_get_required_resources(qc_config_factory):
    config = qc_config_factory(
        metrics={
            "mask_obscuration": QualityMetricConfig(
                enabled=True,
                required_resources=[],
            ),
            "goodness_of_fit": QualityMetricConfig(
                enabled=True,
                required_resources=["psf_models.standard"],
            ),
            "shapes": QualityMetricConfig(
                enabled=False,
                required_resources=["psf_models.oversampled"],
            ),
        }
    )

    assert get_required_resources(config) == {"psf_models.standard"}


def test_get_required_resources_combines_unique_resources(qc_config_factory):
    config = qc_config_factory(
        metrics={
            "metric_a": QualityMetricConfig(
                enabled=True,
                required_resources=["psf_models.standard"],
            ),
            "metric_b": QualityMetricConfig(
                enabled=True,
                required_resources=[
                    "psf_models.standard",
                    "psf_models.oversampled",
                ],
            ),
        }
    )

    assert get_required_resources(config) == {
        "psf_models.standard",
        "psf_models.oversampled",
    }


@pytest.mark.parametrize(
    ("required", "provided", "expected_resolved", "expected_missing"),
    [
        (
            {"psf_models.standard"},
            {"psf_models.standard": [1, 1, 1, 1]},
            {"psf_models.standard": [1, 1, 1, 1]},
            set(),
        ),
        (
            {"psf_models.standard"},
            {},
            {},
            {"psf_models.standard"},
        ),
        (
            {"psf_models.standard", "psf_models.oversampled"},
            {"psf_models.standard": [1, 1, 1, 1]},
            {"psf_models.standard": [1, 1, 1, 1]},
            {"psf_models.oversampled"},
        ),
        (
            {"psf_models.standard"},
            {
                "psf_models.standard": [1, 1, 1, 1],
                "psf_models.oversampled": [2, 2, 2, 2],
            },
            {"psf_models.standard": [1, 1, 1, 1]},
            set(),
        ),
        (
            set(),
            {},
            {},
            set(),
        ),
    ],
)
def test_resolve_resources(
    required,
    provided,
    expected_resolved,
    expected_missing,
):
    assert resolve_resources(required, provided) == (
        expected_resolved,
        expected_missing,
    )


# Resource resolution orchestration tests
def test_resolve_required_resources(qc_config_factory):
    config = qc_config_factory(
        required_resources=["psf_models.standard"],
    )
    model_psfs = [1, 1, 1, 1]
    provided = {"psf_models.standard": model_psfs}

    assert resolve_required_resources(config, provided) == {
        "psf_models.standard": model_psfs,
    }


def test_resolve_required_resources_missing(qc_config_factory):
    config = qc_config_factory(
        required_resources=["psf_models.standard"],
    )

    with pytest.raises(
        NotImplementedError,
        match="Required resources are not available",
    ):
        resolve_required_resources(config)


def test_resolve_required_resources_no_resources_required(qc_config_factory):
    config = qc_config_factory(required_resources=[])

    assert resolve_required_resources(config) == {}
