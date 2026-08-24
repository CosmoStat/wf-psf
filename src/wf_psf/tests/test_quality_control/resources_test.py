"""UNIT TESTS FOR PACKAGE MODULE: Quality Control Resources

This module contains unit tests for the quality control resources module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import pytest
from wf_psf.quality_control.config import QualityMetricConfig
from wf_psf.quality_control.resources import Resources


def test_get_required(qc_config_factory):
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
    resources = Resources(config)
    assert resources.get_required() == {"psf_models.standard"}


def test_get_required_combines_unique_resources(qc_config_factory):
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

    resources = Resources(config)
    assert resources.get_required() == {
        "psf_models.standard",
        "psf_models.oversampled",
    }


# Resource resolution orchestration tests
@pytest.mark.parametrize(
    ("required_resources", "provided", "expected_resolved"),
    [
        (
            {"psf_models.standard"},
            {"psf_models.standard": [1, 1, 1, 1]},
            {"psf_models.standard": [1, 1, 1, 1]},
        ),
        (
            {"psf_models.standard"},
            {
                "psf_models.standard": [1, 1, 1, 1],
                "psf_models.oversampled": [2, 2, 2, 2],
            },
            {"psf_models.standard": [1, 1, 1, 1]},
        ),
        (
            [],
            {},
            {},
        ),
    ],
)
def test_resolve_resources(
    qc_config_factory,
    required_resources,
    provided,
    expected_resolved,
):
    config = qc_config_factory(required_resources=required_resources)
    resources = Resources(config)

    assert resources.resolve(provided) == expected_resolved


def test_resolve_resources_missing(qc_config_factory):
    config = qc_config_factory(
        required_resources=["psf_models.standard"],
    )
    resources = Resources(config)

    with pytest.raises(
        NotImplementedError,
        match="Required resources are not available",
    ):
        resources.resolve()
