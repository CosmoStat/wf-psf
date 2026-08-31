"""UNIT TESTS FOR PACKAGE MODULE: Quality Control Resources

This module contains unit tests for the quality control resources module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
import pytest
from unittest.mock import Mock, call

from wf_psf.quality_control.config import QualityMetricConfig
from wf_psf.quality_control.resources import RESOURCE_PREPARERS, Resources


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


def test_prepare_resources(qc_config_factory, monkeypatch):
    def prepare_side_effect(_dataset, resource_config):
        if resource_config == config.resources.available["psf_models"]["standard"]:
            return np.ones((3, 32, 32))

        if resource_config == config.resources.available["psf_models"]["oversampled"]:
            return 2 * np.ones((3, 32, 32))

        raise AssertionError(f"Unexpected resource config: {resource_config}")

    config = qc_config_factory(
        required_resources=["psf_models.standard", "psf_models.oversampled"],
    )

    missing = {"psf_models.standard", "psf_models.oversampled"}
    dataset = {"data": [1.0, 2.0, 3.0]}

    mock_preparer = Mock(side_effect=prepare_side_effect)
    monkeypatch.setitem(RESOURCE_PREPARERS, "psf_models", mock_preparer)

    resources = Resources(config)

    prepared = resources.prepare_resources(missing, dataset)

    assert np.array_equal(
        prepared["psf_models.standard"],
        np.ones((3, 32, 32)),
    )
    assert np.array_equal(
        prepared["psf_models.oversampled"],
        2 * np.ones((3, 32, 32)),
    )

    assert mock_preparer.call_count == 2
    mock_preparer.assert_has_calls(
        [
            call(
                dataset,
                config.resources.available["psf_models"]["standard"],
            ),
            call(
                dataset,
                config.resources.available["psf_models"]["oversampled"],
            ),
        ],
        any_order=True,
    )


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
            {
                "psf_models.standard": [1, 1, 1, 1],
            },
        ),
        (
            [],
            {},
            {},
        ),
    ],
)
def test_resolve_resources_provided(
    qc_config_factory,
    required_resources,
    provided,
    expected_resolved,
):
    config = qc_config_factory(required_resources=required_resources)
    resources = Resources(config)

    assert resources.resolve(provided) == expected_resolved


def test_resolve_resources_missing(qc_config_factory, monkeypatch):
    config = qc_config_factory(
        required_resources=["psf_models.standard"],
    )
    resources = Resources(config)

    mock_psf_models = np.ones((3, 32, 32))
    dataset = {"data": [1.0, 2.0, 3.0]}
    mock_preparer = Mock(return_value=mock_psf_models)
    monkeypatch.setitem(RESOURCE_PREPARERS, "psf_models", mock_preparer)

    resolved = resources.resolve(provided=None, dataset=dataset)
    assert np.array_equal(
        resolved["psf_models.standard"],
        mock_psf_models,
    )
    mock_preparer.assert_called_once_with(
        dataset,
        config.resources.available["psf_models"]["standard"],
    )


def test_resolve_resources_provided_and_prepare_missing(qc_config_factory, monkeypatch):
    provided = {"psf_models.standard": np.ones((3, 32, 32))}
    dataset = {"data": [1.0, 2.0, 3.0]}

    mock_oversampled_psfs = 3 * np.ones((3, 32, 32))
    mock_preparer = Mock(return_value=mock_oversampled_psfs)
    monkeypatch.setitem(RESOURCE_PREPARERS, "psf_models", mock_preparer)

    required_resources = {"psf_models.standard", "psf_models.oversampled"}
    config = qc_config_factory(required_resources=required_resources)

    resources = Resources(config)
    resolved = resources.resolve(provided=provided, dataset=dataset)

    assert np.array_equal(
        resolved["psf_models.standard"], provided["psf_models.standard"]
    )
    assert np.array_equal(resolved["psf_models.oversampled"], mock_oversampled_psfs)
    mock_preparer.assert_called_once_with(
        dataset,
        config.resources.available["psf_models"]["oversampled"],
    )


def test_resolve_resources_provided_takes_precedence(qc_config_factory, monkeypatch):
    provided = {"psf_models.standard": np.ones((3, 32, 32))}
    dataset = {"data": [1.0, 2.0, 3.0]}
    mock_psf_models = 2 * np.ones((3, 32, 32))

    mock_preparer = Mock(return_value=mock_psf_models)
    monkeypatch.setitem(RESOURCE_PREPARERS, "psf_models", mock_preparer)

    required_resources = {"psf_models.standard"}
    config = qc_config_factory(required_resources=required_resources)
    resources = Resources(config)
    resolved = resources.resolve(provided=provided, dataset=dataset)

    assert np.array_equal(
        resolved["psf_models.standard"], provided["psf_models.standard"]
    )
    mock_preparer.assert_not_called()
