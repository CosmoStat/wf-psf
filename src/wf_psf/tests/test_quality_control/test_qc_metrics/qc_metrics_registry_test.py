"""UNIT TESTS FOR PACKAGE MODULE: Quality Control Metrics Registry.

This module contains unit tests for the MetricsRegistry class.

:Author:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""

import pytest
from wf_psf.quality_control.metrics.base import QualityMetric
from wf_psf.quality_control.metrics.registry import (
    MetricsRegistry,
    build_metrics_registry,
)
from wf_psf.quality_control.metrics.mask_obscuration import MaskObscurationMetric
from wf_psf.quality_control.metrics.goodness_of_fit import GoodnessOfFitMetric


class CustomMetric(QualityMetric):
    name = "custom"

    def compute(self, dataset):
        return None


def test_duplicate_metric_registration_raises():
    registry = MetricsRegistry()

    registry.register_metric(MaskObscurationMetric)

    with pytest.raises(ValueError):
        registry.register_metric(MaskObscurationMetric)


def test_build_metrics_registry():
    registry = build_metrics_registry()

    assert registry.get("mask_obscuration") is MaskObscurationMetric

    assert registry.get("goodness_of_fit") is GoodnessOfFitMetric


def test_unknown_metric_raises():
    registry = build_metrics_registry()

    with pytest.raises(KeyError):
        registry.get("unknown_metric")


def test_custom_metric_registration():
    registry = build_metrics_registry()

    registry.register_metric(CustomMetric)

    assert registry.get("custom") is CustomMetric
