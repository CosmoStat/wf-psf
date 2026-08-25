import numpy as np
from pathlib import Path
import pytest
from unittest.mock import patch

from wf_psf.quality_control.pipeline import QualityControlPipeline
from wf_psf.quality_control.config import QualityControlConfig
from wf_psf.quality_control.metrics.mask_obscuration import MaskObscurationMetric
from wf_psf.quality_control.metrics.goodness_of_fit import GoodnessOfFitMetric
from wf_psf.quality_control.rejection.threshold import ThresholdRejectionPolicy


@pytest.fixture
def pipeline_factory():
    def build(config_file):
        path = Path(__file__).parent / "data" / config_file
        return QualityControlPipeline(path)

    return build


def test_pipeline_constructor(pipeline_factory):
    pipeline = pipeline_factory("valid/quality_control.yaml")

    # Check config
    assert isinstance(pipeline.config, QualityControlConfig)

    # Check metrics registry
    assert pipeline.metrics_registry.get("mask_obscuration") is MaskObscurationMetric
    assert pipeline.metrics_registry.get("goodness_of_fit") is GoodnessOfFitMetric

    # Check rejection registry
    assert pipeline.rejection_registry.get("threshold") is ThresholdRejectionPolicy


def test_pipeline_instantiate_metrics_valid(pipeline_factory):
    pipeline = pipeline_factory("valid/quality_control.yaml")

    metrics = pipeline._instantiate_metrics()

    assert len(metrics) == 2
    assert isinstance(metrics["mask_obscuration"], MaskObscurationMetric)
    assert isinstance(metrics["goodness_of_fit"], GoodnessOfFitMetric)


def test_pipeline_instantiate_metrics_unknown_metric(pipeline_factory):
    pipeline = pipeline_factory("valid/imaginary_metric.yaml")

    with pytest.raises(KeyError):
        pipeline._instantiate_metrics()


def test_pipeline_instantiate_rejection_policy_valid(pipeline_factory):
    pipeline = pipeline_factory("valid/quality_control.yaml")

    rejection_policies = pipeline._instantiate_rejection_policies()

    assert len(rejection_policies) == 1
    assert isinstance(rejection_policies["mask_obscuration"], ThresholdRejectionPolicy)
    assert rejection_policies["mask_obscuration"].value == 3.0

    assert "goodness_of_fit" not in rejection_policies


# Test pipeline runner
def test_pipeline_run(pipeline_factory):
    metric_result = np.array([1.0, 2.0, 3.0])
    rejection_mask = np.array([True, False, True])

    with (
        patch.object(
            MaskObscurationMetric,
            "compute",
            return_value=metric_result,
        ) as mock_mask_compute,
        patch.object(
            GoodnessOfFitMetric,
            "compute",
            return_value=metric_result,
        ) as mock_gof_compute,
        patch.object(
            ThresholdRejectionPolicy,
            "apply",
            return_value=rejection_mask,
        ) as mock_apply,
    ):
        pipeline = pipeline_factory("valid/quality_control.yaml")

        dataset = np.array([1.0, 2.0, 3.0])
        provided_resources = {"psf_models.standard": np.array([1.0, 2.0, 3.0])}

        result = pipeline.run(
            dataset=dataset,
            provided_resources=provided_resources,
        )

        mock_mask_compute.assert_called_once()
        mock_gof_compute.assert_called_once()
        mock_apply.assert_called_once_with(metric_result)

        assert np.array_equal(
            result.metrics["mask_obscuration"],
            np.array([1.0, 2.0, 3.0]),
        )

        assert np.array_equal(
            result.metrics["goodness_of_fit"],
            np.array([1.0, 2.0, 3.0]),
        )

        assert np.array_equal(
            result.rejection_masks["mask_obscuration"],
            np.array([True, False, True]),
        )

        assert "goodness_of_fit" not in result.rejection_masks
        assert "shapes" not in result.metrics
        assert "shapes" not in result.rejection_masks

        assert np.array_equal(
            result.valid_mask,
            np.array([True, False, True]),
        )
