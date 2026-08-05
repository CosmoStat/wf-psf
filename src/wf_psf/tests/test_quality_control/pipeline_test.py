from pathlib import Path
import pytest

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
