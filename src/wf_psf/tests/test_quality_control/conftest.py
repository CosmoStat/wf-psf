import pytest

from wf_psf.tests.test_quality_control.test_utils import PSFDataset

from wf_psf.quality_control.config import (
    QualityControlConfig,
    QualityMetricConfig,
    RejectionPolicyConfig,
    ResourcesConfig,
)
from wf_psf.quality_control.context import QualityControlContext


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
                    },
                    "oversampled": {"inference_config": "inference_oversampled.yaml"},
                }
            }
        )

        rejection_default = {
            rejection_metric or "goodness_of_fit": RejectionPolicyConfig(
                enabled=True,
                diagnostic="reduced_chi_square",
                policy={
                    "threshold": {
                        "value": 0.25,
                    },
                },
            )
        }

        return QualityControlConfig(
            metrics=metric_default if metrics is None else metrics,
            resources=resources_default if resources is None else resources,
            rejection=rejection_default if rejection is None else rejection,
        )

    return factory


@pytest.fixture
def dataset_factory():
    def factory(n_src=10, n_pix=32):
        return PSFDataset(n_src=n_src, n_pix=n_pix)

    return factory


@pytest.fixture
def context():
    return QualityControlContext(dataset=PSFDataset())
