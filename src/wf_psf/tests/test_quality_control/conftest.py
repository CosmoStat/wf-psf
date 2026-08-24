import pytest
from wf_psf.quality_control.config import (
    QualityControlConfig,
    QualityMetricConfig,
    RejectionPolicyConfig,
    ResourcesConfig,
)


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
