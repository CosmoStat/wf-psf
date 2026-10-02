from contextlib import nullcontext as does_not_raise
import numpy as np
from pathlib import Path
import pytest
from unittest.mock import call, patch

from wf_psf.quality_control.pipeline import QualityControlPipeline
from wf_psf.quality_control.config import (
    QualityControlConfig,
    QualityMetricConfig,
    RejectionPolicyConfig,
)


from wf_psf.quality_control.metrics.pixel_masks import PixelMaskMetric
from wf_psf.quality_control.metrics.goodness_of_fit import GoodnessOfFitMetric
from wf_psf.quality_control.rejection.threshold import ThresholdRejectionPolicy
from wf_psf.quality_control.resource_identifier import ResourceIdentifier


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
    assert pipeline.metrics_registry.get("pixel_mask") is PixelMaskMetric
    assert pipeline.metrics_registry.get("goodness_of_fit") is GoodnessOfFitMetric

    # Check rejection registry
    assert pipeline.rejection_registry.get("threshold") is ThresholdRejectionPolicy

    # Check config validation
    with does_not_raise():
        pipeline.validate_configuration()


def test_pipeline_instantiate_metrics_valid(pipeline_factory):
    pipeline = pipeline_factory("valid/quality_control.yaml")

    metrics = pipeline._instantiate_metrics()

    assert len(metrics) == 2
    assert isinstance(metrics["pixel_mask"], PixelMaskMetric)
    assert isinstance(metrics["goodness_of_fit"], GoodnessOfFitMetric)


def test_pipeline_instantiate_metrics_unknown_metric(pipeline_factory):
    pipeline = pipeline_factory("valid/imaginary_metric.yaml")

    with pytest.raises(KeyError):
        pipeline._instantiate_metrics()


def test_pipeline_instantiate_rejection_policy_valid(pipeline_factory):
    pipeline = pipeline_factory("valid/quality_control.yaml")

    rejection_policies = pipeline._instantiate_rejection_policies()

    assert len(rejection_policies) == 1
    assert isinstance(rejection_policies["pixel_mask"], ThresholdRejectionPolicy)
    assert rejection_policies["pixel_mask"].value == 3.0

    assert "goodness_of_fit" not in rejection_policies


# Tests for validation methods
def test_validate_metric_resources_all_valid(pipeline_factory):
    pipeline = pipeline_factory("valid/quality_control.yaml")
    with does_not_raise():
        pipeline.validate_metric_resource_requirements()


@pytest.mark.parametrize(
    "required_resource",
    [
        ResourceIdentifier.from_string("images.segmentation_maps"),
        ResourceIdentifier.from_string("psf_models.imaginary"),
    ],
)
@patch("wf_psf.quality_control.pipeline.QualityControlConfigHandler")
@patch("wf_psf.quality_control.pipeline.QualityControlPipeline.validate_configuration")
def test_validate_metric_resources_unknown_resource(
    mock_validate_configuration,
    mock_config_handler,
    qc_config_factory,
    required_resource,
):
    config = qc_config_factory(required_resources=[required_resource])
    mock_config_handler.return_value.load.return_value = config
    mock_validate_configuration.return_value = None

    with pytest.raises(
        ValueError,
        match=(
            f"Metric 'goodness_of_fit' requires unknown resource '{required_resource}'."
        ),
    ):
        pipeline = QualityControlPipeline(qc_config_path="foo.yaml")
        pipeline.validate_metric_resource_requirements()


@patch("wf_psf.quality_control.pipeline.QualityControlConfigHandler")
@patch("wf_psf.quality_control.pipeline.QualityControlPipeline.validate_configuration")
def test_validate_rejection_policy_metrics_all_valid(
    mock_validate_configuration, mock_config_handler, qc_config_factory
):
    mock_config_handler.return_value.load.return_value = qc_config_factory()
    mock_validate_configuration.return_value = None

    pipeline = QualityControlPipeline(qc_config_path="foo.yaml")
    with does_not_raise():
        pipeline.validate_rejection_policy_metrics()


@patch("wf_psf.quality_control.pipeline.QualityControlConfigHandler")
@patch("wf_psf.quality_control.pipeline.QualityControlPipeline.validate_configuration")
def test_validate_rejection_policy_disabled_policy(
    mock_validate_configuration, mock_config_handler, qc_config_factory
):
    # Define an invalid rejection policy that is disabled
    rejection_policy = {
        "foo_metric": RejectionPolicyConfig(
            enabled=False,
            diagnostic="bar",
            policy={},
        )
    }
    config = qc_config_factory(rejection=rejection_policy)
    mock_config_handler.return_value.load.return_value = config
    mock_validate_configuration.return_value = None

    # Verify no error is raised because disabled policies are skipped
    pipeline = QualityControlPipeline(qc_config_path="foo.yaml")
    pipeline.validate_rejection_policy_metrics()


@patch("wf_psf.quality_control.pipeline.QualityControlConfigHandler")
@patch("wf_psf.quality_control.pipeline.QualityControlPipeline.validate_configuration")
def test_validate_rejection_policy_metrics_unknown_metric(
    mock_validate_configuration, mock_config_handler, qc_config_factory
):
    rejection_policy = {
        "foo_metric": RejectionPolicyConfig(
            enabled=True,
            diagnostic="bar",
            policy={},
        )
    }

    config = qc_config_factory(rejection=rejection_policy)
    mock_config_handler.return_value.load.return_value = config
    mock_validate_configuration.return_value = None

    pipeline = QualityControlPipeline(qc_config_path="foo.yaml")

    with pytest.raises(
        ValueError,
        match="Rejection policy configured for unknown metric 'foo_metric'.",
    ):
        pipeline.validate_rejection_policy_metrics()


@patch("wf_psf.quality_control.pipeline.QualityControlConfigHandler")
@patch("wf_psf.quality_control.pipeline.QualityControlPipeline.validate_configuration")
def test_validate_rejection_policy_metrics_metric_not_enabled(
    mock_validate_configuration, mock_config_handler, qc_config_factory
):
    metric = {
        "goodness_of_fit": QualityMetricConfig(
            enabled=False,
            required_resources=[],
        )
    }

    config = qc_config_factory(metrics=metric)
    mock_config_handler.return_value.load.return_value = config
    mock_validate_configuration.return_value = None

    pipeline = QualityControlPipeline(qc_config_path="foo.yaml")

    with pytest.raises(
        ValueError,
        match="Rejection policy cannot be enabled because metric 'goodness_of_fit' is disabled.",
    ):
        pipeline.validate_rejection_policy_metrics()


@patch("wf_psf.quality_control.pipeline.QualityControlConfigHandler")
@patch("wf_psf.quality_control.pipeline.QualityControlPipeline.validate_configuration")
def test_validate_rejection_policy_metrics_metric_not_registered(
    mock_validate_configuration, mock_config_handler, qc_config_factory
):
    rejection_policy = {
        "foo_metric": RejectionPolicyConfig(
            enabled=True,
            diagnostic="bar",
            policy={},
        )
    }

    metric = {
        "foo_metric": QualityMetricConfig(
            enabled=True,
            required_resources=[],
        )
    }
    config = qc_config_factory(metrics=metric, rejection=rejection_policy)
    mock_config_handler.return_value.load.return_value = config
    mock_validate_configuration.return_value = None

    pipeline = QualityControlPipeline(qc_config_path="foo.yaml")

    with pytest.raises(
        KeyError,
        match="Key 'foo_metric' not found.",
    ):
        pipeline.validate_rejection_policy_metrics()


@patch("wf_psf.quality_control.pipeline.QualityControlConfigHandler")
@patch("wf_psf.quality_control.pipeline.QualityControlPipeline.validate_configuration")
def test_validate_rejection_policy_metrics_diagnostic_not_found(
    mock_validate_configuration, mock_config_handler, qc_config_factory
):
    rejection_policy = {
        "pixel_mask": RejectionPolicyConfig(
            enabled=True,
            diagnostic="bad_diagnostic",
            policy={},
        )
    }
    metric = {
        "pixel_mask": QualityMetricConfig(
            enabled=True,
            required_resources=[],
        )
    }
    config = qc_config_factory(rejection=rejection_policy, metrics=metric)
    mock_config_handler.return_value.load.return_value = config
    mock_validate_configuration.return_value = None
    pipeline = QualityControlPipeline(qc_config_path="foo.yaml")

    with pytest.raises(
        ValueError,
        match="Diagnostic 'bad_diagnostic' is not provided by the metric 'pixel_mask'.",
    ):
        pipeline.validate_rejection_policy_metrics()


# Test pipeline runner
def test_pipeline_run_single_rejection_policy(pipeline_factory):
    pixel_mask_metric_results = {
        "total_masked_pixels": np.array([10.0, 20.0, 30.0]),
        "total_masked_fraction": np.array([0.1, 0.2, 0.3]),
        "aperture_masked_pixels": np.array([1.0, 2.0, 3.0]),
        "aperture_masked_fraction": np.array([0.1, 0.2, 0.3]),
    }

    gof_metric_results = {
        "chi_square": np.array([123.0, 345.0, 678.0]),
        "reduced_chi_square": np.array([1.2, 1.4, 1.1]),
    }

    validity_mask = np.array([True, False, True])

    with (
        patch.object(
            PixelMaskMetric,
            "compute",
            return_value=pixel_mask_metric_results,
        ) as mock_mask_compute,
        patch.object(
            GoodnessOfFitMetric,
            "compute",
            return_value=gof_metric_results,
        ) as mock_gof_compute,
        patch.object(
            ThresholdRejectionPolicy,
            "apply",
            return_value=validity_mask,
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
        mock_apply.assert_called_once_with(
            pixel_mask_metric_results["aperture_masked_fraction"]
        )

        for diagnostic_id, diagnostic_result in pixel_mask_metric_results.items():
            assert np.array_equal(
                result.metrics["pixel_mask"][diagnostic_id], diagnostic_result
            )

        for diagnostic_id, diagnostic_result in gof_metric_results.items():
            assert np.array_equal(
                result.metrics["goodness_of_fit"][diagnostic_id],
                diagnostic_result,
            )

        assert np.array_equal(
            result.validity_masks["pixel_mask"],
            np.array([True, False, True]),
        )

        assert "goodness_of_fit" not in result.validity_masks
        assert "shapes" not in result.metrics
        assert "shapes" not in result.validity_masks

        assert np.array_equal(
            result.valid_mask,
            np.array([True, False, True]),
        )


def test_pipeline_run_multiple_rejection_policies(pipeline_factory):
    pixel_mask_metric_results = {
        "total_masked_pixels": np.array([10.0, 20.0, 30.0]),
        "total_masked_fraction": np.array([0.1, 0.2, 0.3]),
        "aperture_masked_pixels": np.array([1.0, 2.0, 3.0]),
        "aperture_masked_fraction": np.array([0.1, 0.2, 0.3]),
    }

    gof_metric_results = {
        "chi_square": np.array([123.0, 345.0, 678.0]),
        "reduced_chi_square": np.array([1.2, 1.4, 1.1]),
    }
    validity_masks = [
        np.array([True, True, False]),
        np.array([True, False, True]),
    ]

    with (
        patch.object(
            PixelMaskMetric,
            "compute",
            return_value=pixel_mask_metric_results,
        ) as mock_mask_compute,
        patch.object(
            GoodnessOfFitMetric,
            "compute",
            return_value=gof_metric_results,
        ) as mock_gof_compute,
        patch.object(
            ThresholdRejectionPolicy,
            "apply",
            side_effect=validity_masks,
        ) as mock_apply,
    ):
        pipeline = pipeline_factory(
            "valid/quality_control_multiple_rejection_policies.yaml"
        )

        dataset = np.array([1.0, 2.0, 3.0])
        provided_resources = {"psf_models.standard": np.array([1.0, 2.0, 3.0])}

        result = pipeline.run(
            dataset=dataset,
            provided_resources=provided_resources,
        )

        mock_mask_compute.assert_called_once()
        mock_gof_compute.assert_called_once()
        mock_apply.assert_has_calls(
            [
                call(pixel_mask_metric_results["aperture_masked_fraction"]),
                call(gof_metric_results["reduced_chi_square"]),
            ],
            any_order=True,
        )

        for diagnostic_id, diagnostic_result in pixel_mask_metric_results.items():
            assert np.array_equal(
                result.metrics["pixel_mask"][diagnostic_id], diagnostic_result
            )

        for diagnostic_id, diagnostic_result in gof_metric_results.items():
            assert np.array_equal(
                result.metrics["goodness_of_fit"][diagnostic_id],
                diagnostic_result,
            )

        assert np.array_equal(
            result.validity_masks["pixel_mask"],
            np.array([True, True, False]),
        )

        assert np.array_equal(
            result.validity_masks["goodness_of_fit"],
            np.array([True, False, True]),
        )

        assert np.array_equal(
            result.valid_mask,
            np.array([True, False, False]),
        )


def test_pipeline_run_rejection_policy_disabled(pipeline_factory):
    pixel_mask_metric_results = {
        "total_masked_pixels": np.array([10.0, 20.0, 30.0]),
        "total_masked_fraction": np.array([0.1, 0.2, 0.3]),
        "aperture_masked_pixels": np.array([1.0, 2.0, 3.0]),
        "aperture_masked_fraction": np.array([0.1, 0.2, 0.3]),
    }

    gof_metric_results = {
        "chi_square": np.array([123.0, 345.0, 678.0]),
        "reduced_chi_square": np.array([1.2, 1.4, 1.1]),
    }

    with (
        patch.object(
            PixelMaskMetric,
            "compute",
            return_value=pixel_mask_metric_results,
        ) as mock_mask_compute,
        patch.object(
            GoodnessOfFitMetric,
            "compute",
            return_value=gof_metric_results,
        ) as mock_gof_compute,
        patch.object(
            ThresholdRejectionPolicy,
            "apply",
        ) as mock_apply,
    ):
        pipeline = pipeline_factory(
            "valid/quality_control_rejection_policy_disabled.yaml"
        )

        dataset = np.array([1.0, 2.0, 3.0])
        provided_resources = {"psf_models.standard": np.array([1.0, 2.0, 3.0])}

        result = pipeline.run(
            dataset=dataset,
            provided_resources=provided_resources,
        )

        mock_mask_compute.assert_called_once()
        mock_gof_compute.assert_called_once()
        mock_apply.assert_not_called()

        for diagnostic_id, diagnostic_result in pixel_mask_metric_results.items():
            assert np.array_equal(
                result.metrics["pixel_mask"][diagnostic_id], diagnostic_result
            )

        for diagnostic_id, diagnostic_result in gof_metric_results.items():
            assert np.array_equal(
                result.metrics["goodness_of_fit"][diagnostic_id],
                diagnostic_result,
            )

        assert "shapes" not in result.metrics

        assert result.validity_masks == {}
        assert np.array_equal(result.valid_mask, np.ones(3, dtype=bool))


def test_pipeline_run_pixel_mask_metrics_without_rejection(
    dataset_factory, pipeline_factory
):
    dataset = dataset_factory(n_src=2, n_pix=5)
    dataset.masks = np.array(
        [
            [
                [True, False, False, False, True],
                [False, False, False, False, False],
                [False, True, False, False, False],
                [False, True, False, False, False],
                [True, False, False, False, True],
            ],
            [
                [True, False, False, False, True],
                [False, True, True, False, False],
                [False, True, True, False, False],
                [False, True, True, False, False],
                [True, False, False, False, False],
            ],
        ]
    )

    pipeline = pipeline_factory("valid/quality_control_pixel_masks_metrics.yaml")

    result = pipeline.run(
        dataset=dataset,
    )

    expected = {
        "total_masked_pixels": np.array([6, 9]),
        "total_masked_fraction": np.array([6 / 25, 9 / 25]),
        "aperture_masked_pixels": np.array([2, 6]),
        "aperture_masked_fraction": np.array([2 / 9, 2 / 3]),
    }

    for diagnostic_id, diagnostic_result in expected.items():
        assert np.array_equal(
            result.metrics["pixel_mask"][diagnostic_id], diagnostic_result
        )
    assert result.validity_masks == {}
    assert np.array_equal(result.valid_mask, np.ones(2, dtype=bool))
