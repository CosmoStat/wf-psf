"""Quality Control Pipeline.

Defines the orchestration layer for dataset quality control.

The QualityControlPipeline coordinates quality metric evaluation and
sample rejection. Individual quality metrics and rejection policies are
provided through their respective interfaces, allowing new methods to
be added without modifying the pipeline implementation.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from dataclasses import dataclass
import numpy as np

from wf_psf.quality_control.config import QualityControlConfigHandler
from wf_psf.quality_control.context import QualityControlContext
from wf_psf.quality_control.metrics.base import QualityMetric
from wf_psf.quality_control.metrics.registry import build_metrics_registry
from wf_psf.quality_control.rejection.base import RejectionPolicy
from wf_psf.quality_control.rejection.registry import build_rejection_policy_registry
from wf_psf.quality_control.resources import Resources

import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityControlResult:
    """Results produced by the quality control pipeline.

    Attributes
    ----------
    metrics
        Computed quality diagnostics indexed by metric name and diagnostic name.

    validity_masks
        Boolean validity masks produced by each rejection policy.

    valid_mask
        Combined boolean validity mask obtained by applying all enabled
        rejection policies.
    """

    metrics: dict[str, dict[str, np.ndarray]]

    validity_masks: dict[str, np.ndarray]

    valid_mask: np.ndarray


class QualityControlPipeline:
    """Coordinate quality metric evaluation and sample rejection.

    The pipeline evaluates all configured quality metrics, applies the
    corresponding rejection policies, combines the resulting validity
    masks, and returns the quality control results.

    Dataset filtering and reporting may optionally be performed as part
    of the pipeline execution.
    """

    def __init__(self, qc_config_path):
        self.config = QualityControlConfigHandler(qc_config_path).load()
        self.metrics_registry = build_metrics_registry()
        self.rejection_registry = build_rejection_policy_registry()
        self.validate_configuration()

    def _instantiate_metrics(self) -> dict[str, QualityMetric]:
        """Instantiate enabled quality metric implementations from configuration.

        Returns
        -------
        dict[str, QualityMetric]
            Enabled quality metric implementations keyed by metric name.

        Notes
        -----
        The quality control configuration is assumed to have been validated
        before policy instantiation.
        """
        metrics = {}

        for name, metric_config in self.config.metrics.items():
            if not metric_config.enabled:
                logger.debug("Skipping metric %s: not enabled.", name)
                continue

            metric_cls = self.metrics_registry.get(name)

            metrics[name] = metric_cls(metric_config.params)

        logger.debug("Instantiated metrics: %s", list(metrics))

        return metrics

    def _instantiate_rejection_policies(self) -> dict[str, RejectionPolicy]:
        """Instantiate enabled rejection policy implementations.

        Returns
        -------
        dict[str, RejectionPolicy]
            Enabled rejection policy implementations keyed by metric name.

        Notes
        -----
        The quality control configuration is assumed to have been validated
        before policy instantiation.
        """
        rejection_policies = {}

        for metric_name, rejection_config in self.config.rejection.items():
            if not rejection_config.enabled:
                logger.debug("Skipping rejection policy %s: not enabled.", metric_name)
                continue

            policy_name, policy_params = next(iter(rejection_config.policy.items()))
            policy_cls = self.rejection_registry.get(policy_name)

            rejection_policies[metric_name] = policy_cls(**policy_params)

        logger.debug("Instantiated rejection policies: %s", list(rejection_policies))

        return rejection_policies

    def _resolve_resources(self, provided_resources):
        """Resolve resources required by enabled quality metrics.

        Parameters
        ----------
        provided_resources : Mapping[str, Any] or None
            Ready-to-use resources supplied by the pipeline caller, keyed by
            resource identifier.

        Returns
        -------
        dict[str, Any]
            Resolved resources required by enabled quality metrics.
        """
        resource_manager = Resources(self.config)
        return resource_manager.resolve(provided_resources)

    def validate_configuration(self) -> None:
        """Validate internal consistency of a quality control configuration.

        Raises
        ------
        ValueError
            If any cross-section configuration dependency is invalid.
        """
        self.validate_metric_resource_requirements()
        self.validate_rejection_policy_metrics()

    def validate_metric_resource_requirements(self) -> None:
        """Validate that resource requirements of enabled metrics are configured.

        Raises
        ------
        ValueError
            If a required resource identifier is not available in the configured resources.

        Notes
        -----
        A configured resource can be overriden by the pipeline
        caller using the `provided_resources` argument.  This static validation ensures that
        each resource identifier required by an enabled metric is declared in the
        resources configuration, regardless of whethe resource will be ultimately supplied
        by the caller or prepared by the pipeline.

        """
        resources = self.config.resources.available

        for metric_name, metric in self.config.metrics.items():
            if not metric.enabled:
                continue

            for resource_id in metric.required_resources:
                if (
                    resource_id.family not in resources
                    or resource_id.variant not in resources[resource_id.family]
                ):
                    raise ValueError(
                        f"Metric '{metric_name}' requires unknown resource '{resource_id}'."
                    )

    def validate_rejection_policy_metrics(self) -> None:
        """Validate rejection policies against configured quality metrics.

        Raises
        ------
        ValueError
            If an enabled rejection policy references an unknown or disabled
            quality metric, or specifies an invalid diagnostic.
        """
        for metric_name, rejection_policy in self.config.rejection.items():
            if not rejection_policy.enabled:
                continue

            if metric_name not in self.config.metrics:
                raise ValueError(
                    f"Rejection policy configured for unknown metric '{metric_name}'."
                )

            if not self.config.metrics[metric_name].enabled:
                raise ValueError(
                    f"Rejection policy cannot be enabled because metric '{metric_name}' is disabled."
                )

            metric_cls = self.metrics_registry.get(metric_name)

            if metric_cls is None:
                raise ValueError(f"Quality metric '{metric_name}' is not registered.")

            if rejection_policy.diagnostic not in metric_cls.diagnostics:
                raise ValueError(
                    f"Diagnostic '{rejection_policy.diagnostic}' is not provided by the metric '{metric_name}'."
                )

    def run(self, dataset, provided_resources=None):
        """Run quality control pipeline.

        Parameters
        ----------
        dataset : Any
            Dataset or data container supplied to the quality control pipeline.

        provided_resources : Mapping[str, Any] or None
            Ready-to-use resources supplied by the pipeline caller, keyed by resource identifier.

        Notes
        -----
        The pipeline is expected to be invoked only when at least one quality metric is enabled in the quality control configuration.
        """
        resolved_resources = self._resolve_resources(
            provided_resources=provided_resources
        )

        context = QualityControlContext(dataset, resolved_resources)

        metrics = self._instantiate_metrics()

        metric_results = {
            name: metric.compute(context) for name, metric in metrics.items()
        }

        rejection_policies = self._instantiate_rejection_policies()

        validity_masks = {}
        for name, policy in rejection_policies.items():
            diagnostic_identifier = self.config.rejection[name].diagnostic
            assert diagnostic_identifier is not None

            diagnostic = metric_results[name][diagnostic_identifier]
            validity_masks[name] = policy.apply(diagnostic)

        if validity_masks != {}:
            # True indicates a valid sample. A sample is valid only if it passes
            # every enabled rejection policy.
            valid_mask = np.logical_and.reduce(list(validity_masks.values()))
        else:
            # If rejection policy not enabled, generate boolean unity mask
            metric_result = next(iter(metric_results.values()))
            diagnostic_result = next(iter(metric_result.values()))
            valid_mask = np.ones(diagnostic_result.shape, dtype=bool)

        return QualityControlResult(
            metrics=metric_results,
            validity_masks=validity_masks,
            valid_mask=valid_mask,
        )
