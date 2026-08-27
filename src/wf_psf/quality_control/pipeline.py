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
        Computed quality metrics indexed by metric name.

    validity_masks
        Boolean validity masks produced by each rejection policy.

    valid_mask
        Combined boolean validity mask obtained by applying all enabled
        rejection policies.
    """

    metrics: dict[str, np.ndarray]

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

            metrics[name] = metric_cls()

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

        validity_masks = {
            name: policy.apply(metric_results[name])
            for name, policy in rejection_policies.items()
        }

        if validity_masks:
            # True indicates a valid sample. A sample is valid only if it passes
            # every enabled rejection policy.
            valid_mask = np.logical_and.reduce(list(validity_masks.values()))
        else:
            metric_result = next(iter(metric_results.values()))
            valid_mask = np.ones(metric_result.shape, dtype=bool)

        return QualityControlResult(
            metrics=metric_results,
            validity_masks=validity_masks,
            valid_mask=valid_mask,
        )
