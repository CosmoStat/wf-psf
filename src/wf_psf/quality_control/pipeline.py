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
from wf_psf.quality_control.metrics.base import QualityMetric
from wf_psf.quality_control.metrics.registry import build_metrics_registry
from wf_psf.quality_control.rejection.registry import build_rejection_policy_registry


@dataclass
class QualityControlResult:
    """Results produced by the quality control pipeline.

    Attributes
    ----------
    metrics
        Computed quality metrics indexed by metric name.

    rejection_masks
        Boolean validity masks produced by each rejection policy.

    valid_mask
        Combined boolean validity mask obtained by applying all enabled
        rejection policies.
    """

    metrics: dict[str, np.ndarray]

    rejection_masks: dict[str, np.ndarray]

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
        """Instantiate enabled quality metric implementation from configuration."""
        metrics = {}

        for name, metric_config in self.config.metrics.items():
            if not metric_config.enabled:
                continue

            metric_cls = self.metrics_registry.get(name)

            metrics[name] = metric_cls()

        return metrics

    def run(self, dataset):
        """Run quality control pipeline."""
        #        metrics = self._instantiate_metrics()
        ...
        return QualityControlResult(...)
