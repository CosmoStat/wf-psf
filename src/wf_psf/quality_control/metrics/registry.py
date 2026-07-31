"""Quality metric registry.

Defines the MetricsRegistry responsible for registering and retrieving
quality metric implementations used to assess dataset sample quality.

:Authors:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""

from .mask_obscuration import MaskObscurationMetric
from .goodness_of_fit import GoodnessOfFitMetric
from .base import QualityMetric
from wf_psf.utils.registry import Registry


class MetricsRegistry(Registry[str, type[QualityMetric]]):
    """Metrics registry class."""

    def register_metric(self, metric_cls: type[QualityMetric]) -> None:
        """Register a quality metric implementation.

        Registers a class of type[QualityMetric] using the attribute
        `name` as a key, and the class as a value.

        Parameters
        ----------
        metric_cls : type[QualityMetric]
            Quality metric class to register

        """
        self.register(metric_cls.name, metric_cls)


def build_metrics_registry():
    """Build metrics registry."""
    registry = MetricsRegistry()
    registry.register_metric(MaskObscurationMetric)
    registry.register_metric(GoodnessOfFitMetric)
    return registry
