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
    """Registry to store available quality metrics implementations.

    Quality metric implementation may be added using
    :meth:`register_metric`.

    """

    def register_metric(self, metric_cls: type[QualityMetric]) -> None:
        """Register a quality metric implementation.

        Registers a ``QualityMetric`` subclass using its ``name``
        attribute as the registry key.

        Parameters
        ----------
        metric_cls : type[QualityMetric]
            Quality metric implementation to register.

        """
        self.register(metric_cls.name, metric_cls)


def build_metrics_registry():
    """Create the registry of built-in quality metrics."""
    registry = MetricsRegistry()
    registry.register_metric(MaskObscurationMetric)
    registry.register_metric(GoodnessOfFitMetric)
    return registry
