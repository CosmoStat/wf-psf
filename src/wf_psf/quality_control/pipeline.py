"""Quality Control Pipeline.

Defines the orchestration layer for dataset quality control.

The QualityControlPipeline coordinates quality metric evaluation,
application of rejection policies, reporting, and dataset filtering.
Individual quality metrics and rejection policies are provided through
their respective interfaces, allowing new methods to be added without
modifying the pipeline implementation.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from dataclasses import dataclass
import numpy as np

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

    def __init__(
            self,
            metrics,
            rejection_policies,
    ):
        ...

    def run(self, dataset):

        ...

        return QualityControlResult(...)