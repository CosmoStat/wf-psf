"""Quality control context.

Encapsulates contextual information, such as datasets and resolved resources,
required by the quality control pipeline and its metrics.

:Authors:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualityControlContext:
    """Context shared across the quality control pipeline.

    Attributes
    ----------
    dataset : dict[str, Any]
        Dataset or data container supplied to the quality control pipeline.

    resources : dict[str, Any]
        Ready-to-use resources required by enabled quality metrics, keyed by
        resource identifier.
    """

    dataset: Any = None
    resources: dict[str, Any] = field(default_factory=dict)
