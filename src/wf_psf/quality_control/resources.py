"""Resource dependency handling for the quality control pipeline.

Defines helpers for identifying resources required by enabled quality metrics
and resolving those requirements against resources supplied by the caller.

:Authors:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from wf_psf.quality_control.config import QualityControlConfig


class Resources:
    """Manage resources required by quality control metrics.

    Assesses resource requirements and availability for a validated quality
    control configuration.
    """

    def __init__(self, config: QualityControlConfig):
        self.config = config

    def get_required(self) -> set[str]:
        """Return resources required by enabled quality metrics.

        Returns
        -------
        Unique resource identifiers required by enabled quality metrics.

        Notes
        -----
        This configuration is assumed to have been validated for internal consistency between resource requirements and available resources.

        """
        return {
            resource
            for metric in self.config.metrics.values()
            if metric.enabled
            for resource in metric.required_resources
        }

    def resolve(
        self,
        provided: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve resources required by enabled quality metrics.

        Parameters
        ----------
        provided : Mapping[str, Any] or None
            Ready-to-use resources supplied by the pipeline caller.

        Returns
        -------
        dict[str, Any]
            Resources required by enabled quality metrics and supplied by the
            caller.

        Raises
        ------
        NotImplementedError
            If required resources are not supplied by the caller. Preparation of
            missing resources is not yet implemented.
        """
        required = self.get_required()
        provided = {} if provided is None else provided

        missing = required - provided.keys()

        if missing:
            raise NotImplementedError(
                f"Required resources are not available: {sorted(missing)}"
            )

        return {resource: provided[resource] for resource in required}
