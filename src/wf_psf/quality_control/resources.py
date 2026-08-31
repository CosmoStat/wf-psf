"""Resource dependency handling for the quality control pipeline.

Defines helpers for identifying resources required by enabled quality metrics
and resolving those requirements against resources supplied by the caller.

:Authors:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""

from __future__ import annotations
from collections.abc import Mapping, Callable
from typing import Any
from wf_psf.quality_control.config import QualityControlConfig

import logging

logger = logging.getLogger(__name__)

ResourcePreparer = Callable[[Any, dict[str, Any]], Any]
RESOURCE_PREPARERS: dict[str, ResourcePreparer] = {}


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

    def prepare_resources(self, missing: set, dataset: Any | None) -> dict[str, Any]:
        """Prepare resources required by enabled quality metrics.

        Parameters
        ----------
        missing: set
            Unique resource identifiers required by enabled quality metrics.

        dataset: Any
            Dataset or data container required to prepare resources.

        Returns
        -------
        dict[str, Any]
            Prepared resources indexed by resource identifier.
        """
        resources_config = self.config.resources.available

        prepared_resources = {}
        for identifier in missing:
            family, variant = identifier.split(".", 1)

            resource_config = resources_config[family][variant]

            preparer = RESOURCE_PREPARERS[family]
            prepared_resource = preparer(dataset, resource_config)

            prepared_resources[identifier] = prepared_resource

        return prepared_resources

    def resolve(
        self,
        provided: Mapping[str, Any] | None = None,
        dataset: Any | None = None,
    ) -> dict[str, Any]:
        """Resolve resources required by enabled quality metrics.

        Parameters
        ----------
        provided : Mapping[str, Any] or None
            Ready-to-use resources supplied by the pipeline caller.

        dataset : Any or None
            Dataset or data container required to prepare missing resources.

        Returns
        -------
        dict[str, Any]
            Resources required by enabled quality metrics and supplied by the
            caller.

        """
        required = self.get_required()
        provided = {} if provided is None else provided

        resolved = {
            resource: provided[resource]
            for resource in required
            if resource in provided
        }

        missing = required - provided.keys()
        unused = provided.keys() - required

        if missing:
            prepared_resources = self.prepare_resources(missing, dataset)
            resolved.update(prepared_resources)

        logger.debug(
            "Resource resolution: resolved=%s, missing=%s, unused=%s",
            sorted(resolved),
            sorted(missing),
            sorted(unused),
        )

        return resolved
