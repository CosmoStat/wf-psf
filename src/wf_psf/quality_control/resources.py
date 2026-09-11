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
from wf_psf.quality_control.resource_identifier import ResourceIdentifier

import logging

logger = logging.getLogger(__name__)

ResourcePreparer = Callable[[Any, dict[str, Any]], Any]
RESOURCE_PREPARERS: dict[str, ResourcePreparer] = {}


def register_resource_preparer(family: str, *, override: bool = False):
    """Register a resource preparer.

    A decorator to add built-in or custom resource
    preparer methods to the RESOURCE_PREPARERS registry.

    Parameters
    ----------
    family : str
        Resource family used as a key in the resource preparer registry.
    override : bool
        If True, replace an existing preparer registered for this family.
        If False, raise an error if a preparer is already registered.

    Returns
    -------
    callable
        A decorator that registers the decorated resource preparer and returns it unchanged.

    Raises
    ------
    ValueError
        If a preparer is already registered for ``family`` and ``override`` is False.
    """

    def decorator(preparer):
        if not override and family in RESOURCE_PREPARERS:
            raise ValueError(
                f"Resource preparer for family: '{family}' is already registered."
            )

        RESOURCE_PREPARERS[family] = preparer
        return preparer

    return decorator


class Resources:
    """Manage resources required by quality control metrics.

    Assesses resource requirements and availability for a validated quality
    control configuration.
    """

    def __init__(self, config: QualityControlConfig):
        self.config = config

    def get_required(self) -> set[ResourceIdentifier]:
        """Return resources required by enabled quality metrics.

        Returns
        -------
        set[ResourceIdentifier]
            Unique resource identifiers required by enabled quality metrics.
        """
        return {
            resource_id
            for metric in self.config.metrics.values()
            if metric.enabled
            for resource_id in metric.required_resources
        }

    def prepare_resources(
        self, missing: set[ResourceIdentifier], dataset: Any
    ) -> dict[ResourceIdentifier, Any]:
        """Prepare resources required by enabled quality metrics.

        Parameters
        ----------
        missing : set[ResourceIdentifier]
            Resource identifiers required by enabled quality metrics that were not supplied by the pipeline caller.

        dataset : Any
            Dataset or data container required to prepare resources.

        Returns
        -------
        dict[ResourceIdentifier, Any]
            Prepared resources indexed by resource identifier.

        Raises
        ------
        KeyError
            If no resource preparer is registered for a required resource family.
        """
        resources_config = self.config.resources.available

        prepared_resources = {}

        for identifier in missing:
            try:
                preparer = RESOURCE_PREPARERS[identifier.family]
            except KeyError as exc:
                raise KeyError(
                    f"No resource preparer is registered for resource family "
                    f"'{identifier.family}' required by resource '{identifier}'. "
                    "Register a resource preparer using the "
                    "`register_resource_preparer` decorator."
                ) from exc

            resource_config = resources_config[identifier.family][identifier.variant]

            prepared_resource = preparer(dataset, resource_config)

            prepared_resources[identifier] = prepared_resource

        return prepared_resources

    def resolve(
        self,
        provided_resources: Mapping[str, Any] | None = None,
        dataset: Any | None = None,
    ) -> dict[str, Any]:
        """Resolve resources required by enabled quality metrics.

        Parameters
        ----------
        provided_resources : Mapping[str, Any] or None
            Ready-to-use resources supplied by the pipeline caller.

        dataset : Any or None
            Dataset or data container required to prepare missing resources.

        Returns
        -------
        dict[str, Any]
            Resources required by enabled quality metrics, including resources supplied by the
            caller and resources prepared as needed.

        """
        required = self.get_required()
        provided_resources = {} if provided_resources is None else provided_resources

        provided = {
            ResourceIdentifier.from_string(identifier): resource
            for identifier, resource in provided_resources.items()
        }

        resolved = {
            resource_id: provided[resource_id]
            for resource_id in required
            if resource_id in provided
        }

        missing = required - set(provided)
        unused = provided.keys() - required

        if missing:
            prepared_resources = self.prepare_resources(missing, dataset)
            resolved.update(prepared_resources)

        logger.debug(
            "Resource resolution: resolved=%s, missing=%s, unused=%s",
            sorted(str(resource_id) for resource_id in resolved),
            sorted(str(resource_id) for resource_id in missing),
            sorted(str(resource_id) for resource_id in unused),
        )

        return {
            str(resource_id): resource for resource_id, resource in resolved.items()
        }
