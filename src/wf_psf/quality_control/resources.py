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


def get_required_resources(
    config: QualityControlConfig,
) -> set[str]:
    """Return resources required by enabled quality metrics.

    Parameters
    ----------
    config : QualityControlConfig
        Validated quality control configuration.

    Returns
    -------
        Unique resource identifiers required by enabled quality metrics.

    Notes
    -----
    This configuration is assumed to have been validated for internal consistency between resource requirements and available resources.
    """
    return {
        resource
        for metric in config.metrics.values()
        if metric.enabled
        for resource in metric.required_resources
    }


def resolve_resources(
    required: set[str],
    provided: Mapping[str, Any],
) -> tuple[dict[str, Any], set[str]]:
    """Separate supplied resources from resources requiring preparation.

    Parameters
    ----------
    required : set[str]
        Unique resource identifiers required by enabled quality metrics.

    provided : Mapping[str, Any]
        Ready-to-use resources provided by the quality control pipeline caller, keyed by resource identifier.

    Returns
    -------
    tuple[dict[str, Any], set[str]]
        A tuple containing:

        - a dictionary of required resources that were provided, keyed by
          resource identifier;
        - a set of required resource identifiers that were not provided and
          therefore require preparation.

    Notes
    -----
    This function only resolves which required resources are already
    available. It does not prepare or otherwise acquire missing resources.
    """
    resolved = {
        resource: provided[resource] for resource in required if resource in provided
    }
    missing = required.difference(resolved)
    return resolved, missing


def resolve_required_resources(
    config: QualityControlConfig,
    provided: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve resources required by enabled quality metrics.

    Parameters
    ----------
    config : QualityControlConfig
        Quality control configuration.

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
    required = get_required_resources(config)
    provided = {} if provided is None else provided

    resolved, missing = resolve_resources(
        required=required,
        provided=provided,
    )

    if missing:
        raise NotImplementedError(
            f"Required resources are not available: {sorted(missing)}"
        )

    return resolved
