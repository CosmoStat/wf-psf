"""Quality control pipeline configuration.

Defines configuration interfaces for the quality control framework,
including quality metric, rejection policy, and reporting configuration.

:Authors:
    Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wf_psf.utils.read_config import read_yaml


@dataclass
class QualityMetricConfig:
    """Configuration for a single quality metric.

    Attributes
    ----------
    enabled : bool
        Whether the quality metric is enabled.

    params : dict
        Quality metric-specific parameters.

    required_resources : list[str]
        Identifiers of resources required to compute the quality metric.
    """

    enabled: bool = False
    params: dict = field(default_factory=dict)
    required_resources: list[str] = field(default_factory=list)


@dataclass
class RejectionPolicyConfig:
    """Configuration for rejection policy logic for a single check.

    Attributes
    ----------
    enabled : bool
        Whether rejection policy is enabled.

    policy : dict[str, Any]
        Rejection policy configuration keyed by policy type.
    """

    enabled: bool = False
    policy: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportingConfig:
    """Configuration controlling reporting behavior.

    Attributes
    ----------
    save_metrics : bool
        Persist computed metrics to storage.

    log_statistics : bool
        Emit statistics to the logging system.
    """

    save_metrics: bool = False
    log_statistics: bool = False


@dataclass
class ResourcesConfig:
    """Configuration for resources available for quality control execution.

    Attributes
    ----------
    available : dict
        Mapping of resource types to configured resource identifiers and parameters.
    """

    available: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityControlConfig:
    """Top-level quality control configuration.

    Attributes
    ----------
    metrics : dict
        Mapping of metric name to QualityMetricConfig instances.

    rejection : dict
        Mapping of rejection policy name to RejectionPolicyConfig instances.

    reporting : ReportingConfig
        ReportingConfig instance.

    resources : ResourcesConfig
        Available resources used during quality control execution.
    """

    metrics: dict[str, QualityMetricConfig] = field(default_factory=dict)

    rejection: dict[str, RejectionPolicyConfig] = field(default_factory=dict)

    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    resources: ResourcesConfig = field(default_factory=ResourcesConfig)


# config section parsers
def parse_metrics_config(
    section: Mapping[str, Any] | None,
) -> dict[str, QualityMetricConfig]:
    """Parse the quality metrics configuration section.

    Parameters
    ----------
    section : Mapping[str, Any] or None
        Raw quality metric configuration. If None, an empty configuration
        is returned.

    Returns
    -------
    dict[str, QualityMetricConfig]
        Parsed quality metric configurations keyed by metric name.

    Raises
    ------
    TypeError
        If the configuration section or any metric configuration is not a
        mapping, if ``enabled`` is not boolean, if ``params`` is not a
        mapping, or if required resource identifiers is not a list, or if
        its entries are not strings.
    """
    if section is None:
        return {}

    if not isinstance(section, Mapping):
        raise TypeError("Metrics configuration must be a mapping.")

    metrics = {}

    for metric_name, cfg in section.items():
        if not isinstance(cfg, Mapping):
            raise TypeError(f"Metric configuration '{metric_name}' must be a mapping.")

        enabled = cfg.get("enabled", False)

        if not isinstance(enabled, bool):
            raise TypeError(
                f"Metric `enabled` flag for '{metric_name}' must be boolean."
            )

        params = cfg.get("params", {})

        if not isinstance(params, Mapping):
            raise TypeError(f"Metric parameters for '{metric_name}' must be a mapping.")

        required_resources = cfg.get("required_resources", [])

        if not isinstance(required_resources, list):
            raise TypeError(
                f"Required resources for metric '{metric_name}' must be a list."
            )

        if not all(isinstance(resource, str) for resource in required_resources):
            raise TypeError(
                f"Required resources for metric '{metric_name}' must contain only strings."
            )

        metrics[metric_name] = QualityMetricConfig(
            enabled=enabled, params=dict(params), required_resources=required_resources
        )

    return metrics


def parse_rejection_policy_config(
    section: Mapping[str, Any] | None,
) -> dict[str, RejectionPolicyConfig]:
    """Parse the rejection policy configuration section.

    Parameters
    ----------
    section : Mapping[str, Any] or None
        Raw rejection policy configuration. If None, an empty configuration
        is returned.

    Returns
    -------
    dict[str, RejectionPolicyConfig]
        Parsed rejection policy configurations keyed by metric name.

    Raises
    ------
    TypeError
        If the configuration section or a policy configuration is not a
        mapping.

    ValueError
        If the policy configuration for a metric does not specify exactly one policy.
    """
    if section is None:
        return {}

    if not isinstance(section, Mapping):
        raise TypeError("Rejection policy configuration must be a mapping.")

    policies = {}

    for metric_name, cfg in section.items():
        if not isinstance(cfg, Mapping):
            raise TypeError(
                f"Rejection policy configuration '{metric_name}' must be a mapping."
            )

        enabled = cfg.get("enabled", False)

        if not isinstance(enabled, bool):
            raise TypeError(
                f"Rejection policy `enabled` flag for '{metric_name}' must be boolean."
            )

        if not enabled:
            policies[metric_name] = RejectionPolicyConfig(enabled=False)
            continue

        if "policy" not in cfg:
            raise ValueError(
                f"Rejection policy configuration for '{metric_name}' "
                "must specify a `policy`."
            )

        policy = cfg["policy"]

        if not isinstance(policy, Mapping):
            raise TypeError(
                f"Rejection policy `policy` field for '{metric_name}' must be a mapping."
            )

        if len(policy) != 1:
            raise ValueError(
                f"Rejection policy configuration for '{metric_name}' "
                "must specify exactly one policy type."
            )

        policy_name, policy_params = next(iter(policy.items()))

        if not isinstance(policy_name, str):
            raise TypeError(
                f"Rejection policy identifier '{policy_name}' for '{metric_name}' must be a string."
            )

        if not isinstance(policy_params, Mapping):
            raise TypeError(
                f"Rejection policy parameters for '{metric_name}' must be a mapping."
            )

        policies[metric_name] = RejectionPolicyConfig(
            enabled=enabled, policy=dict(policy)
        )

    return policies


def parse_reporting_config(section: Mapping[str, Any] | None) -> ReportingConfig:
    """Parse the reporting configuration section.

    Parameters
    ----------
    section : dict or None
        Raw reporting configuration. If None, the default reporting
        configuration is returned.

    Returns
    -------
    ReportingConfig
        Parsed reporting configuration.

    Raises
    ------
    TypeError
        If the configuration section is not a mapping.
    """
    if section is None:
        return ReportingConfig()

    if not isinstance(section, Mapping):
        raise TypeError("Reporting configuration must be a mapping.")

    return ReportingConfig(**section)


def parse_resources_config(
    config: Mapping[str, Any] | None,
) -> ResourcesConfig:
    """Parse the resources configuration section.

    Parameters
    ----------
    config : Mapping or None
        Raw resource configuration. If None, an empty resource
        configuration is returned.

    Returns
    -------
    ResourcesConfig
        Parsed resource configuration.

    Raises
    ------
    TypeError
        If the configuration section is not a mapping.
    """
    if config is None:
        return ResourcesConfig()

    if not isinstance(config, Mapping):
        raise TypeError("Resources configuration must be a mapping.")

    return ResourcesConfig(available=dict(config))


#  validators for internal consistency of config sections
def validate_quality_control_config(config: QualityControlConfig) -> None:
    """Validate internal consistency of a quality control configuration.

    Parameters
    ----------
    config : QualityControlConfig
        Parsed quality control configuration.

    Raises
    ------
    ValueError
        If any cross-section configuration dependency is invalid.
    """
    validate_metric_resources(config)
    validate_rejection_policy_metrics(config)


def validate_metric_resources(config: QualityControlConfig) -> None:
    """Validate that all metric resource requirements can be resolved.

    Parameters
    ----------
    config : QualityControlConfig
        Parsed quality control configuration.

    Raises
    ------
    ValueError
        If a required resource identifier is malformed or references an
        unavailable resource.
    """
    for metric_name, metric in config.metrics.items():
        for req in metric.required_resources:
            parts = req.split(".")

            if len(parts) != 2 or not all(parts):
                raise ValueError(
                    f"Resource identifier '{req}' must have the form "
                    "'<resource_type>.<resource_name>'."
                )

            resource_type, resource_name = parts
            resources = config.resources.available

            if (
                resource_type not in resources
                or resource_name not in resources[resource_type]
            ):
                raise ValueError(
                    f"Metric '{metric_name}' requires unknown resource '{req}'."
                )


def validate_rejection_policy_metrics(config: QualityControlConfig) -> None:
    """Validate rejection policies against configured quality metrics.

    Parameters
    ----------
    config : QualityControlConfig
        Parsed quality control configuration.

    Raises
    ------
    ValueError
        If an enabled rejection policy references an unknown or disabled
        quality metric.
    """
    for metric_name, metric_rejection_policy in config.rejection.items():
        if not metric_rejection_policy.enabled:
            continue

        if metric_name not in config.metrics:
            raise ValueError(
                f"Rejection policy configured for unknown metric '{metric_name}' "
            )

        if not config.metrics[metric_name].enabled:
            raise ValueError(
                f"Rejection policy cannot be enabled because metric '{metric_name}' is disabled."
            )


SECTION_PARSERS = {
    "metrics": parse_metrics_config,
    "rejection": parse_rejection_policy_config,
    "reporting": parse_reporting_config,
    "resources": parse_resources_config,
}


class QualityControlConfigHandler:
    """QualityControlConfigHandler.

    A class to handle quality control configuration
    parameters.

    Parameters
    ----------
    qc_config : str
        Path of the quality control configuration file
    """

    ids = ("qc_conf",)

    def __init__(self, qc_config_path: str | Path):
        self.qc_config_path = qc_config_path

    def load(self) -> QualityControlConfig:
        """Load, parse, and validate the quality control configuration.

        Returns
        -------
        QualityControlConfig
            Parsed and validated quality control configuration.

        Raises
        ------
        TypeError
            If a configuration section has an invalid structure or type.

        ValueError
            If the parsed configuration contains inconsistent references
            between metrics, resources, or rejection policies.
        """
        qc_config = read_yaml(self.qc_config_path)
        config = {}

        for section, parser in SECTION_PARSERS.items():
            values = qc_config.get(section, {})
            config[section] = parser(values)

        qc = QualityControlConfig(**config)

        validate_quality_control_config(qc)

        return qc
