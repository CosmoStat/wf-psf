"""Quality control pipeline configuration.

Defines configuration interfaces for the quality control framework,
including quality metric, rejection policy, and reporting configuration.

:Authors:
    Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from wf_psf.utils.read_config import read_yaml


@dataclass
class QualityMetricConfig:
    """Configuration for a single quality metric.

    Attributes
    ----------
        enabled: Whether the quality metric is enabled.
        params: Arbitrary quality metric-specific parameters.
    """

    enabled: bool = True
    params: dict = field(default_factory=dict)


@dataclass
class RejectionPolicyConfig:
    """Configuration for rejection policy logic for a single check.

    Attributes
    ----------
        enabled: Whether rejection policy is enabled.
        threshold: Numeric threshold used to trigger rejection policy, or None.
    """

    enabled: bool = False
    threshold: float | None = None


@dataclass
class ReportingConfig:
    """Configuration controlling reporting behavior.

    Attributes
    ----------
        save_metrics: Persist computed metrics to storage.
        log_statistics: Emit statistics to the logging system.
    """

    save_metrics: bool = False
    log_statistics: bool = False


@dataclass
class QualityControlConfig:
    """Top-level quality control configuration.

    Attributes
    ----------
        metrics: Mapping of metric name -> QualityMetricConfig.
        rejection: Mapping of check name -> RejectionPolicyConfig.
        reporting: ReportingConfig instance.
    """

    metrics: dict[str, QualityMetricConfig] = field(default_factory=dict)

    rejection: dict[str, RejectionPolicyConfig] = field(default_factory=dict)

    reporting: ReportingConfig = field(default_factory=ReportingConfig)


# config section parsers
def parse_metrics_config(
    section: dict[str, dict] | None,
) -> dict[str, QualityMetricConfig]:
    """Parse quality metric configuration section."""
    if section is None:
        return {}

    if not isinstance(section, dict):
        raise TypeError("Metrics configuration must be a mapping")

    metrics = {}

    for name, cfg in section.items():
        if not isinstance(cfg, dict):
            raise TypeError(f"Metric configuration '{name}' must be a mapping.")

        enabled = cfg.get("enabled", True)

        if not isinstance(enabled, bool):
            raise TypeError(f"Metric enabled flag for '{name}' must be boolean.")

        params = cfg.get("params", {})

        if not isinstance(params, dict):
            raise TypeError(f"Metric parameters for '{name}' must be a mapping.")

        metrics[name] = QualityMetricConfig(
            enabled=enabled,
            params=params,
        )

    return metrics


def parse_rejection_policy_config(
    config: dict[str, dict] | None,
) -> dict[str, RejectionPolicyConfig]:
    """Parse rejection policy configuration section."""
    if config is None:
        return {}

    if not isinstance(config, dict):
        raise TypeError("Rejection policy configuration must be a mapping.")

    policies = {}

    for name, cfg in config.items():
        if not isinstance(cfg, dict):
            raise TypeError(
                f"Rejection policy configuration '{name}' must be a mapping."
            )

        policies[name] = RejectionPolicyConfig(**cfg)

    return policies


def parse_reporting_config(config: dict | None) -> ReportingConfig:
    """Parse reporting configuration section."""
    if config is None:
        return ReportingConfig()

    if not isinstance(config, dict):
        raise TypeError("Reporting configuration must be a mapping.")

    return ReportingConfig(**config)


SECTION_PARSERS = {
    "metrics": parse_metrics_config,
    "rejection": parse_rejection_policy_config,
    "reporting": parse_reporting_config,
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
        """Load and parse configuration file."""
        qc_config = read_yaml(self.qc_config_path)
        config = {}

        for section, parser in SECTION_PARSERS.items():
            values = qc_config.get(section, {})
            config[section] = parser(values)

        return QualityControlConfig(**config)
