"""Quality control pipeline configuration.

Defines configuration interfaces for the quality control framework,
including metric, rejection policy, and reporting configuration.

:Authors:
    Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from __future__ import annotations
from dataclasses import dataclass, field
from types import NoneType
from wf_psf.utils.read_config import read_yaml


@dataclass
class MetricConfig:
    enabled: bool = True
    params: dict = field(default_factory=dict)

@dataclass
class RejectionConfig:
    enabled: bool = False
    threshold: float | None = None

@dataclass
class ReportingConfig:
    save_metrics: bool = False
    log_statistics: bool = False

@dataclass
class QualityControlConfig:
    metrics: dict[str, MetricConfig]
    rejection: dict[str, RejectionConfig]
    reporting: ReportingConfig

# config section parsers
def parse_metrics_config(config):
    """Parse Metrics Configuration Section."""
    return {
        name: MetricConfig(
            enabled=cfg.get("enabled", True),
            params={
                k: v
                for k, v in cfg.items()
                if k != "enabled"
            },
        )
        for name, cfg in config.items()
    } 

def parse_rejection_config(config):
    """Parse Rejection Configuration Section."""
    return {
        name: RejectionConfig(**cfg)
        for name, cfg in config.items()
    }

def parse_reporting_config(config):
    """Parse Reporting Configuration Section."""
    return ReportingConfig(**config)

SECTION_PARSERS = {
    "metrics": parse_metrics_config, 
    "rejection": parse_rejection_config, 
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

    def __init__(self, qc_config_path):
       self.qc_config_path = qc_config_path

    def load(self):
        self.qc_config = read_yaml(self.qc_config_path)
        config = {}

        for section, parser in SECTION_PARSERS.items():
            values = self.qc_config.get(section, {})
            config[section] = parser(values)
            
        return QualityControlConfig(**config)
            


   
