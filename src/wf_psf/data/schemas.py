"""
Dataset schema definitions for canonical dataset handling.

This module defines dataset validation and conversion schemas used across
training, evaluation, and inference workflows. Schemas specify which
canonical dataset fields are required, which are optional, and whether
missing required fields should raise an exception.

These schemas provide a centralized contract between dataset adapters,
preprocessing pipelines, and TensorFlow conversion utilities.

Canonical dataset field names are defined in ``constants.py`` and represent
the normalized internal dataset interface used throughout the library,
independent of external dataset naming conventions.

Author(s): Jennifer Pollack <jennifer.pollack@cea.fr>
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Any
from wf_psf.data.constants import (
    CANONICAL_DATASET_KEYS,
    OPTIONAL_KEYS as CONST_OPTIONAL_KEYS,
    SED_DOMAIN,
)
from wf_psf.data.handlers import process_seds_handler


class DatasetMode(Enum):
    """
    Enumeration of supported dataset operation modes.

    These modes define the expected dataset contract for different stages
    of the wf-psf workflow.

    Attributes
    ----------
    TRAIN
        Dataset schema used during model training.
    EVALUATION
        Dataset schema used during evaluation.
    INFERENCE
        Dataset schema used during inference or prediction.
    """

    TRAIN = auto()
    EVALUATION = auto()
    INFERENCE = auto()


@dataclass(frozen=True)
class DatasetSchema:
    """
    Definition of a canonical dataset schema.

    A dataset schema specifies which canonical dataset fields are required,
    which fields are optional, and whether missing required fields should
    raise an exception during validation or conversion.

    Parameters
    ----------
    id: str
        Schema identifier (e.g. "train", "evaluation", "inference")
    required_keys : tuple[str, ...]
        Canonical dataset fields that must be present.
    optional_keys : tuple[str, ...]
        Canonical dataset fields that may be present and will be processed
        if available.
    strict : bool, optional
        If ``True``, missing required fields raise an exception.
        If ``False``, missing required fields generate warnings and are
        skipped. Default is ``True``.
    handlers : dict[str, Callable[..., Any]] = None
        Handler for specific dataset fields (e.g. seds)
    """
    id: str
    required_keys: tuple[str, ...]
    optional_keys: tuple[str, ...]
    strict: bool = True
    handlers: dict[str, Callable[..., Any]] = field(default_factory=dict)


TRAIN_SCHEMA = DatasetSchema(
    id="train",
    required_keys=CANONICAL_DATASET_KEYS,
    optional_keys=CONST_OPTIONAL_KEYS,
    strict=True,
    handlers={SED_DOMAIN: process_seds_handler}
)
"""
Dataset schema used during model training.

All canonical dataset fields are required during training.
Missing required fields raise exceptions.
"""


EVALUATION_SCHEMA = DatasetSchema(
    id="evaluation",
    required_keys=CANONICAL_DATASET_KEYS,
    optional_keys=CONST_OPTIONAL_KEYS,
    strict=True,
    handlers={"seds": process_seds_handler}
)
"""
Dataset schema used during model evaluation.

All canonical dataset fields are required during evaluation.
Missing required fields raise exceptions.
"""


INFERENCE_SCHEMA = DatasetSchema(
    id="inference",
    required_keys=(
        "seds",
        "positions",
    ),
    optional_keys=CONST_OPTIONAL_KEYS,
    strict=False,
    handlers={"seds": process_seds_handler}
)
"""
Dataset schema used during inference.

Inference requires only the minimal subset of canonical fields needed
for prediction. Missing required fields generate warnings rather than
raising exceptions.
"""


SCHEMAS = {
    DatasetMode.TRAIN: TRAIN_SCHEMA,
    DatasetMode.EVALUATION: EVALUATION_SCHEMA,
    DatasetMode.INFERENCE: INFERENCE_SCHEMA,
}
"""
Registry mapping dataset operation modes to dataset schemas.

This dictionary provides centralized access to workflow-specific dataset
contracts used throughout the preprocessing and conversion pipeline.
"""