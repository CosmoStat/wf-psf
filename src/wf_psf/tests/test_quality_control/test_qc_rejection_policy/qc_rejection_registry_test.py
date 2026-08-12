"""UNIT TESTS FOR PACKAGE MODULE: Quality Control Rejection Policy Registry.

This module contains unit tests for the RejectionPolicyRegistry class.

:Author:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""

import numpy as np
import pytest
from wf_psf.quality_control.rejection.base import RejectionPolicy
from wf_psf.quality_control.rejection.registry import (
    RejectionPolicyRegistry,
    build_rejection_policy_registry,
)
from wf_psf.quality_control.rejection.threshold import ThresholdRejectionPolicy


class CustomRejectionPolicy(RejectionPolicy):
    """Test rejection policy used to verify registry extensibility."""

    name = "custom_policy"

    def apply(self, metric) -> np.ndarray:
        """Return a dummy validity mask for testing."""
        return np.ones(metric.shape, dtype=bool)


def test_build_rejection_registry():
    registry = build_rejection_policy_registry()

    assert registry.get("threshold") is ThresholdRejectionPolicy


def test_duplicate_rejection_policy_registration_raises():
    registry = RejectionPolicyRegistry()

    registry.register_rejection_policy(ThresholdRejectionPolicy)

    with pytest.raises(ValueError):
        registry.register_rejection_policy(ThresholdRejectionPolicy)


def test_unknown_policy_raises():
    registry = build_rejection_policy_registry()

    with pytest.raises(KeyError):
        registry.get("unknown_policy")


def test_custom_policy_registration():
    registry = build_rejection_policy_registry()

    registry.register_rejection_policy(CustomRejectionPolicy)

    policy_cls = registry.get("custom_policy")
    policy = policy_cls()

    metric = np.array([0.1, 0.5, 0.9])
    result = policy.apply(metric)

    assert isinstance(result, np.ndarray)
    assert result.shape == metric.shape
    assert result.dtype == bool
