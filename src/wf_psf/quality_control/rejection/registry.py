"""Quality control rejection policy registry.

Provides the registry responsible for storing and retrieving rejection
policy implementations.

The registry decouples rejection policy lookup from pipeline logic,
allowing new rejection policies to be registered without modifying the
quality control pipeline.

:Authors:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""

from wf_psf.quality_control.rejection.base import RejectionPolicy
from wf_psf.quality_control.rejection.threshold import ThresholdRejectionPolicy
from wf_psf.utils.registry import Registry


class RejectionPolicyRegistry(Registry[str, type[RejectionPolicy]]):
    """Registry storing available sample rejection policies."""

    def register_rejection_policy(self, rejection_cls: type[RejectionPolicy]) -> None:
        """Register a rejection policy implementation.

        Registers a class of type[RejectionPolicy] using the attribute
        `name` as a key, and the class as a value.

        Parameters
        ----------
        rejection_cls : type[RejectionPolicy]
            Rejection policy class to register

        """
        self.register(rejection_cls.name, rejection_cls)


def build_rejection_policy_registry() -> RejectionPolicyRegistry:
    """Build rejection policy registry."""
    registry = RejectionPolicyRegistry()

    registry.register_rejection_policy(ThresholdRejectionPolicy)

    return registry
