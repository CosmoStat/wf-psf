"""Resource identifiers for the quality control pipeline.

Defines the resource identifier representation used to parse and resolve
resources required by enabled quality metrics.

:Authors:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Self


@dataclass(frozen=True)
class ResourceIdentifier:
    """Parsed resource identifier.

    Represents a resource identifier as a resource family and variant.

    Attributes
    ----------
    family : str
        Resource family used as a key in the resource preparer registry.
    variant : str
        Resource variant identifying a particular resource configuration
        within the family.
    """

    family: str
    variant: str

    @classmethod
    def from_string(cls, identifier: str) -> Self:
        """Create a resource identifier from its string representation.

        Parameters
        ----------
        identifier : str
            Resource identifier in the form
            ``<resource_family>.<resource_variant>``.

        Returns
        -------
        ResourceIdentifier
            Parsed resource identifier containing the resource family and
            variant.

        Raises
        ------
        ValueError
            If the identifier does not have the required
            ``<resource_family>.<resource_variant>`` format.
        """
        parts = identifier.split(".")

        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"Resource identifier '{identifier}' must have the form "
                "'<resource_family>.<resource_variant>'."
            )

        return cls(family=parts[0], variant=parts[1])

    def __str__(self) -> str:
        """Return the string representation of the resource identifier."""
        return f"{self.family}.{self.variant}"
