"""UNIT TESTS FOR PACKAGE MODULE: Quality Control Resources Identifiers

This module contains unit tests for the quality control resources identifiers module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import pytest
from wf_psf.quality_control.resource_identifier import ResourceIdentifier


@pytest.mark.parametrize(
    ("identifier", "family", "variant"),
    [
        ("psf_models.standard", "psf_models", "standard"),
        ("psf_models.oversampled", "psf_models", "oversampled"),
        ("foo.bar", "foo", "bar"),
    ],
)
def test_resource_identifier_from_string_valid(identifier, family, variant):
    resource_id = ResourceIdentifier.from_string(identifier)

    assert resource_id.family == family
    assert resource_id.variant == variant


@pytest.mark.parametrize(
    "identifier",
    [
        "psf_model_standard",
        "psf_models.standard.foo",
        "psf_models.",
        ".standard",
    ],
)
def test_resource_identifier_from_string_invalid(identifier):
    with pytest.raises(
        ValueError,
        match=(
            f"Resource identifier '{identifier}' must have the form "
            "'<resource_family>.<resource_variant>'."
        ),
    ):
        ResourceIdentifier.from_string(identifier)


def test_resource_identifier_str():
    resource_id = ResourceIdentifier(
        family="psf_models",
        variant="standard",
    )

    assert str(resource_id) == "psf_models.standard"
