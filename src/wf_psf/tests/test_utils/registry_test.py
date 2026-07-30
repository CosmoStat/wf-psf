"""UNIT TESTS FOR PACKAGE MODULE: Generic Registry Utility.

This module contains unit tests for the generic Registry class.

:Author:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""

import pytest

from wf_psf.utils.registry import Registry


def test_register_and_get():

    registry = Registry[str, int]()

    registry.register("one", 1)

    assert registry.get("one") == 1


def test_register_duplicate_key_raises():

    registry = Registry[str, int]()

    registry.register("one", 1)

    with pytest.raises(
        ValueError,
        match="Key 'one' is already registered.",
    ):
        registry.register("one", 2)


def test_get_unknown_key_raises():

    registry = Registry[str, int]()

    with pytest.raises(
        KeyError,
        match="Key 'missing' not found in registry.",
    ):
        registry.get("missing")


def test_unregister():

    registry = Registry[str, int]()

    registry.register("one", 1)

    registry.unregister("one")

    assert "one" not in registry


def test_unregister_unknown_key_raises():

    registry = Registry[str, int]()

    with pytest.raises(
        KeyError,
        match="Key 'missing' not found in registry.",
    ):
        registry.unregister("missing")


def test_contains():

    registry = Registry[str, int]()

    registry.register("one", 1)

    assert "one" in registry
    assert "two" not in registry


def test_keys():

    registry = Registry[str, int]()

    registry.register("one", 1)
    registry.register("two", 2)

    assert set(registry.keys()) == {"one", "two"}


def test_values():

    registry = Registry[str, int]()

    registry.register("one", 1)
    registry.register("two", 2)

    assert set(registry.values()) == {1, 2}


def test_items():

    registry = Registry[str, int]()

    registry.register("one", 1)
    registry.register("two", 2)

    assert set(registry.items()) == {
        ("one", 1),
        ("two", 2),
    }


def test_iteration():

    registry = Registry[str, int]()

    registry.register("one", 1)
    registry.register("two", 2)

    assert set(registry) == {
        ("one", 1),
        ("two", 2),
    }


def test_get_falsey_value():

    registry = Registry[str, int]()

    registry.register("zero", 0)

    assert registry.get("zero") == 0


def test_get_false_boolean():

    registry = Registry[str, bool]()

    registry.register("flag", False)

    assert registry.get("flag") is False
