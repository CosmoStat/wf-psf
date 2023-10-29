"""UNIT TESTS FOR PACKAGE MODULE: VALIDATE_CONFIG.

This module contains unit tests for the wf_psf.utils validate_config module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""

from typing import Literal
from pathlib import Path
import pytest

from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.utils.validate_config import ValidateConfig
from wf_psf.utils.validate_config_dicts import validate_dict


@pytest.fixture
def path_to_tmp_config_dir(tmp_path: Path):
    return tmp_path


@pytest.mark.parametrize(
    "test_input,expected",
    [(8, True), (-888, False), (38, True), ("test", False), (382.38, False)],
)
def test_check_if_positive_int(test_input, expected):
    assert ValidateConfig.check_if_positive_int(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected", [(8.0, True), (-88.8, False), (3, False), ("test", False)]
)
def test_check_if_positive_float(test_input, expected):
    assert ValidateConfig.check_if_positive_float(test_input) == expected


@pytest.mark.parametrize(
    "key,expected",
    [("id_name", "_sample_w_bis1_2k")],
)
def test_get_attr_from_RN(training_params, key, expected):
    assert ValidateConfig.get_attr_from_RN(training_params, key) == expected


@pytest.mark.parametrize(
    "input,expected", [(None, True), ("metrics_config.yaml", False)]
)
def test_check_none(input, expected):
    assert ValidateConfig.check_none(input) is expected


@pytest.mark.parametrize(
    "input,expected",
    [
        (["poly", "poly"], True),
        (
            [
                "complete",
                [
                    "parametric",
                    "non-parametric",
                    "complete",
                    "only-non-parametric",
                    "only-parametric",
                ],
            ],
            True,
        ),
        (["poly", "mccd"], False),
    ],
)
def test_check_if_equals_name(input, expected):
    assert ValidateConfig.check_if_equals_name(input[0], input[1]) is expected


@pytest.mark.parametrize("input,expected", [(True, True), (False, True), (1923, False)])
def test_check_if_bool(input, expected):
    assert ValidateConfig.check_if_bool(input) is expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            [
                RecursiveNamespace(
                    model_name="poly",
                    n_bins_lda=8,
                ),
                {
                    "model_name": {
                        "function": ("check_if_equals_name",),
                        "name": ["poly"],
                    },
                    "n_bins_lda": {
                        "function": ("check_if_positive_int",),
                        "name": None,
                    },
                },
            ],
            [("model_name", "poly is OK"), ("n_bins_lda", "8 is OK")],
        )
    ],
)
def test_validate_recursiveNamespace(file_handler, input, expected):
    validate_obj = ValidateConfig(
        file_handler, RecursiveNamespace(conf="training"), "training_conf"
    )
    results = []
    results = validate_obj.validate_recursiveNamespace(input[0], input[1], results)
    assert results == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            [
                True,
                {
                    "function": ("check_if_bool",),
                    "name": None,
                },
            ],
            ("True is OK"),
        ),
        (
            [False, {"function": ("check_if_bool",), "name": None}],
            ("False is OK"),
        ),
        (
            ["yes", {"function": ("check_if_bool",), "name": None}],
            ("yes is NOK. check_if_bool not fulfilled."),
        ),
    ],
)
def test_validate(file_handler, input, expected):
    validate_obj = ValidateConfig(
        file_handler, RecursiveNamespace(conf="training"), "training_conf"
    )
    result = validate_obj.validate(input[0], input[1])
    assert result == expected
