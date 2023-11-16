"""UNIT TESTS FOR PACKAGE MODULE: PSF MODELS.

This module contains unit tests for the wf_psf.psf_models psf_models module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""

import pytest
from wf_psf.psf_models import psf_models
from wf_psf.utils.io import FileIOHandler
import os


def test_get_psf_model_weights_filepath():
    weights_filepath = "src/wf_psf/tests/data/validation/main_random_seed/checkpoint/checkpoint*_poly*_sample_w_bis1_2k_cycle2*"

    ans = psf_models.get_psf_model_weights_filepath(weights_filepath)
    assert (
        ans
        == "src/wf_psf/tests/data/validation/main_random_seed/checkpoint/checkpoint_callback_poly_sample_w_bis1_2k_cycle2"
    )
