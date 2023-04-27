"""FIXTURES FOR GENERATING TESTS FOR WF-PSF MODULES: CONFTEST.

This module contains fixtures to use in unit tests for 
various wf_psf packages.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""
import pytest
from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.training.train import TrainingParamsHandler

training_config=RecursiveNamespace(model_params)

chkp_dir = "/wf-outputs/checkpoint/"

@pytest.fixture(scope="module", params=[training_config, chkp_dir])
def training_params():
    return TrainingParamsHandler(training_params_set, chkp_dir)
