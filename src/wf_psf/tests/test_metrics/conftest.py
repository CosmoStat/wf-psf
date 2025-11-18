"""FIXTURES FOR GENERATING TESTS FOR WF-PSF METRICS PACKAGE: CONFTEST.

This module contains fixtures to use in unit tests for the
wf_psf metrics package.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""
import pytest
from unittest.mock import patch, MagicMock
import tensorflow as tf


# Mock PSF model class
class TFSemiParametricField(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define layers and model components here

    def call(self, inputs, **kwargs):
        # Implement the forward pass
        return inputs  # Replace with actual logic

    def load_weights(self, *args, **kwargs):
        # Simulate the weight loading
        pass

class TFGroundTruthSemiParametricField(TFSemiParametricField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define additional components if necessary

    def call(self, inputs, **kwargs):
        return inputs

@pytest.fixture
def mock_psf_model():
    # Return a mock instance of TFSemiParametricField
    psf_model = TFSemiParametricField()
    psf_model.load_weights = MagicMock()  # Mock load_weights method
    return psf_model

@pytest.fixture
def mock_get_psf_model(mock_psf_model):
    with patch('wf_psf.psf_models.psf_models.get_psf_model', return_value=mock_psf_model) as mock_method:
        yield mock_method

