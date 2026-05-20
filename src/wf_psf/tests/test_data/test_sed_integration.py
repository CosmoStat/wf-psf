import pytest
import numpy as np
import tensorflow as tf
from wf_psf.data.tensorflow_converter import TensorFlowDatasetConverter
from wf_psf.data.schemas import DatasetMode


@pytest.fixture
def simple_sed_dataset():
    return {
        "positions": np.array([[0.0, 0.0]], dtype=np.float32),
        "sources": np.ones((1, 32, 32), dtype=np.float32),
        "seds": np.array([
            [
                [550.0, 1.0],
                [700.0, 0.8],
                [900.0, 0.5],
            ]
        ], dtype=np.float32),
    }

@pytest.fixture
def converter():
    """TensorFlowDatasetConverter instance with mock simPSF."""
    return TensorFlowDatasetConverter()



def test_real_sed_processing_pipeline(
    converter,
    simple_sed_dataset,
    simPSF,
):
    result = converter.convert_dataset(
        simple_sed_dataset,
        simPSF,
        n_bins_lambda=10,
        mode=DatasetMode.TRAIN,
    )

    seds = result["seds"]

    assert isinstance(seds, tf.Tensor)

    assert seds.shape[0] == len(simple_sed_dataset["seds"])
    assert seds.shape[1] == 10

    assert tf.reduce_all(tf.math.is_finite(seds))