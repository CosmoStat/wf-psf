import pytest
import numpy as np
import tensorflow as tf
from wf_psf.data.tensorflow_converter import TensorFlowDatasetConverter
from wf_psf.data.data_utils import DatasetContainer


@pytest.fixture
def mock_simPSF(mocker):
    """Mock simPSF instance to avoid real SED processing."""
    mock = mocker.Mock(name="SimPSFToolkit")
    return mock


@pytest.fixture
def converter():
    """TensorFlowDatasetConverter instance with mock simPSF."""
    return TensorFlowDatasetConverter()


@pytest.fixture
def mock_process_seds(mocker, converter):
    """
    Mock process_seds to avoid real SED processing in converter tests.

    Returns a tensor of shape (N, n_bins_lambda, 3) to mimic real output.
    """

    def _mock_process_seds(sed_data, mock_simPSF, n_bins_lambda):
        n_sources = len(sed_data)
        n_bins_lambda = 10
        return tf.zeros((n_sources, n_bins_lambda, 3), dtype=tf.float32)

    return mocker.patch.object(
        converter, "process_seds", side_effect=_mock_process_seds
    )


@pytest.fixture
def dataset_dict():
    """Mock dataset dict."""
    return {
        "positions": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "sources": np.random.randn(2, 32, 32).astype(np.float32),
        "seds": np.random.randn(2, 3, 2).astype(np.float32),
        "masks": np.ones((2, 32, 32), dtype=np.float32),
    }


@pytest.fixture
def dataset_container(dataset_dict):
    """Wrap dataset_dict in DatasetContainer."""
    return DatasetContainer(dataset_dict)


def test_convert_dataset_with_dict(
    converter, dataset_dict, mock_simPSF, mock_process_seds
):
    """Test conversion when dataset is a plain dict."""
    result = converter.convert_dataset(dataset_dict, mock_simPSF, n_bins_lambda=10)

    # Required keys
    for key in converter.REQUIRED_KEYS:
        assert key in result
        assert isinstance(result[key], tf.Tensor)

    # Optional keys present
    for key in converter.OPTIONAL_KEYS:
        if key in dataset_dict:
            assert isinstance(result[key], tf.Tensor)


def test_convert_dataset_with_container(
    converter, dataset_container, mock_simPSF, mock_process_seds
):
    """Test conversion when dataset is a DatasetContainer."""
    result = converter.convert_dataset(dataset_container, mock_simPSF, n_bins_lambda=10)

    # Check required keys
    for key in converter.REQUIRED_KEYS:
        assert key in result
        assert isinstance(result[key], tf.Tensor)

    # Optional keys present
    for key in converter.OPTIONAL_KEYS:
        if hasattr(dataset_container, key):
            assert isinstance(result[key], tf.Tensor)


@pytest.mark.parametrize("missing_key", TensorFlowDatasetConverter.REQUIRED_KEYS)
def test_missing_required_key_raises(
    converter, dataset_dict, mock_simPSF, mock_process_seds, missing_key
):
    """Ensure ValueError is raised if a required key is missing."""
    data = dict(dataset_dict)
    data.pop(missing_key, None)

    with pytest.raises(
        ValueError, match=f"Required dataset field '{missing_key}' is missing."
    ):
        converter.convert_dataset(data, mock_simPSF, n_bins_lambda=10)


def test_optional_keys_skipped_if_missing(
    converter, dataset_dict, mock_simPSF, mock_process_seds
):
    """Optional keys not present in dataset are skipped without error."""
    data = dict(dataset_dict)
    # Remove optional keys
    for key in converter.OPTIONAL_KEYS:
        data.pop(key, None)

    result = converter.convert_dataset(data, mock_simPSF, n_bins_lambda=10)
    # Ensure all required keys exist
    for key in converter.REQUIRED_KEYS:
        assert key in result
    # Ensure optional keys are not present
    for key in converter.OPTIONAL_KEYS:
        assert key not in result


def test_process_seds_called_for_seds_key(
    converter, dataset_dict, mock_simPSF, mock_process_seds, mocker
):
    """Ensure process_seds is called only for the 'seds' key."""
    spy = mocker.spy(converter, "process_seds")
    converter.convert_dataset(dataset_dict, mock_simPSF, n_bins_lambda=10)

    # Should have been called exactly once
    spy.assert_called_once()
    args, kwargs = spy.call_args
    # First argument should match dataset seds
    assert np.allclose(args[0], dataset_dict.get("seds"))


def test_custom_required_optional_keys(
    converter, dataset_dict, mock_simPSF, mock_process_seds
):
    """Test using custom required and optional keys."""
    required = ("positions",)
    optional = ("masks",)
    result = converter.convert_dataset(
        dataset_dict,
        mock_simPSF,
        n_bins_lambda=10,
        required_keys=required,
        optional_keys=optional,
    )
    # Check requested keys are in result
    for k in ["positions", "masks"]:
        assert k in result.keys()


def test_seds_tensor_shape(converter, dataset_dict, mock_process_seds, mock_simPSF):
    """Check mocked SED tensor shape returned from convert_dataset."""
    n_bins_lambda = 10
    result = converter.convert_dataset(dataset_dict, mock_simPSF, n_bins_lambda)
    seds_tensor = result["seds"]
    # Mocked shape is (N, n_bins_lambda, 3)
    assert seds_tensor.shape[1] == n_bins_lambda
    assert seds_tensor.shape[2] == 3
