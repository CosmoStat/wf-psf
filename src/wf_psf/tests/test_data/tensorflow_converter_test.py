import pytest
import numpy as np
import tensorflow as tf
from wf_psf.data.tensorflow_converter import TensorFlowDatasetConverter
from wf_psf.data.data_utils import DatasetContainer
from wf_psf.data.schemas import DatasetMode, SCHEMAS

# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------

@pytest.fixture
def mock_simPSF(mocker):
    mock = mocker.Mock()

    # deterministic output for SED pipeline
    mock.calc_SED_wave_values.return_value = (
        np.array([1.0, 2.0, 3.0]),   # feasible_wv
        np.array([0.5])             # SED_norm
    )

    return mock

@pytest.fixture
def mock_process_seds(mocker):
    def _mock_process_seds(sed_data, simPSF, n_bins_lambda):
        n_sources = len(sed_data)

        return tf.zeros(
            (n_sources, n_bins_lambda, 3),
            dtype=tf.float32,
        )

    return mocker.patch.object(
        TensorFlowDatasetConverter,
        "process_seds",
        side_effect=_mock_process_seds,
    )


@pytest.fixture
def converter():
    """TensorFlowDatasetConverter instance with mock simPSF."""
    return TensorFlowDatasetConverter()


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

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def get_schema(mode: DatasetMode):
    return SCHEMAS[mode]

# ------------------------------------------------------------
# CONTRACT TESTS
# ------------------------------------------------------------

@pytest.mark.parametrize("mode", [
    DatasetMode.TRAIN,
    DatasetMode.EVALUATION,
    DatasetMode.INFERENCE,
])
def test_convert_dataset_returns_all_required_fields(
    converter,
    dataset_dict,
    mock_simPSF,
    mock_process_seds,
    mode,
):
    """
    Contract:
    All schema-required fields must exist in output and be tensors.
    """
    result = converter.convert_dataset(
        dataset_dict,
        mock_simPSF,
        n_bins_lambda=10,
        mode=mode,
    )

    schema = get_schema(mode)

    for key in schema.required_keys:
        assert key in result
        assert isinstance(result[key], tf.Tensor)


@pytest.mark.parametrize("mode", [
    DatasetMode.TRAIN,
    DatasetMode.EVALUATION,
    DatasetMode.INFERENCE,
])
def test_optional_fields_are_tensors_when_present(
    converter,
    dataset_dict,
    mock_simPSF,
    mock_process_seds,
    mode,
):
    """
    Contract:
    Optional fields are converted to tensors if present.
    """
    result = converter.convert_dataset(
        dataset_dict,
        mock_simPSF,
        n_bins_lambda=10,
        mode=mode,
    )

    schema = get_schema(mode)

    for key in schema.optional_keys:
        if key in dataset_dict:
            assert isinstance(result[key], tf.Tensor)


def test_missing_required_field_raises(
    converter,
    dataset_dict,
    mock_simPSF,
):
    """
    Contract:
    Missing required field under strict schema raises ValueError.
    """
    schema = SCHEMAS[DatasetMode.TRAIN]
    missing_key = schema.required_keys[0]

    corrupted = dict(dataset_dict)
    corrupted.pop(missing_key, None)

    with pytest.raises(ValueError):
        converter.convert_dataset(
            corrupted,
            mock_simPSF,
            n_bins_lambda=10,
            mode=DatasetMode.TRAIN,
        )

def test_dataset_container_and_dict_equivalence(
    converter,
    dataset_dict,
    dataset_container,
    mock_simPSF,
    mock_process_seds,
):
    """
    Contract:
    DatasetContainer and dict inputs produce identical keys in output.
    """
    dict_result = converter.convert_dataset(
        dataset_dict,
        mock_simPSF,
        n_bins_lambda=10,
        mode=DatasetMode.TRAIN,
    )

    container_result = converter.convert_dataset(
        dataset_container,
        mock_simPSF,
        n_bins_lambda=10,
        mode=DatasetMode.TRAIN,
    )

    assert set(dict_result.keys()) == set(container_result.keys())

def test_seds_are_always_converted_to_tensor(
    converter,
    dataset_dict,
    mock_simPSF,
    mock_process_seds,
):
    """
    Contract:
    SED field must always be converted to TensorFlow tensor.
    """
    result = converter.convert_dataset(
        dataset_dict,
        mock_simPSF,
        n_bins_lambda=10,
        mode=DatasetMode.TRAIN,
    )

    assert isinstance(result["seds"], tf.Tensor)
  

def test_optional_keys_removed_input_does_not_break_pipeline(
    converter,
    dataset_dict,
    mock_simPSF,
    mock_process_seds,
):
    """
    Contract:
    Missing optional keys must not raise errors.
    """
    schema = SCHEMAS[DatasetMode.TRAIN]

    cleaned = dict(dataset_dict)
    for k in schema.optional_keys:
        cleaned.pop(k, None)

    result = converter.convert_dataset(
        cleaned,
        mock_simPSF,
        n_bins_lambda=10,
        mode=DatasetMode.TRAIN,
    )

    for k in schema.required_keys:
        assert k in result

