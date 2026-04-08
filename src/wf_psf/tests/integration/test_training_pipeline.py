import pytest
import numpy as np
from wf_psf.data.data_adapter import RepresentationState
from wf_psf.data.factory import DataAdapterFactory
from wf_psf.training.training_data_adapter import TrainingDataAdapter
from wf_psf.data.data_adapter import LoadedDataset


@pytest.fixture
def smoke_disk_dataset(tmp_path):
    """Create a mini dataset on disk for full pipeline smoke test."""
    # train
    np.save(tmp_path / "train_positions.npy", np.random.rand(8, 2))
    np.save(tmp_path / "train_sources.npy", np.random.rand(8, 16, 16))
    np.save(tmp_path / "train_masks.npy", np.ones((8, 16, 16)))
    np.save(tmp_path / "train_seds.npy", np.random.rand(8, 3, 2))

    # test
    np.save(tmp_path / "test_positions.npy", np.random.rand(2, 2))
    np.save(tmp_path / "test_sources.npy", np.random.rand(2, 16, 16))
    np.save(tmp_path / "test_masks.npy", np.ones((2, 16, 16)))
    np.save(tmp_path / "test_seds.npy", np.random.rand(2, 3, 2))
    return tmp_path


@pytest.fixture
def mock_converter(mocker):
    """Mock TensorFlowDatasetConverter to pass through NumPy."""
    mock = mocker.Mock()
    mock.convert_to_tensorflow.side_effect = (
        lambda d, simPSF, n_bins: d
    )  # just return the dict
    return mock


@pytest.fixture
def fake_simpsf():
    """Mock PSF simulator; just a placeholder object."""

    class FakeSimPSF:
        pass

    return FakeSimPSF()


@pytest.fixture
def n_bins_lambda():
    """Set number of wavelength bins."""
    return 10  # arbitrary for testing


def test_training_pipeline_smoke(smoke_disk_dataset, mocker, mock_converter):
    """Full smoke test: load disk data, build DataAdapter, wrap in TrainingDataAdapter."""

    # Simulate loading via DataWrapper
    class DataWrapper:
        params = type(
            "Params",
            (),
            {
                "target_field": "sources",
                "train_fraction": 0.8,
                "seed": 42,
                "canonical_keys": ["sources", "positions", "masks"],
            },
        )()
        metadata = {"object_id": list(range(10))}

        complete = LoadedDataset(
            train={
                "positions": np.load(smoke_disk_dataset / "train_positions.npy"),
                "sources": np.load(smoke_disk_dataset / "train_sources.npy"),
                "masks": np.load(smoke_disk_dataset / "train_masks.npy"),
                "seds": np.load(smoke_disk_dataset / "train_seds.npy"),
            },
            test={
                "positions": np.load(smoke_disk_dataset / "test_positions.npy"),
                "sources": np.load(smoke_disk_dataset / "test_sources.npy"),
                "masks": np.load(smoke_disk_dataset / "test_masks.npy"),
                "seds": np.load(smoke_disk_dataset / "test_seds.npy"),
            },
        )

    # Patch factory to skip actual file I/O / config resolution
    mocker.patch(
        "wf_psf.data.factory.DataAdapterFactory._resolve_dataset",
        return_value=(DataWrapper.complete, DataWrapper.params, DataWrapper.metadata),
    )

    # Build adapter via factory
    adapter = DataAdapterFactory.build(DataWrapper)
    assert adapter.train_data is not None
    assert adapter.test_data is not None

    # Patch the converter class so that DataAdapter gets mock_converter
    mocker.patch.object(adapter, "convert_to_tensorflow", lambda simPSF, n_bins: None)

    # Convert to TensorFlow representation
    if adapter.representation_state == RepresentationState.NUMPY:
        adapter.convert_to_tensorflow(fake_simpsf, n_bins_lambda)

    # Wrap in TrainingDataAdapter
    tda = TrainingDataAdapter(adapter, loss_type="mask_mse")

    # Check train inputs
    positions, seds = tda.train_inputs
    np.testing.assert_array_equal(positions, adapter.train_data["positions"])
    np.testing.assert_array_equal(seds, adapter.train_data["seds"])

    # Check train targets stacked with masks
    stacked_targets = tda.train_targets
    np.testing.assert_array_equal(
        stacked_targets[..., 0], adapter.train_data["sources"]
    )
    np.testing.assert_array_equal(stacked_targets[..., 1], adapter.train_data["masks"])

    # Validation inputs and targets
    val_inputs = tda.validation_inputs
    np.testing.assert_array_equal(val_inputs[0], adapter.test_data["positions"])
    np.testing.assert_array_equal(val_inputs[1], adapter.test_data["seds"])
    val_targets = tda.validation_targets
    np.testing.assert_array_equal(val_targets[..., 0], adapter.test_data["sources"])
    np.testing.assert_array_equal(val_targets[..., 1], adapter.test_data["masks"])
