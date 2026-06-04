import pytest
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from wf_psf.data.data_adapter import DataAdapter, RepresentationState
from wf_psf.data.factory import DataAdapterFactory
from wf_psf.training.training_data_adapter import TrainingDataAdapter
from wf_psf.data.data_adapter import LoadedDataset
from wf_psf.data.schemas import DatasetMode


@pytest.fixture
def disk_dataset(tmp_path):
    """Simulate a dataset saved to disk as .npy files."""
    train_pos = np.random.rand(8, 2)
    train_src = np.random.rand(8, 16, 16)
    test_pos = np.random.rand(2, 2)
    test_src = np.random.rand(2, 16, 16)

    np.save(tmp_path / "train_positions.npy", train_pos)
    np.save(tmp_path / "train_sources.npy", train_src)
    np.save(tmp_path / "test_positions.npy", test_pos)
    np.save(tmp_path / "test_sources.npy", test_src)

    return tmp_path


@pytest.fixture
def complete_dataset():
    """Create in-memory complete dataset."""
    return LoadedDataset(
        complete={
            "positions": np.random.rand(10, 2),
            "sources": np.random.rand(10, 16, 16),
            "masks": np.ones((10, 16, 16)),
            "seds": np.random.rand(10, 3, 2),
        }
    )


@pytest.fixture
def split_dataset():
    """Create in-memory split dataset."""
    train = {
        "positions": np.random.rand(8, 2),
        "sources": np.random.rand(8, 16, 16),
        "masks": np.ones((8, 16, 16)),
        "seds": np.random.rand(8, 3, 2),
    }
    test = {
        "positions": np.random.rand(2, 2),
        "sources": np.random.rand(2, 16, 16),
        "masks": np.ones((2, 16, 16)),
        "seds": np.random.rand(2, 3, 2),
    }
    return LoadedDataset(train=train, test=test)


@pytest.fixture
def data_params():
    """Set data parameters."""

    class Params:
        target_field = "sources"
        train_fraction = 0.8
        seed = 42
        canonical_keys = ["sources", "masks", "positions"]

    return Params()


@pytest.fixture
def mock_converter(mocker):
    """Mock TensorFlowDatasetConverter to pass through NumPy."""
    mock = mocker.Mock()
    mock.convert_dataset.side_effect = (
        lambda d, simPSF, n_bins, mode, **params: d
    )  # just return the dict
    return mock


@pytest.fixture
def fake_simpsf():
    """Mock PSF simulator."""

    class FakeSimPSF:
        pass

    return FakeSimPSF()


@pytest.fixture
def n_bins_lambda():
    """Set number of wavelength bins."""
    return 10  # arbitrary for testing


@dataclass
class DataClassDataset:
    positions: np.ndarray
    sources: np.ndarray
    masks: np.ndarray
    seds: np.ndarray


@pytest.fixture
def dataclass_dataset():
    return DataClassDataset(
        positions=np.random.rand(10, 2),
        sources=np.random.rand(10, 16, 16),
        masks=np.ones((10, 16, 16)),
        seds=np.random.rand(10, 3, 2),
    )


def test_training_data_adapter_disk_integration(disk_dataset, mocker):
    """Integration test loading dataset from disk via DataAdapterFactory."""

    class DataWrapper:
        """Fake object representing a dataset loader."""

        params = type(
            "Params",
            (),
            {
                "target_field": "sources",
                "train_fraction": 0.8,
                "seed": 42,
                "canonical_keys": ["sources", "positions"],
            },
        )()
        metadata = {"object_id": list(range(10))}

        # Simulate LoadedDataset resolution
        complete = LoadedDataset(
            train={
                "positions": np.load(disk_dataset / "train_positions.npy"),
                "sources": np.load(disk_dataset / "train_sources.npy"),
            },
            test={
                "positions": np.load(disk_dataset / "test_positions.npy"),
                "sources": np.load(disk_dataset / "test_sources.npy"),
            },
        )

    # Patch factory to skip real file I/O
    mocker.patch(
        "wf_psf.data.factory.DataAdapterFactory._resolve_dataset",
        return_value=(DataWrapper.complete, DataWrapper.params, DataWrapper.metadata),
    )

    adapter = DataAdapterFactory.build(DataWrapper)
    assert adapter.train_data is not None
    assert adapter.test_data is not None

    # Wrap in TrainingDataAdapter
    tda = TrainingDataAdapter(adapter)

    # Verify inputs and targets shapes
    np.testing.assert_array_equal(tda.train_inputs[0], adapter.train_data["positions"])
    np.testing.assert_array_equal(tda.train_targets, adapter.train_data["sources"])


# ---- Case: LoadedDataset integration ----
def test_training_data_adapter_integration_complete(
    complete_dataset, data_params, mock_converter
):
    """Full integration test with complete dataset."""
    adapter = DataAdapter(
        dataset=complete_dataset,
        converter=mock_converter,
        params=data_params,
    )

    # split dataset just before training
    adapter.split_data()

    # Convert to TensorFlow representation
    if adapter.representation_state == RepresentationState.NUMPY:
        adapter.convert_to_tensorflow(fake_simpsf, n_bins_lambda, mode=DatasetMode.TRAIN)

    tda = TrainingDataAdapter(adapter, loss_type="mask_mse")

    # Check inputs
    train_inputs = tda.train_inputs
    val_inputs = tda.validation_inputs
    assert len(train_inputs) == 2
    assert len(val_inputs) == 2

    # Check targets
    train_targets = tda.train_targets
    val_targets = tda.validation_targets
    assert isinstance(train_targets, tf.Tensor)
    assert train_targets.shape[-1] == 2  # stacked sources + masks
    np.testing.assert_array_equal(train_targets[..., 0], adapter.train_data["sources"])
    np.testing.assert_array_equal(train_targets[..., 1], adapter.train_data["masks"])

    # Validation targets
    np.testing.assert_array_equal(val_targets[..., 0], adapter.test_data["sources"])
    np.testing.assert_array_equal(val_targets[..., 1], adapter.test_data["masks"])


# ---- Case: Dataclass dataset integration ----
def test_training_data_adapter_integration_dataclass(
    mocker, dataclass_dataset, mock_converter
):
    """Integration test with dataset as a dataclass."""
    data_conf = mocker.Mock()

    # Pretend normalized params
    data_conf.params = type(
        "Params",
        (),
        {
            "target_field": "sources",
            "train_fraction": 0.8,
            "seed": 42,
            "canonical_keys": [
                "sources",
                "masks",
                "positions",
            ],
        },
    )()

    # convert dataclass to LoadedDataset
    dataset_dict = {
        f: getattr(dataclass_dataset, f) for f in dataclass_dataset.__dataclass_fields__
    }
    loaded = LoadedDataset(complete=dataset_dict)

    adapter = DataAdapter(
        dataset=loaded,
        converter=mock_converter,
        params=data_conf.params,
    )

    # Split dataset
    adapter.split_data()

    # Convert to TensorFlow representation
    if adapter.representation_state == RepresentationState.NUMPY:
        adapter.convert_to_tensorflow(fake_simpsf, n_bins_lambda, mode=DatasetMode.TRAIN)

    tda = TrainingDataAdapter(adapter)

    # Inputs
    train_inputs = tda.train_inputs
    val_inputs = tda.validation_inputs

    assert len(train_inputs) == 2
    assert np.array_equal(train_inputs[0], adapter.train_data["positions"])
    assert np.array_equal(train_inputs[1], adapter.train_data["seds"])

    assert len(val_inputs) == 2
    assert np.array_equal(val_inputs[0], adapter.test_data["positions"])
    assert np.array_equal(val_inputs[1], adapter.test_data["seds"])

    # Targets
    train_targets = tda.train_targets
    val_targets = tda.validation_targets
    assert np.array_equal(train_targets, adapter.train_data["sources"])
    assert np.array_equal(val_targets, adapter.test_data["sources"])
