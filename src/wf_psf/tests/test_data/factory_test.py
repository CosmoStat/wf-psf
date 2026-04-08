# test_factory.py
import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional
from wf_psf.data.data_adapter import DataAdapter, LoadedDataset
from wf_psf.data.factory import normalize_data_envelope, DataAdapterFactory
from wf_psf.utils.read_config import RecursiveNamespace


# -----------------------------
# Tests for normalize_data_envelope
# -----------------------------
@dataclass
class MyDataclass:
    x: int
    y: int
    params: dict
    metadata: Optional[dict] = None


def test_extract_from_dataclass():
    obj = MyDataclass(x=1, y=2, params={"lr": 0.01}, metadata={"object_id": 1234})
    env = normalize_data_envelope(obj)
    assert env.params == {"lr": 0.01}
    assert env.data == {"x": 1, "y": 2}
    assert env.metadata == {"object_id": 1234}


def test_normalize_envelope_with_dict():
    obj = {
        "a": np.array([1, 2]),
        "params": {"batch": 32},
        "metadata": {"object_id": 1234},
    }
    env = normalize_data_envelope(obj)

    assert env.params == {"batch": 32}
    assert env.metadata == {"object_id": 1234}
    np.testing.assert_array_equal(env.data["a"], obj["a"])


def test_normalize_envelope_with_generic_object():
    class Obj:
        def __init__(self):
            self.foo = 123
            self.params = {"drop": 0.5}
            self.metadata = None

    obj = Obj()
    env = normalize_data_envelope(obj)
    assert env.params == {"drop": 0.5}
    assert env.data == {"foo": 123}
    assert env.metadata is None


def test_extract_params_only():
    obj = {"params": {"only": True}}
    env = normalize_data_envelope(obj)
    assert env.params == {"only": True}
    assert env.data is None


# -----------------------------
# Tests for DataAdapterFactory._resolve_dataset
# -----------------------------
class TestDataAdapterFactory:
    def test_resolve_dataset_with_in_memory_complete(self):
        class Data:
            complete = np.array([1, 2, 3])
            params = {"lr": 0.1}

        dataset, params, metadata = DataAdapterFactory._resolve_dataset(Data)
        assert isinstance(dataset, LoadedDataset)
        assert params == {"lr": 0.1}
        assert metadata is None

    def test_resolve_dataset_with_train_test(self):
        class Data:
            train = np.array([1, 2])
            test = np.array([3, 4])
            params = {"lr": 0.01}
            metadata = {"object_id": [1, 2]}

        dataset, params, metadata = DataAdapterFactory._resolve_dataset(Data)
        assert isinstance(dataset, LoadedDataset)
        assert hasattr(dataset, "train")
        assert hasattr(dataset, "test")
        assert params == {"lr": 0.01}
        assert metadata == {"object_id": [1, 2]}

    def test_resolve_dataset_with_in_memory_shallow(self):
        class Data:
            positions = np.array([1, 2, 3])
            params = {"lr": 0.1}

        dataset, params, metadata = DataAdapterFactory._resolve_dataset(Data)
        assert isinstance(dataset, LoadedDataset)
        assert params == {"lr": 0.1}
        assert metadata is None

    def test_resolve_dataset_split_params_only_loads(self, monkeypatch, data_params):
        # Patch _load_dataset to return a mock
        monkeypatch.setattr(
            DataAdapterFactory,
            "_load_dataset",
            lambda p: LoadedDataset(complete="loaded"),
        )

        dataset, params, metadata = DataAdapterFactory._resolve_dataset(data_params)
        assert isinstance(dataset, LoadedDataset)
        assert dataset.complete == "loaded"
        assert params == data_params.params
        assert metadata is None

    def test_resolve_dataset_complete_params_only_loads(self, monkeypatch):
        # Create complete data params
        data_params = RecursiveNamespace(
            params=RecursiveNamespace(
                complete=RecursiveNamespace(
                    data_dir="data",
                    file="coherent_euclid_dataset/train_Euclid_res_200_TrainStars_id_001.npy",
                    target_field="noisy_stars",
                ),
                canonical_keys=["sources", "masks", "positions"],
            )
        )
        # Patch _load_dataset to return a mock
        monkeypatch.setattr(
            DataAdapterFactory,
            "_load_dataset",
            lambda p: LoadedDataset(complete="loaded"),
        )

        dataset, params, metadata = DataAdapterFactory._resolve_dataset(data_params)
        assert isinstance(dataset, LoadedDataset)
        assert dataset.complete == "loaded"
        assert params == data_params.params
        assert metadata is None

    def test_resolve_dataset_no_data_raises(self):
        class Wrapper:
            params = {"something": 1}
            # No data attributes

        with pytest.raises(ValueError):
            DataAdapterFactory._resolve_dataset(Wrapper())

    def test_resolve_dataset_shallow_params_only_loads(self, monkeypatch):
        # Create complete data params
        data_params = RecursiveNamespace(
            params=RecursiveNamespace(
                data_dir="data",
                file="coherent_euclid_dataset/train_Euclid_res_200_TrainStars_id_001.npy",
                target_field="noisy_stars",
                canonical_keys=["sources", "masks", "positions"],
            )
        )
        # Patch _load_dataset to return a mock
        monkeypatch.setattr(
            DataAdapterFactory,
            "_load_dataset",
            lambda p: LoadedDataset(complete="loaded"),
        )

        dataset, params, metadata = DataAdapterFactory._resolve_dataset(data_params)
        assert isinstance(dataset, LoadedDataset)
        assert dataset.complete == "loaded"
        assert params == data_params.params
        assert metadata is None

    # -----------------------------
    # Optional: test DataAdapterFactory.build (mocked)
    # -----------------------------

    def test_build_calls_resolve_dataset(self, monkeypatch, data_params):
        class Data:
            complete = {"sources": np.array([1, 2])}
            params = data_params
            metadata = {"object_id": [1]}

        def fake_resolve_dataset(data):
            return (
                LoadedDataset(complete=data.complete),
                data.params,
                data.metadata,
            )

        monkeypatch.setattr(
            DataAdapterFactory,
            "_resolve_dataset",
            fake_resolve_dataset,
        )
        monkeypatch.setattr(
            "wf_psf.data.factory.TensorFlowDatasetConverter", lambda: "converter"
        )

        adapter = DataAdapterFactory.build(Data)
        # Should return a mock adapter object
        assert isinstance(adapter, DataAdapter)
        assert adapter._converter == "converter"
        assert adapter.params == data_params
        assert adapter.metadata == {"object_id": [1]}
