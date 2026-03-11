import pytest
import numpy as np
from wf_psf.data.data_adapter import DataAdapter, StructureState, RepresentationState
from wf_psf.data.data_utils import DatasetContainer
from wf_psf.utils.read_config import RecursiveNamespace


class FakeLoadedDataset:
    """Minimal LoadedDataset stub."""

    def __init__(self, complete=None, train=None, test=None):
        self.complete = complete
        self.train = train
        self.test = test

    def is_complete(self):
        return self.complete is not None

    def is_split(self):
        return self.train is not None and self.test is not None


def shallow_params():
    return RecursiveNamespace(
        target_field="sources",
        train_fraction=0.8,
        seed=42,
        canonical_keys=["sources", "masks", "positions"],
    )


def complete_params():
    return RecursiveNamespace(
        complete=RecursiveNamespace(target_field="sources"),
        canonical_keys=["sources", "masks", "positions"],
    )


def split_params():
    return RecursiveNamespace(
        train=RecursiveNamespace(target_field="noisy_stars"),
        test=RecursiveNamespace(target_field="stars"),
        canonical_keys=["sources", "masks", "positions"],
    )


@pytest.fixture
def fake_metadata():
    return {"object_id": list(np.arange(10))}


@pytest.fixture
def numpy_dataset():
    """Simple dataset of numpy arrays."""
    return {
        "positions": np.random.rand(10, 2),
        "sources": np.random.rand(10, 32, 32),
        "masks": np.ones((10, 32, 32)),
    }


@pytest.fixture
def numpy_train_dataset():
    """Simple dataset of numpy arrays."""
    return {
        "positions": np.random.rand(10, 2),
        "noisy_stars": np.random.rand(10, 32, 32),
        "masks": np.ones((10, 32, 32)),
    }


@pytest.fixture
def numpy_test_dataset():
    """Simple dataset of numpy arrays."""
    return {
        "positions": np.random.rand(10, 2),
        "stars": np.random.rand(10, 32, 32),
        "masks": np.ones((10, 32, 32)),
    }


@pytest.fixture
def loaded_complete(numpy_dataset):
    """LoadedDataset with complete data."""
    return FakeLoadedDataset(complete=numpy_dataset)


@pytest.fixture
def loaded_split(numpy_train_dataset, numpy_test_dataset):
    """LoadedDataset already split."""
    train = {k: v[:8] for k, v in numpy_train_dataset.items()}
    test = {k: v[8:] for k, v in numpy_test_dataset.items()}
    return FakeLoadedDataset(train=train, test=test)


@pytest.fixture
def mock_converter(mocker):
    """Mock TensorFlowDatasetConverter."""
    mock = mocker.Mock()
    mock.convert_dataset.side_effect = lambda d, *_: {"converted": True}
    return mock


@pytest.fixture
def adapter(request, loaded_complete, loaded_split, mock_converter):
    dataset, params = request.param

    return DataAdapter(
        dataset=dataset,
        converter=mock_converter,
        params=params,
    )


@pytest.mark.parametrize(
    "params,split,expected",
    [
        (shallow_params(), None, "sources"),
        (complete_params(), None, "sources"),
        (split_params(), "train", "noisy_stars"),
        (split_params(), "test", "stars"),
    ],
)
def test_resolve_target_field(params, split, expected, loaded_complete, mock_converter):
    adapter = DataAdapter(
        dataset=loaded_complete,
        converter=mock_converter,
        params=params,
    )

    assert adapter._resolve_target_field(split) == expected


@pytest.mark.parametrize(
    "params",
    [
        shallow_params(),
        complete_params(),
    ],
)
def test_init_complete_dataset(params, loaded_complete, mock_converter, fake_metadata):
    adapter_complete = DataAdapter(
        dataset=loaded_complete,
        converter=mock_converter,
        params=params,
        metadata=fake_metadata,
    )

    assert adapter_complete.structure_state.name == "COMPLETE"
    assert adapter_complete.representation_state.name == "NUMPY"

    assert isinstance(adapter_complete.complete_data, DatasetContainer)
    assert adapter_complete.train_data is None
    assert adapter_complete.test_data is None

    assert adapter_complete.metadata == fake_metadata
    assert adapter_complete.params == params

    assert "sources" in adapter_complete.complete_data
    assert "stars" not in adapter_complete.complete_data

    assert set(adapter_complete.complete_data.keys()) == set(
        loaded_complete.complete.keys()
    )

    for k, v in loaded_complete.complete.items():
        np.testing.assert_array_equal(adapter_complete.complete_data[k], v)


def test_init_split(loaded_split, mock_converter):
    """Adapter initializes correctly for complete dataset with no metadata."""
    adapter_split = DataAdapter(
        dataset=loaded_split,
        converter=mock_converter,
        params=split_params(),
    )
    assert adapter_split.structure_state.name == "SPLIT"
    assert adapter_split.representation_state.name == "NUMPY"

    assert adapter_split.complete_data is None
    assert adapter_split.metadata is None
    assert isinstance(adapter_split.train_data, DatasetContainer)
    assert isinstance(adapter_split.test_data, DatasetContainer)

    assert "sources" in adapter_split.train_data
    assert "sources" in adapter_split.test_data
    assert "noisy_stars" not in adapter_split.test_data
    assert "stars" not in adapter_split.test_data

    np.testing.assert_array_equal(
        adapter_split.train_data["sources"], loaded_split.train["noisy_stars"]
    )

    np.testing.assert_array_equal(
        adapter_split.test_data["sources"], loaded_split.test["stars"]
    )

    np.testing.assert_array_equal(
        adapter_split.train_data["positions"], loaded_split.train["positions"]
    )

    np.testing.assert_array_equal(
        adapter_split.test_data["positions"], loaded_split.test["positions"]
    )


class TestDataAdapterCanonicalization:
    """Test canonicalization of dataset fields in DataAdapter."""

    @pytest.fixture
    def dataset_complete_custom(self):
        """Complete dataset with a non-canonical 'sources' key."""
        return {
            "positions": np.random.rand(5, 2),
            "noisy_stars": np.random.rand(5, 16, 16),
            "masks": np.ones((5, 16, 16)),
            "SEDs": np.random.randn(5, 3, 2).astype(np.float32),
        }

    @pytest.fixture
    def loaded_complete_custom(self, dataset_complete_custom):
        return FakeLoadedDataset(complete=dataset_complete_custom)

    @pytest.fixture
    def adapter_complete_custom(self, loaded_complete_custom, mock_converter):
        """Adapter for dataset with custom target field."""

        class ParamsWithTarget:
            target_field = "noisy_stars"

        return DataAdapter(
            dataset=loaded_complete_custom,
            converter=mock_converter,
            params=ParamsWithTarget(),
        )

    def test_canonicalize_complete_custom(
        self, adapter_complete_custom, dataset_complete_custom
    ):
        """Canonicalization maps custom key to 'sources' correctly."""
        adapter = adapter_complete_custom

        np.testing.assert_array_equal(
            adapter.complete_data["sources"], dataset_complete_custom["noisy_stars"]
        )

        assert "noisy_stars" not in adapter.complete_data
        assert "sources" in adapter.complete_data

        # Other keys intact
        np.testing.assert_array_equal(
            adapter.complete_data["positions"], dataset_complete_custom["positions"]
        )
        np.testing.assert_array_equal(
            adapter.complete_data["masks"], dataset_complete_custom["masks"]
        )

    @pytest.fixture
    def dataset_split_custom(self):
        """Split dataset with custom keys."""
        train = {
            "positions": np.random.rand(6, 2),
            "stars": np.random.rand(6, 16, 16),
            "masks": np.ones((6, 16, 16)),
        }
        test = {
            "positions": np.random.rand(4, 2),
            "stars": np.random.rand(4, 16, 16),
            "masks": np.ones((4, 16, 16)),
        }
        return FakeLoadedDataset(train=train, test=test)

    @pytest.fixture
    def adapter_split_custom(self, dataset_split_custom, mock_converter):
        class ParamsWithSplitTarget:
            train = type("TrainParams", (), {"target_field": "stars"})
            test = type("TestParams", (), {"target_field": "stars"})

        return DataAdapter(
            dataset=dataset_split_custom,
            converter=mock_converter,
            params=ParamsWithSplitTarget(),
        )

    def test_canonicalize_split_custom(
        self, adapter_split_custom, dataset_split_custom
    ):
        """Canonicalization works for split datasets."""
        # Train
        np.testing.assert_array_equal(
            adapter_split_custom.train_data["sources"],
            dataset_split_custom.train["stars"],
        )
        assert "stars" not in adapter_split_custom.train_data
        # Test
        np.testing.assert_array_equal(
            adapter_split_custom.test_data["sources"],
            dataset_split_custom.test["stars"],
        )
        assert "stars" not in adapter_split_custom.test_data

    def test_missing_target_field_raises(self, mock_converter):
        dataset = FakeLoadedDataset(
            complete={
                "positions": np.random.rand(5, 2),
                "stars": np.random.rand(5, 16, 16),
            }
        )

        class BadParams:
            target_field = "noisy_stars"

        with pytest.raises(KeyError) as exc_info:
            _ = DataAdapter(
                dataset=dataset,
                converter=mock_converter,
                params=BadParams(),
            )

        assert f"Target field '{BadParams.target_field}' not found." in str(
            exc_info.value
        )


def test_adapter_does_not_mutate_input_dataset(mock_converter):
    dataset = {
        "positions": np.random.rand(5, 2),
        "noisy_stars": np.random.rand(5, 16, 16),
    }

    original_keys = set(dataset.keys())

    class Params:
        target_field = "noisy_stars"

    DataAdapter(
        dataset=FakeLoadedDataset(complete=dataset),
        converter=mock_converter,
        params=Params(),
    )

    assert set(dataset.keys()) == original_keys
    assert "noisy_stars" in dataset
    assert "sources" not in dataset


class TestDataAdapterSplitData:
    """Unit tests for split_data method."""

    @pytest.mark.parametrize(
        "ratio,seed,expected_train_len",
        [
            (0.6, 42, 6),  # 10 * 0.6 = 6
            (0.8, 123, 8),  # 10 * 0.8 = 8
            (0.5, 0, 5),  # 50/50 split
            (None, 42, 8),  # default ratio from params (0.8)
        ],
    )
    def test_split_data_parametrized(
        self,
        ratio,
        seed,
        expected_train_len,
        loaded_complete,
        mock_converter,
    ):
        """Split complete dataset with different ratios and seeds."""

        norm_params = RecursiveNamespace(
            target_field="sources",
            train_fraction=0.8,
            seed=42,
        )

        # Two adapters for repeatability assertion
        adapter1 = DataAdapter(
            dataset=loaded_complete,
            converter=mock_converter,
            params=norm_params,
        )
        adapter2 = DataAdapter(
            dataset=loaded_complete,
            converter=mock_converter,
            params=norm_params,
        )

        adapter1.split_data(ratio=ratio, seed=seed)
        adapter2.split_data(ratio=ratio, seed=seed)

        for k in adapter1.train_data:
            np.testing.assert_array_equal(
                adapter1.train_data[k], adapter2.train_data[k]
            )
            np.testing.assert_array_equal(adapter1.test_data[k], adapter2.test_data[k])

            assert len(adapter1.train_data[k]) == expected_train_len

    def test_split_data_error_on_split(self, loaded_split, mock_converter):
        """Splitting an already split dataset raises RuntimeError."""
        # Two adapters for repeatability assertion
        adapter_split = DataAdapter(
            dataset=loaded_split,
            converter=mock_converter,
            params=split_params(),
        )

        with pytest.raises(
            RuntimeError, match="Split only allowed from COMPLETE state"
        ):
            adapter_split.split_data()

    def test_split_data_idempotent(self, loaded_split, mock_converter):
        """Test split_data is idempotent."""
        params = RecursiveNamespace(
            train=RecursiveNamespace(target_field="noisy_stars"),
            test=RecursiveNamespace(target_field="stars"),
            canonical_keys=["sources", "masks", "positions"],
            train_fraction=0.8,
            seed=42,
        )
        adapter_split = DataAdapter(
            dataset=loaded_split,
            converter=mock_converter,
            params=params,
        )
        assert adapter_split.complete_data is None
        adapter_split.join_data()
        assert adapter_split.structure_state == StructureState.COMPLETE
        adapter_split.split_data()
        assert adapter_split.structure_state == StructureState.SPLIT
        np.testing.assert_array_equal(
            adapter_split.train_data["sources"], loaded_split.train["noisy_stars"]
        )
        np.testing.assert_array_equal(
            adapter_split.test_data["sources"], loaded_split.test["stars"]
        )

        for k in ["positions", "masks"]:
            np.testing.assert_array_equal(
                adapter_split.train_data[k], loaded_split.train[k]
            )
            np.testing.assert_array_equal(
                adapter_split.test_data[k], loaded_split.test[k]
            )


class TestDataAdapterJoinData:
    """Unit tests for join_data method."""

    @pytest.mark.parametrize(
        "keys_to_join",
        [
            None,  # use default canonical keys
            ["sources", "positions", "masks"],
            ["sources", "positions"],  # explicit subset of keys
        ],
    )
    def test_join_data(self, loaded_split, mock_converter, keys_to_join):
        """Joining split datasets produces complete dataset."""
        adapter = DataAdapter(
            dataset=loaded_split,
            converter=mock_converter,
            params=split_params(),
        )

        # Optionally override canonical keys for test
        if keys_to_join:
            adapter._canonical_keys = keys_to_join

        train_copy = {k: v.copy() for k, v in adapter.train_data.items()}
        test_copy = {k: v.copy() for k, v in adapter.test_data.items()}

        adapter.join_data(keys=keys_to_join)
        assert adapter.structure_state == StructureState.COMPLETE
        for k in adapter._canonical_keys:
            expected = np.concatenate([train_copy[k], test_copy[k]], axis=0)
            np.testing.assert_array_equal(adapter.complete_data[k], expected)

        # Subsequent join raises RuntimeError
        with pytest.raises(RuntimeError, match="Join only allowed from SPLIT state"):
            adapter.join_data(keys=keys_to_join)

    def test_join_data_error_on_complete(self, loaded_complete, mock_converter):
        """Joining a COMPLETE dataset raises RuntimeError."""
        adapter_complete = DataAdapter(
            dataset=loaded_complete, converter=mock_converter
        )
        with pytest.raises(RuntimeError, match="Join only allowed from SPLIT state"):
            adapter_complete.join_data()


class TestDataAdapterConvertTensorFlow:
    """Unit tests for convert_to_tensorflow."""

    def test_convert_to_tensorflow_complete(self, loaded_complete, mock_converter):
        """Convert complete dataset to TensorFlow representation."""
        adapter = DataAdapter(dataset=loaded_complete, converter=mock_converter)
        adapter.convert_to_tensorflow(simPSF=None, n_bins_lambda=10)
        assert adapter.representation_state == RepresentationState.TENSORFLOW
        assert adapter._complete_tf == {"converted": True}
        # Train/test remain None
        assert adapter._train_tf is None
        assert adapter._test_tf is None

    def test_convert_to_tensorflow_split(self, loaded_split, mock_converter):
        """Convert split dataset to TensorFlow representation."""
        adapter = DataAdapter(
            dataset=loaded_split, converter=mock_converter, params=split_params()
        )
        adapter.convert_to_tensorflow(simPSF=None, n_bins_lambda=10)
        assert adapter.representation_state == RepresentationState.TENSORFLOW
        assert adapter._train_tf == {"converted": True}
        assert adapter._test_tf == {"converted": True}
        assert adapter._complete_tf is None

    def test_convert_to_tensorflow_error_no_converter(
        self, loaded_complete, mock_converter
    ):
        """convert_to_tensorflow raises error if converter is missing."""

        adapter = DataAdapter(
            dataset=loaded_complete, converter=mock_converter, params=complete_params()
        )

        # Simulate invalid state
        adapter._converter = None

        with pytest.raises(RuntimeError, match="No converter provided"):
            adapter.convert_to_tensorflow(simPSF=None, n_bins_lambda=10)
