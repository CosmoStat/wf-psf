import pytest
import numpy as np
import tensorflow as tf
from wf_psf.data.simulation_data_loader import SimulationDataLoader
from wf_psf.utils.read_config import RecursiveNamespace


@pytest.fixture
def data_dir(tmp_path):
    """Create temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def data_params(data_dir):
    """RecursiveNamespace data params pointing to tmp directory."""
    return RecursiveNamespace(data_dir=str(data_dir), file="dataset.npy")


@pytest.fixture
def training_dataset(mock_np_dataset):
    """
    Valid training dataset with noisy_stars.
    """
    return {
        "positions": mock_np_dataset["positions"],
        "noisy_stars": mock_np_dataset["noisy_stars"],
        "SEDs": mock_np_dataset["SEDs"],
    }


@pytest.fixture
def test_dataset(mock_np_dataset):
    """
    Valid test dataset with stars.
    """
    return {
        "positions": mock_np_dataset["positions"],
        "stars": mock_np_dataset["stars"],
        "SEDs": mock_np_dataset["SEDs"],
    }


@pytest.fixture
def mock_process_seds(mocker):
    """Mock process_seds on TensorFlowDatasetConverter."""

    def _mock_process_seds(sed_data):
        n_sources = len(sed_data)
        return tf.zeros(
            (n_sources, 8, 3), dtype=tf.float32
        )  # n_bins_lambda=8 from conftest

    return mocker.patch(
        "wf_psf.data.simulation_data_loader.TensorFlowDatasetConverter.process_seds",
        side_effect=_mock_process_seds,
    )


@pytest.fixture
def mock_convert_dict(mocker):
    """Mock convert_dict on TensorFlowDatasetConverter."""

    def _mock_convert_dict(dataset, dataset_type):
        return {
            k: tf.constant(v) if isinstance(v, np.ndarray) else v
            for k, v in dataset.items()
        }

    return mocker.patch(
        "wf_psf.data.simulation_data_loader.TensorFlowDatasetConverter.convert_dict",
        side_effect=_mock_convert_dict,
    )


# Helpers
def save_dataset(data_dir, dataset, filename="dataset.npy"):
    """Save dataset dict to .npy file in data_dir."""
    np.save(str(data_dir / filename), dataset)


def make_loader(dataset_type, data_params, simPSF, n_bins_lambda=8):
    """
    Create a SimulationDataLoader instance.

    Uses n_bins_lambda=8 to match simPSF fixture from conftest.py
    (training_config.model_params.n_bins_lda=8).
    """
    return SimulationDataLoader(
        dataset_type=dataset_type,
        data_params=data_params,
        simPSF=simPSF,
        n_bins_lambda=n_bins_lambda,
    )


# ============================================================================
# TestSimulationDataLoaderInit
# ============================================================================


class TestSimulationDataLoaderInit:
    """Tests for SimulationDataLoader initialisation."""

    @pytest.mark.parametrize(
        "dataset_type", ["training", "test"], ids=["training", "test"]
    )
    def test_attributes_set_on_init(self, dataset_type, data_params, simPSF):
        """Test all attributes are set correctly on initialisation."""

        loader = make_loader(dataset_type, data_params, simPSF)

        assert loader.dataset_type == dataset_type
        assert loader.data_params == data_params
        assert loader.n_bins_lambda == 8
        assert loader.dataset is None
        assert loader.sed_data is None

    def test_converter_instantiated_on_init(self, data_params, simPSF):
        """Test TensorFlowDatasetConverter is instantiated on init."""
        loader = make_loader("training", data_params, simPSF)

        assert loader.converter is not None


# ============================================================================
# TestSimulationDataLoaderLoad
# ============================================================================


class TestSimulationDataLoaderLoad:
    """Tests for SimulationDataLoader.load()."""

    @pytest.mark.parametrize(
        "dataset_type, dataset_fixture",
        [
            ("training", "training_dataset"),
            ("test", "test_dataset"),
        ],
        ids=["training", "test"],
    )
    def test_load_returns_dataset_and_sed_data(
        self,
        request,
        dataset_type,
        dataset_fixture,
        data_dir,
        data_params,
        simPSF,
        mock_process_seds,
        mock_convert_dict,
    ):
        """Test load() returns dataset and sed_data for both dataset types."""
        dataset = request.getfixturevalue(dataset_fixture)
        save_dataset(data_dir, dataset)

        loader = make_loader(dataset_type, data_params, simPSF)
        returned_dataset, returned_sed_data = loader.load()

        assert returned_dataset is not None
        assert returned_sed_data is not None

    @pytest.mark.parametrize(
        "dataset_type, dataset_fixture",
        [
            ("training", "training_dataset"),
            ("test", "test_dataset"),
        ],
        ids=["training", "test"],
    )
    def test_load_sets_instance_attributes(
        self,
        request,
        dataset_type,
        dataset_fixture,
        data_dir,
        data_params,
        simPSF,
        mock_process_seds,
        mock_convert_dict,
    ):
        """Test load() sets dataset and sed_data instance attributes."""
        dataset = request.getfixturevalue(dataset_fixture)
        save_dataset(data_dir, dataset)

        loader = make_loader(dataset_type, data_params, simPSF)
        loader.load()

        assert loader.dataset is not None
        assert loader.sed_data is not None

    @pytest.mark.parametrize(
        "dataset_type, dataset_fixture",
        [
            ("training", "training_dataset"),
            ("test", "test_dataset"),
        ],
        ids=["training", "test"],
    )
    def test_load_calls_process_seds_once(
        self,
        request,
        dataset_type,
        dataset_fixture,
        data_dir,
        data_params,
        simPSF,
        mock_process_seds,
        mock_convert_dict,
    ):
        """Test load() calls process_seds exactly once."""
        dataset = request.getfixturevalue(dataset_fixture)
        save_dataset(data_dir, dataset)

        loader = make_loader(dataset_type, data_params, simPSF)
        loader.load()

        mock_process_seds.assert_called_once()

    @pytest.mark.parametrize(
        "dataset_type, dataset_fixture",
        [
            ("training", "training_dataset"),
            ("test", "test_dataset"),
        ],
        ids=["training", "test"],
    )
    def test_load_calls_convert_dict_with_correct_type(
        self,
        request,
        dataset_type,
        dataset_fixture,
        data_dir,
        data_params,
        simPSF,
        mock_process_seds,
        mock_convert_dict,
    ):
        """Test load() calls convert_dict with correct dataset_type."""
        dataset = request.getfixturevalue(dataset_fixture)
        save_dataset(data_dir, dataset)

        loader = make_loader(dataset_type, data_params, simPSF)
        loader.load()

        mock_convert_dict.assert_called_once()
        actual_dataset_type = mock_convert_dict.call_args[0][
            1
        ]  # dataset_type is second positional argument
        assert actual_dataset_type == dataset_type


# ============================================================================
# TestSimulationDataLoaderValidation
# ============================================================================


class TestSimulationDataLoaderValidation:
    """Tests for SimulationDataLoader._validate_structure()."""

    @pytest.mark.parametrize(
        "dataset_type, missing_field, dataset",
        [
            (
                "training",
                "positions",
                {
                    "noisy_stars": np.random.randn(2, 32, 32),
                    "SEDs": np.random.randn(2, 3, 2),
                },
            ),
            (
                "training",
                "noisy_stars",
                {"positions": np.array([[1.0, 2.0]]), "SEDs": np.random.randn(1, 3, 2)},
            ),
            (
                "test",
                "positions",
                {"stars": np.random.randn(2, 32, 32), "SEDs": np.random.randn(2, 3, 2)},
            ),
            (
                "test",
                "stars",
                {"positions": np.array([[1.0, 2.0]]), "SEDs": np.random.randn(1, 3, 2)},
            ),
        ],
        ids=[
            "training_missing_positions",
            "training_missing_noisy_stars",
            "test_missing_positions",
            "test_missing_stars",
        ],
    )
    def test_missing_required_field_raises_value_error(
        self,
        dataset_type,
        missing_field,
        dataset,
        data_dir,
        data_params,
        simPSF,
        mock_process_seds,
        mock_convert_dict,
    ):
        """Test ValueError raised for each missing required field."""
        save_dataset(data_dir, dataset)

        loader = make_loader(dataset_type, data_params, simPSF)

        with pytest.raises(ValueError, match=missing_field):
            loader.load()

    def test_unrecognized_dataset_type_raises_value_error(
        self, data_dir, data_params, simPSF, mock_process_seds, mock_convert_dict
    ):
        """Test ValueError raised for unrecognized dataset_type."""

        dataset = {
            "positions": np.array([[1.0, 2.0]]),
            "SEDs": np.random.randn(1, 3, 2),
        }
        save_dataset(data_dir, dataset)

        loader = make_loader("invalid_type", data_params, simPSF)

        with pytest.raises(ValueError, match="Unrecognized dataset_type"):
            loader.load()

    def test_missing_seds_raises_value_error(
        self, data_dir, data_params, simPSF, mock_process_seds, mock_convert_dict
    ):
        """Test ValueError raised when SEDs are missing."""
        dataset = {
            "positions": np.array([[1.0, 2.0]]),
            "noisy_stars": np.random.randn(1, 32, 32),
            # No SEDs
        }
        save_dataset(data_dir, dataset)

        loader = make_loader("training", data_params, simPSF)

        with pytest.raises(ValueError, match="SEDs"):
            loader.load()

    def test_file_not_found_raises_error(self, data_params, simPSF):
        """Test error raised when .npy file does not exist."""
        loader = make_loader("training", data_params, simPSF)

        with pytest.raises(FileNotFoundError):
            loader.load()
