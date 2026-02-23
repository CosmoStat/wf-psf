import pytest
import numpy as np
from pathlib import Path
import tensorflow as tf
from wf_psf.data.simulation_data_loader import SimulationDataLoader


@pytest.fixture
def data_dir(tmp_path):
    """Create temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def training_dataset(mock_np_dataset):
    """
    Valid training dataset with noisy_stars.
    """
    return {
        "positions": mock_np_dataset["positions"],
        "noisy_stars": mock_np_dataset["noisy_stars"],
        "SEDs": mock_np_dataset["SEDs"],
        "target_field": "noisy_stars",
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
        "target_field": "stars",
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
def mock_convert_psf_dict(mocker):
    """Mock convert_psf_dict on TensorFlowDatasetConverter."""

    def _mock_convert_psf_dict(dataset, source_field):
        return {
            k: tf.constant(v) if isinstance(v, np.ndarray) else v
            for k, v in dataset.items()
        }

    return mocker.patch(
        "wf_psf.data.simulation_data_loader.TensorFlowDatasetConverter.convert_psf_dict",
        side_effect=_mock_convert_psf_dict,
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

        config = getattr(data_params, dataset_type)

        loader = make_loader(dataset_type, config, simPSF)

        assert loader.data_params == config
        assert loader.n_bins_lambda == 8
        assert loader.dataset is None
        assert loader.sed_data is None

    def test_converter_instantiated_on_init(self, data_params, simPSF):
        """Test TensorFlowDatasetConverter is instantiated on init."""
        dataset_type = "training"
        config = getattr(data_params, dataset_type)
        loader = make_loader(dataset_type, config, simPSF)

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
        mock_convert_psf_dict,
    ):
        """Test load() returns dataset and sed_data for both dataset types."""
        dataset = request.getfixturevalue(dataset_fixture)
        save_dataset(data_dir, dataset)

        config = getattr(data_params, dataset_type)

        loader = make_loader(dataset_type, config, simPSF)
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
        mock_convert_psf_dict,
    ):
        """Test load() sets dataset and sed_data instance attributes."""
        dataset = request.getfixturevalue(dataset_fixture)
        save_dataset(data_dir, dataset)

        config = getattr(data_params, dataset_type)
        loader = make_loader(dataset_type, config, simPSF)
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
    ):
        """Test load() calls process_seds exactly once."""
        dataset = request.getfixturevalue(dataset_fixture)
        save_dataset(data_dir, dataset)

        config = getattr(data_params, dataset_type)

        loader = make_loader(dataset_type, config, simPSF)
        loader.load()

        mock_process_seds.assert_called_once()

    @pytest.mark.parametrize(
        "dataset_type, dataset_fixture, source_field",
        [
            ("training", "training_dataset", "noisy_stars"),
            ("test", "test_dataset", "stars"),
        ],
        ids=["training", "test"],
    )
    def test_load_calls_convert_psf_dict_with_correct_type(
        self,
        request,
        dataset_type,
        dataset_fixture,
        source_field,
        data_dir,
        data_params,
        simPSF,
        mock_convert_psf_dict,
    ):
        """Test load() calls convert_psf_dict with correct dataset_type."""
        dataset = request.getfixturevalue(dataset_fixture)
        save_dataset(data_dir, dataset)

        config = getattr(data_params, dataset_type)
        loader = make_loader(dataset_type, config, simPSF)
        loader.load()

        mock_convert_psf_dict.assert_called_once()
        actual_source = mock_convert_psf_dict.call_args[0][
            1
        ]  # source_field is second positional argument
        assert actual_source == source_field


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
    ):
        """Test ValueError raised for each missing required field."""

        config = getattr(data_params, dataset_type)

        config.data_dir = str(data_dir)
        config.file = "test.npy"

        save_path = Path(data_dir) / "test.npy"
        np.save(save_path, dataset)

        loader = make_loader(dataset_type, config, simPSF)

        with pytest.raises(ValueError, match=missing_field):
            loader.load()

    def test_missing_seds_raises_value_error(self, data_dir, data_params, simPSF):
        """Test ValueError raised when SEDs are missing."""

        dataset_type = "training"
        config = getattr(data_params, dataset_type)
        config.data_dir = str(data_dir)
        config.file = "train.npy"

        dataset = {
            "positions": np.array([[1.0, 2.0]]),
            "noisy_stars": np.random.randn(1, 32, 32),
            # No SEDs
        }

        save_path = Path(data_dir) / config.file
        np.save(save_path, dataset)

        loader = make_loader(dataset_type, config, simPSF)

        with pytest.raises(ValueError, match="SEDs"):
            loader.load()

    def test_file_not_found_raises_error(self, data_params, simPSF):
        """Test error raised when .npy file does not exist."""
        dataset_type = "training"
        config = getattr(data_params, dataset_type)
        config.file = "no_exist.npy"

        loader = make_loader(dataset_type, config, simPSF)

        with pytest.raises(FileNotFoundError):
            loader.load()
