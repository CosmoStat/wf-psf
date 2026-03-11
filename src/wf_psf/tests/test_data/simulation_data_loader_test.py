import pytest
import numpy as np
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


# Helpers
def save_dataset(data_dir, dataset, filename="dataset.npy"):
    """Save dataset dict to .npy file in data_dir."""
    np.save(str(data_dir / filename), dataset)


def make_loader(data_params):
    """
    Create a SimulationDataLoader instance.

    Uses n_bins_lambda=8 to match simPSF fixture from conftest.py
    (training_config.model_params.n_bins_lda=8).
    """
    return SimulationDataLoader(
        data_params=data_params,
    )


# ============================================================================
# TestSimulationDataLoaderInit
# ============================================================================


class TestSimulationDataLoaderInit:
    """Tests for SimulationDataLoader initialisation."""

    def test_attributes_set_on_init(self, data_params):
        """Test all attributes are set correctly on initialisation."""

        loader = make_loader(data_params.params)

        assert loader.data_params == data_params.params


# ============================================================================
# TestSimulationDataLoaderLoad
# ============================================================================


class TestSimulationDataLoaderLoad:
    """Tests for SimulationDataLoader.load()."""

    @pytest.mark.parametrize(
        "dataset_type, dataset_fixture",
        [
            ("train", "training_dataset"),
            ("test", "test_dataset"),
        ],
        ids=["training", "test"],
    )
    def test_load_returns_dataset(
        self,
        request,
        dataset_type,
        dataset_fixture,
        data_dir,
        data_params,
    ):
        """Test load() returns dataset both dataset types."""
        params = getattr(data_params.params, dataset_type)

        dataset = request.getfixturevalue(dataset_fixture)
        save_dataset(data_dir, dataset)
        loader = make_loader(params)
        loader.load()
        returned_dataset = loader.dataset

        assert returned_dataset is not None
