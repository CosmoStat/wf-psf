"""UNIT TESTS FOR PACKAGE MODULE: CONFIGS_HANDLER.

This module contains unit tests for the wf_psf.utils configs_handler module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import pytest
from wf_psf.data.data_handler import DataHandler
from wf_psf.utils import configs_handler
from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.utils.io import FileIOHandler
from wf_psf.data.data_config_handler import DataConfigHandler
from wf_psf.training.training_config_handler import TrainingConfigHandler
from wf_psf.metrics.metrics_config_handler import MetricsConfigHandler
from wf_psf.plotting.plotting_config_handler import PlottingConfigHandler
from wf_psf.utils.configs_handler import ConfigHandler, register_configclass


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_data_read_conf(mocker):
    return mocker.patch(
        "wf_psf.data.data_config_handler.read_conf",
        return_value=RecursiveNamespace(
            data=RecursiveNamespace(
                data_type="simulation",
                training=RecursiveNamespace(
                    data_dir="/path/to/train_data",
                    file="train_data.npy",
                    target_field="noisy_stars",
                ),
                test=RecursiveNamespace(
                    data_dir="/path/to/test_data",
                    file="test_data.npy",
                    target_field="stars",
                ),
            ),
        ),
    )


@pytest.fixture
def mock_training_conf(mocker):
    return RecursiveNamespace(
        training=RecursiveNamespace(
            id_name="_test_",
            data_config="data_config.yaml",
            load_data_on_init=True,
            metrics_config=None,
            model_params=RecursiveNamespace(
                model_name="poly",
                n_bins_lda=10,
                param_hparams=RecursiveNamespace(
                    random_seed=3877572,
                ),
                nonparam_hparams=RecursiveNamespace(
                    d_max_nonparam=5,
                ),
            ),
            training_hparams=RecursiveNamespace(batch_size=32),
        ),
    )


@pytest.fixture
def mock_data_conf(mocker):
    """Mock DataConfigHandler instance."""
    data_conf = mocker.Mock()
    data_conf.training_data = mocker.Mock()
    data_conf.test_data = mocker.Mock()
    return data_conf


@pytest.fixture
def test_config_class():
    """Fixture that registers and returns a test config class."""

    @register_configclass
    class TestConfigClass(ConfigHandler):
        ids = ("test_conf",)

        def __init__(self, config_params, file_handler):
            self.config_param = config_params
            self.file_handler = file_handler

        def run(self):
            pass

    return TestConfigClass


# ============================================================================
# Test Registry Pattern
# ============================================================================


class TestConfigRegistry:
    """Tests for config handler registration and dispatch."""

    def test_register_configclass(self, test_config_class):
        """Test that @register_configclass adds handler to registry."""
        assert configs_handler.CONFIG_CLASS["test_conf"] == test_config_class

    def test_set_run_config_with_test_handler(self, test_config_class):
        """Test set_run_config finds test handler by name."""
        config_class = configs_handler.set_run_config("test_conf")
        assert config_class == test_config_class

    @pytest.mark.parametrize(
        "config_name, expected_class",
        [
            ("training_conf", TrainingConfigHandler),
            ("metrics_conf", MetricsConfigHandler),
            ("plotting_conf", PlottingConfigHandler),
        ],
        ids=["training", "metrics", "plotting"],
    )
    def test_set_run_config_with_real_handlers(self, config_name, expected_class):
        """Test set_run_config finds real handlers by name."""
        config_class = configs_handler.set_run_config(config_name)
        assert config_class == expected_class

    def test_set_run_config_invalid_name_raises_exception(self):
        """Test set_run_config raises for unrecognized config name."""
        with pytest.raises(Exception):  # Could be KeyError or IndexError
            configs_handler.set_run_config("nonexistent_conf")

    def test_get_run_config_instantiates(
        self, test_config_class, path_to_tmp_output_dir, path_to_config_dir
    ):
        """Test get_run_config returns instance of correct handler."""
        test_file_handler = FileIOHandler(path_to_tmp_output_dir, path_to_config_dir)

        config_instance = configs_handler.get_run_config(
            "test_conf", "fake_config.yaml", test_file_handler
        )

        assert isinstance(config_instance, test_config_class)
        assert config_instance.config_param == "fake_config.yaml"
        assert config_instance.file_handler == test_file_handler


# ============================================================================
# Test ConfigHandler ABC
# ============================================================================


class TestConfigHandlerABC:
    """Tests for ConfigHandler abstract base class."""

    def test_config_handler_requires_run_method(self):
        """Test that handlers must implement run()."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            @register_configclass
            class InvalidHandler(ConfigHandler):
                ids = ("invalid_conf",)
                # Missing run() method!

            InvalidHandler()

    def test_config_handler_requires_ids(self):
        """Test that handlers must define ids attribute."""

        class NoIdsHandler(ConfigHandler):
            def run(self):
                pass

        # Should fail when trying to register (no ids attribute)
        with pytest.raises(AttributeError):
            register_configclass(NoIdsHandler)

    def test_all_registered_handlers_inherit_from_base(self):
        """Test all registered handlers inherit from ConfigHandler."""
        for handler_class in configs_handler.CONFIG_CLASS.values():
            assert issubclass(handler_class, ConfigHandler)

    def test_all_registered_handlers_have_run(self):
        """Test all registered handlers implement run()."""
        for handler_class in configs_handler.CONFIG_CLASS.values():
            assert hasattr(handler_class, "run")
            assert callable(handler_class.run)


# ============================================================================
# Test DataConfigHandler (moved from here, should be in test_data_config_handler.py)
# ============================================================================


@pytest.mark.skip(reason="Skipped - deprecated DataHandler pending removal")
def test_data_config_handler_init(mock_training_conf, mock_data_read_conf, mocker):
    """
    Test DataConfigHandler initialization.

    NOTE: This test is skipped because it tests deprecated DataHandler
    behavior. Once DataHandler is removed, this test should be updated
    or moved to test_data_config_handler.py with proper mocking.
    """
    # Mock read_conf function
    mock_data_read_conf()

    # Mock SimPSF instance
    mock_simPSF_instance = mocker.Mock(name="SimPSFToolkit")
    mocker.patch(
        "wf_psf.psf_models.psf_models.simPSF", return_value=mock_simPSF_instance
    )

    # Patch process_sed_data method
    mocker.patch.object(DataHandler, "process_sed_data")

    # Patch validate_and_process_datasetmethod
    mocker.patch.object(DataHandler, "validate_and_process_dataset")

    # Patch load_dataset to assign dataset
    def mock_load_dataset(self):
        self.dataset = {
            "SEDs": ["dummy_sed_data"],
            "positions": ["dummy_positions_data"],
        }

    mocker.patch.object(DataHandler, "load_dataset", new=mock_load_dataset)

    # Create DataConfigHandler instance
    data_config_handler = DataConfigHandler(
        "/path/to/data_config.yaml",
        mock_training_conf.training.model_params,
        mock_training_conf.training.training_hparams.batch_size,
    )

    # Check that attributes are set correctly
    assert isinstance(data_config_handler.data_conf, RecursiveNamespace)
    assert isinstance(data_config_handler.simPSF, object)
    assert (
        data_config_handler.training_data.n_bins_lambda
        == mock_training_conf.training.model_params.n_bins_lda
    )
    assert (
        data_config_handler.test_data.n_bins_lambda
        == mock_training_conf.training.model_params.n_bins_lda
    )
    assert (
        data_config_handler.batch_size
        == mock_training_conf.training.training_hparams.batch_size
    )


# ============================================================================
# Test TrainingConfigHandler (should be in test_training_config_handler.py)
# ============================================================================


class TestTrainingConfigHandler:
    """Tests for TrainingConfigHandler."""

    def test_run_method_calls_train_with_correct_arguments(
        self, mocker, mock_training_conf, mock_data_conf
    ):
        """Test that run() calls train.train() with correct arguments."""
        # Patch the TrainingConfigHandler.__init__() method
        mocker.patch(
            "wf_psf.training.training_config_handler.TrainingConfigHandler.__init__",
            return_value=None,
        )
        mock_th = TrainingConfigHandler(None, None)

        # Set attributes of the mock_th
        mock_th.training_conf = mock_training_conf
        mock_th.data_conf = mock_data_conf
        mock_th.data_conf.training_data = mock_data_conf.training_data
        mock_th.data_conf.test_data = mock_data_conf.test_data
        mock_th.checkpoint_dir = "/mock/checkpoint/dir"
        mock_th.optimizer_dir = "/mock/optimizer/dir"
        mock_th.psf_model_dir = "/mock/psf/model/dir"

        # Patch the train.train() function
        mock_train_function = mocker.patch("wf_psf.training.train.train")

        # Create a spy for the run method
        spy = mocker.spy(mock_th, "run")

        # Call the run method
        mock_th.run()

        # Assert that run() is called once
        spy.assert_called_once()

        # Assert that train.train() is called with the correct arguments
        mock_train_function.assert_called_once_with(
            mock_th.training_conf.training,
            mock_th.data_conf,
            mock_th.checkpoint_dir,
            mock_th.optimizer_dir,
            mock_th.psf_model_dir,
        )
