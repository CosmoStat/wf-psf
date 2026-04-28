"""UNIT TESTS FOR PACKAGE MODULE: CONFIGS_HANDLER.

This module contains unit tests for the wf_psf.utils configs_handler module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import pytest
from wf_psf.utils import configs_handler
from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.utils.io import FileIOHandler
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
            params=RecursiveNamespace(
                train=RecursiveNamespace(
                    data_dir="data",
                    file="coherent_euclid_dataset/train_Euclid_res_200_TrainStars_id_001.npy",
                    target_field="noisy_stars",
                ),
                test=RecursiveNamespace(
                    data_dir="data",
                    file="coherent_euclid_dataset/test_Euclid_res_id_001.npy",
                    target_field="stars",
                ),
                canonical_keys=["sources", "masks", "positions"],
            )
        ),
    )


@pytest.fixture
def mock_training_conf(mocker):
    return RecursiveNamespace(
        training=RecursiveNamespace(
            id_name="_test_",
            data_config="data_config.yaml",
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
            training_hparams=RecursiveNamespace(batch_size=32, loss="mask_mse"),
        ),
    )


@pytest.fixture
def mock_data_conf(mocker):
    """Mock DataConfigHandler instance."""
    data_conf = mocker.Mock()
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
# Test TrainingConfigHandler (should be in test_training_config_handler.py)
# ============================================================================


class TestTrainingConfigHandler:
    """Tests for TrainingConfigHandler."""

    def test_run_method_calls_train_with_correct_arguments(
        self, mocker, mock_training_conf, mock_data_conf
    ):
        """Test that run() calls train.train() with correct arguments."""
        # Mock SimPSF instance
        mock_simPSF_instance = mocker.Mock(name="SimPSFToolkit")
        mocker.patch(
            "wf_psf.psf_models.psf_models.simPSF", return_value=mock_simPSF_instance
        )

        # Patch the TrainingConfigHandler.__init__() method
        mocker.patch(
            "wf_psf.training.training_config_handler.TrainingConfigHandler.__init__",
            return_value=None,
        )
        mock_th = TrainingConfigHandler(None, None)

        # Patch prepare_training_inputs method
        mock_ta = mocker.Mock()
        mock_psfm = mocker.Mock()
        mocker.patch(
            "wf_psf.training.training_config_handler.prepare_training_inputs",
            return_value=(mock_ta, mock_psfm),
        )

        # Set attributes of the mock_th
        mock_th.training_conf = mock_training_conf.training
        mock_th.data_params = mock_data_conf
        mock_th.simPSF = mock_simPSF_instance
        mock_th.n_bins_lambda = 10
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
            mock_th.training_conf,
            mock_ta,
            mock_psfm,
            mock_th.checkpoint_dir,
            mock_th.optimizer_dir,
            mock_th.psf_model_dir,
        )
