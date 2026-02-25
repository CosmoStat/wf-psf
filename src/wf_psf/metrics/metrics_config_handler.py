"""MetricsConfigHandler.

A module containing the MetricsConfigHandler class, which is responsible for managing the parameters of the metrics configuration file and running the
metrics evaluation of a trained PSF model

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import os
from wf_psf.utils.configs_handler import (
    ConfigHandler,
    register_configclass,
    ConfigParameterError,
)
from wf_psf.data.data_config_handler import DataConfigHandler
from wf_psf.utils.read_config import read_conf
from wf_psf.metrics.metrics_interface import evaluate_model
from wf_psf.psf_models.psf_model_loader import load_trained_psf_model
from wf_psf.plotting.plotting_config_handler import PlottingConfigHandler
import logging

logger = logging.getLogger(__name__)


@register_configclass
class MetricsConfigHandler(ConfigHandler):
    """MetricsConfigHandler.

    A class to handle metrics configuation
    parameters.

    Parameters
    ----------
    ids: tuple
        A tuple containing a string id for the Configuration Class
    metrics_conf: str
        Path to the metrics configuration file
    file_handler: object
        An instance of the FileIOHandler
    training_conf: RecursiveNamespace object
        RecursiveNamespace object containing the training configuration parameters


    """

    ids = ("metrics_conf",)

    def __init__(self, metrics_conf, file_handler, training_conf=None):
        self._metrics_conf = read_conf(metrics_conf)
        self._file_handler = file_handler
        self.training_conf = training_conf
        self.data_conf = self._load_data_conf()
        self.data_conf.run_type = "metrics"
        self.metrics_dir = self._file_handler.get_metrics_dir(
            self._file_handler._run_output_dir
        )
        self.trained_psf_model = self._load_trained_psf_model()

    @property
    def metrics_conf(self):
        """Get Metrics Conf.

        A function to return the metrics configuration file name.

        Returns
        -------
        RecursiveNamespace
            An instance of the metrics configuration file.
        """
        return self._metrics_conf

    @property
    def training_conf(self):
        """Returns the loaded training configuration."""
        return self._training_conf

    @training_conf.setter
    def training_conf(self, training_conf):
        """
        Set the training configuration.

        If None is provided, attempts to load it
        from the trained_model_path in the metrics configuration.
        """
        if training_conf is None:
            try:
                training_conf_path = self._get_training_conf_path_from_metrics()
                logger.info(
                    f"Loading training config from inferred path: {training_conf_path}"
                )
                self._training_conf = read_conf(training_conf_path)
            except Exception as e:
                logger.error(f"Failed to load training config: {e}")
                raise
        else:
            self._training_conf = training_conf

    @property
    def plotting_conf(self):
        """Get Plotting Conf.

        A function to return the plotting configuration file name.

        Returns
        -------
        str
            Name of plotting configuration file
        """
        return self.metrics_conf.metrics.plotting_config

    def _load_trained_psf_model(self):
        trained_model_path = self._get_trained_model_path()
        try:
            model_subdir = self.metrics_conf.metrics.model_save_path
            cycle = self.metrics_conf.metrics.saved_training_cycle
        except AttributeError as e:
            raise KeyError("Missing required model config fields.") from e

        model_name = self.training_conf.training.model_params.model_name
        id_name = self.training_conf.training.id_name

        weights_path_pattern = os.path.join(
            trained_model_path,
            model_subdir,
            (f"{model_subdir}*_{model_name}*{id_name}_cycle{cycle}*"),
        )
        return load_trained_psf_model(
            self.training_conf,
            self.data_conf,
            weights_path_pattern,
        )

    def _get_training_conf_path_from_metrics(self):
        """Get training config path from metrics config.

        Retrieves the full path to the training config based on the metrics configuration.

        Returns
        -------
        str
            Full path to the training configuration file.

        Raises
        ------
        KeyError
            If 'trained_model_config' key is missing.
        FileNotFoundError
            If the file does not exist at the constructed path.
        """
        trained_model_path = self._get_trained_model_path()

        try:
            training_conf_filename = self._metrics_conf.metrics.trained_model_config
        except AttributeError as e:
            raise KeyError(
                "Missing 'trained_model_config' key in metrics configuration."
            ) from e

        training_conf_path = os.path.join(
            self._file_handler.get_config_dir(trained_model_path),
            training_conf_filename,
        )

        if not os.path.exists(training_conf_path):
            raise FileNotFoundError(
                f"Training config file not found: {training_conf_path}"
            )

        return training_conf_path

    def _get_trained_model_path(self):
        """Get trained model path.

        Determine the trained model path from either:

        1. The metrics configuration file (i.e., for metrics-only runs after training), or
        2. The runtime-generated file handler paths (i.e., for single runs that perform both training and evaluation).

        Returns
        -------
        str
            Path to the trained model directory.

        Raises
        ------
        ConfigParameterError
            If the path specified in the metrics config is invalid or missing.
        """
        trained_model_path = getattr(
            self._metrics_conf.metrics, "trained_model_path", None
        )

        if trained_model_path:
            if not os.path.isdir(trained_model_path):
                raise ConfigParameterError(
                    f"The trained model path provided in the metrics config is not a valid directory: {trained_model_path}"
                )
            logger.info(
                f"Using trained model path from metrics config: {trained_model_path}"
            )
            return trained_model_path

        # Fallback for single-run training + metrics evaluation mode
        fallback_path = os.path.join(
            self._file_handler.output_path,
            self._file_handler.parent_output_dir,
            self._file_handler.workdir,
        )
        logger.info(
            f"Using fallback trained model path from runtime file handler: {fallback_path}"
        )
        return fallback_path

    def _load_data_conf(self):
        """Load Data Conf.

        A method to load the data configuration file
        and return an instance of DataConfigHandler class.

        Returns
        -------
        An instance of the DataConfigHandler class.
        """
        try:
            return DataConfigHandler(
                os.path.join(
                    self._file_handler.config_path,
                    self.training_conf.training.data_config,
                ),
                self.training_conf.training.model_params,
            )
        except TypeError as e:
            logger.exception(e)
            raise ConfigParameterError("Data configuration loading error.")

    def call_plot_config_handler_run(self, model_metrics):
        """Make Metrics Plots.

        A function to call the PlottingConfigHandler run
        command to generate metrics plots.

        Parameters
        ----------
        model_metrics: dict
            A dictionary storing the metrics
            output generated during evaluation
            of the trained PSF model.

        """
        self._plotting_conf = os.path.join(
            self._file_handler.config_path,
            self.plotting_conf,
        )

        plots_config_handler = PlottingConfigHandler(
            self._plotting_conf,
            self._file_handler,
        )

        # Update metrics_confs dict with latest result
        plots_config_handler.metrics_confs[self._file_handler.workdir] = (
            self.metrics_conf
        )

        # Update metric results dict with latest result
        plots_config_handler.list_of_metrics_dict[self._file_handler.workdir] = [
            {
                self.training_conf.training.model_params.model_name
                + self.training_conf.training.id_name: [model_metrics]
            }
        ]

        plots_config_handler.run()

    def run(self):
        """Run.

        A function to run WaveDiff according to the
        input configuration.

        """
        logger.info("Running metrics evaluation on trained PSF model...")

        model_metrics = evaluate_model(
            self.metrics_conf.metrics,
            self.training_conf.training,
            self.data_conf,
            self.trained_psf_model,
            self.metrics_dir,
        )

        if self.plotting_conf is not None:
            self._file_handler.copy_conffile_to_output_dir(
                self.metrics_conf.metrics.plotting_config
            )
            self.call_plot_config_handler_run(model_metrics)
