"""Configs_Handler.

A module which provides general utility methods
to manage the parameters of the config files

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
import logging
import os
import re
import glob
from wf_psf.data.data_handler import DataHandler
from wf_psf.metrics.metrics_interface import evaluate_model
from wf_psf.plotting.plots_interface import plot_metrics
from wf_psf.psf_models import psf_models
from wf_psf.psf_models.psf_model_loader import load_trained_psf_model
from wf_psf.training import train
from wf_psf.utils.read_config import read_conf


logger = logging.getLogger(__name__)

CONFIG_CLASS = {}


def register_configclass(config_class):
    """Register Config Class.

    A wrapper function to register all config classes
    in a dictionary.

    Parameters
    ----------
    config_class: type
        Config Class

    Returns
    -------
    config_class: type
        Config class

    """
    for id in config_class.ids:
        CONFIG_CLASS[id] = config_class

    return config_class


def set_run_config(config_name):
    """Set Run Configuration Class.

    A function to retrieve the appropriate configuration
    class based on the provided config name.

    Parameters
    ----------
    config_name: str
        Name of config

    Returns
    -------
    config_class: class
        Name of config class

    """
    try:
        config_id = [id for id in CONFIG_CLASS.keys() if re.search(id, config_name)][0]
        config_class = CONFIG_CLASS[config_id]
    except KeyError:
        logger.exception("Invalid config name. Check your config settings.")
        exit()

    return config_class


def get_run_config(run_config_name, *config_params):
    """Get Run Configuration Instance.

    A function to retrieve an instance of
    the appropriate configuration class for
    a WF-PSF run.

    Parameters
    ----------
    run_config_name: str
        Name of the run configuraton
    *config_params: str
        Run configuration parameters used for class instantiation.

    Returns
    -------
    config_class: object
        A class instance of the selected configuration class.

    """
    config_class = set_run_config(run_config_name)

    return config_class(*config_params)


class ConfigParameterError(Exception):
    """Custom Config Parameter Error exception class for specific error scenarios."""

    def __init__(self, message="An error with your config settings occurred."):
        self.message = message
        super().__init__(self.message)


class DataConfigHandler:
    """DataConfigHandler.

    A class to handle data configuration
    parameters.

    Parameters
    ----------
    data_conf : str
        Path of the data configuration file
    training_model_params : Recursive Namespace object
        Recursive Namespace object containing the training model parameters
    batch_size : int
       Training hyperparameter used for batched pre-processing of data.

    """

    def __init__(self, data_conf, training_model_params, batch_size=16, load_data=True):
        try:
            self.data_conf = read_conf(data_conf)
        except (FileNotFoundError, TypeError) as e:
            logger.exception(e)
            exit()

        self.simPSF = psf_models.simPSF(training_model_params)

        # Extract sub-configs early
        train_params = self.data_conf.data.training
        test_params = self.data_conf.data.test

        self.training_data = DataHandler(
            dataset_type="training",
            data_params=train_params,
            simPSF=self.simPSF,
            n_bins_lambda=training_model_params.n_bins_lda,
            load_data=load_data,
        )
        self.test_data = DataHandler(
            dataset_type="test",
            data_params=test_params,
            simPSF=self.simPSF,
            n_bins_lambda=training_model_params.n_bins_lda,
            load_data=load_data,
        )

        self.batch_size = batch_size


@register_configclass
class TrainingConfigHandler:
    """TrainingConfigHandler.

    A class to handle training configuration
    parameters.

    Parameters
    ----------
    ids: tuple
        A tuple containing a string id for the Configuration Class
    training_conf: str
        Path of the training configuration file
    file_handler: object
        A instance of the FileIOHandler class

    """

    ids = ("training_conf",)

    def __init__(self, training_conf, file_handler):
        self.training_conf = read_conf(training_conf)
        self.file_handler = file_handler
        self.data_conf = DataConfigHandler(
            os.path.join(
                file_handler.config_path, self.training_conf.training.data_config
            ),
            self.training_conf.training.model_params,
            self.training_conf.training.training_hparams.batch_size,
            self.training_conf.training.load_data_on_init,
        )
        self.data_conf.run_type = "training"
        self.file_handler.copy_conffile_to_output_dir(
            self.training_conf.training.data_config
        )
        self.checkpoint_dir = file_handler.get_checkpoint_dir(
            self.file_handler._run_output_dir
        )
        self.optimizer_dir = file_handler.get_optimizer_dir(
            self.file_handler._run_output_dir
        )
        self.psf_model_dir = file_handler.get_psf_model_dir(
            self.file_handler._run_output_dir
        )

    def run(self):
        """Run.

        A function to run wavediff according to the
        input configuration.

        """
        train.train(
            self.training_conf.training,
            self.data_conf,
            self.checkpoint_dir,
            self.optimizer_dir,
            self.psf_model_dir,
        )

        if self.training_conf.training.metrics_config is not None:
            self.file_handler.copy_conffile_to_output_dir(
                self.training_conf.training.metrics_config
            )

            metrics = MetricsConfigHandler(
                os.path.join(
                    self.file_handler.config_path,
                    self.training_conf.training.metrics_config,
                ),
                self.file_handler,
                self.training_conf,
            )

            metrics.run()


@register_configclass
class MetricsConfigHandler:
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
        Sets the training configuration. If None is provided, attempts to load it
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
            (f"{model_subdir}*_{model_name}" f"*{id_name}_cycle{cycle}*"),
        )
        return load_trained_psf_model(
            self.training_conf,
            self.data_conf,
            weights_path_pattern,
        )

    def _get_training_conf_path_from_metrics(self):
        """
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
        """
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


@register_configclass
class PlottingConfigHandler:
    """PlottingConfigHandler.

    A class to handle plotting config
    settings.

    Parameters
    ----------
    ids: tuple
        A tuple containing a string id for the Configuration Class
    plotting_conf: str
        Name of plotting configuration file
    file_handler: obj
        An instance of the FileIOHandler class

    """

    ids = ("plotting_conf",)

    def __init__(self, plotting_conf, file_handler):
        self.plotting_conf = read_conf(plotting_conf)
        self.file_handler = file_handler
        self.metrics_confs = {}
        self.check_and_update_metrics_confs()
        self.list_of_metrics_dict = self.make_dict_of_metrics()
        self.plots_dir = self.file_handler.get_plots_dir(
            self.file_handler._run_output_dir
        )

    def check_and_update_metrics_confs(self):
        """Check and Update Metrics Confs.

        A function to check if user provided inputs metrics
        dir to add to metrics_confs dictionary.

        """
        if self.plotting_conf.plotting_params.metrics_dir:
            self._update_metrics_confs()

    def make_dict_of_metrics(self):
        """Make dictionary of metrics.

        A function to create a dictionary for each metrics per run.

        Returns
        -------
        dict
         A dictionary containing metrics or an empty dictionary.

        """
        if self.plotting_conf.plotting_params.metrics_dir:
            return self.load_metrics_into_dict()
        else:
            return {}

    def _update_metrics_confs(self):
        """Update Metrics Configurations.

        A method to update the metrics_confs dictionary
        with each set of metrics configuration parameters
        provided as input.

        """
        for wf_dir, metrics_conf in zip(
            self.plotting_conf.plotting_params.metrics_dir,
            self.plotting_conf.plotting_params.metrics_config,
        ):
            self.metrics_confs[wf_dir] = read_conf(
                os.path.join(
                    self.plotting_conf.plotting_params.metrics_output_path,
                    self.file_handler.get_config_dir(wf_dir),
                    metrics_conf,
                )
            )

    def _metrics_run_id_name(self, wf_outdir, metrics_params):
        """Get Metrics Run ID Name.

        A function to generate run id name
        for the metrics of a trained model

        Parameters
        ----------
        wf_outdir: str
            Name of the wf-psf run output directory
        metrics_params: RecursiveNamespace Object
            RecursiveNamespace object containing the metrics parameters used to evaluated the trained model.

        Returns
        -------
        metrics_run_id_name: list
            List containing the model name and id for each training run
        """
        try:
            training_conf = read_conf(
                os.path.join(
                    metrics_params.metrics.trained_model_path,
                    metrics_params.metrics.trained_model_config,
                )
            )
            id_name = training_conf.training.id_name
            model_name = training_conf.training.model_params.model_name
            return [model_name + id_name]

        except (TypeError, FileNotFoundError):
            logger.info("Trained model path not provided...")
            logger.info(
                f"Trying to retrieve training config file from workdir: {wf_outdir}"
            )

            training_confs = [
                read_conf(training_conf)
                for training_conf in glob.glob(
                    os.path.join(
                        self.plotting_conf.plotting_params.metrics_output_path,
                        self.file_handler.get_config_dir(wf_outdir),
                        "training*",
                    )
                )
            ]

            run_ids = [
                training_conf.training.model_params.model_name
                + training_conf.training.id_name
                for training_conf in training_confs
            ]

            return run_ids
        except FileNotFoundError:
            logger.exception("File not found.")

    def load_metrics_into_dict(self):
        """Load Metrics into Dictionary.

        A method to load a metrics file of
        a trained model from a previous run into a
        dictionary.

        Returns
        -------
        metrics_files_dict: dict
            A dictionary containing all of the metrics from the loaded metrics files.

        """
        metrics_dict = {}

        for k, v in self.metrics_confs.items():
            run_id_names = self._metrics_run_id_name(k, v)

            metrics_dict[k] = []
            for run_id_name in run_id_names:
                output_path = os.path.join(
                    self.plotting_conf.plotting_params.metrics_output_path,
                    k,
                    "metrics",
                    "metrics-" + run_id_name + ".npy",
                )
                logger.info(
                    f"Attempting to read in trained model config file...{output_path}"
                )
                try:
                    metrics_dict[k].append(
                        {run_id_name: [np.load(output_path, allow_pickle=True)[()]]}
                    )
                except FileNotFoundError:
                    logger.error(
                        "The required file for the plots was not found. Please check your configs settings."
                    )

        return metrics_dict

    def run(self):
        """Run.

        A function to run wave-diff according to the
        input configuration.

        """
        logger.info("Generating metric plots...")
        plot_metrics(
            self.plotting_conf.plotting_params,
            self.list_of_metrics_dict,
            self.metrics_confs,
            self.plots_dir,
        )
