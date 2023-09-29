"""Configs_Handler.

A module which provides general utility methods
to manage the parameters of the config files

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""
import numpy as np
from wf_psf.utils.read_config import read_conf
from wf_psf.data.training_preprocessing import TrainingDataHandler, TestDataHandler
from wf_psf.training import train
from wf_psf.psf_models import psf_models
from wf_psf.metrics.metrics_interface import evaluate_model
from wf_psf.plotting.plots_interface import plot_metrics
import logging
from wf_psf.utils.io import FileIOHandler
import os
import re
import glob


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
    """Set Config Class.

    A function to select the class of
    a configuration from CONFIG_CLASS dictionary.

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
    except KeyError as e:
        logger.exception("Config name entered is invalid. Check your config settings.")
        exit()

    return config_class


def get_run_config(run_config, config_params, file_handler):
    """Get Run Configuration.

    A function to get the configuration
    for a wf-psf run.

    Parameters
    ----------
    run_config: str
        Name of the type of run configuraton
    config_params: str
        Path of the run configuration file
    file_handler: object
        A class instance of FileIOHandler

    Returns
    -------
    config_class: object
        A class instance of the selected configuration class.

    """
    config_class = set_run_config(run_config)
    return config_class(config_params, file_handler)


class DataConfigHandler:
    """DataConfigHandler.

    A class to handle data configuration
    parameters.

    Parameters
    ----------
    data_conf: str
        Path of the data configuration file
    training_model_params: Recursive Namespace object
        Recursive Namespace object containing the training model parameters

    """

    def __init__(self, data_conf, training_model_params):
        try:
            self.data_conf = read_conf(data_conf)
        except FileNotFoundError as e:
            logger.exception(e)
            exit()
        except TypeError as e:
            logger.exception(e)
            exit()

        self.simPSF = psf_models.simPSF(training_model_params)
        self.training_data = TrainingDataHandler(
            self.data_conf.data.training,
            self.simPSF,
            training_model_params.n_bins_lda,
        )
        self.test_data = TestDataHandler(
            self.data_conf.data.test,
            self.simPSF,
            training_model_params.n_bins_lda,
        )


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
        )
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

        A function to run wave-diff according to the
        input configuration.

        """
        train.train(
            self.training_conf.training,
            self.data_conf.training_data,
            self.data_conf.test_data,
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
        self.metrics_conf = read_conf(metrics_conf)
        self.metrics_dir = file_handler.get_metrics_dir(file_handler._run_output_dir)
        self.file_handler = file_handler

        if training_conf is None:
            try:
                self.training_conf = read_conf(
                    os.path.join(
                        self.metrics_conf.metrics.trained_model_path,
                        self.metrics_conf.metrics.trained_model_config,
                    )
                )
            except TypeError:
                logger.info(
                    "Check metrics config file trained model path or config values are empty."
                )
                exit()

            self.trained_model_workdir = os.path.join(
                self.metrics_conf.metrics.trained_model_path
            )

        else:
            self.training_conf = training_conf

            self.trained_model_workdir = os.path.join(
                self.file_handler.output_path,
                self.file_handler.parent_output_dir,
                self.file_handler.workdir,
            )

        self.plotting_conf = self.metrics_conf.metrics.plotting_config

        try:
            self.data_conf = DataConfigHandler(
                os.path.join(
                    self.file_handler.config_path,
                    self.training_conf.training.data_config,
                ),
                self.training_conf.training.model_params,
            )
        except TypeError as e:
            logger.exception(e)

        self.psf_model = psf_models.get_psf_model(
            self.training_conf.training.model_params,
            self.training_conf.training.training_hparams,
        )
        self.weights_path = self.get_psf_model_weights_filepath()

    def get_psf_model_weights_filepath(self):
        """Get PSF model weights filepath.

        A function to return the basename of the user-specified psf model weights path.

        Returns
        -------
            weights_filepath: str
                The basename of the psf model weights to be loaded.

        """
        weights_basename = (
            self.metrics_conf.metrics.model_save_path
            + "*_"
            + self.training_conf.training.model_params.model_name
            + "*"
            + self.training_conf.training.id_name
            + "_cycle"
            + str(self.metrics_conf.metrics.saved_training_cycle)
            + "*"
        )
        try:
            weights_filepath = glob.glob(
                os.path.join(
                    self.trained_model_workdir,
                    self.metrics_conf.metrics.model_save_path,
                    weights_basename,
                )
            )[0].split(".")[0]
        except IndexError:
            logger.error(
                "PSF weights file not found. Check that you've specified the correct weights file in the metrics config file."
            )
            exit()

        return weights_filepath

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
        self.plotting_conf = os.path.join(
            self.file_handler.config_path,
            self.metrics_conf.metrics.plotting_config,
        )

        plots_config_handler = PlottingConfigHandler(
            self.plotting_conf,
            self.file_handler,
        )

        # Update metrics_confs dict with latest result
        plots_config_handler.metrics_confs[
            self.file_handler.workdir
        ] = self.metrics_conf

        # Update metric results dict with latest result
        plots_config_handler.list_of_metrics_dict[self.file_handler.workdir] = {
            self.training_conf.training.model_params.model_name
            + self.training_conf.training.id_name: [model_metrics]
        }

        plots_config_handler.run()

    def run(self):
        """Run.

        A function to run wave-diff according to the
        input configuration.

        """
        logger.info(
            "Running metrics evaluation on psf model: {}".format(self.weights_path)
        )
        model_metrics = evaluate_model(
            self.metrics_conf.metrics,
            self.training_conf.training,
            self.data_conf.training_data,
            self.data_conf.test_data,
            self.psf_model,
            self.weights_path,
            self.metrics_dir,
        )

        if self.plotting_conf is not None:
            self.file_handler.copy_conffile_to_output_dir(
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
    metrics_conf: dict
        A dictionary containing the metrics configuration parameters
    """

    ids = ("plotting_conf",)

    def __init__(self, plotting_conf, file_handler, metrics_conf=dict()):
        self.plotting_conf = read_conf(plotting_conf)
        self.filehandler = file_handler
        self.metrics_confs = {}
        self.check_and_update_metrics_confs()
        self.list_of_metrics_dict = self.make_dict_of_metrics()
        self.plots_dir = file_handler.get_plots_dir(file_handler._run_output_dir)

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
            try:
                self.metrics_confs[wf_dir] = read_conf(
                    os.path.join(
                        self.plotting_conf.plotting_params.metrics_output_path,
                        wf_dir,
                        "config",
                        metrics_conf,
                    )
                )
            except:
                logger.info("Problem with config file.")
                exit()

    def _metrics_run_id_name(self, wf_outdir, metrics_params):
        """Get Metrics Run ID Name.

        A function to generate run id name
        for the metrics of a trained model

        Parameters
        ----------
        metrics_params: RecursiveNamespace Object
            RecursiveNamespace object containing the metrics parameters used to evaluated the trained model.

        Returns
        -------
        metrics_run_id_name: str
            String containing the model name and id of the training run
        """
        try:
            training_conf = read_conf(
                os.path.join(
                    metrics_params.metrics.trained_model_path,
                    metrics_params.metrics.trained_model_config,
                )
            )
        except TypeError:
            logger.info("Trained model path not provided...")
            logger.info(
                "Trying to retrieve training config file from workdir: {}".format(
                    wf_outdir
                )
            )

            training_conf = read_conf(
                glob.glob(
                    os.path.join(
                        self.filehandler.output_path,
                        self.filehandler.parent_output_dir,
                        wf_outdir,
                        "config/training*",
                    )
                )[0]
            )
        except:
            logger.info("File not found.")
            exit()

        id_name = training_conf.training.id_name
        model_name = training_conf.training.model_params.model_name
        return model_name + id_name

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
        logger.info("Attempting to read in trained model config file...")
        for k, v in self.metrics_confs.items():
            run_id_name = self._metrics_run_id_name(k, v)
            output_path = os.path.join(
                self.plotting_conf.plotting_params.metrics_output_path,
                k,
                "metrics",
                "metrics-" + run_id_name + ".npy",
            )

            try:
                metrics_dict[k] = {
                    run_id_name: [np.load(output_path, allow_pickle=True)[()]]
                }
            except FileNotFoundError:
                print(
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
