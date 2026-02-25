"""PlottingConfigHandler.

A module containing the PlottingConfigHandler class, which is responsible for managing the parameters of the plotting configuration file and running the
plotting of the metrics of a trained PSF model

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>
"""

import os
import glob
import numpy as np
from typing import Any
from wf_psf.utils.configs_handler import ConfigHandler, register_configclass
from wf_psf.utils.read_config import read_conf
from wf_psf.plotting.plots_interface import plot_metrics
import logging

logger = logging.getLogger(__name__)


@register_configclass
class PlottingConfigHandler(ConfigHandler):
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
            run_id_names: list[Any] | None = self._metrics_run_id_name(k, v)

            metrics_dict[k] = []
            for run_id_name in run_id_names:  # pyright: ignore[reportOptionalIterable]
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
