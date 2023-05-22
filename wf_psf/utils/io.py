"""IO.

A module which defines methods to
manage wf-psf inputs and outputs.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import pathlib
import os
import logging
from datetime import datetime


class FileIOHandler:
    """FileIOHandler.

    A class to manage the output
    file structure.

    Parameters
    ----------
    repodir_path: str
        Absolute path to the code repository directory
    output_path: str
        Absolute path to output directory

    """

    def __init__(self, repodir_path, output_path):
        self._repodir_path = repodir_path
        self._output_path = output_path
        self._wf_outputs = "wf-outputs"
        self._checkpoint = "checkpoint"
        self._log_files = "log-files"
        self._metrics = "metrics"
        self._optimizer = "optim-hist"
        self._plots = "plots"

    def setup_outputs(self):
        """Setup Outputs.

        A function to call
        specific functions
        to set up output
        directories and logging.

        Parameters
        ----------
        output_path: str
            Path to output directory

        """
        self._make_output_dir()
        self._setup_dirs()
        self._setup_logging()

    def _make_output_dir(self):
        """Make Output Directory.

        A function to make the
        output directory "wf-outputs".

        """
        pathlib.Path(os.path.join(self._output_path, self._wf_outputs)).mkdir(
            exist_ok=True
        )

    def _setup_dirs(self):
        """Setup Directories.

        A function to setup the output
        directories.

        """
        list_of_dirs = (
            self._checkpoint,
            self._log_files,
            self._metrics,
            self._optimizer,
            self._plots,
        )
        for dir in list_of_dirs:
            self._make_dir(dir)

    def _setup_logging(self):
        """Setup Logger.

        A function to set up
        logging.

        """
        logfile = datetime.now().strftime("train_%Y%m%d%H%M.log")

        logfile = os.path.join(
            self._output_path, self._wf_outputs, self._log_files, logfile
        )

        logging.config.fileConfig(
            os.path.join(self._repodir_path, "config/logging.conf"),
            defaults={"filename": logfile},
            disable_existing_loggers=False,
        )

    def _make_dir(self, dir_name):
        """Make Directory.

        A function to make a subdirectory
        inside the parent directory "wf-outputs".

        Parameters
        ----------
        dir_name: str
            Name of directory

        """
        pathlib.Path(os.path.join(self._output_path, self._wf_outputs, dir_name)).mkdir(
            exist_ok=True
        )

    def get_checkpoint_dir(self):
        """Get Checkpoint Directory.

        A function that returns path
        of checkpoint directory.

        Returns
        -------
        str
            Absolute path to checkpoint directory

        """
        return os.path.join(self._output_path, self._wf_outputs, self._checkpoint)

    def get_optimizer_dir(self):
        """Get Optimizer Directory.

        A function that returns path
        of optimizer directory.

        Returns
        -------
        str
            Absolute path to optimizer directory

        """
        return os.path.join(self._output_path, self._wf_outputs, self._optimizer)

    def get_metrics_dir(self):
        """Get Metrics Directory.

        A function that returns path
        of metrics directory.

        Returns
        -------
        str
            Absolute path to metrics directory

        """
        return os.path.join(self._output_path, self._wf_outputs, self._metrics)
