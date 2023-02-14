"""IO.

A module which defines methods to
manage wf-psf inputs and outputs.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import pathlib
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

load_dotenv("./.env")

_workdir = os.getenv("WORKDIR")
_wf_outputs = "wf-outputs"
_checkpoint = "checkpoint"
_log_files = "log-files"
_metrics = "metrics"
_optim_hist = "optim-hist"
_plots = "plots"


def setup_outputs():
    """Set up Outputs.

    A function to call
    specific functions
    to set up output
    directories and logging.

    """
    _setup_dirs()
    _setup_logging()


def _setup_dirs():
    """Setup Directories.

    A function to setup the output
    directories.

    """
    _make_output_dir()

    list_of_dirs = (_checkpoint, _log_files, _metrics, _optim_hist, _plots)
    for dir in list_of_dirs:
        _make_dir(dir)


def _setup_logging():
    """Setup Logger.

    A function to set up
    logging.

    """
    repodir = os.getenv("REPODIR")

    logfile = datetime.now().strftime("train_%Y%m%d%H%M.log")

    logfile = os.path.join(_workdir, _wf_outputs, _log_files, logfile)

    logging.config.fileConfig(
        os.path.join(repodir, "config/logging.conf"),
        defaults={"filename": logfile},
        disable_existing_loggers=False,
    )


def _make_output_dir():
    """Make Output Directory.

    A function to make the
    output directory "wf-outputs"
    inside the directory defined
    by the $WORKDIR environment
    variable.

    """
    pathlib.Path(os.path.join(_workdir, _wf_outputs)).mkdir(exist_ok=True)


def _make_dir(dir_name):
    """Make Directory.

    A function to make a subdirectory
    inside the parent directory "wf-outputs".

    Parameters
    ----------
    dir_name: str
        Name of directory

    """
    pathlib.Path(os.path.join(_workdir, _wf_outputs, dir_name)).mkdir(exist_ok=True)


def get_checkpoint_dir():
    """Get Checkpoint Directory.

    A function that returns path
    of checkpoint directory.

    Returns
    -------
    str
        Absolute path to checkpoint directory

    """
    return os.path.join(_workdir, _checkpoint)


if __name__ == "__main__":
    setup_dirs()
