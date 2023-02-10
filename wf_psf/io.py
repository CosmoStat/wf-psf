"""
:file: wf_psf/io.py

:date: 18/01/23
:author: jpollack

"""

import pathlib
import os
from dotenv import load_dotenv

load_dotenv("./.env")
_workdir = os.getenv('WORKDIR')
_wf_outputs = "wf-outputs"
_chkp = "chkp"
_log_files = "log-files"
_metrics = "metrics"
_optim_hist = "optim-hist"
_plots = "plots"


def make_wfpsf_file_struct():
    # A function to launch a set of commands
    # to the nested output file structure
    # for the wave-diff psf code.

    _make_wf_outputs_dir()
    _make_chkp_dir()
    _make_logfiles_dir()
    _make_wf_metrics_dir()
    _make_optimhist_dir()
    _make_plots_dir()


def _make_wf_outputs_dir():
    """
    A function to make wf-outputs dir.
    """
    pathlib.Path(os.path.join(_workdir, _wf_outputs)
                 ).mkdir(exist_ok=True)


def _make_chkp_dir():
    pathlib.Path(os.path.join(_workdir, _wf_outputs,
                              _chkp)).mkdir(exist_ok=True)


def _make_logfiles_dir():
    pathlib.Path(os.path.join(_workdir, _wf_outputs,
                              _log_files)).mkdir(exist_ok=True)


def _make_wf_metrics_dir():
    pathlib.Path(os.path.join(_workdir, _wf_outputs,
                              _metrics)).mkdir(exist_ok=True)


def _make_optimhist_dir():
    pathlib.Path(os.path.join(_workdir, _wf_outputs,
                 _optim_hist)).mkdir(exist_ok=True)


def _make_plots_dir():
    pathlib.Path(os.path.join(_workdir, _wf_outputs,
                              _plots)).mkdir(exist_ok=True)


def _get_log_save_file():
    pass


def get_model_save_file():
    return os.path.join(_workdir, _chkp)


def get_optim_hist_file():
    workdir = os.getenv('WORKDIR')
    return os.path.join(_workdir, _optim_hist)


if __name__ == "__main__":
    make_wfpsf_file_struct()
    print(get_model_save_file())
