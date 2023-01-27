"""
:file: wf_psf/io.py

:date: 18/01/23
:author: jpollack

"""

import pathlib
import os


workdir = os.getenv('WORKDIR')
wf_outputs = "wf-outputs"
chkp = "chkp"
log_files = "log-files"
metrics = "metrics"
optim_hist = "optim-hist"
plots = "plots"


def make_wfpsf_file_struct():
    """
    A function to produce the output file structure
    for wave-diff psf code.
    """
    make_wf_outputs()
    make_chkp_dir()
    make_logfiles()
    make_wf_metrics()
    make_optimhist_dir()
    make_plots()


def make_wf_outputs():
    pathlib.Path(os.path.join(workdir, wf_outputs)
                 ).mkdir(exist_ok=True)


def make_chkp_dir():
    pathlib.Path(os.path.join(workdir, wf_outputs, chkp)
                 ).mkdir(exist_ok=True)


def make_logfiles():
    pathlib.Path(os.path.join(workdir, wf_outputs,
                              log_files)).mkdir(exist_ok=True)


def make_wf_metrics():
    pathlib.Path(os.path.join(workdir, wf_outputs, metrics)
                 ).mkdir(exist_ok=True)


def make_optimhist_dir():
    pathlib.Path(os.path.join(workdir, wf_outputs,
                              optim_hist)).mkdir(exist_ok=True)


def make_plots():
    pathlib.Path(os.path.join(workdir, wf_outputs, plots)
                 ).mkdir(exist_ok=True)


def get_log_save_file():
    pass


def get_model_save_file():
    return os.path.join(workdir, chkp)


def get_optim_hist_file():
    return os.path.join(workdir, optim_hist)

    # Define paths -> move to io
  #  log_save_file = args['base_path'] + args['log_folder']
  #  model_save_file = args['base_path'] + args['model_folder']
  #  optim_hist_file = args['base_path'] + args['optim_hist_folder']


if __name__ == "__main__":
    make_wfpsf_file_struct()
    print(get_model_save_file())
