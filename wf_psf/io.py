"""
:file: wf_psf/io.py

:date: 18/01/23
:author: jpollack

    "--base_path",
    default="/gpfswork/rech/ynx/ulx23va/wf-outputs/",
    type=str,
    help="Base path for saving files.")
        "--log_folder",
    default="log-files/",
    type=str,
    help="Folder name to save log files.")
@click.option(
    "--model_folder",
    default="chkp/",
    type=str,
    help="Folder name to save trained models.")
@click.option(
    "--optim_hist_folder",
    default="optim-hist/",
    type=str,
    help="Folder name to save optimisation history files.")
# Define paths -> move to io
    log_save_file = args['base_path'] + args['log_folder']
    model_save_file = args['base_path'] + args['model_folder']
    optim_hist_file = args['base_path'] + args['optim_hist_folder']

"""

import pathlib
import os


def set_env_workdir(workdir):
    os.environ['WORKDIR'] = workdir


def set_env_repodir(repodir):
    os.environ['REPODIR'] = repodir


class WFPSF_FileStructure():
    """
    A class to build WF-PSF 
    output file structure.
    """

    def __init__(self):
        self.workdir = os.getenv('WORKDIR')
        self.wf_outputs = "wf-outputs"
        self.chkp = "chkp"
        self.log_files = "log-files"
        # not sure these folders
        # should be created here
        # adding for now
        self.metrics = "metrics"
        self.optim_hist = "optim-hist"
        self.plots = "plots"

    def make_wfpsf_file_struct(self):
        """
        A function to produce the output file structure
        for wave-diff psf code.
        """
        self.make_wf_outputs()
        self.make_chkp_dir()
        self.make_logfiles()
        self.make_wf_metrics()
        self.make_optimhist_dir()
        self.make_plots()

    def make_wf_outputs(self):
        pathlib.Path(os.path.join(self.workdir, self.wf_outputs)
                     ).mkdir(exist_ok=True)

    def make_chkp_dir(self):
        pathlib.Path(os.path.join(self.workdir, self.wf_outputs, self.chkp)
                     ).mkdir(exist_ok=True)

    def make_logfiles(self):
        pathlib.Path(os.path.join(self.workdir, self.wf_outputs,
                                  self.log_files)).mkdir(exist_ok=True)

    def make_wf_metrics(self):
        pathlib.Path(os.path.join(self.workdir, self.wf_outputs, self.metrics)
                     ).mkdir(exist_ok=True)

    def make_optimhist_dir(self):
        pathlib.Path(os.path.join(self.workdir, self.wf_outputs,
                                  self.optim_hist)).mkdir(exist_ok=True)

    def make_plots(self):
        pathlib.Path(os.path.join(self.workdir, self.wf_outputs, self.plots)
                     ).mkdir(exist_ok=True)

def get_log_save_file():
    pass

def get_model_save_file():
    pass

def get_optim_hist_file():
    pass

      # Define paths -> move to io
    log_save_file = args['base_path'] + args['log_folder']
    model_save_file = args['base_path'] + args['model_folder']
    optim_hist_file = args['base_path'] + args['optim_hist_folder']

if __name__ == "__main__":
    make_wfpsf_file_struct()
