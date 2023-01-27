"""
:file: wf_psf/run.py

:date: 19/01/23
:author: jpollack

"""
import argparse
from wf_psf.read_config import read_stream, read_conf
import wf_psf.io as io
import os
import logging.config
import logging
from .training import train


parser = argparse.ArgumentParser()

parser.add_argument('--conffile', '-c', type=str, required=True,
                    help="a configuration file containing program settings.")

parser.add_argument('--workspace', '-w', type=str, required=True,
                    help="Workspace directory containing configuration files.")

args = parser.parse_args()
config = vars(args)

logging.config.fileConfig(os.path.join(args.workspace, 'config/logging.conf'),
                          defaults={"filename": "test.log"},
                          disable_existing_loggers=False)

logger = logging.getLogger("wavediff")

logger.info('#')
logger.info('# Entering wavediff mainMethod()')
logger.info('#')

configs = read_stream(os.path.join(args.workspace, config['conffile']))

for conf in configs:
    if hasattr(conf, "env_vars"):
        io.set_env_workdir(conf.env_vars.workdir)
        io.set_env_repodir(conf.env_vars.repodir)
        io.make_wfpsf_file_struct()

    if hasattr(conf, "training_conf"):
        # load training_conf
        training_params = read_conf(os.path.join(
            args.workspace, conf.training_conf))
        logger.info(training_params)

        training_params = train.TrainingParams(training_params.training)
        print(training_params.model_save_file)
        training_params.train()
