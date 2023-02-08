"""
:file: wf_psf/run.py

:date: 19/01/23
:author: jpollack

"""
import argparse
from wf_psf.read_config import read_stream, read_conf
from dotenv import load_dotenv
import wf_psf.io as io
from .psf_models import psf_models
import os
import logging.config
import logging
from .training import train

# load .env variables
load_dotenv("./.env")

repodir = os.getenv("REPODIR")

parser = argparse.ArgumentParser()

parser.add_argument('--conffile', '-c', type=str, required=True,
                    help="a configuration file containing program settings.")

args = parser.parse_args()
config = vars(args)


logging.config.fileConfig(os.path.join(repodir, 'config/logging.conf'),
                          defaults={"filename": "test.log"},
                          disable_existing_loggers=False)


logger = logging.getLogger("wavediff")

logger.info('#')
logger.info('# Entering wavediff mainMethod()')
logger.info('#')

# make wf-psf output dirs
io.make_wfpsf_file_struct()

configs = read_stream(os.path.join(repodir, config['conffile']))

for conf in configs:

    if hasattr(conf, "training_conf"):
        # load training_conf
        training_params = read_conf(os.path.join(
            repodir, conf.training_conf))
        logger.info(training_params)


train.train(training_params)
