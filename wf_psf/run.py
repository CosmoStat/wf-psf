#!/usr/bin/env python
"""WF_PSF Run.

This module setups the run of the WF_PSF pipeline.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""
import argparse
from wf_psf.read_config import read_stream, read_conf
from dotenv import load_dotenv
from wf_psf.io import FileIOHandler
import os
import logging.config
import logging
from wf_psf.training import train

# load .env variables
load_dotenv("./.env")

# set repo directory
repodir = os.getenv("REPODIR")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--conffile",
    "-c",
    type=str,
    required=True,
    help="a configuration file containing program settings.",
)

parser.add_argument(
    "--outputdir",
    "-o",
    type=str,
    required=True,
    help="the path of the output directory.",
)

args = parser.parse_args()

# make wf-psf output dirs
file_handler = FileIOHandler(args.outputdir)
file_handler.setup_outputs()

logger = logging.getLogger("wavediff")

logger.info("#")
logger.info("# Entering wavediff mainMethod()")
logger.info("#")

configs = read_stream(os.path.join(repodir, args.conffile))

for conf in configs:
    if hasattr(conf, "training_conf"):
        # load training_conf
        training_params = read_conf(os.path.join(repodir, conf.training_conf))
        logger.info(training_params)

train.train(training_params, file_handler)
