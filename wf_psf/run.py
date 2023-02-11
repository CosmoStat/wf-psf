#!/usr/bin/env python
"""WF_PSF Run

This module setups the run of the WF_PSF pipeline.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""
import argparse
from wf_psf.read_config import read_stream, read_conf
from dotenv import load_dotenv
import wf_psf.io as io
import os
import logging.config
import logging
from wf_psf.training import train

# load .env variables
load_dotenv("./.env")

repodir = os.getenv("REPODIR")
# make wf-psf output dirs
io.setup_outputs()


parser = argparse.ArgumentParser()

parser.add_argument(
    "--conffile",
    "-c",
    type=str,
    required=True,
    help="a configuration file containing program settings.",
)

args = parser.parse_args()
config = vars(args)

logger = logging.getLogger("wavediff")

logger.info("#")
logger.info("# Entering wavediff mainMethod()")
logger.info("#")


configs = read_stream(os.path.join(repodir, config["conffile"]))

for conf in configs:
    if hasattr(conf, "training_conf"):
        # load training_conf
        training_params = read_conf(os.path.join(repodir, conf.training_conf))
        logger.info(training_params)


train.train(training_params)
