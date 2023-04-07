"""WF_PSF Run.

This module setups the run of the WF_PSF pipeline.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""
import argparse
from wf_psf.utils.read_config import read_stream, read_conf
from wf_psf.utils.io import FileIOHandler
import os
import logging.config
import logging
from wf_psf.training import train


def setProgramOptions():
    """Define Program Options.

    Set command-line options for
    this program.

    Returns
    -------
    args: type
        Argument Parser Namespace

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--conffile",
        "-c",
        type=str,
        required=True,
        help="a configuration file containing program settings.",
    )

    parser.add_argument(
        "--repodir",
        "-r",
        type=str,
        required=True,
        help="the path of the code repository directory.",
    )

    parser.add_argument(
        "--outputdir",
        "-o",
        type=str,
        required=True,
        help="the path of the output directory.",
    )

    args = parser.parse_args()

    return args


def mainMethod():
    """Main Method.

    The main entry point to wavediff program.


    """

    args = setProgramOptions()

    file_handler = FileIOHandler(args.repodir, args.outputdir)
    file_handler.setup_outputs()

    logger = logging.getLogger("wavediff")

    logger.info("#")
    logger.info("# Entering wavediff mainMethod()")
    logger.info("#")

    configs = read_stream(os.path.join(args.repodir, args.conffile))

    for conf in configs:
        if hasattr(conf, "training_conf"):
            training_params = read_conf(os.path.join(args.repodir, conf.training_conf))
            logger.info(training_params.training)

        if hasattr(conf, "metrics_conf"):
            metrics_params = read_conf(os.path.join(args.repodir, conf.metrics_conf))
            logger.info(metrics_params.metrics)

    try:
        train.train(training_params.training, file_handler)
    except NameError:
        logger.info("Training not correctly set. Please check your config file.")   
        

    train.train(training_params.training, file_handler)

    logger.info("#")
    logger.info("# Exiting wavediff mainMethod()")
    logger.info("#")


if __name__ == "__main__":
    mainMethod()
