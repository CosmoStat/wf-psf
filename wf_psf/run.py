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
from wf_psf.utils.configs_handler import get_run_config
from wf_psf.data.training_preprocessing import TrainingDataHandler, TestDataHandler
from wf_psf.training import train
from wf_psf.psf_models import psf_models
from wf_psf.metrics.metrics_interface import evaluate_model


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

    configs_path = os.path.dirname(args.conffile)
    configs = read_stream(args.conffile)
    configs_file = os.path.basename(args.conffile)
    file_handler.copy_conffile_to_output_dir(configs_path, configs_file)

    config_types = {
        #       "data_conf": None,
        "training_conf": None,
        "metrics_conf": None,
        "plotting_conf": None,
    }

    for conf in configs:
        for k, v in conf.items():
            try:
                if k in config_types:
                    config_class = get_run_config(
                        k, os.path.join(configs_path, v), file_handler
                    )
                    logger.info(config_class)
                    file_handler.copy_conffile_to_output_dir(configs_path, v)
                    config_class.run()
                else:
                    raise KeyError("Incorrect key values in configs.yaml file.")
            except FileNotFoundError as e:
                logger.exception("Check your config file settings.")
                exit()
            except TypeError as e:
                if v is not None:
                    logger.exception(e)
                    exit()
            except KeyError as e:
                logger.exception(e)
                exit()

    logger.info("#")
    logger.info("# Exiting wavediff mainMethod()")
    logger.info("#")


if __name__ == "__main__":
    mainMethod()
